"""architecture/simulation_layer.py
=====================================================================
Simulation Layer (Improvement #19)
=====================================================================
Allows the bot to run "what-if" scenarios in parallel with live
trading — testing new strategies, parameters, and risk models on
real-time data without risking capital.

Three Simulation Modes:

1. SHADOW MODE
   New strategy variant runs in parallel with the live strategy.
   Generates signals but doesn't execute. Logs what WOULD have
   happened. After N trades, compare to live strategy and decide
   whether to promote.

2. MONTE CARLO MODE
   Take the current strategy + historical trades, run 10,000
   randomized orderings to compute:
   - Probability of ruin (account goes to 0)
   - 95% confidence interval for drawdown
   - Expected max drawdown
   - Risk of consecutive losses

3. STRESS TEST MODE
   Replay historical crisis events (March 2020 COVID crash, May 2021
   China ban, Nov 2022 FTX collapse) against the current strategy +
   portfolio to estimate survival.

Usage:
    sim = SimulationLayer()
    sim.start_shadow("aggressive_momentum_v2", strategy_fn=...)
    # After 100 shadow trades:
    stats = sim.shadow_stats("aggressive_momentum_v2")
    if stats["sharpe"] > live_sharpe:
        sim.promote_shadow("aggressive_momentum_v2")

    # Monte Carlo
    mc = sim.monte_carlo(historical_trades, n_runs=10000)
    print(f"Risk of ruin: {mc['ruin_probability']:.1%}")
    print(f"95% max DD: {mc['max_drawdown_95']:.2f}%")

    # Stress test
    stress = sim.stress_test(portfolio, scenario="covid_crash_2020")
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.simulation")


@dataclass
class ShadowTrade:
    """A trade taken in shadow mode (no real execution)."""
    timestamp: str = ""
    symbol: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lots: float = 0.0
    pnl: float = 0.0
    r_multiple: float = 0.0
    hold_time_s: float = 0.0
    open: bool = True


@dataclass
class ShadowStats:
    """Performance stats for a shadow strategy."""
    name: str = ""
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_r: float = 0.0


@dataclass
class MonteCarloResult:
    """Results of a Monte Carlo simulation."""
    n_runs: int = 0
    ruin_probability: float = 0.0      # P(account → 0)
    median_final_equity: float = 0.0
    p05_final_equity: float = 0.0      # 5th percentile (worst case)
    p95_final_equity: float = 0.0      # 95th percentile (best case)
    median_max_drawdown_pct: float = 0.0
    max_drawdown_95: float = 0.0       # 95th percentile max DD
    expected_max_consecutive_losses: int = 0
    median_sharpe: float = 0.0


class SimulationLayer:
    """Runs shadow strategies, Monte Carlo, and stress tests."""

    # Historical crisis scenarios for stress testing
    STRESS_SCENARIOS = {
        "covid_crash_2020": {
            "description": "March 2020 COVID crash",
            "btc_drop_pct": -50.0,
            "duration_days": 7,
            "max_daily_drop_pct": -40.0,
            "volatility_multiplier": 4.0,
        },
        "china_ban_2021": {
            "description": "May 2021 China crypto ban",
            "btc_drop_pct": -35.0,
            "duration_days": 14,
            "max_daily_drop_pct": -30.0,
            "volatility_multiplier": 3.0,
        },
        "ftx_collapse_2022": {
            "description": "November 2022 FTX collapse",
            "btc_drop_pct": -25.0,
            "duration_days": 5,
            "max_daily_drop_pct": -15.0,
            "volatility_multiplier": 2.5,
        },
        "luna_crash_2022": {
            "description": "May 2022 Luna/UST death spiral",
            "btc_drop_pct": -20.0,
            "duration_days": 3,
            "max_daily_drop_pct": -20.0,
            "volatility_multiplier": 3.5,
        },
    }

    def __init__(self,
                 bus: Optional[EventBus] = None,
                 shadow_db_path: str = "data/shadow_trades.db"):
        self._lock = threading.RLock()
        self._bus = bus or get_bus()
        # Shadow strategies in flight
        self._shadow_strategies: Dict[str, Dict[str, Any]] = {}
        # Open shadow positions: name -> list of ShadowTrade
        self._shadow_open: Dict[str, List[ShadowTrade]] = {}
        # Closed shadow trades: name -> list
        self._shadow_closed: Dict[str, List[ShadowTrade]] = {}

    # ------------------------------------------------------------------
    # SHADOW MODE
    # ------------------------------------------------------------------
    def start_shadow(self,
                     name: str,
                     strategy_fn: Optional[Callable] = None,
                     config: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new shadow strategy to run in parallel."""
        with self._lock:
            if name in self._shadow_strategies:
                log.warning("sim: shadow %s already exists", name)
                return False
            self._shadow_strategies[name] = {
                "strategy_fn": strategy_fn,
                "config": config or {},
                "started_at": time.time(),
                "trade_count": 0,
            }
            self._shadow_open[name] = []
            self._shadow_closed[name] = []
        log.info("sim: shadow strategy '%s' started", name)
        return True

    def shadow_signal(self,
                      name: str,
                      symbol: str,
                      direction: str,
                      entry_price: float,
                      sl: float,
                      tp: float,
                      lots: float) -> bool:
        """Record a signal from a shadow strategy (open a shadow trade)."""
        with self._lock:
            if name not in self._shadow_strategies:
                return False
            trade = ShadowTrade(
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                sl=sl, tp=tp, lots=lots,
            )
            self._shadow_open[name].append(trade)
            self._shadow_strategies[name]["trade_count"] += 1
        return True

    def shadow_update_prices(self, prices: Dict[str, float]) -> None:
        """Update prices for shadow positions, check SL/TP, close if hit."""
        with self._lock:
            for name, opens in self._shadow_open.items():
                still_open = []
                for t in opens:
                    if t.symbol not in prices:
                        still_open.append(t)
                        continue
                    price = prices[t.symbol]
                    # Check SL/TP
                    hit_sl = (t.direction == "BUY" and price <= t.sl) or \
                            (t.direction == "SELL" and price >= t.sl)
                    hit_tp = (t.direction == "BUY" and price >= t.tp) or \
                            (t.direction == "SELL" and price <= t.tp)
                    if hit_sl or hit_tp:
                        t.exit_price = price
                        t.open = False
                        direction = 1 if t.direction == "BUY" else -1
                        t.pnl = (price - t.entry_price) * t.lots * direction
                        risk = abs(t.entry_price - t.sl) * t.lots
                        t.r_multiple = t.pnl / risk if risk > 0 else 0
                        self._shadow_closed[name].append(t)
                    else:
                        still_open.append(t)
                self._shadow_open[name] = still_open

    def shadow_stats(self, name: str) -> Optional[ShadowStats]:
        with self._lock:
            closed = self._shadow_closed.get(name, [])
            if not closed:
                return None
            pnls = [t.pnl for t in closed]
            rs = [t.r_multiple for t in closed]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            # Sharpe (per-trade, not annualized)
            arr = np.array(pnls)
            sharpe = float(arr.mean() / arr.std()) if arr.std() > 0 else 0.0
            # Max drawdown
            equity = np.cumsum(pnls)
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity)
            max_dd_pct = float((dd.max() / max(peak.max(), 1)) * 100) if len(peak) > 0 else 0
            return ShadowStats(
                name=name,
                trade_count=len(closed),
                wins=len(wins), losses=len(losses),
                win_rate=len(wins) / max(len(closed), 1),
                total_pnl=sum(pnls),
                avg_pnl=sum(pnls) / len(pnls),
                sharpe=sharpe,
                max_drawdown_pct=max_dd_pct,
                profit_factor=sum(wins) / max(abs(sum(losses)), 0.01),
                avg_r=sum(rs) / len(rs),
            )

    def promote_shadow(self, name: str) -> bool:
        """Promote a shadow strategy to live (caller must wire it in)."""
        stats = self.shadow_stats(name)
        if stats is None or stats.trade_count < 20:
            log.warning("sim: cannot promote %s — insufficient trades", name)
            return False
        log.info("sim: PROMOTING %s to live (Sharpe=%.2f, WR=%.1f%%, %d trades)",
                 name, stats.sharpe, stats.win_rate * 100, stats.trade_count)
        return True

    def stop_shadow(self, name: str) -> None:
        with self._lock:
            self._shadow_strategies.pop(name, None)
            self._shadow_open.pop(name, None)
            self._shadow_closed.pop(name, None)
        log.info("sim: shadow strategy '%s' stopped", name)

    # ------------------------------------------------------------------
    # MONTE CARLO
    # ------------------------------------------------------------------
    def monte_carlo(self,
                    historical_trades: List[Dict[str, Any]],
                    n_runs: int = 10000,
                    initial_capital: float = 10000.0,
                    risk_per_trade: float = 0.02,
                    ruin_threshold_pct: float = 0.5) -> MonteCarloResult:
        """Run Monte Carlo simulation on historical trade outcomes.

        historical_trades: list of {"pnl": float, "r_multiple": float}
        n_runs: number of randomized orderings to simulate
        ruin_threshold_pct: equity below this % of initial = ruin
        """
        if not historical_trades:
            return MonteCarloResult()

        pnls = np.array([t.get("pnl", 0) for t in historical_trades])
        n_trades = len(pnls)
        ruin_threshold = initial_capital * ruin_threshold_pct

        final_equities = np.zeros(n_runs)
        max_dds = np.zeros(n_runs)
        max_consec_losses = np.zeros(n_runs, dtype=int)
        ruined = 0

        for run in range(n_runs):
            # Random permutation of trade outcomes
            perm = np.random.permutation(pnls)
            equity = initial_capital
            peak = initial_capital
            max_dd = 0.0
            consec_losses = 0
            max_consec = 0
            ruined_run = False
            for pnl in perm:
                # Position-size relative to current equity
                sized_pnl = pnl * (equity / initial_capital) if initial_capital > 0 else 0
                equity += sized_pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / max(peak, 1) * 100
                if dd > max_dd:
                    max_dd = dd
                if pnl < 0:
                    consec_losses += 1
                    if consec_losses > max_consec:
                        max_consec = consec_losses
                else:
                    consec_losses = 0
                if equity < ruin_threshold:
                    ruined_run = True
                    break
            final_equities[run] = equity
            max_dds[run] = max_dd
            max_consec_losses[run] = max_consec
            if ruined_run:
                ruined += 1

        # Per-run Sharpe (annualized assumption: 252*24*4 = 24,192 bars/year at 15min)
        return MonteCarloResult(
            n_runs=n_runs,
            ruin_probability=ruined / n_runs,
            median_final_equity=float(np.median(final_equities)),
            p05_final_equity=float(np.percentile(final_equities, 5)),
            p95_final_equity=float(np.percentile(final_equities, 95)),
            median_max_drawdown_pct=float(np.median(max_dds)),
            max_drawdown_95=float(np.percentile(max_dds, 95)),
            expected_max_consecutive_losses=int(np.median(max_consec_losses)),
            median_sharpe=0.0,  # simplified — would need returns series
        )

    # ------------------------------------------------------------------
    # STRESS TEST
    # ------------------------------------------------------------------
    def stress_test(self,
                    portfolio: Any,
                    scenario: str = "covid_crash_2020",
                    initial_equity: float = 10000.0) -> Dict[str, Any]:
        """Stress test the current portfolio against a historical crisis."""
        sc = self.STRESS_SCENARIOS.get(scenario)
        if sc is None:
            return {"error": f"unknown scenario {scenario}"}

        # Get current open positions
        positions = []
        if hasattr(portfolio, "all_positions"):
            positions = portfolio.all_positions()
        elif isinstance(portfolio, list):
            positions = portfolio

        # Apply the shock: every long position loses X%, every short gains X%
        shock_pct = sc["btc_drop_pct"] / 100.0
        vol_mult = sc["volatility_multiplier"]
        total_pnl = 0.0
        for p in positions:
            notional = p.get("volume", 0) * p.get("current_price", 0)
            direction = 1 if p.get("side", "BUY").upper() == "BUY" else -1
            # Simple model: position loses shock_pct * vol_mult
            position_pnl = notional * shock_pct * vol_mult * direction * -1
            total_pnl += position_pnl

        stressed_equity = initial_equity + total_pnl
        drawdown = (initial_equity - stressed_equity) / max(initial_equity, 1) * 100

        result = {
            "scenario": scenario,
            "description": sc["description"],
            "initial_equity": initial_equity,
            "stressed_equity": stressed_equity,
            "drawdown_pct": drawdown,
            "would_survive": stressed_equity > 0,
            "positions_stressed": len(positions),
            "volatility_multiplier": vol_mult,
        }
        log.info("sim: stress test %s — DD=%.2f%%, survived=%s",
                 scenario, drawdown, result["would_survive"])
        return result

    def list_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {"name": k, **v}
            for k, v in self.STRESS_SCENARIOS.items()
        ]

    # ------------------------------------------------------------------
    # Stats summary
    # ------------------------------------------------------------------
    def all_shadow_stats(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for name in self._shadow_strategies:
                s = self.shadow_stats(name)
                if s is not None:
                    out.append({
                        "name": s.name, "trade_count": s.trade_count,
                        "win_rate": s.win_rate, "sharpe": s.sharpe,
                        "total_pnl": s.total_pnl,
                        "max_drawdown_pct": s.max_drawdown_pct,
                        "profit_factor": s.profit_factor,
                    })
            return out
