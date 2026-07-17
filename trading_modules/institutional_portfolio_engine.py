"""trading_modules/institutional_portfolio_engine.py
=====================================================================
Institutional Portfolio Engine (Principle #137, #153, #159)
=====================================================================
Top-level portfolio orchestrator that combines all portfolio intelligence
into a single decision framework.

Combines:
    - PortfolioIntelligenceLayer (correlation, VaR, diversification)
    - PortfolioExposureAnalyzer (currency/sector exposure)
    - RiskBudgetManager (daily/weekly/monthly limits)
    - Strategy diversification (don't put all capital in one strategy)
    - Capital allocation optimizer (Kelly + risk parity)

What It Provides:
    1. PORTFOLIO HEAT MAP — visual representation of risk concentration
    2. CAPITAL ALLOCATION — optimal split across strategies
    3. RISK PARITY — equalize risk contribution per position
    4. STRATEGY DIVERSIFICATION — ensure strategy variety
    5. CAPACITY ANALYSIS — can we scale this strategy?
    6. CORRELATION HEATMAP — pairwise correlations
    7. SCENARIO ANALYSIS — what happens if market drops 5%?

Usage:
    engine = InstitutionalPortfolioEngine(equity=10000)

    # Add positions + strategies
    engine.add_position("BTCUSD", "BUY", 0.5, 43250, strategy="momentum")
    engine.add_position("ETHUSD", "BUY", 3.0, 2580, strategy="trend")

    # Get portfolio report
    report = engine.report()
    # report = {
    #     "portfolio_heat": 0.35,  # 0=safe, 1=extreme
    #     "diversification_score": 0.72,
    #     "strategy_concentration": {"momentum": 0.6, "trend": 0.4},
    #     "capital_allocation": {"momentum": 60%, "trend": 40%},
    #     "risk_parity_weights": {...},
    #     "scenario_loss_5pct": -250.00,
    #     "capacity_remaining": 0.45,
    # }
"""
from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger
from trading_modules.portfolio_exposure_analyzer import (
    PortfolioExposureAnalyzer, classify_symbol,
)

log = get_logger("trading_bot.institutional_portfolio_engine")


@dataclass
class PortfolioReport:
    """Complete institutional portfolio report."""
    # Heat map (0=safe, 1=extreme risk)
    portfolio_heat: float = 0.0

    # Diversification
    diversification_score: float = 0.0     # 0-1
    strategy_diversification: float = 0.0  # 0-1
    effective_positions: float = 0.0

    # Concentration
    strategy_concentration: Dict[str, float] = field(default_factory=dict)
    currency_concentration: Dict[str, float] = field(default_factory=dict)
    sector_concentration: Dict[str, float] = field(default_factory=dict)

    # Capital allocation
    capital_allocation: Dict[str, float] = field(default_factory=dict)  # by strategy
    risk_parity_weights: Dict[str, float] = field(default_factory=dict)  # by symbol
    kelly_optimal: Dict[str, float] = field(default_factory=dict)

    # Scenario analysis
    scenario_loss_5pct: float = 0.0   # estimated loss if market drops 5%
    scenario_loss_10pct: float = 0.0
    scenario_gain_5pct: float = 0.0

    # Capacity
    capacity_remaining_pct: float = 0.0  # how much more capital can we deploy?
    max_strategy_capacity: Dict[str, float] = field(default_factory=dict)

    # Risk metrics
    portfolio_var_95: float = 0.0
    portfolio_beta: float = 0.0
    sharpe_estimate: float = 0.0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    rebalancing_needed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "portfolio_heat": round(self.portfolio_heat, 3),
            "diversification_score": round(self.diversification_score, 3),
            "strategy_diversification": round(self.strategy_diversification, 3),
            "effective_positions": round(self.effective_positions, 2),
            "strategy_concentration": {k: round(v, 3) for k, v in self.strategy_concentration.items()},
            "currency_concentration": self.currency_concentration,
            "sector_concentration": {k: round(v, 3) for k, v in self.sector_concentration.items()},
            "capital_allocation": {k: round(v, 3) for k, v in self.capital_allocation.items()},
            "risk_parity_weights": {k: round(v, 3) for k, v in self.risk_parity_weights.items()},
            "kelly_optimal": {k: round(v, 3) for k, v in self.kelly_optimal.items()},
            "scenario_loss_5pct": round(self.scenario_loss_5pct, 2),
            "scenario_loss_10pct": round(self.scenario_loss_10pct, 2),
            "scenario_gain_5pct": round(self.scenario_gain_5pct, 2),
            "capacity_remaining_pct": round(self.capacity_remaining_pct, 3),
            "portfolio_var_95": round(self.portfolio_var_95, 2),
            "portfolio_beta": round(self.portfolio_beta, 3),
            "sharpe_estimate": round(self.sharpe_estimate, 3),
            "recommendations": self.recommendations,
            "rebalancing_needed": self.rebalancing_needed,
        }


class InstitutionalPortfolioEngine:
    """Top-level portfolio orchestrator."""

    def __init__(self,
                 equity: float = 10000.0,
                 max_portfolio_heat: float = 0.10,
                 max_strategy_concentration: float = 0.50,
                 max_single_position_pct: float = 0.25,
                 target_sharpe: float = 2.0):
        """Initialize portfolio engine.

        Args:
            equity: current account equity
            max_portfolio_heat: max total risk as fraction of equity
            max_strategy_concentration: max % in one strategy
            max_single_position_pct: max % in one position
            target_sharpe: target Sharpe ratio
        """
        self.equity = equity
        self.max_heat = max_portfolio_heat
        self.max_strat_conc = max_strategy_concentration
        self.max_position = max_single_position_pct
        self.target_sharpe = target_sharpe

        self._lock = threading.RLock()
        self._positions: List[dict] = []
        self._exposure_analyzer = PortfolioExposureAnalyzer()
        self._price_history: Dict[str, pd.DataFrame] = {}
        self._strategy_stats: Dict[str, dict] = {}  # strategy → {win_rate, avg_r, ...}

    # ------------------------------------------------------------------
    # Position + strategy management
    # ------------------------------------------------------------------
    def add_position(self, symbol: str, side: str, volume: float,
                     entry_price: float, strategy: str = "",
                     sl: float = 0, current_price: float = 0) -> None:
        """Add a position."""
        with self._lock:
            self._positions.append({
                "symbol": symbol, "side": side.upper(),
                "volume": volume, "entry_price": entry_price,
                "current_price": current_price or entry_price,
                "strategy": strategy, "sl": sl,
                "notional_usd": volume * (current_price or entry_price),
                "risk_usd": abs(entry_price - sl) * volume if sl > 0 else 0,
            })
            self._exposure_analyzer.add_position(symbol, side, volume, entry_price, current_price)

    def remove_position(self, symbol: str) -> None:
        """Remove all positions for a symbol."""
        with self._lock:
            self._positions = [p for p in self._positions if p["symbol"] != symbol]
            self._exposure_analyzer.remove_position(symbol)

    def set_price_history(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Set price history for correlation/scenario analysis."""
        with self._lock:
            self._price_history = dfs

    def set_strategy_stats(self, strategy: str, win_rate: float,
                          avg_win_r: float, avg_loss_r: float = 1.0,
                          sample_size: int = 0) -> None:
        """Set performance stats for a strategy (for Kelly calculation)."""
        with self._lock:
            self._strategy_stats[strategy] = {
                "win_rate": win_rate,
                "avg_win_r": avg_win_r,
                "avg_loss_r": avg_loss_r,
                "sample_size": sample_size,
            }

    def update_equity(self, equity: float) -> None:
        self.equity = equity

    # ------------------------------------------------------------------
    # Generate full report
    # ------------------------------------------------------------------
    def report(self) -> PortfolioReport:
        """Generate complete portfolio report."""
        r = PortfolioReport()

        with self._lock:
            positions = list(self._positions)

        if not positions:
            r.recommendations.append("No open positions")
            return r

        # === Portfolio heat ===
        total_risk = sum(p["risk_usd"] for p in positions)
        r.portfolio_heat = total_risk / max(self.equity, 1)

        # === Diversification ===
        weights = [p["notional_usd"] for p in positions]
        total = sum(weights)
        if total > 0:
            norm_weights = [w / total for w in weights]
            herfindahl = sum(w * w for w in norm_weights)
            r.effective_positions = 1.0 / herfindahl if herfindahl > 0 else 0
            r.diversification_score = max(0, 1 - herfindahl)

        # === Strategy concentration ===
        by_strategy: Dict[str, float] = defaultdict(float)
        for p in positions:
            by_strategy[p["strategy"] or "unknown"] += p["notional_usd"]
        total_notional = sum(by_strategy.values())
        r.strategy_concentration = {
            s: v / max(total_notional, 1) for s, v in by_strategy.items()
        }

        # Strategy diversification (Herfindahl of strategy weights)
        if r.strategy_concentration:
            strat_h = sum(w * w for w in r.strategy_concentration.values())
            r.strategy_diversification = max(0, 1 - strat_h)

        # === Currency + sector concentration ===
        exposure_report = self._exposure_analyzer.analyze()
        r.currency_concentration = exposure_report.currency_exposure
        r.sector_concentration = {
            k: v / max(total_notional, 1)
            for k, v in exposure_report.asset_class_exposure.items()
        }

        # === Capital allocation (by strategy) ===
        r.capital_allocation = r.strategy_concentration.copy()

        # === Risk parity weights ===
        r.risk_parity_weights = self._compute_risk_parity(positions)

        # === Kelly optimal ===
        r.kelly_optimal = self._compute_kelly_weights()

        # === Scenario analysis ===
        r.scenario_loss_5pct, r.scenario_loss_10pct, r.scenario_gain_5pct = \
            self._scenario_analysis(positions)

        # === Capacity ===
        r.capacity_remaining_pct = max(0, 1 - r.portfolio_heat / self.max_heat)
        r.max_strategy_capacity = {
            s: self.max_strat_conc - w
            for s, w in r.strategy_concentration.items()
            if w < self.max_strat_conc
        }

        # === VaR + Beta ===
        r.portfolio_var_95 = self._compute_var(positions)
        r.portfolio_beta = self._compute_beta(positions)

        # === Sharpe estimate ===
        r.sharpe_estimate = self._estimate_sharpe(r)

        # === Recommendations ===
        r.recommendations = self._recommend(r)
        r.rebalancing_needed = any("REBALANCE" in rec for rec in r.recommendations)

        return r

    # ------------------------------------------------------------------
    # Computations
    # ------------------------------------------------------------------
    def _compute_risk_parity(self, positions: list) -> Dict[str, float]:
        """Compute risk parity weights (equalize risk contribution)."""
        risks = {p["symbol"]: max(p["risk_usd"], 1) for p in positions}
        total_risk = sum(risks.values())
        if total_risk == 0:
            return {p["symbol"]: 1.0 / len(positions) for p in positions}
        # Inverse risk weighting
        inv_risks = {s: 1.0 / r for s, r in risks.items()}
        total_inv = sum(inv_risks.values())
        return {s: r / total_inv for s, r in inv_risks.items()}

    def _compute_kelly_weights(self) -> Dict[str, float]:
        """Compute Kelly-optimal weights per strategy."""
        kelly = {}
        for strat, stats in self._strategy_stats.items():
            wr = stats["win_rate"]
            b = stats["avg_win_r"] / max(stats["avg_loss_r"], 1e-10)
            kelly_fraction = (b * wr - (1 - wr)) / max(b, 1e-10)
            # Quarter Kelly for safety
            kelly[strat] = max(0, min(0.25, kelly_fraction * 0.25))
        return kelly

    def _scenario_analysis(self, positions: list) -> Tuple[float, float, float]:
        """Estimate P&L under market scenarios.

        Returns (loss_5pct, loss_10pct, gain_5pct)
        """
        loss_5 = 0.0
        loss_10 = 0.0
        gain_5 = 0.0

        for p in positions:
            notional = p["notional_usd"]
            direction = 1 if p["side"] == "BUY" else -1
            # If market drops 5%, longs lose 5%, shorts gain 5%
            loss_5 += notional * (-0.05) * direction
            loss_10 += notional * (-0.10) * direction
            gain_5 += notional * 0.05 * direction

        return loss_5, loss_10, gain_5

    def _compute_var(self, positions: list) -> float:
        """Compute portfolio VaR (simplified)."""
        if not self._price_history or not positions:
            return 0.0
        # Sum position volatilities (ignoring correlation for simplicity)
        total_var = 0.0
        for p in positions:
            df = self._price_history.get(p["symbol"])
            if df is not None and not df.empty and "close" in df:
                returns = df["close"].pct_change().dropna().tail(100)
                if len(returns) > 20:
                    vol = float(returns.std())
                    total_var += (p["notional_usd"] * vol * 1.645) ** 2  # 95% VaR
        return float(np.sqrt(total_var))

    def _compute_beta(self, positions: list) -> float:
        """Compute portfolio beta (simplified)."""
        # Without benchmark, assume beta = 1 for crypto, 0.5 for forex
        total_weighted_beta = 0.0
        total_notional = sum(p["notional_usd"] for p in positions)
        if total_notional == 0:
            return 0.0
        for p in positions:
            asset_class = classify_symbol(p["symbol"])
            beta = 1.0 if asset_class == "crypto" else 0.5 if asset_class == "forex" else 0.7
            weight = p["notional_usd"] / total_notional
            direction = 1 if p["side"] == "BUY" else -1
            total_weighted_beta += beta * weight * direction
        return total_weighted_beta

    def _estimate_sharpe(self, r: PortfolioReport) -> float:
        """Estimate portfolio Sharpe from diversification + strategies."""
        # Simplified: better diversification → higher Sharpe
        diversification_bonus = r.diversification_score * 0.5
        # Average strategy Sharpe (if available)
        strat_sharpes = []
        for s, stats in self._strategy_stats.items():
            if stats["sample_size"] >= 10:
                # Rough Sharpe from win rate + R
                wr = stats["win_rate"]
                avg_r = stats["avg_win_r"] * wr - stats["avg_loss_r"] * (1 - wr)
                strat_sharpes.append(avg_r * 2)  # rough scaling
        avg_strat_sharpe = sum(strat_sharpes) / max(len(strat_sharpes), 1) if strat_sharpes else 1.0
        return avg_strat_sharpe * (1 + diversification_bonus)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def _recommend(self, r: PortfolioReport) -> List[str]:
        """Generate portfolio recommendations."""
        recs = []
        if r.portfolio_heat > self.max_heat:
            recs.append(f"REBALANCE: Portfolio heat {r.portfolio_heat:.1%} > limit {self.max_heat:.1%} — reduce positions")
        if r.diversification_score < 0.3:
            recs.append("REBALANCE: Poor diversification — add uncorrelated positions")
        for s, w in r.strategy_concentration.items():
            if w > self.max_strat_conc:
                recs.append(f"REBALANCE: Strategy '{s}' at {w:.0%} > limit {self.max_strat_conc:.0%}")
        if r.scenario_loss_5pct < -self.equity * 0.05:
            recs.append(f"REBALANCE: 5% market drop would cost ${abs(r.scenario_loss_5pct):.0f} — hedge or reduce")
        if r.strategy_diversification < 0.4:
            recs.append("Add strategy diversity — currently too concentrated in one approach")
        if not recs:
            recs.append("Portfolio balanced — no rebalancing needed")
        return recs
