"""monitoring.trading_monitor
=====================================================================
Day 72 — Trading-layer monitor.

Tracks PnL + exposure per strategy + per symbol:
  - Per-strategy realised + unrealised PnL
  - Drawdown curves per strategy
  - Exposure heatmaps (per symbol, per side)
  - Win rate, profit factor, average trade duration
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger("monitoring.trading")


@dataclass
class StrategyPnLAttribution:
    strategy_name: str
    realized_pnl: float
    unrealized_pnl: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    profit_factor: float
    avg_pnl: float
    max_drawdown_pct: float
    current_consecutive_losses: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass
class TradingHealth:
    status: str
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_open_trades: int
    per_strategy: dict[str, dict[str, Any]] = field(default_factory=dict)
    exposure_heatmap: dict[str, dict[str, float]] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "total_realized_pnl": self.total_realized_pnl,
            "total_unrealized_pnl": self.total_unrealized_pnl,
            "total_open_trades": self.total_open_trades,
            "per_strategy": dict(self.per_strategy),
            "exposure_heatmap": dict(self.exposure_heatmap),
            "issues": list(self.issues),
        }


# ----------------------------------------------------------------------
class TradingMonitor:
    def __init__(self) -> None:
        # Per-strategy trade PnL history (for drawdown + win rate)
        self._trade_pnls: dict[str, list[float]] = defaultdict(list)
        self._realized: dict[str, float] = defaultdict(float)
        self._unrealized: dict[str, float] = defaultdict(float)
        self._open_counts: dict[str, int] = defaultdict(int)
        self._consecutive_losses: dict[str, int] = defaultdict(int)
        # Per-symbol exposure: {symbol: {"long": lots, "short": lots}}
        self._exposure: dict[str, dict[str, float]] = defaultdict(
            lambda: {"long": 0.0, "short": 0.0}
        )

    # ----------------------------------------------------------------
    def record_trade_close(self, strategy_name: str, pnl: float,
                           symbol: str, side: str, lots: float) -> None:
        self._trade_pnls[strategy_name].append(float(pnl))
        self._realized[strategy_name] += float(pnl)
        if pnl < 0:
            self._consecutive_losses[strategy_name] += 1
        else:
            self._consecutive_losses[strategy_name] = 0
        # Update exposure (position closed)
        side_key = "long" if side == "long" else "short"
        self._exposure[symbol][side_key] = max(0.0,
            self._exposure[symbol][side_key] - lots)

    def record_trade_open(self, strategy_name: str, symbol: str,
                          side: str, lots: float) -> None:
        self._open_counts[strategy_name] += 1
        side_key = "long" if side == "long" else "short"
        self._exposure[symbol][side_key] += lots

    def update_unrealized(self, strategy_name: str, unrealized: float) -> None:
        self._unrealized[strategy_name] = float(unrealized)

    # ----------------------------------------------------------------
    def health(self) -> TradingHealth:
        per_strategy: dict[str, dict[str, Any]] = {}
        issues: list[str] = []
        total_real = 0.0
        total_unreal = 0.0
        total_open = 0
        for name, pnls in self._trade_pnls.items():
            arr = np.array(pnls) if pnls else np.array([0.0])
            n_trades = len(pnls)
            n_wins = int((arr > 0).sum())
            n_losses = int((arr <= 0).sum())
            win_rate = float(n_wins / n_trades) if n_trades else 0.0
            gross_w = float(arr[arr > 0].sum())
            gross_l = float(-arr[arr < 0].sum())
            pf = (gross_w / gross_l) if gross_l > 0 else float("inf") if gross_w > 0 else 0.0
            avg = float(arr.mean()) if n_trades else 0.0
            # Max drawdown
            # Major #3 fix: the old code divided by `abs(arr.mean()) * 100 + 1`
            # which is not a standard drawdown measure. The correct method:
            # compute cumulative PnL, running maximum, then drawdown = cum - peak.
            # The drawdown is reported as the absolute dollar amount of the
            # largest peak-to-trough decline in cumulative PnL.
            cum = np.cumsum(arr)
            running_max = np.maximum.accumulate(cum)
            dd = (cum - running_max)
            if dd.size > 0:
                max_dd_pct = float(abs(dd.min()))
            else:
                max_dd_pct = 0.0

            attribution = StrategyPnLAttribution(
                strategy_name=name,
                realized_pnl=self._realized[name],
                unrealized_pnl=self._unrealized[name],
                n_trades=n_trades,
                n_wins=n_wins,
                n_losses=n_losses,
                win_rate=win_rate,
                profit_factor=pf,
                avg_pnl=avg,
                max_drawdown_pct=max_dd_pct,
                current_consecutive_losses=self._consecutive_losses[name],
            )
            per_strategy[name] = attribution.to_dict()
            total_real += self._realized[name]
            total_unreal += self._unrealized[name]
            total_open += self._open_counts[name]
            # Flag issues
            if self._consecutive_losses[name] >= 5:
                issues.append(f"{name}: {self._consecutive_losses[name]} consecutive losses")
            if win_rate < 0.30 and n_trades >= 10:
                issues.append(f"{name}: win_rate {win_rate:.0%} < 30%")
            if pf < 1.0 and n_trades >= 10:
                issues.append(f"{name}: profit factor {pf:.2f} < 1.0")

        status = "ok"
        if issues:
            status = "degraded" if len(issues) <= 2 else "critical"

        return TradingHealth(
            status=status,
            total_realized_pnl=total_real,
            total_unrealized_pnl=total_unreal,
            total_open_trades=total_open,
            per_strategy=per_strategy,
            exposure_heatmap={s: dict(v) for s, v in self._exposure.items()},
            issues=issues,
        )
