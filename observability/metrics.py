"""observability.metrics
=====================================================================
Day 28 — Metrics collector.

Tracks rolling performance metrics every cycle:
  - Equity curve
  - Realised + unrealised PnL
  - Win rate, profit factor
  - Max drawdown
  - Sharpe / Sortino (annualised)
  - Exposure (gross / net / long / short)
  - Per-strategy PnL attribution

Snapshots are appended to `data/metrics.jsonl` so the dashboard
renderer can plot them.
"""
from __future__ import annotations

import json
import math
import os
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("observability.metrics")


@dataclass
class MetricsSnapshot:
    timestamp: str
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    n_open_trades_total: int
    win_rate: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    per_strategy_pnl: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class MetricsCollector:
    def __init__(self, path: str = "data/metrics.jsonl",
                 lookback: int = 1000) -> None:
        self.path = path
        self.lookback = int(lookback)
        self._equity_curve: deque[float] = deque(maxlen=lookback)
        self._trade_pnls: deque[float] = deque(maxlen=lookback)
        self._per_strategy_pnl: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ----------------------------------------------------------------
    def record_equity(self, equity: float) -> None:
        with self._lock:
            self._equity_curve.append(float(equity))

    def record_trade_pnl(self, pnl: float, strategy: str = "") -> None:
        with self._lock:
            self._trade_pnls.append(float(pnl))
            if strategy:
                self._per_strategy_pnl[strategy] += float(pnl)

    # ----------------------------------------------------------------
    def snapshot(self, portfolio_snapshot: dict[str, Any]) -> MetricsSnapshot:
        with self._lock:
            eqs = list(self._equity_curve)
            pnls = list(self._trade_pnls)
            per_strat = dict(self._per_strategy_pnl)

        win_rate = self._win_rate(pnls)
        pf = self._profit_factor(pnls)
        max_dd = self._max_drawdown(eqs)
        sharpe, sortino = self._risk_adj_returns(eqs)

        snap = MetricsSnapshot(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            equity=float(portfolio_snapshot.get("equity", 0.0)),
            realized_pnl=float(portfolio_snapshot.get("realized_pnl", 0.0)),
            unrealized_pnl=float(portfolio_snapshot.get("unrealized_pnl", 0.0)),
            gross_exposure=float(portfolio_snapshot.get("gross_exposure", 0.0)),
            net_exposure=float(portfolio_snapshot.get("net_exposure", 0.0)),
            n_positions=int(portfolio_snapshot.get("n_positions", 0)),
            n_open_trades_total=len(pnls),
            win_rate=win_rate,
            profit_factor=pf,
            max_drawdown_pct=max_dd,
            sharpe=sharpe,
            sortino=sortino,
            per_strategy_pnl=per_strat,
        )
        self._persist(snap)
        return snap

    # ----------------------------------------------------------------
    def _persist(self, snap: MetricsSnapshot) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snap.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("metrics persist failed: %r", e)

    # ----------------------------------------------------------------
    @staticmethod
    def _win_rate(pnls: list[float]) -> float:
        if not pnls:
            return 0.0
        wins = sum(1 for p in pnls if p > 0)
        return float(wins / len(pnls))

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float:
        if not pnls:
            return 0.0
        gross_w = sum(p for p in pnls if p > 0)
        gross_l = -sum(p for p in pnls if p < 0)
        if gross_l <= 0:
            return float("inf") if gross_w > 0 else 0.0
        return float(gross_w / gross_l)

    @staticmethod
    def _max_drawdown(eqs: list[float]) -> float:
        if len(eqs) < 2:
            return 0.0
        arr = np.array(eqs, dtype=float)
        running_max = np.maximum.accumulate(arr)
        dd = (arr - running_max) / np.where(running_max > 0, running_max, 1.0)
        return float(abs(dd.min()) if dd.size else 0.0)

    @staticmethod
    def _risk_adj_returns(eqs: list[float]) -> tuple[float, float]:
        if len(eqs) < 3:
            return 0.0, 0.0
        arr = np.array(eqs, dtype=float)
        rets = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)
        if rets.std() == 0:
            return 0.0, 0.0
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        downside = rets[rets < 0]
        if len(downside) == 0 or downside.std() == 0:
            sortino = float("inf") if rets.mean() > 0 else 0.0
        else:
            sortino = float(rets.mean() / downside.std() * math.sqrt(252))
        return sharpe, sortino
