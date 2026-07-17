"""factory.decay_detector
=====================================================================
Day 64-66 — Decay Detector.

Most strategies die slowly. Without explicit drift detection, you'll
think a strategy works for weeks after its edge has actually eroded.

This module tracks rolling performance for each strategy and computes:
  - Sharpe decay        : recent_sharpe / baseline_sharpe
  - Hit-rate decay      : recent_win_rate - baseline_win_rate
  - Volatility expansion: recent_vol / baseline_vol
  - Signal frequency    : signals per cycle (drops = stale strategy)

A composite `decay_score` in [0, 1] summarises the drift:
  1.0 = no decay
  0.0 = severe decay

When `decay_score` drops below `retirement_threshold`, the auto-retirement
module disables the strategy.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("factory.decay")


@dataclass
class DecayReport:
    strategy_name: str
    decay_score: float           # 1.0 = healthy, 0.0 = dead
    sharpe_baseline: float
    sharpe_recent: float
    win_rate_baseline: float
    win_rate_recent: float
    vol_baseline: float
    vol_recent: float
    signal_frequency_baseline: float
    signal_frequency_recent: float
    components: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""     # "ok" | "watch" | "retire"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "decay_score": self.decay_score,
            "sharpe_baseline": self.sharpe_baseline,
            "sharpe_recent": self.sharpe_recent,
            "win_rate_baseline": self.win_rate_baseline,
            "win_rate_recent": self.win_rate_recent,
            "vol_baseline": self.vol_baseline,
            "vol_recent": self.vol_recent,
            "signal_frequency_baseline": self.signal_frequency_baseline,
            "signal_frequency_recent": self.signal_frequency_recent,
            "components": dict(self.components),
            "recommendation": self.recommendation,
        }


# ----------------------------------------------------------------------
class DecayDetector:
    def __init__(self,
                 baseline_window: int = 500,
                 recent_window: int = 100,
                 retirement_threshold: float = 0.4,
                 watch_threshold: float = 0.65) -> None:
        self.baseline_window = int(baseline_window)
        self.recent_window = int(recent_window)
        self.retirement_threshold = float(retirement_threshold)
        self.watch_threshold = float(watch_threshold)
        # Per-strategy rolling trade PnLs + signal counts
        self._pnls: dict[str, deque] = {}
        self._signals: dict[str, deque] = {}  # 1.0 if signal fired, 0.0 otherwise

    # ----------------------------------------------------------------
    def record_trade_pnl(self, strategy_name: str, pnl: float) -> None:
        d = self._pnls.setdefault(strategy_name,
                                  deque(maxlen=self.baseline_window))
        d.append(float(pnl))

    def record_cycle(self, strategy_name: str, signal_fired: bool) -> None:
        d = self._signals.setdefault(strategy_name,
                                     deque(maxlen=self.baseline_window))
        d.append(1.0 if signal_fired else 0.0)

    # ----------------------------------------------------------------
    def evaluate(self, strategy_name: str) -> Optional[DecayReport]:
        pnls = self._pnls.get(strategy_name)
        sigs = self._signals.get(strategy_name)
        if pnls is None or len(pnls) < self.recent_window:
            return None
        pnls_arr = np.array(pnls)
        sigs_arr = np.array(sigs) if sigs else np.array([])

        baseline = pnls_arr[:-self.recent_window] if len(pnls_arr) > self.recent_window else pnls_arr
        recent = pnls_arr[-self.recent_window:]

        def sharpe(a: np.ndarray) -> float:
            if len(a) < 2 or a.std() == 0:
                return 0.0
            return float(a.mean() / a.std())

        def win_rate(a: np.ndarray) -> float:
            if len(a) == 0:
                return 0.0
            return float((a > 0).mean())

        def vol(a: np.ndarray) -> float:
            if len(a) < 2:
                return 0.0
            return float(a.std())

        s_base = sharpe(baseline)
        s_recent = sharpe(recent)
        wr_base = win_rate(baseline)
        wr_recent = win_rate(recent)
        v_base = vol(baseline)
        v_recent = vol(recent)

        # Signal frequency (signals per cycle)
        if len(sigs_arr) >= self.recent_window:
            sig_base = float(sigs_arr[:-self.recent_window].mean()) if len(sigs_arr) > self.recent_window else 0.0
            sig_recent = float(sigs_arr[-self.recent_window:].mean())
        else:
            sig_base = sig_recent = 0.0

        # Sharpe decay component
        if s_base > 0:
            sharpe_ratio = max(0.0, min(1.5, s_recent / s_base)) / 1.5
        elif s_recent > 0:
            sharpe_ratio = 1.0
        else:
            sharpe_ratio = 0.0

        # Win rate decay
        wr_decay = max(0.0, 1.0 - max(0.0, wr_base - wr_recent) * 4)

        # Vol expansion (penalise vol growth)
        if v_base > 0:
            vol_ratio = v_recent / v_base
            vol_score = max(0.0, 1.0 - max(0.0, vol_ratio - 1.0))
        else:
            vol_score = 1.0

        # Signal frequency decay (fewer signals = stale)
        if sig_base > 0:
            sig_ratio = sig_recent / sig_base
            sig_score = max(0.0, min(1.0, sig_ratio))
        else:
            sig_score = 1.0

        # Composite decay score
        decay_score = (
            0.40 * sharpe_ratio
            + 0.25 * wr_decay
            + 0.15 * vol_score
            + 0.20 * sig_score
        )

        # Recommendation
        if decay_score < self.retirement_threshold:
            rec = "retire"
        elif decay_score < self.watch_threshold:
            rec = "watch"
        else:
            rec = "ok"

        return DecayReport(
            strategy_name=strategy_name,
            decay_score=float(decay_score),
            sharpe_baseline=s_base, sharpe_recent=s_recent,
            win_rate_baseline=wr_base, win_rate_recent=wr_recent,
            vol_baseline=v_base, vol_recent=v_recent,
            signal_frequency_baseline=sig_base,
            signal_frequency_recent=sig_recent,
            components={
                "sharpe_ratio": float(sharpe_ratio),
                "win_rate_decay": float(wr_decay),
                "vol_score": float(vol_score),
                "sig_score": float(sig_score),
            },
            recommendation=rec,
        )

    # ----------------------------------------------------------------
    def evaluate_all(self) -> list[DecayReport]:
        return [r for r in (self.evaluate(n) for n in self._pnls) if r is not None]
