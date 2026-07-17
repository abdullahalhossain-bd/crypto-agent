"""engine.candlestick.pattern_decay
=====================================================================
Day 134 — Per-pattern decay tracker.

Similar to factory.decay_detector but tracks decay PER PATTERN TYPE,
not per strategy. A pin bar that worked 6 months ago may no longer
work — this module detects that.

Decay signals:
  - Win rate of recent N trades < baseline win rate by > threshold
  - Avg PnL of recent trades < baseline by > threshold
  - Signal frequency dropping (pattern stops appearing)
  - Sharpe of recent trades collapsed

When decay is detected, the confluence engine should DOWNWEIGHT
that pattern's contribution to the final score.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.logger import get_logger

log = get_logger("candlestick.decay")


@dataclass
class PatternDecayReport:
    pattern: str
    decay_score: float           # 1.0 = healthy, 0.0 = severe decay
    baseline_win_rate: float
    recent_win_rate: float
    baseline_avg_pnl: float
    recent_avg_pnl: float
    n_recent: int
    recommendation: str          # "ok" / "watch" / "downweight" / "retire"
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "decay_score": self.decay_score,
            "baseline_win_rate": self.baseline_win_rate,
            "recent_win_rate": self.recent_win_rate,
            "baseline_avg_pnl": self.baseline_avg_pnl,
            "recent_avg_pnl": self.recent_avg_pnl,
            "n_recent": self.n_recent,
            "recommendation": self.recommendation,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class PatternDecayTracker:
    def __init__(self,
                 baseline_window: int = 200,
                 recent_window: int = 50,
                 min_samples: int = 30,
                 downweight_threshold: float = 0.5,
                 retire_threshold: float = 0.3) -> None:
        self.baseline_window = int(baseline_window)
        self.recent_window = int(recent_window)
        self.min_samples = int(min_samples)
        self.downweight_threshold = float(downweight_threshold)
        self.retire_threshold = float(retire_threshold)
        # Per-pattern trade history
        self._history: dict[str, deque] = {}
        # H7 fix: lock guarding _history — record_trade and evaluate can
        # be called concurrently from the strategy runner's thread pool.
        self._lock = threading.RLock()

    # ----------------------------------------------------------------
    def record_trade(self, pattern: str, pnl_pct: float, won: bool) -> None:
        # H7 fix: hold the lock while mutating the deque.
        with self._lock:
            d = self._history.setdefault(pattern, deque(maxlen=self.baseline_window))
            d.append({"pnl": float(pnl_pct), "won": bool(won)})

    # ----------------------------------------------------------------
    def evaluate(self, pattern: str) -> PatternDecayReport:
        # H7 fix: hold the lock while reading the deque (a concurrent
        # append could otherwise mutate it mid-iteration).
        with self._lock:
            history = self._history.get(pattern)
        if history is None or len(history) < self.min_samples:
            return PatternDecayReport(
                pattern=pattern, decay_score=1.0,
                baseline_win_rate=0.0, recent_win_rate=0.0,
                baseline_avg_pnl=0.0, recent_avg_pnl=0.0,
                n_recent=0, recommendation="ok",
                components={"reason": "insufficient_data"},
            )
        arr = list(history)
        n = len(arr)
        baseline = arr[:-self.recent_window] if n > self.recent_window else arr
        recent = arr[-self.recent_window:]
        baseline_wins = sum(1 for t in baseline if t["won"])
        recent_wins = sum(1 for t in recent if t["won"])
        baseline_wr = baseline_wins / max(1, len(baseline))
        recent_wr = recent_wins / max(1, len(recent))
        baseline_pnl = float(np.mean([t["pnl"] for t in baseline])) if baseline else 0.0
        recent_pnl = float(np.mean([t["pnl"] for t in recent])) if recent else 0.0

        # Decay score components
        if baseline_wr > 0:
            wr_ratio = max(0.0, recent_wr / baseline_wr)
        else:
            wr_ratio = 1.0 if recent_wr == 0 else 1.5
        wr_ratio = min(1.5, wr_ratio)
        wr_score = min(1.0, wr_ratio)

        if baseline_pnl > 0:
            pnl_ratio = max(0.0, recent_pnl / baseline_pnl)
        elif recent_pnl > 0:
            pnl_ratio = 1.5
        else:
            pnl_ratio = 0.0
        pnl_score = min(1.0, pnl_ratio)

        # Sharpe component (recent vs baseline)
        baseline_pnls = [t["pnl"] for t in baseline]
        recent_pnls = [t["pnl"] for t in recent]
        if len(baseline_pnls) > 1 and np.std(baseline_pnls) > 0:
            base_sharpe = float(np.mean(baseline_pnls) / np.std(baseline_pnls))
        else:
            base_sharpe = 0.0
        if len(recent_pnls) > 1 and np.std(recent_pnls) > 0:
            recent_sharpe = float(np.mean(recent_pnls) / np.std(recent_pnls))
        else:
            recent_sharpe = 0.0
        if base_sharpe > 0:
            sharpe_ratio = max(0.0, recent_sharpe / base_sharpe)
        else:
            sharpe_ratio = 1.0
        sharpe_score = min(1.0, sharpe_ratio)

        decay_score = (
            0.4 * wr_score
            + 0.3 * pnl_score
            + 0.3 * sharpe_score
        )

        if decay_score < self.retire_threshold:
            rec = "retire"
        elif decay_score < self.downweight_threshold:
            rec = "downweight"
        elif decay_score < 0.75:
            rec = "watch"
        else:
            rec = "ok"

        return PatternDecayReport(
            pattern=pattern,
            decay_score=float(decay_score),
            baseline_win_rate=float(baseline_wr),
            recent_win_rate=float(recent_wr),
            baseline_avg_pnl=float(baseline_pnl),
            recent_avg_pnl=float(recent_pnl),
            n_recent=len(recent),
            recommendation=rec,
            components={
                "wr_score": float(wr_score),
                "pnl_score": float(pnl_score),
                "sharpe_score": float(sharpe_score),
                "baseline_sharpe": float(base_sharpe),
                "recent_sharpe": float(recent_sharpe),
            },
        )

    # ----------------------------------------------------------------
    def evaluate_all(self) -> list[PatternDecayReport]:
        return [self.evaluate(p) for p in self._history]

    # ----------------------------------------------------------------
    def downweight_factor(self, pattern: str) -> float:
        """Return a multiplier [0, 1] for the confluence engine."""
        report = self.evaluate(pattern)
        if report.recommendation == "retire":
            return 0.0
        if report.recommendation == "downweight":
            return report.decay_score
        if report.recommendation == "watch":
            return min(1.0, report.decay_score + 0.2)
        return 1.0
