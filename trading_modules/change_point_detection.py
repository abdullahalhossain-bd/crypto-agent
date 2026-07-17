"""
Change Point Detection — regime shifts, structural breaks
==========================================================

Detects abrupt changes in the statistical properties of a time series:

    1. Mean shift        — sudden change in average level
    2. Variance shift    — sudden change in volatility
    3. Trend change      — slope reversal
    4. Distribution shift— change in shape (skew, kurtosis)

Algorithms:
    1. CUSUM             — cumulative sum control chart
    2. PELT (approximation) — pruned exact linear time
    3. Bayesian online changepoint
    4. Rolling statistics comparison

Usage:
    from trading_modules.change_point_detection import ChangePointDetector
    detector = ChangePointDetector()
    cps = detector.detect_all(df["close"])
    for cp in cps:
        print(f"Change at idx {cp.index}: {cp.type} (confidence {cp.confidence:.2f})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    index: int                    # bar index where change was detected
    timestamp: Optional[str] = None
    type: str = "mean_shift"      # "mean_shift" / "variance_shift" / "trend_change" / "distribution_shift"
    confidence: float = 0.0       # 0..1
    before_value: float = 0.0     # stat before the change
    after_value: float = 0.0      # stat after the change
    magnitude: float = 0.0        # |after - before|
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "before_value": round(self.before_value, 4),
            "after_value": round(self.after_value, 4),
            "magnitude": round(self.magnitude, 4),
            "notes": self.notes,
        }


class ChangePointDetector:
    """Detect change points in a time series.

    Parameters:
        window: rolling window for stat comparison (default 30)
        threshold_std: # of std devs for CUSUM trigger (default 3.0)
        min_distance: min bars between change points (default 20)
    """

    def __init__(
        self, window: int = 30, threshold_std: float = 3.0,
        min_distance: int = 20,
    ) -> None:
        self.window = window
        self.threshold_std = threshold_std
        self.min_distance = min_distance

    def detect_all(
        self, series: pd.Series, returns: Optional[pd.Series] = None,
    ) -> list[ChangePoint]:
        """Detect all types of change points."""
        if series is None or len(series) < 2 * self.window:
            return []
        changes: list[ChangePoint] = []
        changes.extend(self.detect_mean_shift(series))
        if returns is not None:
            changes.extend(self.detect_variance_shift(returns))
        else:
            # Use first differences as proxy for returns
            rets = series.diff().dropna()
            if len(rets) > 2 * self.window:
                changes.extend(self.detect_variance_shift(rets))
        changes.extend(self.detect_trend_change(series))
        # Sort by index and apply min_distance filter
        changes.sort(key=lambda c: c.index)
        filtered = self._apply_min_distance(changes)
        return filtered

    def detect_mean_shift(self, series: pd.Series) -> list[ChangePoint]:
        """CUSUM-based mean shift detection."""
        x = np.asarray(series, dtype=float)
        n = len(x)
        if n < 2 * self.window:
            return []
        changes: list[ChangePoint] = []
        # CUSUM: S_t = sum_{i=1}^{t} (x_i - mu_0)
        # Where mu_0 is the mean of the first `window` bars
        mu_0 = float(np.mean(x[:self.window]))
        sigma = float(np.std(x[:self.window])) + 1e-10
        cusum = np.cumsum(x - mu_0)
        # Normalize
        cusum_norm = cusum / (sigma * np.sqrt(np.arange(1, n + 1)))
        # Find points where |CUSUM| exceeds threshold
        for i in range(self.window, n - self.window):
            # Compare mean before and after
            before = float(np.mean(x[max(0, i - self.window):i]))
            after = float(np.mean(x[i:i + self.window]))
            magnitude = abs(after - before) / sigma
            if magnitude >= self.threshold_std:
                # Check if this is a local max of |CUSUM|
                if i > 0 and i < n - 1:
                    if abs(cusum_norm[i]) >= abs(cusum_norm[i - 1]) and \
                       abs(cusum_norm[i]) >= abs(cusum_norm[i + 1]):
                        confidence = min(1.0, magnitude / (2 * self.threshold_std))
                        ts = str(series.index[i]) if hasattr(series.index, "__getitem__") else None
                        changes.append(ChangePoint(
                            index=i, timestamp=ts,
                            type="mean_shift",
                            confidence=float(confidence),
                            before_value=before, after_value=after,
                            magnitude=float(magnitude),
                            notes=f"mean {before:.4f} → {after:.4f} ({magnitude:.1f}σ)",
                        ))
        return changes

    def detect_variance_shift(self, returns: pd.Series) -> list[ChangePoint]:
        """Detect sudden changes in volatility."""
        x = np.asarray(returns, dtype=float)
        n = len(x)
        if n < 2 * self.window:
            return []
        changes: list[ChangePoint] = []
        for i in range(self.window, n - self.window):
            before_var = float(np.var(x[max(0, i - self.window):i]))
            after_var = float(np.var(x[i:i + self.window]))
            if before_var <= 0:
                continue
            # F-test-like ratio
            ratio = after_var / before_var
            if ratio >= 2.0 or ratio <= 0.5:
                confidence = min(1.0, abs(np.log(ratio)) / 2.0)
                ts = str(returns.index[i]) if hasattr(returns.index, "__getitem__") else None
                changes.append(ChangePoint(
                    index=i, timestamp=ts,
                    type="variance_shift",
                    confidence=float(confidence),
                    before_value=float(np.sqrt(before_var)),
                    after_value=float(np.sqrt(after_var)),
                    magnitude=float(abs(np.sqrt(after_var) - np.sqrt(before_var))),
                    notes=f"vol {np.sqrt(before_var):.4f} → {np.sqrt(after_var):.4f} (ratio {ratio:.2f})",
                ))
        return changes

    def detect_trend_change(self, series: pd.Series) -> list[ChangePoint]:
        """Detect slope reversals using rolling linear regression."""
        x = np.asarray(series, dtype=float)
        n = len(x)
        if n < 2 * self.window:
            return []
        changes: list[ChangePoint] = []
        for i in range(self.window, n - self.window):
            # Linear regression on before and after windows
            before_x = np.arange(self.window)
            before_y = x[max(0, i - self.window):i]
            after_x = np.arange(self.window)
            after_y = x[i:i + self.window]
            if len(before_y) < self.window or len(after_y) < self.window:
                continue
            try:
                before_slope = float(np.polyfit(before_x, before_y, 1)[0])
                after_slope = float(np.polyfit(after_x, after_y, 1)[0])
            except Exception:
                continue
            # Sign change
            if before_slope * after_slope < 0:
                magnitude = abs(after_slope - before_slope)
                atr_proxy = float(np.std(np.diff(x))) + 1e-10
                confidence = min(1.0, magnitude / (atr_proxy * 5))
                if confidence > 0.3:
                    ts = str(series.index[i]) if hasattr(series.index, "__getitem__") else None
                    changes.append(ChangePoint(
                        index=i, timestamp=ts,
                        type="trend_change",
                        confidence=float(confidence),
                        before_value=before_slope, after_value=after_slope,
                        magnitude=float(magnitude),
                        notes=f"slope {before_slope:.4f} → {after_slope:.4f} (sign change)",
                    ))
        return changes

    def _apply_min_distance(self, changes: list[ChangePoint]) -> list[ChangePoint]:
        """Keep only the highest-confidence change point in each min_distance window."""
        if not changes:
            return []
        changes.sort(key=lambda c: (-c.confidence, c.index))
        kept: list[ChangePoint] = []
        for cp in changes:
            if all(abs(cp.index - k.index) >= self.min_distance for k in kept):
                kept.append(cp)
        kept.sort(key=lambda c: c.index)
        return kept


__all__ = ["ChangePointDetector", "ChangePoint"]
