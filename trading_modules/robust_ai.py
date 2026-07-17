"""
Robust AI — handle missing/corrupted data and unexpected events
=================================================================

Real-world data is messy. This module provides robustness utilities:

    1. Missing data imputation  — forward-fill, interpolation, model-based
    2. Outlier handling         — winsorize, clip, robust statistics
    3. Corrupted data detection — identify stale/zero/extreme values
    4. Fail-safe wrappers       — catch exceptions, return safe defaults
    5. Graceful degradation     — fall back when modules fail
    6. Data quality scoring     — 0..1 score for each bar

Usage:
    from trading_modules.robust_ai import (
        impute_missing, winsorize, DataQualityChecker, fail_safe
    )
    clean_df = impute_missing(df, method="forward_fill")
    winsorized = winsorize(df["close"], limits=(0.01, 0.99))
    quality = DataQualityChecker().check(df)
"""
from __future__ import annotations

import logging
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. Missing data imputation
# ──────────────────────────────────────────────────────────────────────
def impute_missing(
    df: pd.DataFrame, method: str = "forward_fill",
    max_gap: int = 5,
) -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    Args:
        df: input DataFrame
        method: "forward_fill" / "linear" / "cubic" / "mean" / "median"
        max_gap: maximum gap size to fill (larger gaps left as NaN)
    """
    if df is None or df.empty:
        return df
    result = df.copy()
    if method == "forward_fill":
        result = result.ffill(limit=max_gap)
    elif method == "linear":
        result = result.interpolate(method="linear", limit=max_gap)
    elif method == "cubic":
        result = result.interpolate(method="cubic", limit=max_gap)
    elif method == "mean":
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].fillna(result[col].mean())
    elif method == "median":
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].fillna(result[col].median())
    return result


# ──────────────────────────────────────────────────────────────────────
# 2. Outlier handling
# ──────────────────────────────────────────────────────────────────────
def winsorize(
    series: pd.Series, limits: tuple[float, float] = (0.01, 0.99),
) -> pd.Series:
    """Winsorize — clip values to percentile bounds.

    Args:
        series: 1-D array
        limits: (lower_pct, upper_pct) e.g., (0.01, 0.99) clips to 1st and 99th percentiles
    """
    s = pd.Series(series)
    lower = s.quantile(limits[0])
    upper = s.quantile(limits[1])
    return s.clip(lower=lower, upper=upper)


def robust_zscore(series: pd.Series, window: int = 50) -> pd.Series:
    """Robust z-score using median + MAD instead of mean + std."""
    s = pd.Series(series)
    median = s.rolling(window).median()
    mad = (s - median).abs().rolling(window).median()
    modified_z = 0.6745 * (s - median) / mad.replace(0, np.nan)
    return modified_z


# ──────────────────────────────────────────────────────────────────────
# 3. Data quality checking
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DataQualityReport:
    total_rows: int = 0
    missing_count: int = 0
    missing_pct: float = 0.0
    stale_bars: int = 0              # bars with no change (zero OHLC change)
    extreme_bars: int = 0            # bars with extreme moves (>10%)
    zero_volume_bars: int = 0
    duplicate_timestamps: int = 0
    quality_score: float = 0.0       # 0..1
    issues: list[str] = field(default_factory=list)


class DataQualityChecker:
    """Check data quality for OHLCV data."""

    def __init__(
        self, stale_threshold: float = 0.0001,
        extreme_move_pct: float = 0.10,
        max_zero_volume_pct: float = 0.1,
    ) -> None:
        self.stale_threshold = stale_threshold
        self.extreme_move_pct = extreme_move_pct
        self.max_zero_volume_pct = max_zero_volume_pct

    def check(self, df: pd.DataFrame) -> DataQualityReport:
        """Check OHLCV data quality."""
        report = DataQualityReport()
        if df is None or df.empty:
            report.issues.append("empty dataframe")
            return report
        report.total_rows = len(df)
        # Missing values
        if "close" in df.columns:
            report.missing_count = int(df["close"].isna().sum())
            report.missing_pct = report.missing_count / report.total_rows
        # Stale bars (no change)
        if "close" in df.columns:
            rets = df["close"].pct_change().abs()
            report.stale_bars = int((rets < self.stale_threshold).sum())
        # Extreme moves
        if "close" in df.columns:
            rets = df["close"].pct_change().abs()
            report.extreme_bars = int((rets > self.extreme_move_pct).sum())
        # Zero volume
        if "volume" in df.columns:
            report.zero_volume_bars = int((df["volume"] == 0).sum())
            zero_vol_pct = report.zero_volume_bars / report.total_rows
            if zero_vol_pct > self.max_zero_volume_pct:
                report.issues.append(f"high zero-volume rate: {zero_vol_pct:.1%}")
        # Duplicate timestamps
        if "time" in df.columns:
            report.duplicate_timestamps = int(df["time"].duplicated().sum())
        # Quality score (0..1)
        score = 1.0
        score -= report.missing_pct * 2
        score -= (report.stale_bars / report.total_rows) * 0.5
        score -= (report.extreme_bars / report.total_rows) * 1.0
        if "volume" in df.columns:
            score -= (report.zero_volume_bars / report.total_rows) * 0.5
        score -= (report.duplicate_timestamps / report.total_rows) * 1.0
        report.quality_score = max(0.0, min(1.0, score))
        # Issues
        if report.missing_pct > 0.05:
            report.issues.append(f"high missing rate: {report.missing_pct:.1%}")
        if report.stale_bars > report.total_rows * 0.3:
            report.issues.append(f"high stale bar rate: {report.stale_bars / report.total_rows:.1%}")
        if report.extreme_bars > 10:
            report.issues.append(f"{report.extreme_bars} extreme moves (>{self.extreme_move_pct:.0%})")
        if report.duplicate_timestamps > 0:
            report.issues.append(f"{report.duplicate_timestamps} duplicate timestamps")
        if not report.issues:
            report.issues.append("data quality OK")
        return report


# ──────────────────────────────────────────────────────────────────────
# 4. Fail-safe decorator
# ──────────────────────────────────────────────────────────────────────
def fail_safe(
    default: Any = None, log_errors: bool = True,
) -> Callable:
    """Decorator that catches exceptions and returns a default value.

    Usage:
        @fail_safe(default=0.0)
        def compute_metric(df):
            ...  # might raise
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(
                        "fail_safe: %s raised %s: %s — returning default %r",
                        func.__name__, type(e).__name__, e, default,
                    )
                return default
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────────────────
# 5. Graceful degradation
# ──────────────────────────────────────────────────────────────────────
class GracefulDegradation:
    """Run a list of modules in priority order; fall back if any fail.

    Usage:
        gd = GracefulDegradation()
        gd.register("primary", primary_module, priority=1)
        gd.register("fallback", fallback_module, priority=2)
        result = gd.run("analyze", df)
    """

    def __init__(self) -> None:
        self.modules: list[tuple[str, Any, int]] = []

    def register(self, name: str, module: Any, priority: int = 1) -> None:
        self.modules.append((name, module, priority))
        self.modules.sort(key=lambda x: x[2])  # sort by priority

    def run(self, method_name: str, *args, **kwargs) -> Any:
        """Try each module in priority order; return first success."""
        last_error = None
        for name, module, _ in self.modules:
            try:
                method = getattr(module, method_name)
                return method(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "GracefulDegradation: module '%s' failed on '%s': %s",
                    name, method_name, e,
                )
                last_error = e
        if last_error:
            logger.error("All modules failed for '%s'", method_name)
        return None


# ──────────────────────────────────────────────────────────────────────
# 6. Safe execution with bounds checking
# ──────────────────────────────────────────────────────────────────────
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division — returns default if b is zero or NaN."""
    if b == 0 or not np.isfinite(b) or not np.isfinite(a):
        return default
    return a / b


def safe_pct_change(
    series: pd.Series, periods: int = 1, fill: float = 0.0,
) -> pd.Series:
    """Safe percentage change — handles zero division and NaN."""
    shifted = series.shift(periods)
    pct = (series - shifted) / shifted.replace(0, np.nan)
    return pct.fillna(fill)


def clip_to_range(
    value: float, lower: float, upper: float,
) -> float:
    """Clip a value to [lower, upper] range."""
    return max(lower, min(upper, value))


__all__ = [
    "impute_missing", "winsorize", "robust_zscore",
    "DataQualityReport", "DataQualityChecker",
    "fail_safe", "GracefulDegradation",
    "safe_divide", "safe_pct_change", "clip_to_range",
]
