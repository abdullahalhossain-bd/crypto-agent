"""utils/indicators/validation.py
=====================================================================
Data Quality Validation (Improvement #14)
=====================================================================
Automatic detection of:
    - NaN values
    - Inf values
    - Missing data (gaps in index)
    - Duplicate index entries
    - Invalid candles (high < low, negative volume, etc.)
    - Outliers (Z-score, IQR, MAD methods)

Every indicator function should call `validate_ohlcv(df)` at entry to
fail fast on bad data rather than producing garbage silently.

Usage:
    from utils.indicators.validation import validate_ohlcv, ValidationReport
    report = validate_ohlcv(df)
    if not report.ok:
        print(report.issues)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Result of validating an OHLCV DataFrame."""
    ok: bool = True
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0  # 0-1, 1=perfect
    row_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    duplicate_index_count: int = 0
    invalid_candle_count: int = 0
    outlier_count: int = 0

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        # Warnings don't set ok=False but lower the quality score
        self.data_quality_score = max(0.0, self.data_quality_score - 0.05)


def validate_ohlcv(df: pd.DataFrame,
                   required_cols: Optional[List[str]] = None,
                   outlier_method: str = "zscore",
                   outlier_threshold: float = 4.0) -> ValidationReport:
    """Validate an OHLCV DataFrame.

    Checks:
        1. Required columns present (open, high, low, close, volume)
        2. No NaN values
        3. No Inf values
        4. No duplicate index
        5. No gaps in time index (if DatetimeIndex)
        6. Valid candles: high >= max(open, close), low <= min(open, close)
        7. Non-negative volume
        8. Outlier detection on close prices

    Returns ValidationReport with detailed issues + data_quality_score.
    """
    report = ValidationReport(row_count=len(df))

    if df is None or df.empty:
        report.add_issue("DataFrame is empty or None")
        return report

    required = required_cols or ["open", "high", "low", "close"]
    # Case-insensitive column check
    cols_lower = {c.lower(): c for c in df.columns}
    for col in required:
        if col not in cols_lower:
            report.add_issue(f"missing required column: {col}")

    # NaN check
    nan_count = int(df.isna().sum().sum())
    report.nan_count = nan_count
    if nan_count > 0:
        report.add_issue(f"{nan_count} NaN values found")
        report.data_quality_score = max(0.0, report.data_quality_score - 0.1)

    # Inf check
    inf_count = int(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())
    report.inf_count = inf_count
    if inf_count > 0:
        report.add_issue(f"{inf_count} Inf values found")
        report.data_quality_score = max(0.0, report.data_quality_score - 0.1)

    # Duplicate index
    if df.index.duplicated().any():
        dup_count = int(df.index.duplicated().sum())
        report.duplicate_index_count = dup_count
        report.add_issue(f"{dup_count} duplicate index entries")

    # Invalid candles
    if all(c in cols_lower for c in ["open", "high", "low", "close"]):
        o = df[cols_lower["open"]]
        h = df[cols_lower["high"]]
        l = df[cols_lower["low"]]
        c = df[cols_lower["close"]]
        invalid = (h < o) | (h < c) | (l > o) | (l > c) | (h < l)
        bad = int(invalid.sum())
        report.invalid_candle_count = bad
        if bad > 0:
            report.add_issue(f"{bad} invalid candles (high < low or similar)")

    # Negative volume
    if "volume" in cols_lower:
        v = df[cols_lower["volume"]]
        neg = int((v < 0).sum())
        if neg > 0:
            report.add_issue(f"{neg} negative volume values")

    # Outlier detection on close
    if "close" in cols_lower:
        c = df[cols_lower["close"]].dropna()
        if len(c) > 30:
            if outlier_method == "zscore":
                z = (c - c.mean()) / max(c.std(), 1e-10)
                outliers = int((z.abs() > outlier_threshold).sum())
            elif outlier_method == "iqr":
                q1, q3 = c.quantile(0.25), c.quantile(0.75)
                iqr = q3 - q1
                outliers = int(((c < q1 - 1.5 * iqr) | (c > q3 + 1.5 * iqr)).sum())
            else:  # MAD
                median = c.median()
                mad = (c - median).abs().median()
                if mad > 0:
                    outliers = int(((c - median).abs() / mad > outlier_threshold).sum())
                else:
                    outliers = 0
            report.outlier_count = outliers
            if outliers > 0:
                report.add_warning(f"{outliers} outliers detected via {outlier_method}")

    # Time index gap check
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 2:
        diffs = df.index.to_series().diff().dropna()
        if len(diffs) > 0:
            median_diff = diffs.median()
            gaps = (diffs > median_diff * 2).sum()
            if gaps > 0:
                report.add_warning(f"{int(gaps)} time gaps detected (>2x median interval)")

    return report


def clean_ohlcv(df: pd.DataFrame,
                fill_method: str = "ffill",
                drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean an OHLCV DataFrame by filling NaN, removing duplicates.

    fill_method: ffill, bfill, interpolate, drop
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if drop_duplicates and out.index.duplicated().any():
        out = out[~out.index.duplicated(keep="last")]
    if out.isna().any().any():
        if fill_method == "ffill":
            out = out.ffill().bfill()
        elif fill_method == "bfill":
            out = out.bfill().ffill()
        elif fill_method == "interpolate":
            out = out.interpolate().bfill().ffill()
        elif fill_method == "drop":
            out = out.dropna()
    return out


def assert_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Validate + clean. Raises ValueError if critical issues remain."""
    report = validate_ohlcv(df)
    if not report.ok:
        cleaned = clean_ohlcv(df)
        report2 = validate_ohlcv(cleaned)
        if not report2.ok:
            raise ValueError(f"OHLCV validation failed: {report2.issues}")
        return cleaned
    return df
