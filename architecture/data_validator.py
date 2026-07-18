"""architecture.data_validator — Phase 8 candle sanity checker.

Runs on every candle batch before it reaches feature/signal computation.
Rejects or flags:
  - Zero-volume candles on symbols that shouldn't have them
  - Price gaps beyond a configurable sane threshold
  - Out-of-order or duplicate timestamps
  - Stale data (last candle older than expected given the poll interval)

Returns a DataValidationResult with the verdict + list of specific issues.
The caller (TradingBot._process_symbol) decides whether to skip the cycle
(reject) or log a warning (flag) based on severity.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.architecture.data_validator")


@dataclass
class DataIssue:
    """One issue found during validation."""
    severity: str  # "error" (reject) or "warning" (flag)
    issue_type: str  # "zero_volume", "price_gap", "duplicate_timestamp", etc.
    bar_index: int = -1
    timestamp: str = ""
    detail: str = ""


@dataclass
class DataValidationResult:
    """Result of validating a candle batch."""
    valid: bool  # True = no errors (warnings may still exist)
    issues: list[DataIssue] = field(default_factory=list)
    bars_checked: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == "warning" for i in self.issues)

    def summary(self) -> str:
        errs = sum(1 for i in self.issues if i.severity == "error")
        warns = sum(1 for i in self.issues if i.severity == "warning")
        return f"{errs} errors, {warns} warnings ({self.bars_checked} bars checked)"


class DataValidator:
    """Validates OHLCV candle batches before they reach the pipeline.

    Phase 8 req #48: reject/flag candles with impossible values before
    they reach strategy logic.

    Config (via constructor or config.yaml data.validation section):
      - max_price_gap_pct: reject if |close[t] - close[t-1]| / close[t-1]
        > this (default 0.20 = 20% — catches data feed glitches, not
        normal volatility)
      - min_volume: symbols expected to have volume should not have zero
        (default 0 — zero-volume candles on liquid pairs are suspicious)
      - max_staleness_s: last candle should not be older than this
        (default 300s = 5 min for M15 candles + buffer)
      - check_duplicates: reject duplicate timestamps (default True)
      - check_ordering: reject out-of-order timestamps (default True)
    """

    def __init__(
        self,
        max_price_gap_pct: float = 0.20,
        min_volume: float = 0.0,
        max_staleness_s: float = 300.0,
        check_duplicates: bool = True,
        check_ordering: bool = True,
    ) -> None:
        self.max_price_gap_pct = float(max_price_gap_pct)
        self.min_volume = float(min_volume)
        self.max_staleness_s = float(max_staleness_s)
        self.check_duplicates = bool(check_duplicates)
        self.check_ordering = bool(check_ordering)

    def validate(self, df: pd.DataFrame, symbol: str = "",
                 expected_interval_s: float = 900.0) -> DataValidationResult:
        """Validate a candle DataFrame. Returns DataValidationResult.

        Args:
            df: OHLCV DataFrame with columns time, open, high, low, close, volume
            symbol: for logging
            expected_interval_s: expected seconds between candles (900 for M15)

        Returns:
            DataValidationResult with valid=True if no errors (warnings ok)
        """
        result = DataValidationResult(valid=True, bars_checked=len(df))

        if df is None or df.empty:
            result.issues.append(DataIssue(
                severity="error", issue_type="empty_data",
                detail="DataFrame is None or empty"))
            result.valid = False
            return result

        # Required columns
        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            result.issues.append(DataIssue(
                severity="error", issue_type="missing_columns",
                detail=f"Missing required columns: {required - set(df.columns)}"))
            result.valid = False
            return result

        # 1. Zero-volume candles (warning — some symbols legitimately have 0)
        if "volume" in df.columns and self.min_volume > 0:
            zero_vol = df[df["volume"] < self.min_volume]
            for idx in zero_vol.index[-5:]:  # report last 5 only
                result.issues.append(DataIssue(
                    severity="warning", issue_type="zero_volume",
                    bar_index=int(idx),
                    detail=f"volume={df.loc[idx, 'volume']:.0f} < min={self.min_volume}"))

        # 2. Price gaps (error if beyond threshold)
        closes = df["close"].astype(float)
        if len(closes) > 1:
            pct_changes = closes.pct_change().abs()
            big_gaps = pct_changes[pct_changes > self.max_price_gap_pct]
            for idx in big_gaps.index:
                result.issues.append(DataIssue(
                    severity="error", issue_type="price_gap",
                    bar_index=int(idx),
                    detail=f"price gap {pct_changes[idx]*100:.1f}% > "
                          f"{self.max_price_gap_pct*100:.1f}%"))
                result.valid = False

        # 3. OHLC sanity: high >= max(open,close), low <= min(open,close)
        for idx in df.index:
            row = df.loc[idx]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            if h < max(o, c) or l > min(o, c):
                result.issues.append(DataIssue(
                    severity="error", issue_type="ohlc_invalid",
                    bar_index=int(idx),
                    detail=f"H={h} < max(O,C)={max(o,c)} or L={l} > min(O,C)={min(o,c)}"))
                result.valid = False
                if len(result.issues) > 20:  # cap reporting
                    break

        # 4. Timestamp checks
        if "time" in df.columns:
            times = pd.to_datetime(df["time"])
            # Duplicate timestamps
            if self.check_duplicates:
                dups = times.duplicated()
                for idx in times[dups].index:
                    result.issues.append(DataIssue(
                        severity="error", issue_type="duplicate_timestamp",
                        bar_index=int(idx),
                        timestamp=str(times[idx]),
                        detail="duplicate timestamp"))
                    result.valid = False
            # Out-of-order timestamps
            if self.check_ordering and len(times) > 1:
                not_sorted = times.diff().dt.total_seconds() < 0
                for idx in times[not_sorted].index:
                    result.issues.append(DataIssue(
                        severity="error", issue_type="out_of_order_timestamp",
                        bar_index=int(idx),
                        timestamp=str(times[idx]),
                        detail="timestamp goes backwards"))
                    result.valid = False
            # Staleness check (last candle age)
            if len(times) > 0:
                last_time = times.iloc[-1]
                if hasattr(last_time, "tz") and last_time.tz is None:
                    last_time = last_time.tz_localize("UTC")
                now = datetime.now(tz=timezone.utc)
                if hasattr(last_time, "tz") and last_time.tz is not None:
                    age_s = (now - last_time).total_seconds()
                    if age_s > self.max_staleness_s + expected_interval_s:
                        result.issues.append(DataIssue(
                            severity="warning", issue_type="stale_data",
                            detail=f"last candle is {age_s:.0f}s old > "
                                  f"{self.max_staleness_s + expected_interval_s:.0f}s"))

        if result.has_errors:
            log.warning("DataValidator[%s]: REJECT — %s", symbol, result.summary())
        elif result.has_warnings:
            log.debug("DataValidator[%s]: OK with warnings — %s",
                     symbol, result.summary())
        return result
