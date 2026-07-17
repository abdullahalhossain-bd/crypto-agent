"""
Correlation Filter — "Don't take the same bet twice"
======================================================

If you already have a long EURUSD and you open a long GBPUSD, you are
not diversifying — you are doubling down on USD weakness. When USD
reverses, both positions lose together.

This module tracks open positions and rejects new entries that are
highly correlated with existing exposure.

Correlation is computed from a rolling window of returns. The caller
supplies a returns dataframe (one column per symbol). The filter then
checks the average correlation of the candidate symbol against all
currently-open symbols.

Usage:
    from trading_modules.correlation_filter import CorrelationFilter

    cf = CorrelationFilter(max_correlation=0.7, lookback=100)
    # Build a returns df once per cycle (one column per symbol)
    returns_df = build_returns_df()  # pandas DataFrame

    if cf.is_correlated("BTCUSD", open_symbols=["ETHUSD"], returns_df=returns_df):
        # skip — too correlated
    else:
        # OK to take the trade
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    allowed: bool
    max_correlation: float
    correlated_with: Optional[str]
    avg_correlation: float
    reason: str


class CorrelationFilter:
    """
    Reject new entries that are highly correlated with existing exposure.

    Parameters:
        max_correlation: reject if |corr| >= this (default 0.7)
        lookback: # bars of returns to use (default 100)
        min_samples: minimum bars needed before correlation is enforced (default 30)
    """

    def __init__(
        self,
        max_correlation: float = 0.7,
        lookback: int = 100,
        min_samples: int = 30,
    ) -> None:
        self.max_correlation = max_correlation
        self.lookback = lookback
        self.min_samples = min_samples

    def check(
        self,
        candidate_symbol: str,
        open_symbols: list[str],
        returns_df: pd.DataFrame,
    ) -> CorrelationResult:
        """Check if candidate is too correlated with any open symbol."""
        if not open_symbols:
            return CorrelationResult(
                allowed=True, max_correlation=0.0, correlated_with=None,
                avg_correlation=0.0, reason="no open positions",
            )

        # Need candidate + at least one open symbol in returns_df
        available = [s for s in [candidate_symbol] + open_symbols if s in returns_df.columns]
        if candidate_symbol not in available:
            return CorrelationResult(
                allowed=True, max_correlation=0.0, correlated_with=None,
                avg_correlation=0.0, reason=f"candidate {candidate_symbol} not in returns data",
            )

        # Trim to lookback window
        recent = returns_df[available].tail(self.lookback)
        if len(recent) < self.min_samples:
            return CorrelationResult(
                allowed=True, max_correlation=0.0, correlated_with=None,
                avg_correlation=0.0,
                reason=f"insufficient samples ({len(recent)} < {self.min_samples})",
            )

        # Compute pairwise correlations
        corr_matrix = recent.corr()
        candidate_corrs: list[tuple[str, float]] = []
        for sym in open_symbols:
            if sym in corr_matrix.columns and sym != candidate_symbol:
                c = corr_matrix.loc[candidate_symbol, sym]
                if not np.isnan(c):
                    candidate_corrs.append((sym, float(c)))

        if not candidate_corrs:
            return CorrelationResult(
                allowed=True, max_correlation=0.0, correlated_with=None,
                avg_correlation=0.0, reason="no overlapping open symbols in data",
            )

        # Find max |correlation|
        max_sym, max_val = max(candidate_corrs, key=lambda x: abs(x[1]))
        avg_val = float(np.mean([abs(c) for _, c in candidate_corrs]))

        if abs(max_val) >= self.max_correlation:
            return CorrelationResult(
                allowed=False,
                max_correlation=max_val,
                correlated_with=max_sym,
                avg_correlation=avg_val,
                reason=(f"{candidate_symbol} corr with {max_sym} = {max_val:.2f} "
                        f">= {self.max_correlation}"),
            )

        return CorrelationResult(
            allowed=True,
            max_correlation=max_val,
            correlated_with=max_sym,
            avg_correlation=avg_val,
            reason=f"max |corr| = {abs(max_val):.2f} < {self.max_correlation}",
        )

    def is_correlated(
        self,
        candidate: str,
        open_symbols: list[str],
        returns_df: pd.DataFrame,
    ) -> bool:
        """Convenience: True if correlated (should skip)."""
        return not self.check(candidate, open_symbols, returns_df).allowed


def build_returns_df(candles_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Helper: build a returns dataframe from a dict of OHLCV dataframes.

    Each value must have a 'close' column. The returned dataframe has one
    column per symbol, indexed by timestamp, containing pct_change returns.
    """
    series: dict[str, pd.Series] = {}
    for sym, df in candles_by_symbol.items():
        if df is None or "close" not in df.columns or len(df) < 2:
            continue
        if "time" in df.columns:
            s = df.set_index("time")["close"].pct_change()
        else:
            s = df["close"].pct_change()
        series[sym] = s
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series)
