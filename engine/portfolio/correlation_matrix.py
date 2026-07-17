"""engine.portfolio.correlation_matrix
=====================================================================
Day 12 — Rolling correlation matrix between symbols.

Used by the portfolio engine to detect when two strategies are
effectively doubling up on the same underlying risk (e.g. BTC and ETH
are 0.85+ correlated, so holding both long is closer to a 2x BTC bet
than a diversified bet).

The matrix is computed from log-returns and updated incrementally as
new bars arrive. We cap the lookback to keep the calculation cheap.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("engine.portfolio.correlation")


class CorrelationMatrix:
    """Rolling pairwise correlation matrix across symbols."""

    def __init__(self, lookback: int = 500, min_history: int = 50) -> None:
        self.lookback = int(lookback)
        self.min_history = int(min_history)
        # Per-symbol ring buffer of log returns
        self._returns: dict[str, list[float]] = {}
        self._matrix: Optional[pd.DataFrame] = None
        self._dirty = True

    # ----------------------------------------------------------------
    def add_returns(self, symbol: str, returns: list[float] | pd.Series) -> None:
        """Append a series of log returns for `symbol`."""
        if isinstance(returns, pd.Series):
            returns = returns.dropna().tolist()
        buf = self._returns.setdefault(symbol, [])
        buf.extend(returns)
        # Trim to lookback
        if len(buf) > self.lookback:
            self._returns[symbol] = buf[-self.lookback:]
        self._dirty = True

    def add_bar(self, symbol: str, prev_close: float, curr_close: float) -> None:
        """Append a single bar's log return."""
        if prev_close <= 0 or curr_close <= 0:
            return
        r = float(np.log(curr_close / prev_close))
        buf = self._returns.setdefault(symbol, [])
        buf.append(r)
        if len(buf) > self.lookback:
            buf.pop(0)
        self._dirty = True

    # ----------------------------------------------------------------
    def matrix(self) -> pd.DataFrame:
        """Return the current correlation matrix as a DataFrame."""
        if not self._dirty and self._matrix is not None:
            return self._matrix
        # Align all symbols to the shortest available history
        if not self._returns:
            self._matrix = pd.DataFrame()
            return self._matrix
        max_len = max(len(v) for v in self._returns.values())
        if max_len < self.min_history:
            self._matrix = pd.DataFrame()
            return self._matrix
        data = {}
        for sym, rets in self._returns.items():
            if len(rets) < self.min_history:
                continue
            # Right-align to max_len
            padded = [np.nan] * (max_len - len(rets)) + rets
            data[sym] = padded
        if not data:
            self._matrix = pd.DataFrame()
            return self._matrix
        df = pd.DataFrame(data)
        self._matrix = df.corr(method="pearson")
        self._dirty = False
        return self._matrix

    # ----------------------------------------------------------------
    def pairwise(self, a: str, b: str) -> float:
        """Return current correlation between two symbols (NaN if unknown)."""
        m = self.matrix()
        if m.empty or a not in m.index or b not in m.columns:
            return float("nan")
        return float(m.loc[a, b])

    def correlated_with(self, symbol: str, threshold: float = 0.7) -> list[str]:
        """List symbols whose |corr| with `symbol` >= `threshold`."""
        m = self.matrix()
        if m.empty or symbol not in m.columns:
            return []
        col = m[symbol].abs()
        return [s for s, v in col.items() if v >= threshold and s != symbol]

    def to_dict(self) -> dict[str, dict[str, float]]:
        m = self.matrix()
        if m.empty:
            return {}
        return {r: {c: float(m.loc[r, c]) for c in m.columns} for r in m.index}
