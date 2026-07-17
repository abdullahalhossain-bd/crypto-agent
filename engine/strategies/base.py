"""engine.strategies.base
=====================================================================
Day 8 — Standardised strategy interface.

Every strategy in the pool MUST inherit from `Strategy`. This guarantees
the runner can call `generate_signal(df)` uniformly and that metadata
(name, version, expected warmup, regime affinity) is available for the
portfolio + regime layers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from engine.signals import Signal


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyMetadata:
    """Static descriptor of a strategy.

    `regime_affinity` is a dict mapping regime labels ("trend", "chop",
    "high_vol") to a weight in [0, 1] describing how well the strategy
    performs in that regime. The adaptive allocator uses this to scale
    exposure.
    """
    name: str
    version: str
    author: str = ""
    description: str = ""
    min_bars: int = 50
    expected_runtime_ms: float = 5.0
    tags: tuple[str, ...] = field(default_factory=tuple)
    regime_affinity: dict[str, float] = field(default_factory=lambda: {
        "trend": 0.5, "chop": 0.5, "high_vol": 0.3,
    })


# ----------------------------------------------------------------------
class Strategy(ABC):
    """Abstract base — every strategy implements `generate_signal`."""

    metadata: StrategyMetadata

    def __init__(self, symbol: str, timeframe: str,
                 params: Optional[dict[str, Any]] = None) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.params = params or {}

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> Signal:
        """Produce a `Signal` for the latest bar of `df`.

        MUST NOT mutate `df`. MUST NOT use any data after the last bar.
        MUST be deterministic given the same `df` and `params`.
        """

    # Convenience wrapper kept for backwards compat with the Day-2 API.
    def evaluate(self, df: pd.DataFrame) -> Signal:  # noqa: D401
        return self.generate_signal(df)

    # ----------------------------------------------------------------
    # Helpers shared across strategies
    # ----------------------------------------------------------------
    @staticmethod
    def _last_price(df: pd.DataFrame) -> float:
        return float(df["close"].iloc[-1]) if not df.empty else 0.0

    @staticmethod
    def _last_time(df: pd.DataFrame) -> Optional[datetime]:
        if df.empty:
            return None
        t = df["time"].iloc[-1]
        if isinstance(t, pd.Timestamp):
            return t.to_pydatetime()
        return t

    @staticmethod
    def _hold(symbol: str, timeframe: str, df: pd.DataFrame,
              reason: str = "no signal") -> Signal:
        return Signal.hold(
            symbol=symbol,
            timeframe=timeframe,
            price=Strategy._last_price(df),
            bar_time=Strategy._last_time(df),
            reason=reason,
        )

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} "
                f"{self.metadata.name} v{self.metadata.version} "
                f"{self.symbol}/{self.timeframe}>")
