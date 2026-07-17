"""engine.strategies package — pluggable strategy pool.

Every strategy inherits from `Strategy` in `base.py` and is registered
with `engine.strategy_registry.register()`. The `StrategyRunner`
instantiates and runs all registered strategies in parallel against
the same OHLCV stream, returning a `SignalPool`.
"""
from engine.strategies.base import Strategy, StrategyMetadata  # noqa: F401
from engine.strategies.sma_cross import SmaCrossoverStrategy  # noqa: F401
from engine.strategies.breakout import BreakoutStrategy  # noqa: F401
from engine.strategies.mean_reversion import MeanReversionStrategy  # noqa: F401

__all__ = [
    "Strategy",
    "StrategyMetadata",
    "SmaCrossoverStrategy",
    "BreakoutStrategy",
    "MeanReversionStrategy",
]
