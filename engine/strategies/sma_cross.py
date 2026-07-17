"""engine.strategies.sma_cross
=====================================================================
SMA-crossover strategy, ported to the Day-8 Strategy interface.

Buy when SMA(fast) crosses above SMA(slow). Sell on the opposite cross.
Optional RSI confirmation filter avoids buying into overbought and
selling into oversold conditions.

Regime affinity: this is a trend-following strategy, so it scores
high on "trend" regimes and poorly on "chop".
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.signals import Signal
from engine.strategies.base import Strategy, StrategyMetadata
from utils.indicators import rsi, sma


class SmaCrossoverStrategy(Strategy):
    metadata = StrategyMetadata(
        name="sma_crossover",
        version="2.0.0",
        author="quant desk",
        description="SMA(fast)/SMA(slow) crossover with optional RSI filter",
        min_bars=60,
        tags=("trend", "crossover"),
        regime_affinity={"trend": 0.9, "chop": 0.2, "high_vol": 0.4},
    )

    def __init__(self, symbol: str, timeframe: str,
                 params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, timeframe, params)
        p = self.params
        self.sma_fast = int(p.get("sma_fast", 20))
        self.sma_slow = int(p.get("sma_slow", 50))
        if self.sma_fast >= self.sma_slow:
            raise ValueError("sma_fast must be < sma_slow")
        self.rsi_period = int(p.get("rsi_period", 14))
        self.rsi_overbought = float(p.get("rsi_overbought", 70))
        self.rsi_oversold = float(p.get("rsi_oversold", 30))
        self.use_rsi_filter = bool(p.get("use_rsi_filter", True))
        self.min_strength = float(p.get("min_strength", 0.5))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.sma_slow + 2:
            return self._hold(self.symbol, self.timeframe, df, "warmup")

        fast = sma(df["close"], self.sma_fast)
        slow = sma(df["close"], self.sma_slow)
        if fast.isna().iloc[-1] or slow.isna().iloc[-1] or fast.isna().iloc[-2] or slow.isna().iloc[-2]:
            return self._hold(self.symbol, self.timeframe, df, "indicator warmup")

        fast_now, fast_prev = float(fast.iloc[-1]), float(fast.iloc[-2])
        slow_now, slow_prev = float(slow.iloc[-1]), float(slow.iloc[-2])

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_dn = fast_prev >= slow_prev and fast_now < slow_now

        rsi_series = rsi(df["close"], self.rsi_period) if self.use_rsi_filter else None
        rsi_val = float(rsi_series.iloc[-1]) if rsi_series is not None and not rsi_series.isna().iloc[-1] else None

        rsi_blocks_buy = rsi_val is not None and rsi_val >= self.rsi_overbought
        rsi_blocks_sell = rsi_val is not None and rsi_val <= self.rsi_oversold

        price = self._last_price(df)
        bt = self._last_time(df)

        if crossed_up and not rsi_blocks_buy:
            strength = self._strength(fast_now, slow_now)
            if strength < self.min_strength:
                return self._hold(self.symbol, self.timeframe, df, "weak cross up")
            return Signal.buy(self.symbol, self.timeframe, strength,
                              price=price, bar_time=bt,
                              fast=fast_now, slow=slow_now, rsi=rsi_val,
                              strategy=self.metadata.name)

        if crossed_dn and not rsi_blocks_sell:
            strength = self._strength(fast_now, slow_now)
            if strength < self.min_strength:
                return self._hold(self.symbol, self.timeframe, df, "weak cross dn")
            return Signal.sell(self.symbol, self.timeframe, strength,
                               price=price, bar_time=bt,
                               fast=fast_now, slow=slow_now, rsi=rsi_val,
                               strategy=self.metadata.name)

        bias_up = fast_now > slow_now
        reason = "trend up, no cross" if bias_up else ("trend dn, no cross" if fast_now < slow_now else "flat")
        return self._hold(self.symbol, self.timeframe, df, reason)

    @staticmethod
    def _strength(fast: float, slow: float) -> float:
        if slow == 0:
            return 0.0
        return float(min(1.0, abs(fast - slow) / slow * 100.0))
