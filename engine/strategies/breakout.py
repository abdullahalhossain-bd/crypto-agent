"""engine.strategies.breakout
=====================================================================
Donchian-channel breakout strategy.

Buy when close breaks above N-bar high (excl. current bar).
Sell when close breaks below N-bar low.

Performs well in trending regimes; gets chopped up in ranges.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.signals import Signal
from engine.strategies.base import Strategy, StrategyMetadata


class BreakoutStrategy(Strategy):
    metadata = StrategyMetadata(
        name="donchian_breakout",
        version="1.0.0",
        author="quant desk",
        description="Donchian channel breakout",
        min_bars=30,
        tags=("trend", "breakout"),
        regime_affinity={"trend": 0.85, "chop": 0.15, "high_vol": 0.6},
    )

    def __init__(self, symbol: str, timeframe: str,
                 params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, timeframe, params)
        p = self.params
        self.window = int(p.get("window", 20))
        if self.window < 5:
            raise ValueError("window must be >= 5")
        self.min_strength = float(p.get("min_strength", 0.0))

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.window + 2:
            return self._hold(self.symbol, self.timeframe, df, "warmup")

        # Exclude current bar from the rolling high/low to avoid trivial breakouts
        prev = df.iloc[-self.window - 1:-1]
        rolling_high = float(prev["high"].max())
        rolling_low = float(prev["low"].min())
        last_close = float(df["close"].iloc[-1])

        price = self._last_price(df)
        bt = self._last_time(df)

        # H9 fix: scale the breakout threshold with ATR so the strategy
        # adapts to volatility — in high-volatility markets, a small
        # penetration of the channel high is noise; in calm markets,
        # it's a genuine breakout. We require the close to exceed the
        # rolling high by at least 0.1×ATR (a small but non-zero buffer).
        try:
            from utils.indicators import atr
            atr_val = float(atr(df, 14).iloc[-1])
            if atr_val != atr_val:  # NaN check
                atr_val = 0.0
        except Exception:
            atr_val = 0.0
        breakout_buffer = atr_val * 0.1  # 10% of ATR as minimum penetration

        # Breakout only counts if the close clears the level by the buffer
        upside_break = last_close > (rolling_high + breakout_buffer)
        downside_break = last_close < (rolling_low - breakout_buffer)

        # Channel width as a strength proxy
        width = rolling_high - rolling_low
        if rolling_low == 0:
            strength = 0.0
        else:
            strength = float(min(1.0, width / rolling_low * 10.0))

        if upside_break and strength >= self.min_strength:
            return Signal.buy(self.symbol, self.timeframe, strength,
                              price=price, bar_time=bt,
                              rolling_high=rolling_high,
                              rolling_low=rolling_low,
                              strategy=self.metadata.name)
        if downside_break and strength >= self.min_strength:
            return Signal.sell(self.symbol, self.timeframe, strength,
                               price=price, bar_time=bt,
                               rolling_high=rolling_high,
                               rolling_low=rolling_low,
                               strategy=self.metadata.name)
        return self._hold(self.symbol, self.timeframe, df,
                          f"inside channel [{rolling_low:.4f}, {rolling_high:.4f}]")
