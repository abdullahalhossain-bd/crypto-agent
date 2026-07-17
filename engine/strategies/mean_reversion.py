"""engine.strategies.mean_reversion
=====================================================================
RSI + Bollinger-band mean-reversion strategy.

BUY when:
  - RSI < oversold AND close < lower band
SELL when:
  - RSI > overbought AND close > upper band
EXIT (HOLD) once price returns to the mid band.

Performs well in choppy/regime-flat markets; dies in trends.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.signals import Signal
from engine.strategies.base import Strategy, StrategyMetadata
from utils.indicators import rsi, sma


class MeanReversionStrategy(Strategy):
    metadata = StrategyMetadata(
        name="mean_reversion",
        version="1.0.0",
        author="quant desk",
        description="RSI + Bollinger-band mean reversion",
        min_bars=40,
        tags=("mean_reversion", "bbands"),
        regime_affinity={"trend": 0.15, "chop": 0.9, "high_vol": 0.3},
    )

    def __init__(self, symbol: str, timeframe: str,
                 params: dict[str, Any] | None = None) -> None:
        super().__init__(symbol, timeframe, params)
        p = self.params
        self.bb_period = int(p.get("bb_period", 20))
        self.bb_std = float(p.get("bb_std", 2.0))
        self.rsi_period = int(p.get("rsi_period", 14))
        self.rsi_overbought = float(p.get("rsi_overbought", 70))
        self.rsi_oversold = float(p.get("rsi_oversold", 30))
        if self.bb_std <= 0:
            raise ValueError("bb_std must be > 0")

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.bb_period + 2:
            return self._hold(self.symbol, self.timeframe, df, "warmup")

        mid = sma(df["close"], self.bb_period)
        if mid.isna().iloc[-1]:
            return self._hold(self.symbol, self.timeframe, df, "bb warmup")
        # Std dev computed over the same rolling window
        std = df["close"].rolling(self.bb_period, min_periods=self.bb_period).std()
        if std.isna().iloc[-1]:
            return self._hold(self.symbol, self.timeframe, df, "std warmup")

        mid_v = float(mid.iloc[-1])
        std_v = float(std.iloc[-1])
        upper = mid_v + self.bb_std * std_v
        lower = mid_v - self.bb_std * std_v

        r = rsi(df["close"], self.rsi_period)
        if r.isna().iloc[-1] or r.isna().iloc[-2]:
            return self._hold(self.symbol, self.timeframe, df, "rsi warmup")
        rsi_now, rsi_prev = float(r.iloc[-1]), float(r.iloc[-2])

        last_close = float(df["close"].iloc[-1])
        price = self._last_price(df)
        bt = self._last_time(df)

        # BUY when RSI was below oversold and is now turning up,
        # AND price closed below lower band
        buy_trigger = (rsi_prev <= self.rsi_oversold and rsi_now > self.rsi_oversold
                       and last_close <= lower)
        sell_trigger = (rsi_prev >= self.rsi_overbought and rsi_now < self.rsi_overbought
                        and last_close >= upper)

        if buy_trigger:
            # Strength: how far below the lower band we are, in std units
            dev = max(0.0, lower - last_close) / (std_v or 1.0)
            strength = float(min(1.0, dev / 2.0 + 0.3))
            return Signal.buy(self.symbol, self.timeframe, strength,
                              price=price, bar_time=bt,
                              bb_mid=mid_v, bb_upper=upper, bb_lower=lower,
                              rsi=rsi_now, strategy=self.metadata.name)
        if sell_trigger:
            dev = max(0.0, last_close - upper) / (std_v or 1.0)
            strength = float(min(1.0, dev / 2.0 + 0.3))
            return Signal.sell(self.symbol, self.timeframe, strength,
                               price=price, bar_time=bt,
                               bb_mid=mid_v, bb_upper=upper, bb_lower=lower,
                               rsi=rsi_now, strategy=self.metadata.name)
        return self._hold(self.symbol, self.timeframe, df,
                          f"rsi={rsi_now:.1f} close={last_close:.4f}")
