"""engine.strategy
===============================================================================
Day 90 — Institutional Strategy Engine

Purpose
-------
Transforms normalized market data (OHLCV + indicators + context) into immutable
Signal objects that are passed to the Risk Engine, Portfolio Manager, and
Execution Layer.

This module implements deterministic trading strategies only.
Capital allocation, position sizing, SL/TP optimization, and execution
validation are handled by downstream components.

Current Strategies
------------------
• SMA Crossover (trend-following)
• RSI Mean Reversion
• Multi-Factor Momentum (primary — 4-factor scoring)
• Strategy Factory / Dispatcher

Institutional Design Goals
--------------------------
- Deterministic and reproducible signals
- No repainting
- Closed-candle confirmation only
- Strategy isolation
- Immutable Signal output
- Multi-symbol / multi-timeframe compatible
- Risk-engine independent
- Backtest/live identical behaviour
- Structured logging
- Easy plug-in architecture for future strategies
  (Smart Money, ICT, Order Blocks, FVG, ML Ensemble, etc.)

Pipeline
--------
Market Data
      │
      ▼
Indicators
      │
      ▼
Strategy
      │
      ▼
Signal
      │
      ▼
Risk Engine
      │
      ▼
Wisdom Gate (120 principles)
      │
      ▼
Execution Layer
===============================================================================
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from engine.signals import Signal
from utils.indicators import atr, ema, rsi, sma
from utils.logger import get_logger

log = get_logger("engine.strategy")


# ----------------------------------------------------------------------
# Base interface
# ----------------------------------------------------------------------
class Strategy(ABC):
    """All strategies implement `evaluate(df) -> Signal`."""

    name: str = "abstract"

    def __init__(self, symbol: str, timeframe: str) -> None:
        self.symbol = symbol
        self.timeframe = timeframe

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Signal:
        ...

    def _hold(self, price: float, bar_time: Optional[datetime], reason: str) -> Signal:
        return Signal.hold(self.symbol, self.timeframe, price=price,
                           bar_time=bar_time, reason=reason)


# ----------------------------------------------------------------------
# SMA crossover with optional RSI confirmation
# ----------------------------------------------------------------------
class SmaCrossoverStrategy(Strategy):
    name = "sma_crossover"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        sma_fast: int = 20,
        sma_slow: int = 50,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        use_rsi_filter: bool = True,
        min_strength: float = 0.5,
    ) -> None:
        super().__init__(symbol, timeframe)
        if sma_fast >= sma_slow:
            raise ValueError("sma_fast must be < sma_slow")
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.use_rsi_filter = use_rsi_filter
        self.min_strength = float(min_strength)

    def evaluate(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.sma_slow + 2:
            return self._hold(self._last_price(df), self._last_time(df),
                              "not enough history")

        fast = sma(df["close"], self.sma_fast)
        slow = sma(df["close"], self.sma_slow)
        rsi_series = rsi(df["close"], self.rsi_period) if self.use_rsi_filter else None

        # Crossover detection: compare last two CLOSED rows (P0-2 FIX)
        # CRITICAL: Use iloc[-2] and iloc[-3] for signal generation to avoid
        # repainting. iloc[-1] is the current unclosed candle which can change.
        if fast.isna().iloc[-2] or slow.isna().iloc[-2]:
            return self._hold(self._last_price(df), self._last_time(df),
                              "indicator warm-up")

        fast_now, fast_prev = fast.iloc[-2], fast.iloc[-3]
        slow_now, slow_prev = slow.iloc[-2], slow.iloc[-3]

        crossed_up = fast_prev <= slow_prev and fast_now > slow_now
        crossed_dn = fast_prev >= slow_prev and fast_now < slow_now
        # State-based bias (avoids flapping when exactly equal)
        bias_up = fast_now > slow_now
        bias_dn = fast_now < slow_now

        last_price = self._last_price(df)
        last_time = self._last_time(df)

        # RSI filter: only take BUY when not overbought; SELL when not oversold (P0-2 FIX)
        # Use iloc[-2] for closed candle, not iloc[-1] (current candle)
        rsi_val = rsi_series.iloc[-2] if rsi_series is not None and len(rsi_series) > 2 else None
        rsi_blocks_buy = self.use_rsi_filter and rsi_val is not None and rsi_val >= self.rsi_overbought
        rsi_blocks_sell = self.use_rsi_filter and rsi_val is not None and rsi_val <= self.rsi_oversold

        # v6.3: Debug log — show ALL indicator values every call
        spread_pct = ((fast_now - slow_now) / slow_now * 100) if slow_now != 0 else 0
        log.info(
            "  [%s] SMA fast=%.2f slow=%.2f spread=%+.3f%% | crossed_up=%s crossed_dn=%s | "
            "RSI=%.1f (blocks_buy=%s blocks_sell=%s) | bias=%s",
            self.symbol, fast_now, slow_now, spread_pct,
            crossed_up, crossed_dn,
            rsi_val if rsi_val is not None else -1, rsi_blocks_buy, rsi_blocks_sell,
            "UP" if bias_up else ("DN" if bias_dn else "FLAT"),
        )

        if crossed_up and not rsi_blocks_buy:
            strength = self._strength(fast_now, slow_now)
            if strength < self.min_strength:
                log.debug("  [%s] → HOLD (weak cross up, strength=%.3f < %.3f)",
                         self.symbol, strength, self.min_strength)
                return self._hold(last_price, last_time, "weak cross up")
            log.info("  [%s] → BUY signal! fast=%.2f slow=%.2f rsi=%.1f strength=%.3f",
                     self.symbol, fast_now, slow_now, rsi_val or -1, strength)
            return Signal.buy(self.symbol, self.timeframe, strength,
                              price=last_price, bar_time=last_time,
                              fast=fast_now, slow=slow_now, rsi=rsi_val)

        if crossed_dn and not rsi_blocks_sell:
            strength = self._strength(fast_now, slow_now)
            if strength < self.min_strength:
                log.debug("  [%s] → HOLD (weak cross dn, strength=%.3f < %.3f)",
                         self.symbol, strength, self.min_strength)
                return self._hold(last_price, last_time, "weak cross dn")
            log.info("  [%s] → SELL signal! fast=%.2f slow=%.2f rsi=%.1f strength=%.3f",
                     self.symbol, fast_now, slow_now, rsi_val or -1, strength)
            return Signal.sell(self.symbol, self.timeframe, strength,
                               price=last_price, bar_time=last_time,
                               fast=fast_now, slow=slow_now, rsi=rsi_val)

        reason = "trend up, no cross" if bias_up else ("trend dn, no cross" if bias_dn else "flat")
        log.info("  [%s] → HOLD (%s)", self.symbol, reason)
        return self._hold(last_price, last_time, reason)

    # ----------------------------------------------------------------
    @staticmethod
    def _strength(fast: float, slow: float) -> float:
        """Normalise |fast-slow|/slow into [0, 1]."""
        if slow == 0:
            return 0.0
        return float(min(1.0, abs(fast - slow) / slow * 100.0))

    @staticmethod
    def _last_price(df: pd.DataFrame) -> float:
        return float(df["close"].iloc[-1]) if not df.empty else 0.0

    @staticmethod
    def _last_time(df: pd.DataFrame) -> Optional[datetime]:
        if df.empty:
            return None
        t = df["time"].iloc[-1]
        if isinstance(t, pd.Timestamp):
            return t.to_pydatetime().astimezone(timezone.utc) if t.tzinfo else t.to_pydatetime()
        return t


# ----------------------------------------------------------------------
# RSI mean-reversion (bonus, used for testing & diversification)
# ----------------------------------------------------------------------
class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"

    def __init__(self, symbol: str, timeframe: str,
                 rsi_period: int = 14,
                 oversold: float = 30.0, overbought: float = 70.0) -> None:
        super().__init__(symbol, timeframe)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def evaluate(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.rsi_period + 2:
            return self._hold(self._last_price(df), self._last_time(df), "warm-up")
        r = rsi(df["close"], self.rsi_period)
        if r.isna().iloc[-1] or r.isna().iloc[-2]:
            return self._hold(self._last_price(df), self._last_time(df), "rsi warm-up")
        now, prev = r.iloc[-1], r.iloc[-2]
        price = self._last_price(df)
        bt = self._last_time(df)
        # v6.3: Debug log
        log.info("  [%s] RSI now=%.1f prev=%.1f | oversold=%.0f overbought=%.0f | "
                 "cross_up=%s cross_dn=%s",
                 self.symbol, now, prev, self.oversold, self.overbought,
                 prev <= self.oversold and now > self.oversold,
                 prev >= self.overbought and now < self.overbought)
        # BUY when RSI crosses back up out of oversold
        if prev <= self.oversold and now > self.oversold:
            log.info("  [%s] → BUY! RSI crossed up from oversold (%.1f → %.1f)",
                     self.symbol, prev, now)
            return Signal.buy(self.symbol, self.timeframe,
                              strength=float(min(1.0, (self.oversold - now + 10) / 30)),
                              price=price, bar_time=bt, rsi=now)
        if prev >= self.overbought and now < self.overbought:
            log.info("  [%s] → SELL! RSI crossed down from overbought (%.1f → %.1f)",
                     self.symbol, prev, now)
            return Signal.sell(self.symbol, self.timeframe,
                               strength=float(min(1.0, (now - self.overbought + 10) / 30)),
                               price=price, bar_time=bt, rsi=now)
        log.info("  [%s] → HOLD (rsi=%.1f, waiting for cross)", self.symbol, now)
        return self._hold(price, bt, f"rsi={now:.1f}")

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


# ----------------------------------------------------------------------
# Momentum Strategy — v6.3 NEW (produces more signals)
# ----------------------------------------------------------------------
class MomentumStrategy(Strategy):
    """Multi-factor momentum strategy — produces more signals than SMA crossover.

    Combines:
      - EMA bias (fast > slow = bullish)
      - RSI direction (rising = bullish)
      - Price momentum (recent return > 0 = bullish)
      - Volume confirmation (above average = strong)

    A signal fires when >= 3 of 4 factors agree.
    """
    name = "momentum"

    def __init__(
        self, symbol: str, timeframe: str,
        ema_fast: int = 9, ema_slow: int = 21,
        rsi_period: int = 14,
        momentum_lookback: int = 5,
        volume_period: int = 20,
        min_factors: int = 3,
    ) -> None:
        super().__init__(symbol, timeframe)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.momentum_lookback = momentum_lookback
        self.volume_period = volume_period
        self.min_factors = min_factors

    def evaluate(self, df: pd.DataFrame) -> Signal:
        if len(df) < max(self.ema_slow, self.rsi_period, self.volume_period) + 5:
            return self._hold(self._last_price(df), self._last_time(df), "warm-up")

        close = df["close"]
        # v6.3.2: Robust volume handling — handle missing column, Series, or DataFrame
        if "volume" in df.columns:
            vol = df["volume"]
        elif "tick_volume" in df.columns:
            vol = df["tick_volume"]
        elif "real_volume" in df.columns:
            vol = df["real_volume"]
        else:
            vol = pd.Series([1.0] * len(df), index=df.index)

        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        vol = pd.to_numeric(vol, errors="coerce").fillna(1.0)

        # Indicators
        ema_f = ema(close, self.ema_fast)
        ema_s = ema(close, self.ema_slow)
        rsi_series = rsi(close, self.rsi_period)
        # P0-2 FIX: Use closed candle for price (iloc[-2]), not current (iloc[-1])
        last_price = float(close.iloc[-2]) if len(close) > 2 else float(close.iloc[-1])
        last_time = self._last_time(df)

        # P0-2 FIX: Check NA on closed candles (iloc[-2], iloc[-3])
        if ema_f.isna().iloc[-2] or ema_s.isna().iloc[-2] or rsi_series.isna().iloc[-2]:
            return self._hold(last_price, last_time, "indicator warm-up")

        # P0-2 FIX: Use closed candle values for signal generation
        ema_f_now = float(ema_f.iloc[-2])
        ema_s_now = float(ema_s.iloc[-2])
        rsi_now = float(rsi_series.iloc[-2])
        rsi_prev = float(rsi_series.iloc[-3])

        # Factor 1: EMA bias
        ema_bull = ema_f_now > ema_s_now
        ema_bear = ema_f_now < ema_s_now

        # Factor 2: RSI direction
        rsi_rising = rsi_now > rsi_prev
        rsi_falling = rsi_now < rsi_prev

        # Factor 3: Price momentum (last N bars)
        if len(close) > self.momentum_lookback:
            ref_price = float(close.iloc[-self.momentum_lookback - 1])
            momentum = (last_price - ref_price) / ref_price if ref_price > 0 else 0.0
        else:
            momentum = 0.0
        mom_bull = momentum > 0.001
        mom_bear = momentum < -0.001

        # Factor 4: Volume confirmation
        # C14 fix: replaced the arbitrary `vol_now > vol_avg * 1.2` threshold
        # with a percentile-based test — vol_now must be above the 70th
        # percentile of the recent volume window. This adapts to each
        # symbol's own volume distribution instead of a fixed multiplier.
        vol_strong = False
        if len(vol) > self.volume_period:
            try:
                # P0-2 FIX: Use closed candle volume (iloc[-2])
                vol_now = float(vol.iloc[-2])
                vol_slice = vol.iloc[-self.volume_period - 1:-1]
                if len(vol_slice) > 0:
                    # C14 fix: percentile-based threshold (70th pctile).
                    vol_p70 = float(vol_slice.quantile(0.70))
                    vol_strong = vol_now > vol_p70 if vol_p70 > 0 else False
            except (TypeError, ValueError, IndexError):
                vol_strong = False

        # Score
        bull_factors = sum([ema_bull, rsi_rising, mom_bull, vol_strong])
        bear_factors = sum([ema_bear, rsi_falling, mom_bear, vol_strong])

        # v6.4: Professional logging — only on signal change or every 10 cycles
        _last_signal = getattr(self, '_last_signal', 'HOLD')
        signal_type = "BUY" if (bull_factors >= self.min_factors and bull_factors > bear_factors) else \
                       "SELL" if (bear_factors >= self.min_factors and bear_factors > bull_factors) else "HOLD"

        # Only log when signal changes, or every 10th HOLD
        should_log = (signal_type != _last_signal) or \
                     (signal_type == "HOLD" and not hasattr(self, '_last_hold_logged')) or \
                     (signal_type == "HOLD" and getattr(self, '_hold_count', 0) % 10 == 0)

        if signal_type == "BUY" and bull_factors >= self.min_factors and bull_factors > bear_factors:
            strength = min(1.0, bull_factors / 4.0 + 0.1)
            if should_log:
                log.info("SIGNAL %s BUY  | EMA:%s RSI:%.1f MOM:%+.3f%% VOL:%s | bull=%d/4 bear=%d/4 | price=%.2f",
                         self.symbol, "BULL" if ema_bull else "BEAR", rsi_now,
                         momentum * 100, "HIGH" if vol_strong else "AVG",
                         bull_factors, bear_factors, last_price)
            self._last_signal = "BUY"
            self._hold_count = 0
            return Signal.buy(self.symbol, self.timeframe, strength,
                              price=last_price, bar_time=last_time,
                              ema_fast=ema_f_now, ema_slow=ema_s_now, rsi=rsi_now)

        if signal_type == "SELL" and bear_factors >= self.min_factors and bear_factors > bull_factors:
            strength = min(1.0, bear_factors / 4.0 + 0.1)
            if should_log:
                log.info("SIGNAL %s SELL | EMA:%s RSI:%.1f MOM:%+.3f%% VOL:%s | bull=%d/4 bear=%d/4 | price=%.2f",
                         self.symbol, "BEAR" if ema_bear else "BULL", rsi_now,
                         momentum * 100, "HIGH" if vol_strong else "AVG",
                         bull_factors, bear_factors, last_price)
            self._last_signal = "SELL"
            self._hold_count = 0
            return Signal.sell(self.symbol, self.timeframe, strength,
                               price=last_price, bar_time=last_time,
                               ema_fast=ema_f_now, ema_slow=ema_s_now, rsi=rsi_now)

        # HOLD
        self._hold_count = getattr(self, '_hold_count', 0) + 1
        if should_log:
            log.info("SIGNAL %s HOLD | EMA:%s RSI:%.1f MOM:%+.3f%% VOL:%s | bull=%d/4 bear=%d/4 | price=%.2f",
                     self.symbol, "BULL" if ema_bull else "BEAR", rsi_now,
                     momentum * 100, "HIGH" if vol_strong else "AVG",
                     bull_factors, bear_factors, last_price)
            self._last_signal = "HOLD"
            self._last_hold_logged = True
        return self._hold(last_price, last_time,
                          f"bull={bull_factors} bear={bear_factors}")

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


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
def build_strategy(cfg: dict[str, Any], symbol: str, timeframe: str) -> Strategy:
    """Factory used by main.py. `cfg` is the `strategy:` block of config.yaml."""
    stype = cfg.get("type", "sma_crossover").lower()
    if stype == "sma_crossover":
        return SmaCrossoverStrategy(
            symbol=symbol, timeframe=timeframe,
            sma_fast=int(cfg.get("sma_fast", 20)),
            sma_slow=int(cfg.get("sma_slow", 50)),
            rsi_period=int(cfg.get("rsi_period", 14)),
            rsi_overbought=float(cfg.get("rsi_overbought", 70)),
            rsi_oversold=float(cfg.get("rsi_oversold", 30)),
            use_rsi_filter=bool(cfg.get("use_rsi_filter", True)),
            min_strength=float(cfg.get("min_strength", 0.5)),
        )
    if stype == "rsi_reversion":
        return RsiReversionStrategy(
            symbol=symbol, timeframe=timeframe,
            rsi_period=int(cfg.get("rsi_period", 14)),
            oversold=float(cfg.get("rsi_oversold", 30)),
            overbought=float(cfg.get("rsi_overbought", 70)),
        )
    if stype == "momentum":
        return MomentumStrategy(
            symbol=symbol, timeframe=timeframe,
            ema_fast=int(cfg.get("ema_fast", 9)),
            ema_slow=int(cfg.get("ema_slow", 21)),
            rsi_period=int(cfg.get("rsi_period", 14)),
            momentum_lookback=int(cfg.get("momentum_lookback", 5)),
            volume_period=int(cfg.get("volume_period", 20)),
            min_factors=int(cfg.get("min_factors", 3)),
        )
    raise ValueError(f"unknown strategy type: {stype}")
