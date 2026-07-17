"""engine.manipulation_detector
=====================================================================
Detects 4 market manipulation patterns that can destroy a trading
account. Each pattern is a VETO condition — if detected, the trade
is instantly rejected regardless of confluence score.

Patterns:
  1. Pump & Dump  — sudden spike with abnormal candle body (3x avg + 8% move)
  2. Wash Trading — volume spike 4x with price movement < 0.5%
  3. Wick Hunting — long wicks > 65% of candle range (stop-loss hunting)
  4. Fake-News Pump — spike followed by rapid reversal within 3 candles

Inspired by Centina-Quant's ManipulationDetector. Adapted to fit
our confluence engine veto system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("engine.manipulation")

# Thresholds (tunable via config)
PUMP_CANDLE_BODY_MULT = 3.0     # body > 3× avg body → pump candle
PUMP_PRICE_SPIKE_PCT = 0.08     # 8% single-candle move
WASH_VOL_MULT = 4.0             # volume spike ×4 with price movement < 0.5%
WICK_HUNT_RATIO = 0.65          # wick > 65% of total candle range
FAKE_NEWS_LOOKBACK = 3          # candles to look back for reversal after spike
FAKE_NEWS_REVERSAL_PCT = 0.50   # reversal of 50%+ of the spike


@dataclass
class ManipulationResult:
    pump_detected: bool = False
    wash_detected: bool = False
    wick_hunt: bool = False
    fake_news: bool = False
    any_detected: bool = False
    veto: bool = False
    veto_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pump_detected": self.pump_detected,
            "wash_detected": self.wash_detected,
            "wick_hunt": self.wick_hunt,
            "fake_news": self.fake_news,
            "any_detected": self.any_detected,
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "details": dict(self.details),
        }


# ----------------------------------------------------------------------
class ManipulationDetector:
    """Detects market manipulation patterns in OHLCV data."""

    def __init__(self,
                 pump_body_mult: float = PUMP_CANDLE_BODY_MULT,
                 pump_spike_pct: float = PUMP_PRICE_SPIKE_PCT,
                 wash_vol_mult: float = WASH_VOL_MULT,
                 wick_ratio: float = WICK_HUNT_RATIO,
                 fake_news_lookback: int = FAKE_NEWS_LOOKBACK,
                 fake_news_reversal: float = FAKE_NEWS_REVERSAL_PCT) -> None:
        self.pump_body_mult = float(pump_body_mult)
        self.pump_spike_pct = float(pump_spike_pct)
        self.wash_vol_mult = float(wash_vol_mult)
        self.wick_ratio = float(wick_ratio)
        self.fake_news_lookback = int(fake_news_lookback)
        self.fake_news_reversal = float(fake_news_reversal)

    # ----------------------------------------------------------------
    def check(self, df: pd.DataFrame, symbol: str = "") -> ManipulationResult:
        """Check the last bar (and recent context) for manipulation."""
        result = ManipulationResult()
        if len(df) < 20:
            return result
        try:
            result.pump_detected = self._detect_pump(df)
            result.wash_detected = self._detect_wash(df)
            result.wick_hunt = self._detect_wick_hunt(df)
            result.fake_news = self._detect_fake_news(df)
        except Exception as e:  # noqa: BLE001
            log.warning("ManipulationDetector error for %s: %s", symbol, e)

        result.any_detected = any([
            result.pump_detected, result.wash_detected,
            result.wick_hunt, result.fake_news,
        ])
        if result.any_detected:
            detected = []
            if result.pump_detected:
                detected.append("pump_detected")
            if result.wash_detected:
                detected.append("wash_detected")
            if result.wick_hunt:
                detected.append("wick_hunt")
            if result.fake_news:
                detected.append("fake_news")
            result.veto = True
            result.veto_reason = "; ".join(detected)
            log.warning("Manipulation flags for %s: %s", symbol, result.veto_reason)
        return result

    # ----------------------------------------------------------------
    def _detect_pump(self, df: pd.DataFrame) -> bool:
        """Pump & dump: sudden spike with abnormal candle body."""
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        bodies = (close - open_).abs()
        avg_body = float(bodies.iloc[-20:-1].mean()) if len(bodies) >= 21 else 0
        last_body = float(bodies.iloc[-1])
        last_move = abs(float(close.iloc[-1]) - float(close.iloc[-2])) / max(float(close.iloc[-2]), 1e-9)
        body_spike = avg_body > 0 and last_body > avg_body * self.pump_body_mult
        price_spike = last_move > self.pump_spike_pct
        return body_spike and price_spike

    # ----------------------------------------------------------------
    def _detect_wash(self, df: pd.DataFrame) -> bool:
        """Wash trading: volume spike with no real price movement."""
        if "volume" not in df.columns:
            return False
        vol = df["volume"].astype(float)
        close = df["close"].astype(float)
        vol_mean = float(vol.iloc[-20:-1].mean()) if len(vol) >= 21 else 0
        last_vol = float(vol.iloc[-1])
        price_move = abs(float(close.iloc[-1]) - float(close.iloc[-2])) / max(float(close.iloc[-2]), 1e-9)
        vol_spike = vol_mean > 0 and last_vol > vol_mean * self.wash_vol_mult
        no_movement = price_move < 0.005
        return vol_spike and no_movement

    # ----------------------------------------------------------------
    def _detect_wick_hunt(self, df: pd.DataFrame) -> bool:
        """Wick hunting: long wicks designed to trigger stop-losses."""
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        open_ = float(df["open"].iloc[-1])
        close = float(df["close"].iloc[-1])
        range_ = high - low
        if range_ <= 0:
            return False
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        max_wick = max(upper_wick, lower_wick)
        return (max_wick / range_) > self.wick_ratio

    # ----------------------------------------------------------------
    def _detect_fake_news(self, df: pd.DataFrame) -> bool:
        """Fake-news pump: spike followed by rapid reversal within N candles."""
        close = df["close"].astype(float)
        if len(close) < self.fake_news_lookback + 2:
            return False
        # Look at the bar N bars ago — was there a spike?
        spike_idx = -(self.fake_news_lookback + 1)
        spike_close = float(close.iloc[spike_idx])
        prev_close = float(close.iloc[spike_idx - 1])
        if prev_close <= 0:
            return False
        spike_pct = (spike_close - prev_close) / prev_close
        if abs(spike_pct) < 0.03:  # need at least 3% spike
            return False
        # Has price reversed by 50%+ of the spike?
        current_close = float(close.iloc[-1])
        reversal = abs(spike_close - current_close) / max(abs(spike_close - prev_close), 1e-9)
        return reversal >= self.fake_news_reversal
