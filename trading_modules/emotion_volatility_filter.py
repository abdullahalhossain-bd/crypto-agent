"""trading_modules/emotion_volatility_filter.py
=====================================================================
Emotion & Volatility Filter (Principle #117 — Detect Emotional Markets)
=====================================================================
Detects when the market is in an emotional state and adjusts the bot's
trading mode accordingly.

Emotional States Detected:
    - PANIC        — sharp drop + volume spike + wide spread
    - EUPHORIA     — sharp rise + volume spike + extreme RSI
    - FEAR         — elevated volatility + declining price
    - CAPITULATION — massive volume + huge range + at multi-period low
    - GREED        — extreme overbought + narrowing spread + FOMO volume
    - COMPLACENCY  — ultra-low volatility + declining volume

Trading Mode Adjustments per Emotion:
    | State        | Mode         | Action                           |
    |--------------|--------------|----------------------------------|
    | PANIC        | DEFENSIVE    | Close longs, no new entries      |
    | EUPHORIA     | DEFENSIVE    | Close longs, no new longs        |
    | FEAR         | CAUTIOUS     | Reduce size 50%, no new longs    |
    | CAPITULATION | OPPORTUNITY  | Look for reversal longs          |
    | GREED        | CAUTIOUS     | Reduce size 50%, no new longs    |
    | COMPLACENCY  | NORMAL       | No change (but watch for break)  |
    | NEUTRAL      | NORMAL       | Normal trading                   |

Also detects:
    - News spike (sudden volatility with no trend)
    - Low liquidity (wide spread + thin depth)
    - Stop hunt (rapid reversal after liquidity grab)

Usage:
    filt = EmotionVolatilityFilter()
    state = filt.detect(df, spread_bps=8.5, rvol=2.5)

    if state.emotion == "PANIC":
        # Close longs, halt new entries
        bot.enter_mode("DEFENSIVE")
    elif state.emotion == "CAPITULATION":
        # Look for reversal
        bot.scan_for_reversal()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.emotion_volatility_filter")


class Emotion(str, Enum):
    PANIC = "panic"
    EUPHORIA = "euphoria"
    FEAR = "fear"
    CAPITULATION = "capitulation"
    GREED = "greed"
    COMPLACENCY = "complacency"
    NEUTRAL = "neutral"


class TradingMode(str, Enum):
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"
    OPPORTUNITY = "opportunity"
    EMERGENCY = "emergency"


@dataclass
class EmotionState:
    """Result of emotion detection."""
    emotion: Emotion = Emotion.NEUTRAL
    mode: TradingMode = TradingMode.NORMAL
    confidence: float = 0.0           # 0-1
    volatility_percentile: float = 0.5
    volume_ratio: float = 1.0
    spread_bps: float = 0.0
    price_change_pct: float = 0.0     # recent bar % change
    rsi_extreme: bool = False
    news_spike: bool = False
    low_liquidity: bool = False
    stop_hunt_detected: bool = False
    description: str = ""
    actions: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion": self.emotion.value,
            "mode": self.mode.value,
            "confidence": round(self.confidence, 3),
            "volatility_percentile": round(self.volatility_percentile, 3),
            "volume_ratio": round(self.volume_ratio, 2),
            "spread_bps": round(self.spread_bps, 2),
            "price_change_pct": round(self.price_change_pct, 3),
            "rsi_extreme": self.rsi_extreme,
            "news_spike": self.news_spike,
            "low_liquidity": self.low_liquidity,
            "stop_hunt_detected": self.stop_hunt_detected,
            "description": self.description,
            "actions": self.actions,
        }


class EmotionVolatilityFilter:
    """Detects market emotion and recommends trading mode."""

    def __init__(self,
                 panic_threshold_pct: float = -3.0,
                 euphoria_threshold_pct: float = 3.0,
                 high_vol_percentile: float = 0.85,
                 low_vol_percentile: float = 0.15,
                 high_rvol: float = 2.0,
                 high_spread_bps: float = 15.0,
                 low_spread_bps: float = 1.0):
        """Initialize filter with thresholds."""
        self.panic_threshold = panic_threshold_pct
        self.euphoria_threshold = euphoria_threshold_pct
        self.high_vol_pct = high_vol_percentile
        self.low_vol_pct = low_vol_percentile
        self.high_rvol = high_rvol
        self.high_spread = high_spread_bps
        self.low_spread = low_spread_bps

    def detect(self,
               df: pd.DataFrame,
               spread_bps: float = 5.0,
               orderbook_depth_usd: float = 1_000_000,
               news_pending: bool = False) -> EmotionState:
        """Detect current market emotion.

        Args:
            df: OHLCV DataFrame (need at least 50 bars)
            spread_bps: current bid-ask spread
            orderbook_depth_usd: orderbook depth at ±1%
            news_pending: is high-impact news pending?

        Returns:
            EmotionState with detected emotion + recommended mode
        """
        state = EmotionState(spread_bps=spread_bps)

        if df is None or df.empty or len(df) < 30:
            state.description = "insufficient data"
            return state

        close = df["close"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # === 1. Price change (last bar) ===
        state.price_change_pct = float(
            (close.iloc[-1] - close.iloc[-2]) / max(close.iloc[-2], 1e-10) * 100
        )

        # === 2. Volatility percentile ===
        atr = self._compute_atr(df, 14)
        atr_pct = atr / close * 100
        state.volatility_percentile = float(atr_pct.tail(100).rank(pct=True).iloc[-1])

        # === 3. Volume ratio ===
        recent_vol = float(vol.tail(5).mean())
        avg_vol = float(vol.tail(20).mean())
        state.volume_ratio = recent_vol / max(avg_vol, 1)

        # === 4. RSI extreme check ===
        rsi = self._compute_rsi(close, 14)
        current_rsi = float(rsi.iloc[-1])
        state.rsi_extreme = current_rsi > 80 or current_rsi < 20

        # === 5. Low liquidity check ===
        state.low_liquidity = (
            spread_bps > self.high_spread or
            orderbook_depth_usd < 250_000
        )

        # === 6. News spike ===
        state.news_spike = (
            news_pending and
            state.volume_ratio > 1.5 and
            abs(state.price_change_pct) > 1.5
        )

        # === 7. Stop hunt detection ===
        state.stop_hunt_detected = self._detect_stop_hunt(df)

        # === Emotion determination (priority order) ===
        state.emotion, state.mode, state.confidence = self._classify_emotion(state)

        # === Description + actions ===
        state.description = self._describe(state)
        state.actions = self._actions(state)

        return state

    # ------------------------------------------------------------------
    # Emotion classification
    # ------------------------------------------------------------------
    def _classify_emotion(self, state: EmotionState) -> tuple:
        """Classify emotion from state data. Returns (emotion, mode, confidence)."""
        # CAPITULATION: extreme drop + extreme volume + multi-period low
        if (state.price_change_pct < -5 and
            state.volume_ratio > 3.0 and
            state.volatility_percentile > 0.90):
            return Emotion.CAPITULATION, TradingMode.OPPORTUNITY, 0.85

        # PANIC: sharp drop + high volume + high vol
        if (state.price_change_pct < self.panic_threshold and
            state.volume_ratio > self.high_rvol):
            return Emotion.PANIC, TradingMode.DEFENSIVE, 0.80

        # EUPHORIA: sharp rise + high volume + RSI extreme
        if (state.price_change_pct > self.euphoria_threshold and
            state.volume_ratio > self.high_rvol and
            state.rsi_extreme):
            return Emotion.EUPHORIA, TradingMode.DEFENSIVE, 0.75

        # FEAR: declining + elevated vol + moderate volume
        if (state.price_change_pct < -1.0 and
            state.volatility_percentile > self.high_vol_pct):
            return Emotion.FEAR, TradingMode.CAUTIOUS, 0.65

        # GREED: rising + RSI extreme + narrowing spread
        if (state.price_change_pct > 1.0 and
            state.rsi_extreme and
            state.spread_bps < self.low_spread):
            return Emotion.GREED, TradingMode.CAUTIOUS, 0.60

        # COMPLACENCY: ultra-low vol + low volume
        if (state.volatility_percentile < self.low_vol_pct and
            state.volume_ratio < 0.7):
            return Emotion.COMPLACENCY, TradingMode.NORMAL, 0.50

        # News spike
        if state.news_spike:
            return Emotion.FEAR, TradingMode.CAUTIOUS, 0.70

        # Low liquidity
        if state.low_liquidity:
            return Emotion.FEAR, TradingMode.CAUTIOUS, 0.60

        # Stop hunt
        if state.stop_hunt_detected:
            return Emotion.PANIC, TradingMode.DEFENSIVE, 0.70

        return Emotion.NEUTRAL, TradingMode.NORMAL, 0.40

    # ------------------------------------------------------------------
    # Description + actions
    # ------------------------------------------------------------------
    def _describe(self, state: EmotionState) -> str:
        """Human-readable description."""
        descs = {
            Emotion.PANIC: f"PANIC detected — price dropped {state.price_change_pct:.1f}% "
                          f"with {state.volume_ratio:.1f}x volume. Market selling aggressively.",
            Emotion.EUPHORIA: f"EUPHORIA — price surged {state.price_change_pct:.1f}% "
                             f"with RSI extreme. FOMO buying likely.",
            Emotion.FEAR: f"FEAR — elevated volatility "
                         f"(pctile={state.volatility_percentile:.0%}) with declining price.",
            Emotion.CAPITULATION: f"CAPITULATION — massive volume ({state.volume_ratio:.1f}x) "
                                 f"with {state.price_change_pct:.1f}% drop. Reversal likely.",
            Emotion.GREED: "GREED — overbought with tight spread. Late buyers entering.",
            Emotion.COMPLACENCY: "COMPLACENCY — low volatility, low volume. "
                                "Market waiting for direction.",
            Emotion.NEUTRAL: "Market is neutral — normal trading conditions.",
        }
        return descs.get(state.emotion, "Unknown emotion")

    def _actions(self, state: EmotionState) -> list:
        """Recommended actions for this emotion."""
        actions = {
            Emotion.PANIC: [
                "Close all long positions immediately",
                "No new long entries",
                "Reduce position size to 50% for shorts",
                "Wait for volatility to subside",
            ],
            Emotion.EUPHORIA: [
                "Close all long positions",
                "No new long entries",
                "Consider short setups (reversal likely)",
                "Tighten stops on any open longs",
            ],
            Emotion.FEAR: [
                "Reduce position size by 50%",
                "No new long entries",
                "Monitor for capitulation (potential reversal)",
                "Tighten trailing stops",
            ],
            Emotion.CAPITULATION: [
                "Watch for reversal candle (hammer, bullish engulfing)",
                "Prepare for counter-trend long entry",
                "Use reduced size (this is high-risk)",
                "Set tight stop below the panic low",
            ],
            Emotion.GREED: [
                "Take profit on existing longs",
                "No new long entries",
                "Consider short setups",
                "Wait for pullback before re-entering",
            ],
            Emotion.COMPLACENCY: [
                "Normal trading (but watch for breakout)",
                "Be ready for volatility expansion",
                "Set alerts for range breakouts",
            ],
            Emotion.NEUTRAL: [
                "Normal trading",
                "Follow strategy rules",
            ],
        }
        return actions.get(state.emotion, ["No actions defined"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def _compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _detect_stop_hunt(self, df: pd.DataFrame) -> bool:
        """Detect a stop hunt (liquidity sweep + reversal)."""
        if len(df) < 20:
            return False
        high = df["high"]
        low = df["low"]
        close = df["close"]
        # Recent range
        recent_high = high.tail(20).head(19).max()
        recent_low = low.tail(20).head(19).min()
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])
        last_close = float(close.iloc[-1])
        # Stop hunt above: broke above recent high, then closed back below
        if last_high > recent_high * 1.002 and last_close < recent_high:
            return True
        # Stop hunt below: broke below recent low, then closed back above
        if last_low < recent_low * 0.998 and last_close > recent_low:
            return True
        return False