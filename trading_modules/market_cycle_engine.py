"""trading_modules/market_cycle_engine.py
=====================================================================
Market Cycle Engine (Principle #162)
=====================================================================
Detects where the market is in its macro cycle:
    EXPANSION → PEAK → CONSOLIDATION → DECLINE → RECOVERY → (repeat)

Each phase requires a different strategy:
    EXPANSION     — trend following, full size
    PEAK          — take profits, reduce size, prepare for reversal
    CONSOLIDATION — range trading, mean reversion, small size
    DECLINE       — short bias, defensive, or stay in cash
    RECOVERY      — accumulation, gradual long entries

Detection Logic:
    Combines price action, volume, volatility, and breadth:
    - EXPANSION: rising prices + increasing volume + expanding volatility
    - PEAK: high prices + declining volume + extreme RSI + narrowing breadth
    - CONSOLIDATION: range-bound + low volatility + volume declining
    - DECLINE: falling prices + increasing volume + expanding volatility
    - RECOVERY: bottoming pattern + volume picking up + RSI diverging

Usage:
    engine = MarketCycleEngine()
    cycle = engine.detect(df)
    # cycle = {
    #     "phase": "expansion",
    #     "confidence": 0.82,
    #     "duration_bars": 45,
    #     "next_phase_estimate": "peak",
    #     "strategy_recommendation": "trend_following_full_size",
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.market_cycle_engine")


class CyclePhase(str, Enum):
    EXPANSION = "expansion"
    PEAK = "peak"
    CONSOLIDATION = "consolidation"
    DECLINE = "decline"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


# Phase transitions (typical cycle order)
PHASE_TRANSITIONS = {
    CyclePhase.RECOVERY: CyclePhase.EXPANSION,
    CyclePhase.EXPANSION: CyclePhase.PEAK,
    CyclePhase.PEAK: CyclePhase.CONSOLIDATION,
    CyclePhase.CONSOLIDATION: CyclePhase.DECLINE,
    CyclePhase.DECLINE: CyclePhase.RECOVERY,
}


# Strategy recommendations per phase
PHASE_STRATEGIES = {
    CyclePhase.EXPANSION: "trend_following_full_size",
    CyclePhase.PEAK: "take_profits_reduce_size",
    CyclePhase.CONSOLIDATION: "range_trading_small_size",
    CyclePhase.DECLINE: "short_bias_defensive",
    CyclePhase.RECOVERY: "accumulation_gradual_longs",
    CyclePhase.UNKNOWN: "observe_wait",
}


@dataclass
class CycleResult:
    """Market cycle detection result."""
    phase: CyclePhase = CyclePhase.UNKNOWN
    confidence: float = 0.0        # 0-1
    duration_bars: int = 0         # how long in this phase
    next_phase_estimate: CyclePhase = CyclePhase.UNKNOWN
    strategy_recommendation: str = "observe_wait"

    # Per-indicator scores
    price_trend: float = 0.0       # -1 to +1
    volume_trend: str = "stable"   # increasing/decreasing/stable
    volatility_trend: str = "stable"
    breadth_score: float = 0.5     # 0-1
    rsi_extreme: bool = False

    # Phase characteristics
    range_high: float = 0.0
    range_low: float = 0.0
    range_position: float = 0.5    # 0=at low, 1=at high

    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "confidence": round(self.confidence, 3),
            "duration_bars": self.duration_bars,
            "next_phase_estimate": self.next_phase_estimate.value,
            "strategy_recommendation": self.strategy_recommendation,
            "price_trend": round(self.price_trend, 3),
            "volume_trend": self.volume_trend,
            "volatility_trend": self.volatility_trend,
            "breadth_score": round(self.breadth_score, 3),
            "rsi_extreme": self.rsi_extreme,
            "range_high": self.range_high,
            "range_low": self.range_low,
            "range_position": round(self.range_position, 3),
            "description": self.description,
        }


class MarketCycleEngine:
    """Detects macro market cycles from OHLCV data."""

    def __init__(self,
                 lookback: int = 100,
                 min_phase_duration: int = 10):
        """Initialize cycle engine.

        Args:
            lookback: bars to analyze for cycle detection
            min_phase_duration: minimum bars to confirm a phase
        """
        self.lookback = lookback
        self.min_duration = min_phase_duration

    def detect(self, df: pd.DataFrame) -> CycleResult:
        """Detect current market cycle phase.

        Args:
            df: OHLCV DataFrame (need at least 100 bars)

        Returns:
            CycleResult with phase + confidence + recommendation
        """
        result = CycleResult()

        if df is None or df.empty or len(df) < 50:
            result.description = "insufficient data"
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # Use lookback window
        n = min(len(df), self.lookback)
        window = df.tail(n)
        w_close = window["close"]
        w_high = window["high"]
        w_low = window["low"]
        w_vol = window["volume"]

        # === Price trend (50-bar return) ===
        if len(w_close) >= 50:
            ret_50 = float((w_close.iloc[-1] - w_close.iloc[-50]) / max(w_close.iloc[-50], 1e-10))
            ret_20 = float((w_close.iloc[-1] - w_close.iloc[-20]) / max(w_close.iloc[-20], 1e-10))
            result.price_trend = np.clip((ret_50 + ret_20) * 25, -1, 1)

        # === Volume trend ===
        recent_vol = float(w_vol.tail(10).mean())
        older_vol = float(w_vol.tail(40).head(20).mean())
        if recent_vol > older_vol * 1.3:
            result.volume_trend = "increasing"
        elif recent_vol < older_vol * 0.7:
            result.volume_trend = "decreasing"
        else:
            result.volume_trend = "stable"

        # === Volatility trend ===
        tr = pd.concat([
            w_high - w_low,
            (w_high - w_close.shift()).abs(),
            (w_low - w_close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        recent_atr = float(atr.tail(10).mean())
        older_atr = float(atr.tail(40).head(20).mean())
        if recent_atr > older_atr * 1.3:
            result.volatility_trend = "expanding"
        elif recent_atr < older_atr * 0.7:
            result.volatility_trend = "contracting"
        else:
            result.volatility_trend = "stable"

        # === RSI extreme ===
        delta = w_close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        result.rsi_extreme = current_rsi > 75 or current_rsi < 25

        # === Range ===
        result.range_high = float(w_high.max())
        result.range_low = float(w_low.min())
        range_size = result.range_high - result.range_low
        current_price = float(close.iloc[-1])
        result.range_position = (current_price - result.range_low) / max(range_size, 1e-10)

        # === Breadth (simplified) ===
        above_ma = float((w_close > w_close.rolling(20).mean()).tail(20).mean())
        result.breadth_score = above_ma

        # === Phase determination ===
        result.phase, result.confidence = self._classify_phase(result)
        result.next_phase_estimate = PHASE_TRANSITIONS.get(result.phase, CyclePhase.UNKNOWN)
        result.strategy_recommendation = PHASE_STRATEGIES.get(result.phase, "observe_wait")
        result.duration_bars = self._estimate_duration(w_close, result.phase)
        result.description = self._describe(result, current_rsi)

        return result

    # ------------------------------------------------------------------
    # Phase classification
    # ------------------------------------------------------------------
    def _classify_phase(self, r: CycleResult) -> Tuple[CyclePhase, float]:
        """Classify the current cycle phase."""
        confidence = 0.5

        # EXPANSION: rising + volume increasing + vol expanding
        if (r.price_trend > 0.3 and
            r.volume_trend in ("increasing", "stable") and
            r.volatility_trend in ("expanding", "stable") and
            r.range_position > 0.5):
            confidence = 0.8
            if r.volume_trend == "increasing" and r.volatility_trend == "expanding":
                confidence = 0.9
            return CyclePhase.EXPANSION, confidence

        # PEAK: high prices + volume declining + RSI extreme + top of range
        if (r.range_position > 0.85 and
            r.volume_trend == "decreasing" and
            r.rsi_extreme):
            confidence = 0.75
            return CyclePhase.PEAK, confidence

        # DECLINE: falling + volume increasing + vol expanding
        if (r.price_trend < -0.3 and
            r.volume_trend in ("increasing", "stable") and
            r.volatility_trend in ("expanding", "stable") and
            r.range_position < 0.5):
            confidence = 0.8
            if r.volume_trend == "increasing" and r.volatility_trend == "expanding":
                confidence = 0.9
            return CyclePhase.DECLINE, confidence

        # RECOVERY: bottoming + volume picking up + low RSI
        if (r.range_position < 0.25 and
            r.volume_trend in ("increasing", "stable") and
            r.rsi_extreme and
            r.price_trend > -0.2):
            confidence = 0.65
            return CyclePhase.RECOVERY, confidence

        # CONSOLIDATION: range-bound + low vol + volume declining
        if (abs(r.price_trend) < 0.2 and
            r.volatility_trend in ("contracting", "stable") and
            r.volume_trend == "decreasing"):
            confidence = 0.70
            return CyclePhase.CONSOLIDATION, confidence

        # Default: unknown
        return CyclePhase.UNKNOWN, 0.3

    def _estimate_duration(self, close: pd.Series, phase: CyclePhase) -> int:
        """Estimate how long we've been in this phase."""
        # Simple: count bars since last significant trend change
        if len(close) < 20:
            return 0
        changes = 0
        for i in range(len(close) - 1, max(0, len(close) - 100), -1):
            if abs(close.iloc[i] - close.iloc[i - 1]) / max(close.iloc[i - 1], 1e-10) > 0.03:
                changes += 1
                if changes >= 2:
                    return len(close) - i
        return min(len(close), 30)

    def _describe(self, r: CycleResult, rsi: float) -> str:
        """Human-readable description."""
        descs = {
            CyclePhase.EXPANSION: (
                f"EXPANSION phase — price trending up ({r.price_trend:+.2f}), "
                f"volume {r.volume_trend}, vol {r.volatility_trend}. "
                f"Strategy: trend following, full size."
            ),
            CyclePhase.PEAK: (
                f"PEAK phase — at top of range ({r.range_position:.0%}), "
                f"RSI={rsi:.0f} (extreme), volume declining. "
                f"Strategy: take profits, reduce size."
            ),
            CyclePhase.CONSOLIDATION: (
                f"CONSOLIDATION phase — range-bound, low volatility. "
                f"Strategy: range trading, small size."
            ),
            CyclePhase.DECLINE: (
                f"DECLINE phase — price falling ({r.price_trend:+.2f}), "
                f"volume {r.volume_trend}. "
                f"Strategy: short bias or defensive."
            ),
            CyclePhase.RECOVERY: (
                f"RECOVERY phase — bottoming pattern, RSI={rsi:.0f}. "
                f"Strategy: gradual accumulation."
            ),
            CyclePhase.UNKNOWN: "Phase unknown — observe and wait.",
        }
        return descs.get(r.phase, "Unknown phase")
