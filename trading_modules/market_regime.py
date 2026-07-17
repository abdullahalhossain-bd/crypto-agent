"""
Market Regime Detector — "What state is the market in?"
=========================================================

A regime is a higher-order classification than trend. A trend tells you
direction; a regime tells you HOW the market is behaving — and therefore
WHICH strategies will work.

Regimes detected:
    - TRENDING_UP      — strong directional up move, trend strategies work
    - TRENDING_DOWN    — strong directional down move
    - RANGING          — sideways, mean-reversion strategies work
    - HIGH_VOL_BREAKOUT— volatility expanding, breakout strategies work
    - LOW_VOL_DEAD     — ATR collapsed, nothing works — wait
    - CHOPPY           — high volatility, no direction — dangerous

The detector uses:
    1. ADX (trend strength)
    2. EMA slope + stack (direction)
    3. ATR / ATR-baseline ratio (volatility regime)
    4. Price position in recent range (range vs trend)
    5. Efficiency ratio (Kaufman — net move / sum of absolute moves)

Usage:
    from trading_modules.market_regime import MarketRegimeDetector, Regime

    detector = MarketRegimeDetector()
    regime = detector.detect(df_m15, df_h1)
    if regime == Regime.TRENDING_UP:
        # trend strategies active
    elif regime == Regime.RANGING:
        # mean-reversion active, trend strategies paused
    elif regime == Regime.LOW_VOL_DEAD:
        # skip all trading
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOL_BREAKOUT = "high_vol_breakout"
    LOW_VOL_DEAD = "low_vol_dead"
    CHOPPY = "choppy"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    regime: Regime
    adx: float                    # trend strength 0-100
    efficiency_ratio: float       # 0..1 — 1 = perfectly directional
    atr_ratio: float              # current ATR / 50-bar avg ATR
    ema_stack: str                # "bull" / "bear" / "mixed"
    range_position: float         # 0..1 — where price sits in recent range
    confidence: float             # 0..1 — how confident we are in the regime
    description: str              # human-readable summary

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "adx": round(self.adx, 1),
            "efficiency_ratio": round(self.efficiency_ratio, 3),
            "atr_ratio": round(self.atr_ratio, 2),
            "ema_stack": self.ema_stack,
            "range_position": round(self.range_position, 2),
            "confidence": round(self.confidence, 2),
            "description": self.description,
        }


class MarketRegimeDetector:
    """
    Detect the current market regime.

    Parameters:
        adx_period: ADX lookback (default 14)
        adx_trend_threshold: ADX above this = trending (default 25)
        adx_strong_threshold: ADX above this = strong trend (default 35)
        atr_period: ATR lookback (default 14)
        atr_baseline: # bars to average for ATR baseline (default 50)
        atr_low_ratio: below this = dead market (default 0.5)
        atr_high_ratio: above this = high-vol breakout (default 1.8)
        ema_fast / ema_slow: for EMA stack detection
        efficiency_window: Kaufman efficiency ratio lookback (default 20)
        range_lookback: bars for range high/low (default 50)
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_strong_threshold: float = 35.0,
        atr_period: int = 14,
        atr_baseline: int = 50,
        atr_low_ratio: float = 0.5,
        atr_high_ratio: float = 1.8,
        ema_fast: int = 20,
        ema_slow: int = 50,
        efficiency_window: int = 20,
        range_lookback: int = 50,
    ) -> None:
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_strong_threshold = adx_strong_threshold
        self.atr_period = atr_period
        self.atr_baseline = atr_baseline
        self.atr_low_ratio = atr_low_ratio
        self.atr_high_ratio = atr_high_ratio
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.efficiency_window = efficiency_window
        self.range_lookback = range_lookback

    def detect(
        self,
        df_m15: Optional[pd.DataFrame],
        df_h1: Optional[pd.DataFrame] = None,
    ) -> RegimeResult:
        """Detect the regime. Uses H1 for trend confirmation when available."""
        if df_m15 is None or len(df_m15) < max(self.ema_slow + 10, self.range_lookback):
            return RegimeResult(
                regime=Regime.UNKNOWN, adx=0, efficiency_ratio=0,
                atr_ratio=1.0, ema_stack="unknown", range_position=0.5,
                confidence=0.0, description="Insufficient data",
            )

        # Compute indicators
        adx = self._adx(df_m15, self.adx_period).iloc[-1]
        atr_now = self._atr(df_m15, self.atr_period).iloc[-1]
        atr_baseline_series = self._atr(df_m15, self.atr_period).rolling(self.atr_baseline).mean()
        atr_baseline = atr_baseline_series.iloc[-1] if not atr_baseline_series.empty else atr_now
        atr_ratio = atr_now / atr_baseline if atr_baseline and atr_baseline > 0 else 1.0

        # EMA stack
        close = df_m15["close"]
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        last_close = close.iloc[-1]
        if ema_f > ema_s and last_close > ema_f:
            ema_stack = "bull"
        elif ema_f < ema_s and last_close < ema_f:
            ema_stack = "bear"
        else:
            ema_stack = "mixed"

        # Efficiency ratio (Kaufman)
        eff = self._efficiency_ratio(close, self.efficiency_window)

        # Range position
        recent = df_m15.tail(self.range_lookback)
        rng_hi = recent["high"].max()
        rng_lo = recent["low"].min()
        rng = rng_hi - rng_lo
        range_position = (last_close - rng_lo) / rng if rng > 0 else 0.5

        # H1 confirmation
        htf_aligned = True
        if df_h1 is not None and len(df_h1) >= self.ema_slow + 5:
            h1_close = df_h1["close"]
            h1_ema_f = h1_close.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
            h1_ema_s = h1_close.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
            h1_last = h1_close.iloc[-1]
            if ema_stack == "bull" and not (h1_ema_f > h1_ema_s and h1_last > h1_ema_f):
                htf_aligned = False
            elif ema_stack == "bear" and not (h1_ema_f < h1_ema_s and h1_last < h1_ema_f):
                htf_aligned = False

        # Decide regime
        regime, confidence, description = self._classify(
            adx=adx, atr_ratio=atr_ratio, ema_stack=ema_stack,
            efficiency=eff, range_position=range_position,
            htf_aligned=htf_aligned,
        )

        return RegimeResult(
            regime=regime,
            adx=float(adx) if not np.isnan(adx) else 0.0,
            efficiency_ratio=float(eff),
            atr_ratio=float(atr_ratio),
            ema_stack=ema_stack,
            range_position=float(range_position),
            confidence=confidence,
            description=description,
        )

    # ------------------------------------------------------------------
    def _classify(
        self,
        adx: float,
        atr_ratio: float,
        ema_stack: str,
        efficiency: float,
        range_position: float,
        htf_aligned: bool,
    ) -> tuple[Regime, float, str]:
        """Classify regime from indicators. Returns (regime, confidence, desc)."""
        adx = 0 if np.isnan(adx) else adx
        eff = efficiency  # alias for short f-string references

        # Dead market — ATR collapsed
        if atr_ratio < self.atr_low_ratio:
            return (Regime.LOW_VOL_DEAD, 0.85,
                    f"ATR ratio {atr_ratio:.2f} < {self.atr_low_ratio} — dead market")

        # High-vol breakout — ATR exploding
        if atr_ratio > self.atr_high_ratio and adx > self.adx_trend_threshold:
            direction = "up" if ema_stack == "bull" else "down" if ema_stack == "bear" else "mixed"
            return (Regime.HIGH_VOL_BREAKOUT, 0.80,
                    f"Volatility expanding (ATR {atr_ratio:.2f}x), ADX={adx:.0f}, direction={direction}")

        # Strong trend
        if adx >= self.adx_strong_threshold and ema_stack in ("bull", "bear") and efficiency > 0.3:
            if ema_stack == "bull":
                conf = 0.90 if htf_aligned else 0.65
                return (Regime.TRENDING_UP, conf,
                        f"Strong uptrend ADX={adx:.0f} eff={eff:.2f} HTF={'OK' if htf_aligned else 'NO'}")
            else:
                conf = 0.90 if htf_aligned else 0.65
                return (Regime.TRENDING_DOWN, conf,
                        f"Strong downtrend ADX={adx:.0f} eff={eff:.2f} HTF={'OK' if htf_aligned else 'NO'}")

        # Weak trend
        if adx >= self.adx_trend_threshold and ema_stack in ("bull", "bear") and efficiency > 0.15:
            if ema_stack == "bull":
                conf = 0.70 if htf_aligned else 0.45
                return (Regime.TRENDING_UP, conf,
                        f"Weak uptrend ADX={adx:.0f} eff={eff:.2f}")
            else:
                conf = 0.70 if htf_aligned else 0.45
                return (Regime.TRENDING_DOWN, conf,
                        f"Weak downtrend ADX={adx:.0f} eff={eff:.2f}")

        # Ranging — ADX low, efficiency low, price in middle
        if adx < self.adx_trend_threshold and efficiency < 0.2:
            return (Regime.RANGING, 0.75,
                    f"Ranging ADX={adx:.0f} eff={eff:.2f} pos={range_position:.2f}")

        # Choppy — high ATR, low efficiency, no clear trend
        if atr_ratio > 1.3 and efficiency < 0.2:
            return (Regime.CHOPPY, 0.65,
                    f"Choppy ATR={atr_ratio:.2f}x eff={eff:.2f} — dangerous")

        # Default to ranging
        return (Regime.RANGING, 0.50,
                f"Unclear — ADX={adx:.0f} eff={eff:.2f} ATR={atr_ratio:.2f}")

    # ------------------------------------------------------------------
    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        up = h.diff()
        down = -l.diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int) -> float:
        """Kaufman Efficiency Ratio: |net change| / sum(|bar changes|)."""
        if len(close) < window:
            return 0.0
        recent = close.tail(window)
        net = abs(recent.iloc[-1] - recent.iloc[0])
        gross = recent.diff().abs().sum()
        if gross == 0:
            return 0.0
        return float(net / gross)


# ----------------------------------------------------------------------
# Strategy compatibility map — which strategies work in which regime
# ----------------------------------------------------------------------
STRATEGY_REGIME_COMPAT: dict[Regime, list[str]] = {
    Regime.TRENDING_UP:       ["trend_follow", "breakout", "pullback"],
    Regime.TRENDING_DOWN:     ["trend_follow", "breakout", "pullback"],
    Regime.RANGING:           ["mean_reversion", "range_scalp"],
    Regime.HIGH_VOL_BREAKOUT: ["breakout", "momentum"],
    Regime.LOW_VOL_DEAD:      [],  # no strategies — wait
    Regime.CHOPPY:            [],  # no strategies — dangerous
    Regime.UNKNOWN:           [],
}


def regime_allows_trading(regime: Regime) -> bool:
    """True if any strategy works in this regime."""
    return len(STRATEGY_REGIME_COMPAT.get(regime, [])) > 0
