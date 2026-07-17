"""trading_modules/setup_scoring_engine.py
=====================================================================
Dynamic Setup Scoring Engine (Principle #104 — Avoid Random Entries)
=====================================================================
Scores every trade setup on a 0-100 scale across 5 dimensions:

    Trend       : 25 points  — EMA stacking, ADX, slope
    Volume      : 20 points  — OBV direction, RVol, CMF
    Liquidity   : 20 points  — spread, depth, slippage estimate
    Volatility  : 15 points  — ATR percentile, BBands width
    Timing      : 20 points  — session, pullback, news proximity

    ──────────────────────
    Total       : 100 points
    ──────────────────────

    Minimum Score to Trade: 80 (configurable)

Usage:
    engine = SetupScoringEngine(min_score=80)
    score = engine.score(df, signal_action="BUY",
                        spread_bps=2.5, session="london",
                        has_pullback=True, news_minutes=120)
    if score.passed:
        place_trade()
    else:
        log(f"Rejected: score={score.total}/100, weakest={score.weakest_dimension}")

    # Also provides position size multiplier:
    # score 80-85 → 0.8x
    # score 85-90 → 1.0x
    # score 90-95 → 1.2x
    # score 95+   → 1.5x
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.setup_scoring_engine")


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    name: str
    score: float        # 0 to max_points
    max_points: float
    detail: str = ""

    @property
    def pct(self) -> float:
        return self.score / max(self.max_points, 1)


@dataclass
class SetupScore:
    """Complete setup score across all 5 dimensions."""
    trend: DimensionScore = field(default_factory=lambda: DimensionScore("trend", 0, 25))
    volume: DimensionScore = field(default_factory=lambda: DimensionScore("volume", 0, 20))
    liquidity: DimensionScore = field(default_factory=lambda: DimensionScore("liquidity", 0, 20))
    volatility: DimensionScore = field(default_factory=lambda: DimensionScore("volatility", 0, 15))
    timing: DimensionScore = field(default_factory=lambda: DimensionScore("timing", 0, 20))

    # Composite
    total: float = 0.0
    passed: bool = False
    position_multiplier: float = 1.0
    weakest_dimension: str = ""
    strongest_dimension: str = ""
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": {
                "trend": {"score": self.trend.score, "max": self.trend.max_points,
                         "pct": round(self.trend.pct, 3), "detail": self.trend.detail},
                "volume": {"score": self.volume.score, "max": self.volume.max_points,
                          "pct": round(self.volume.pct, 3), "detail": self.volume.detail},
                "liquidity": {"score": self.liquidity.score, "max": self.liquidity.max_points,
                             "pct": round(self.liquidity.pct, 3), "detail": self.liquidity.detail},
                "volatility": {"score": self.volatility.score, "max": self.volatility.max_points,
                              "pct": round(self.volatility.pct, 3), "detail": self.volatility.detail},
                "timing": {"score": self.timing.score, "max": self.timing.max_points,
                          "pct": round(self.timing.pct, 3), "detail": self.timing.detail},
            },
            "total": round(self.total, 1),
            "passed": self.passed,
            "position_multiplier": round(self.position_multiplier, 2),
            "weakest": self.weakest_dimension,
            "strongest": self.strongest_dimension,
            "recommendation": self.recommendation,
        }


class SetupScoringEngine:
    """Scores trade setups on a 0-100 scale.

    Each dimension is scored independently, then combined.
    A minimum score (default 80) is required to pass.
    """

    def __init__(self,
                 min_score: float = 80.0,
                 # Dimension weights (must sum to 100)
                 trend_weight: float = 25.0,
                 volume_weight: float = 20.0,
                 liquidity_weight: float = 20.0,
                 volatility_weight: float = 15.0,
                 timing_weight: float = 20.0):
        self.min_score = min_score
        self.weights = {
            "trend": trend_weight,
            "volume": volume_weight,
            "liquidity": liquidity_weight,
            "volatility": volatility_weight,
            "timing": timing_weight,
        }
        total_weight = sum(self.weights.values())
        assert abs(total_weight - 100) < 0.1, f"weights must sum to 100, got {total_weight}"

    def score(self,
              df: pd.DataFrame,
              signal_action: str = "BUY",
              spread_bps: float = 5.0,
              slippage_estimate_bps: float = 2.0,
              orderbook_depth_usd: float = 1_000_000,
              session: str = "off_hours",
              has_pullback: bool = False,
              news_minutes: float = 999,
              high_impact_news: bool = False,
              confidence: float = 0.5) -> SetupScore:
        """Score a trade setup.

        Args:
            df: OHLCV DataFrame
            signal_action: "BUY" or "SELL"
            spread_bps: current bid-ask spread in bps
            slippage_estimate_bps: expected slippage for our order size
            orderbook_depth_usd: depth at ±1%
            session: "asia", "london", "new_york", "overlap", "off_hours"
            has_pullback: did price pull back to support before entry?
            news_minutes: minutes until next high-impact news
            high_impact_news: is there high-impact news pending?
            confidence: strategy confidence (0-1)

        Returns:
            SetupScore with per-dimension breakdown + recommendation
        """
        result = SetupScore()

        # === 1. TREND (25 points) ===
        result.trend = self._score_trend(df, signal_action)

        # === 2. VOLUME (20 points) ===
        result.volume = self._score_volume(df, signal_action)

        # === 3. LIQUIDITY (20 points) ===
        result.liquidity = self._score_liquidity(
            spread_bps, slippage_estimate_bps, orderbook_depth_usd)

        # === 4. VOLATILITY (15 points) ===
        result.volatility = self._score_volatility(df)

        # === 5. TIMING (20 points) ===
        result.timing = self._score_timing(
            session, has_pullback, news_minutes, high_impact_news, confidence)

        # === Composite ===
        dimensions = [result.trend, result.volume, result.liquidity,
                     result.volatility, result.timing]
        result.total = sum(d.score for d in dimensions)

        # Find weakest + strongest
        result.weakest_dimension = min(dimensions, key=lambda d: d.pct).name
        result.strongest_dimension = max(dimensions, key=lambda d: d.pct).name

        # Pass/fail
        result.passed = result.total >= self.min_score

        # Position multiplier
        result.position_multiplier = self._compute_multiplier(result.total)

        # Recommendation
        if result.passed:
            result.recommendation = (
                f"APPROVE — score {result.total:.0f}/100 "
                f"(strongest: {result.strongest_dimension}, "
                f"size: {result.position_multiplier:.1f}x)"
            )
        else:
            gap = self.min_score - result.total
            result.recommendation = (
                f"REJECT — score {result.total:.0f}/100 "
                f"(need {gap:.0f} more, weakest: {result.weakest_dimension})"
            )

        return result

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------
    def _score_trend(self, df: pd.DataFrame, action: str) -> DimensionScore:
        """Score trend alignment (25 points max)."""
        if df is None or df.empty or len(df) < 50:
            return DimensionScore("trend", 5, 25, "insufficient data")

        close = df["close"]
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(close.iloc[-1])

        score = 0.0
        details = []

        # EMA stacking (10 points)
        if action == "BUY":
            if ema9 > ema21 > ema50:
                score += 10
                details.append("EMA9>21>50 (bullish stack)")
            elif ema9 > ema21:
                score += 5
                details.append("EMA9>21 only")
        elif action == "SELL":
            if ema9 < ema21 < ema50:
                score += 10
                details.append("EMA9<21<50 (bearish stack)")
            elif ema9 < ema21:
                score += 5
                details.append("EMA9<21 only")

        # Price vs EMA50 (5 points)
        if action == "BUY" and price > ema50:
            score += 5
            details.append("price > EMA50")
        elif action == "SELL" and price < ema50:
            score += 5
            details.append("price < EMA50")

        # Slope of EMA50 (5 points)
        ema50_slope = (ema50 - close.ewm(span=50, adjust=False).mean().iloc[-5]) / 5
        if action == "BUY" and ema50_slope > 0:
            score += 5
            details.append("EMA50 rising")
        elif action == "SELL" and ema50_slope < 0:
            score += 5
            details.append("EMA50 falling")

        # ADX-like strength (5 points) — simplified
        recent_range = float(df["high"].tail(20).max() - df["low"].tail(20).min())
        avg_range = float((df["high"] - df["low"]).tail(50).mean())
        if recent_range > avg_range * 1.2:
            score += 5
            details.append("range expanding (trend strong)")

        return DimensionScore("trend", min(score, 25), 25, "; ".join(details))

    def _score_volume(self, df: pd.DataFrame, action: str) -> DimensionScore:
        """Score volume confirmation (20 points max)."""
        if df is None or df.empty or "volume" not in df:
            return DimensionScore("volume", 5, 20, "no volume data")

        vol = df["volume"]
        close = df["close"]
        recent_vol = float(vol.tail(10).mean())
        avg_vol = float(vol.tail(20).mean())
        rvol = recent_vol / max(avg_vol, 1)

        score = 0.0
        details = []

        # RVol (8 points)
        if rvol > 1.5:
            score += 8
            details.append(f"RVol={rvol:.2f} (high)")
        elif rvol > 1.0:
            score += 5
            details.append(f"RVol={rvol:.2f} (normal)")
        else:
            details.append(f"RVol={rvol:.2f} (low)")

        # Volume direction (6 points)
        # Bullish: up bars have higher volume
        up_vol = float(vol[close > close.shift()].tail(10).sum())
        dn_vol = float(vol[close < close.shift()].tail(10).sum())
        if action == "BUY" and up_vol > dn_vol * 1.2:
            score += 6
            details.append("up-volume > down-volume")
        elif action == "SELL" and dn_vol > up_vol * 1.2:
            score += 6
            details.append("down-volume > up-volume")

        # OBV direction (6 points)
        obv = (close.diff().apply(np.sign) * vol).fillna(0).cumsum()
        obv_slope = obv.iloc[-1] - obv.iloc[-10] if len(obv) > 10 else 0
        if action == "BUY" and obv_slope > 0:
            score += 6
            details.append("OBV rising")
        elif action == "SELL" and obv_slope < 0:
            score += 6
            details.append("OBV falling")

        return DimensionScore("volume", min(score, 20), 20, "; ".join(details))

    def _score_liquidity(self, spread_bps: float,
                        slippage_bps: float,
                        depth_usd: float) -> DimensionScore:
        """Score liquidity quality (20 points max)."""
        score = 0.0
        details = []

        # Spread (8 points)
        if spread_bps < 2:
            score += 8
            details.append(f"spread={spread_bps:.1f}bps (excellent)")
        elif spread_bps < 5:
            score += 6
            details.append(f"spread={spread_bps:.1f}bps (good)")
        elif spread_bps < 10:
            score += 3
            details.append(f"spread={spread_bps:.1f}bps (acceptable)")
        else:
            details.append(f"spread={spread_bps:.1f}bps (poor)")

        # Slippage estimate (6 points)
        if slippage_bps < 1:
            score += 6
            details.append(f"slippage={slippage_bps:.1f}bps (low)")
        elif slippage_bps < 3:
            score += 4
            details.append(f"slippage={slippage_bps:.1f}bps (ok)")
        elif slippage_bps < 5:
            score += 2
            details.append(f"slippage={slippage_bps:.1f}bps (moderate)")

        # Orderbook depth (6 points)
        if depth_usd > 5_000_000:
            score += 6
            details.append(f"depth=${depth_usd/1e6:.1f}M (deep)")
        elif depth_usd > 1_000_000:
            score += 4
            details.append(f"depth=${depth_usd/1e6:.1f}M (adequate)")
        elif depth_usd > 250_000:
            score += 2
            details.append(f"depth=${depth_usd/1e6:.1f}M (thin)")

        return DimensionScore("liquidity", score, 20, "; ".join(details))

    def _score_volatility(self, df: pd.DataFrame) -> DimensionScore:
        """Score volatility regime (15 points max).

        Best: normal volatility (not too low, not too high)
        """
        if df is None or df.empty or len(df) < 20:
            return DimensionScore("volatility", 5, 15, "insufficient data")

        # ATR%
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        atr_pct = float(atr.iloc[-1] / max(close.iloc[-1], 1e-10) * 100)

        score = 0.0
        details = []

        # ATR% sweet spot (8 points)
        if 0.5 < atr_pct < 2.0:
            score += 8
            details.append(f"ATR%={atr_pct:.2f} (optimal)")
        elif 0.2 < atr_pct < 3.0:
            score += 5
            details.append(f"ATR%={atr_pct:.2f} (acceptable)")
        elif atr_pct >= 3.0:
            score += 2
            details.append(f"ATR%={atr_pct:.2f} (high vol)")
        else:
            details.append(f"ATR%={atr_pct:.2f} (dead)")

        # Volatility percentile (7 points)
        atr_pctile = float(atr.tail(100).rank(pct=True).iloc[-1])
        if 0.3 < atr_pctile < 0.7:
            score += 7
            details.append(f"vol pctile={atr_pctile:.0%} (normal)")
        elif 0.2 < atr_pctile < 0.8:
            score += 4
            details.append(f"vol pctile={atr_pctile:.0%} (borderline)")
        else:
            details.append(f"vol pctile={atr_pctile:.0%} (extreme)")

        return DimensionScore("volatility", score, 15, "; ".join(details))

    def _score_timing(self, session: str, has_pullback: bool,
                      news_minutes: float, high_impact_news: bool,
                      confidence: float) -> DimensionScore:
        """Score entry timing (20 points max)."""
        score = 0.0
        details = []

        # Session quality (8 points)
        session_scores = {
            "london": 8, "new_york": 8, "overlap": 7,
            "asia": 4, "off_hours": 1,
        }
        s = session_scores.get(session, 2)
        score += s
        details.append(f"session={session} ({s}/8)")

        # Pullback to support (5 points)
        if has_pullback:
            score += 5
            details.append("pullback confirmed")
        else:
            details.append("no pullback")

        # News proximity (7 points)
        if high_impact_news and news_minutes < 30:
            details.append(f"NEWS in {news_minutes:.0f}min — BLOCKED")
        elif news_minutes < 15:
            details.append(f"news in {news_minutes:.0f}min (too close)")
        elif news_minutes < 60:
            score += 2
            details.append(f"news in {news_minutes:.0f}min (caution)")
        elif news_minutes > 120:
            score += 7
            details.append(f"news in {news_minutes:.0f}min (safe)")
        else:
            score += 4
            details.append(f"news in {news_minutes:.0f}min (ok)")

        # Confidence bonus (already in strategy, but timing adds a bit)
        if confidence > 0.7:
            score = min(score + 0, 20)  # cap

        return DimensionScore("timing", min(score, 20), 20, "; ".join(details))

    # ------------------------------------------------------------------
    # Position sizing multiplier
    # ------------------------------------------------------------------
    def _compute_multiplier(self, total_score: float) -> float:
        """Convert score to position size multiplier.

        80-85 → 0.8x
        85-90 → 1.0x
        90-95 → 1.2x
        95+   → 1.5x
        below 80 → 0x (no trade)
        """
        if total_score < 80:
            return 0.0
        elif total_score < 85:
            return 0.8
        elif total_score < 90:
            return 1.0
        elif total_score < 95:
            return 1.2
        else:
            return 1.5
