"""trading_modules/market_context_engine.py
=====================================================================
Market Context Engine (Principle #141, #142)
=====================================================================
Evaluates the FULL market context before any signal is considered.

The same indicator can profit one day and lose the next — because the
CONTEXT changed. This engine aggregates 7 context dimensions into a
single score that gates every trade.

Context Dimensions:
    1. TREND        — is there a clear trend? (EMA stack + ADX)
    2. SESSION      — London/NY/Asia/Overlap?
    3. LIQUIDITY    — spread + depth + volume
    4. NEWS         — high-impact news pending?
    5. VOLATILITY   — ATR percentile (low/normal/high/extreme)
    6. CORRELATION  — how correlated is this asset to others right now?
    7. REGIME       — trend / range / breakout / crisis

Output:
    - context_score (0-100)
    - context_understood (bool) — do we know WHY price is moving?
    - regime classification
    - "trade in this context?" recommendation

Usage:
    engine = MarketContextEngine()
    ctx = engine.evaluate(df, spread_bps=3.0, session="london",
                         news_pending=False, correlated_assets={...})
    if ctx.context_understood and ctx.context_score > 60:
        # Signal can be evaluated
        evaluate_signal()
    else:
        # Skip — context not favorable or not understood
        skip()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.market_context_engine")


class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class MarketRegime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    BREAKOUT = "breakout"
    CRISIS = "crisis"
    UNKNOWN = "unknown"


@dataclass
class ContextResult:
    """Market context evaluation result."""
    context_score: float = 0.0           # 0-100
    context_understood: bool = False     # do we know why price is moving?
    can_trade: bool = False              # is context favorable?

    # Per-dimension scores (0-1)
    trend_score: float = 0.0
    session_score: float = 0.0
    liquidity_score: float = 0.0
    news_score: float = 0.0
    volatility_score: float = 0.0
    correlation_score: float = 0.0
    regime_score: float = 0.0

    # Classifications
    volatility_regime: VolatilityRegime = VolatilityRegime.NORMAL
    market_regime: MarketRegime = MarketRegime.UNKNOWN
    session: str = "off_hours"

    # Why is price moving?
    price_driver: str = "unknown"        # trend/news/liquidity/correlation/institution
    description: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_score": round(self.context_score, 1),
            "context_understood": self.context_understood,
            "can_trade": self.can_trade,
            "trend_score": round(self.trend_score, 3),
            "session_score": round(self.session_score, 3),
            "liquidity_score": round(self.liquidity_score, 3),
            "news_score": round(self.news_score, 3),
            "volatility_score": round(self.volatility_score, 3),
            "correlation_score": round(self.correlation_score, 3),
            "regime_score": round(self.regime_score, 3),
            "volatility_regime": self.volatility_regime.value,
            "market_regime": self.market_regime.value,
            "session": self.session,
            "price_driver": self.price_driver,
            "description": self.description,
            "recommendations": self.recommendations,
        }


class MarketContextEngine:
    """Evaluates full market context before signal evaluation."""

    def __init__(self,
                 min_context_score: float = 50.0,
                 news_blackout_minutes: float = 30.0,
                 high_spread_bps: float = 15.0):
        """Initialize engine.

        Args:
            min_context_score: minimum context score to allow trading
            news_blackout_minutes: no trades within N minutes of news
            high_spread_bps: spread above this = poor liquidity
        """
        self.min_score = min_context_score
        self.news_blackout = news_blackout_minutes
        self.high_spread = high_spread_bps

    def evaluate(self,
                 df: pd.DataFrame,
                 spread_bps: float = 5.0,
                 session: str = "off_hours",
                 news_pending: bool = False,
                 minutes_to_news: float = 999,
                 correlated_returns: Optional[Dict[str, pd.Series]] = None,
                 benchmark_returns: Optional[pd.Series] = None) -> ContextResult:
        """Evaluate full market context.

        Args:
            df: OHLCV DataFrame for the target symbol
            spread_bps: current bid-ask spread
            session: "london", "new_york", "asia", "overlap", "off_hours"
            news_pending: is high-impact news pending?
            minutes_to_news: minutes until next news event
            correlated_returns: returns of correlated assets {symbol: returns}
            benchmark_returns: benchmark returns for correlation calc

        Returns:
            ContextResult with all 7 dimensions scored
        """
        result = ContextResult(session=session)

        if df is None or df.empty or len(df) < 50:
            result.description = "insufficient data"
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # === 1. TREND score ===
        result.trend_score = self._score_trend(df)

        # === 2. SESSION score ===
        result.session_score = self._score_session(session)

        # === 3. LIQUIDITY score ===
        result.liquidity_score = self._score_liquidity(df, spread_bps)

        # === 4. NEWS score ===
        result.news_score = self._score_news(news_pending, minutes_to_news)

        # === 5. VOLATILITY score ===
        result.volatility_score, result.volatility_regime = self._score_volatility(df)

        # === 6. CORRELATION score ===
        result.correlation_score = self._score_correlation(
            close, correlated_returns, benchmark_returns)

        # === 7. REGIME score ===
        result.regime_score, result.market_regime = self._score_regime(df)

        # === Composite context score ===
        weights = {
            "trend": 0.20, "session": 0.10, "liquidity": 0.15,
            "news": 0.15, "volatility": 0.15, "correlation": 0.10,
            "regime": 0.15,
        }
        result.context_score = (
            result.trend_score * weights["trend"] * 100 +
            result.session_score * weights["session"] * 100 +
            result.liquidity_score * weights["liquidity"] * 100 +
            result.news_score * weights["news"] * 100 +
            result.volatility_score * weights["volatility"] * 100 +
            result.correlation_score * weights["correlation"] * 100 +
            result.regime_score * weights["regime"] * 100
        )

        # === Context understood? ===
        result.context_understood = self._is_context_understood(result, df)

        # === Can trade? ===
        result.can_trade = (
            result.context_score >= self.min_score and
            result.context_understood and
            result.news_score > 0.3 and
            result.liquidity_score > 0.3
        )

        # === Price driver ===
        result.price_driver = self._identify_price_driver(result, df)

        # === Description + recommendations ===
        result.description = self._describe(result)
        result.recommendations = self._recommend(result)

        return result

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------
    def _score_trend(self, df: pd.DataFrame) -> float:
        """Score trend clarity (0-1). Higher = clearer trend."""
        close = df["close"]
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(close.iloc[-1])

        score = 0.0
        if ema9 > ema21 > ema50:
            score += 0.4
        if (price > ema21) == (ema9 > ema21):  # price on correct side
            score += 0.3
        # ADX-like: slope strength
        slope = abs(ema21 - ema50) / max(ema50, 1e-10)
        if slope > 0.001:
            score += 0.3
        return min(1.0, score)

    def _score_session(self, session: str) -> float:
        """Score session quality (0-1)."""
        scores = {
            "london": 1.0, "new_york": 1.0, "overlap": 0.9,
            "asia": 0.5, "off_hours": 0.2,
        }
        return scores.get(session, 0.3)

    def _score_liquidity(self, df: pd.DataFrame, spread_bps: float) -> float:
        """Score liquidity quality (0-1)."""
        score = 0.5
        # Spread
        if spread_bps < 2:
            score += 0.3
        elif spread_bps < 5:
            score += 0.2
        elif spread_bps < 10:
            score += 0.1
        else:
            score -= 0.2
        # Volume
        if "volume" in df:
            recent_vol = float(df["volume"].tail(10).mean())
            avg_vol = float(df["volume"].tail(50).mean())
            rvol = recent_vol / max(avg_vol, 1)
            if rvol > 1.0:
                score += 0.2
            elif rvol < 0.5:
                score -= 0.1
        return max(0, min(1, score))

    def _score_news(self, news_pending: bool, minutes_to_news: float) -> float:
        """Score news risk (0-1). Higher = safer (no news nearby)."""
        if news_pending and minutes_to_news < 15:
            return 0.0  # too close to news
        if minutes_to_news < 30:
            return 0.3
        if minutes_to_news < 60:
            return 0.6
        return 1.0

    def _score_volatility(self, df: pd.DataFrame) -> tuple:
        """Score volatility (0-1) + regime classification.

        Returns (score, regime)
        Score: 1.0 = normal vol, lower = too low or too high
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        atr_pct = float(atr.iloc[-1] / max(close.iloc[-1], 1e-10) * 100)

        # Percentile
        percentile = float(atr.tail(100).rank(pct=True).iloc[-1])

        # Regime
        if percentile > 0.95 or atr_pct > 5:
            regime = VolatilityRegime.EXTREME
            score = 0.2
        elif percentile > 0.80:
            regime = VolatilityRegime.HIGH
            score = 0.5
        elif percentile < 0.10:
            regime = VolatilityRegime.LOW
            score = 0.6
        else:
            regime = VolatilityRegime.NORMAL
            score = 1.0

        return score, regime

    def _score_correlation(self, close: pd.Series,
                           correlated: Optional[Dict[str, pd.Series]],
                           benchmark: Optional[pd.Series]) -> float:
        """Score correlation health (0-1).

        Lower correlation = more diversification opportunity = higher score.
        """
        if not correlated and benchmark is None:
            return 0.7  # neutral

        returns = close.pct_change().dropna()
        max_corr = 0.0

        if benchmark is not None:
            common = returns.index.intersection(benchmark.index)
            if len(common) > 20:
                c = abs(float(returns.loc[common].corr(benchmark.loc[common])))
                max_corr = max(max_corr, c)

        for sym, rets in (correlated or {}).items():
            common = returns.index.intersection(rets.index)
            if len(common) > 20:
                c = abs(float(returns.loc[common].corr(rets.loc[common])))
                max_corr = max(max_corr, c)

        # Higher correlation = lower score
        return max(0, 1.0 - max_corr)

    def _score_regime(self, df: pd.DataFrame) -> tuple:
        """Score regime clarity (0-1) + regime classification.

        Returns (score, regime)
        """
        close = df["close"]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(close.iloc[-1])

        # Recent range
        recent_high = float(df["high"].tail(50).max())
        recent_low = float(df["low"].tail(50).min())
        range_pct = (recent_high - recent_low) / max(recent_low, 1e-10)

        # ADX-like
        slope = abs(ema21 - ema50) / max(ema50, 1e-10)

        if slope > 0.005 and price > ema50:
            return 0.9, MarketRegime.TREND_UP
        elif slope > 0.005 and price < ema50:
            return 0.9, MarketRegime.TREND_DOWN
        elif range_pct < 0.03:
            return 0.7, MarketRegime.RANGE
        elif range_pct > 0.08:
            return 0.5, MarketRegime.BREAKOUT
        elif range_pct > 0.15:
            return 0.3, MarketRegime.CRISIS
        return 0.4, MarketRegime.UNKNOWN

    # ------------------------------------------------------------------
    # Context understanding
    # ------------------------------------------------------------------
    def _is_context_understood(self, result: ContextResult, df: pd.DataFrame) -> bool:
        """Determine if we understand WHY price is moving."""
        # Must have at least trend OR regime clarity
        if result.trend_score < 0.3 and result.regime_score < 0.4:
            return False
        # Must not be in crisis without understanding
        if result.market_regime == MarketRegime.CRISIS and result.volatility_regime == VolatilityRegime.EXTREME:
            return False  # crisis is hard to understand
        return True

    def _identify_price_driver(self, result: ContextResult, df: pd.DataFrame) -> str:
        """What is driving the price right now?"""
        if result.news_score < 0.3:
            return "news"
        if result.volatility_regime == VolatilityRegime.EXTREME:
            return "panic_or_euphoria"
        if result.trend_score > 0.7:
            return "trend"
        if result.market_regime == MarketRegime.RANGE:
            return "mean_reversion"
        if result.liquidity_score < 0.3:
            return "low_liquidity"
        return "mixed"

    def _describe(self, r: ContextResult) -> str:
        """Human-readable description."""
        return (
            f"Context score: {r.context_score:.0f}/100 "
            f"({r.market_regime.value}, {r.volatility_regime.value} vol, "
            f"session={r.session}, driver={r.price_driver}). "
            f"{'CAN TRADE' if r.can_trade else 'CANNOT TRADE — context unfavorable'}"
        )

    def _recommend(self, r: ContextResult) -> List[str]:
        """Recommendations based on context."""
        recs = []
        if not r.context_understood:
            recs.append("Context not understood — wait for clarity")
        if r.volatility_regime == VolatilityRegime.EXTREME:
            recs.append("Extreme volatility — reduce size 70% or skip")
        elif r.volatility_regime == VolatilityRegime.HIGH:
            recs.append("High volatility — reduce size 30%")
        if r.liquidity_score < 0.3:
            recs.append("Poor liquidity — wait for better spread/depth")
        if r.news_score < 0.3:
            recs.append("News imminent — no new entries")
        if r.market_regime == MarketRegime.CRISIS:
            recs.append("Crisis regime — defensive mode only")
        if r.can_trade and not recs:
            recs.append("Context favorable — proceed with signal evaluation")
        return recs
