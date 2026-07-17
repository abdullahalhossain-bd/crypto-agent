"""trading_modules/trade_opportunity_ranker.py
=====================================================================
Trade Opportunity Ranking Engine (Principle #181, #182, #186)
=====================================================================
Ranks every potential trade setup on a 0-100 scale so the bot only
takes the BEST opportunities. "Great traders wait more than they trade."

Scoring Dimensions (7 factors, 100 points total):
    1. TREND ALIGNMENT     (20) — is this trade with the trend?
    2. LIQUIDITY QUALITY   (15) — spread + depth + volume
    3. MOMENTUM STRENGTH   (15) — RSI + MACD + ROC alignment
    4. VOLUME CONFIRMATION (15) — OBV + RVol + delta
    5. STRUCTURE QUALITY   (15) — HH/HL + BOS + key levels
    6. TIMING QUALITY      (10) — session + pullback + news proximity
    7. RISK/REWARD         (10) — R:R ratio + Kelly fraction

Ranking Thresholds:
    Score ≥ 85  → PRIORITY (take first)
    Score 70-85 → GOOD (take if no priority)
    Score 55-70 → MARGINAL (skip unless no other options)
    Score < 55  → SKIP

Usage:
    ranker = TradeOpportunityRanker()

    # Rank multiple opportunities
    ranking = ranker.rank_opportunities([
        {"symbol": "BTCUSD", "df": df_btc, "action": "BUY", ...},
        {"symbol": "ETHUSD", "df": df_eth, "action": "BUY", ...},
        {"symbol": "EURUSD", "df": df_eur, "action": "SELL", ...},
    ])
    # ranking = {
    #     "best": "BTCUSD",
    #     "ranking": [{"symbol": "BTCUSD", "score": 87.5}, ...],
    #     "actionable": ["BTCUSD", "ETHUSD"],
    #     "skipped": ["EURUSD"],
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.trade_opportunity_ranker")


@dataclass
class OpportunityScore:
    """Score for a single trade opportunity."""
    symbol: str
    action: str = "HOLD"
    score: float = 0.0          # 0-100
    rank: int = 0               # 1 = best

    # Per-dimension scores
    trend_alignment: float = 0.0     # 0-20
    liquidity_quality: float = 0.0   # 0-15
    momentum_strength: float = 0.0   # 0-15
    volume_confirmation: float = 0.0 # 0-15
    structure_quality: float = 0.0   # 0-15
    timing_quality: float = 0.0      # 0-10
    risk_reward: float = 0.0         # 0-10

    # Classification
    tier: str = "SKIP"  # PRIORITY, GOOD, MARGINAL, SKIP

    # Details
    details: Dict[str, str] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "score": round(self.score, 1),
            "rank": self.rank,
            "dimensions": {
                "trend": round(self.trend_alignment, 1),
                "liquidity": round(self.liquidity_quality, 1),
                "momentum": round(self.momentum_strength, 1),
                "volume": round(self.volume_confirmation, 1),
                "structure": round(self.structure_quality, 1),
                "timing": round(self.timing_quality, 1),
                "rr": round(self.risk_reward, 1),
            },
            "tier": self.tier,
            "recommendation": self.recommendation,
        }


class TradeOpportunityRanker:
    """Ranks trade opportunities to find the best setups."""

    def __init__(self,
                 priority_threshold: float = 85.0,
                 good_threshold: float = 70.0,
                 marginal_threshold: float = 55.0):
        """Initialize ranker.

        Args:
            priority_threshold: score >= this = PRIORITY tier
            good_threshold: score >= this = GOOD tier
            marginal_threshold: score >= this = MARGINAL tier
        """
        self.priority = priority_threshold
        self.good = good_threshold
        self.marginal = marginal_threshold

    def score_opportunity(self,
                          symbol: str,
                          df: pd.DataFrame,
                          action: str = "BUY",
                          spread_bps: float = 5.0,
                          session: str = "off_hours",
                          sl: float = 0.0,
                          tp: float = 0.0,
                          entry_price: float = 0.0,
                          news_minutes: float = 999) -> OpportunityScore:
        """Score a single trade opportunity.

        Args:
            symbol: trading symbol
            df: OHLCV DataFrame
            action: "BUY" or "SELL"
            spread_bps: current spread
            session: current session
            sl: stop loss price
            tp: take profit price
            entry_price: intended entry
            news_minutes: minutes to next news

        Returns:
            OpportunityScore with 7-dimension breakdown
        """
        opp = OpportunityScore(symbol=symbol, action=action)

        if df is None or df.empty or len(df) < 50:
            opp.recommendation = "insufficient data"
            return opp

        # === 1. Trend alignment (20 pts) ===
        opp.trend_alignment = self._score_trend(df, action)
        opp.details["trend"] = f"{opp.trend_alignment:.0f}/20"

        # === 2. Liquidity quality (15 pts) ===
        opp.liquidity_quality = self._score_liquidity(df, spread_bps)
        opp.details["liquidity"] = f"{opp.liquidity_quality:.0f}/15"

        # === 3. Momentum strength (15 pts) ===
        opp.momentum_strength = self._score_momentum(df, action)
        opp.details["momentum"] = f"{opp.momentum_strength:.0f}/15"

        # === 4. Volume confirmation (15 pts) ===
        opp.volume_confirmation = self._score_volume(df, action)
        opp.details["volume"] = f"{opp.volume_confirmation:.0f}/15"

        # === 5. Structure quality (15 pts) ===
        opp.structure_quality = self._score_structure(df, action)
        opp.details["structure"] = f"{opp.structure_quality:.0f}/15"

        # === 6. Timing quality (10 pts) ===
        opp.timing_quality = self._score_timing(session, news_minutes)
        opp.details["timing"] = f"{opp.timing_quality:.0f}/10"

        # === 7. Risk/reward (10 pts) ===
        opp.risk_reward = self._score_rr(entry_price, sl, tp)
        opp.details["rr"] = f"{opp.risk_reward:.0f}/10"

        # === Total ===
        opp.score = (
            opp.trend_alignment + opp.liquidity_quality +
            opp.momentum_strength + opp.volume_confirmation +
            opp.structure_quality + opp.timing_quality + opp.risk_reward
        )

        # === Tier ===
        if opp.score >= self.priority:
            opp.tier = "PRIORITY"
            opp.recommendation = f"PRIORITY ({opp.score:.0f}) — take first"
        elif opp.score >= self.good:
            opp.tier = "GOOD"
            opp.recommendation = f"GOOD ({opp.score:.0f}) — take if no priority"
        elif opp.score >= self.marginal:
            opp.tier = "MARGINAL"
            opp.recommendation = f"MARGINAL ({opp.score:.0f}) — skip unless no other"
        else:
            opp.tier = "SKIP"
            opp.recommendation = f"SKIP ({opp.score:.0f}) — below threshold"

        return opp

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------
    def _score_trend(self, df: pd.DataFrame, action: str) -> float:
        """Score trend alignment (0-20)."""
        close = df["close"]
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(close.iloc[-1])

        score = 0.0
        if action == "BUY":
            if ema9 > ema21 > ema50:
                score += 12
            elif ema9 > ema21:
                score += 6
            if price > ema50:
                score += 4
            if price > ema21:
                score += 4
        elif action == "SELL":
            if ema9 < ema21 < ema50:
                score += 12
            elif ema9 < ema21:
                score += 6
            if price < ema50:
                score += 4
            if price < ema21:
                score += 4
        return min(20, score)

    def _score_liquidity(self, df: pd.DataFrame, spread_bps: float) -> float:
        """Score liquidity (0-15)."""
        score = 0.0
        # Spread
        if spread_bps < 2:
            score += 7
        elif spread_bps < 5:
            score += 5
        elif spread_bps < 10:
            score += 3
        # Volume
        if "volume" in df:
            recent_vol = float(df["volume"].tail(10).mean())
            avg_vol = float(df["volume"].tail(50).mean())
            rvol = recent_vol / max(avg_vol, 1)
            if rvol > 1.5:
                score += 5
            elif rvol > 1.0:
                score += 3
            else:
                score += 1
        else:
            score += 3
        return min(15, score)

    def _score_momentum(self, df: pd.DataFrame, action: str) -> float:
        """Score momentum (0-15)."""
        close = df["close"]
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not pd.isna(rs.iloc[-1]) else 50

        score = 0.0
        if action == "BUY":
            if 50 < rsi < 70:
                score += 7
            elif rsi >= 70:
                score += 3  # overbought
            elif rsi > 45:
                score += 4
        elif action == "SELL":
            if 30 < rsi < 50:
                score += 7
            elif rsi <= 30:
                score += 3  # oversold
            elif rsi < 55:
                score += 4

        # MACD
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_slope = macd_line.iloc[-1] - macd_line.iloc[-3]
        if action == "BUY" and macd_slope > 0:
            score += 4
        elif action == "SELL" and macd_slope < 0:
            score += 4
        # ROC
        if len(close) > 10:
            roc = (close.iloc[-1] - close.iloc[-10]) / max(close.iloc[-10], 1e-10)
            if action == "BUY" and roc > 0:
                score += 4
            elif action == "SELL" and roc < 0:
                score += 4
        return min(15, score)

    def _score_volume(self, df: pd.DataFrame, action: str) -> float:
        """Score volume confirmation (0-15)."""
        if "volume" not in df:
            return 5
        close = df["close"]
        vol = df["volume"]
        # Up/down volume ratio
        up_vol = float(vol[close > close.shift()].tail(10).sum())
        dn_vol = float(vol[close < close.shift()].tail(10).sum())
        score = 0.0
        if action == "BUY" and up_vol > dn_vol * 1.3:
            score += 7
        elif action == "SELL" and dn_vol > up_vol * 1.3:
            score += 7
        else:
            score += 3
        # OBV direction
        obv = (close.diff().apply(np.sign) * vol).fillna(0).cumsum()
        obv_slope = obv.iloc[-1] - obv.iloc[-10] if len(obv) > 10 else 0
        if action == "BUY" and obv_slope > 0:
            score += 4
        elif action == "SELL" and obv_slope < 0:
            score += 4
        # RVol
        recent = float(vol.tail(5).mean())
        avg = float(vol.tail(20).mean())
        rvol = recent / max(avg, 1)
        if rvol > 1.3:
            score += 4
        return min(15, score)

    def _score_structure(self, df: pd.DataFrame, action: str) -> float:
        """Score market structure (0-15)."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        price = float(close.iloc[-1])
        # Recent swing high/low
        recent_high = float(high.tail(20).max())
        recent_low = float(low.tail(20).min())
        score = 0.0
        if action == "BUY":
            # Higher highs + higher lows?
            if price > recent_high * 0.98:
                score += 5  # near breakout
            if price > close.tail(50).mean():
                score += 5  # above midpoint
            # Distance from support
            dist_from_low = (price - recent_low) / max(recent_high - recent_low, 1e-10)
            if 0.3 < dist_from_low < 0.7:
                score += 5  # mid-range, good entry
        elif action == "SELL":
            if price < recent_low * 1.02:
                score += 5
            if price < close.tail(50).mean():
                score += 5
            dist_from_high = (recent_high - price) / max(recent_high - recent_low, 1e-10)
            if 0.3 < dist_from_high < 0.7:
                score += 5
        return min(15, score)

    def _score_timing(self, session: str, news_minutes: float) -> float:
        """Score timing (0-10)."""
        score = 0.0
        session_scores = {"london": 4, "new_york": 4, "overlap": 4,
                         "asia": 2, "off_hours": 1}
        score += session_scores.get(session, 2)
        if news_minutes > 120:
            score += 4
        elif news_minutes > 60:
            score += 2
        elif news_minutes > 30:
            score += 0
        else:
            score -= 2  # too close to news
        # Pullback bonus (would need to be passed in)
        score += 2  # default
        return max(0, min(10, score))

    def _score_rr(self, entry: float, sl: float, tp: float) -> float:
        """Score risk/reward (0-10)."""
        if entry <= 0 or sl <= 0 or tp <= 0:
            return 3  # default
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / max(risk, 1e-10)
        if rr > 3:
            return 10
        elif rr > 2:
            return 8
        elif rr > 1.5:
            return 6
        elif rr > 1:
            return 4
        else:
            return 2

    # ------------------------------------------------------------------
    # Rank multiple opportunities
    # ------------------------------------------------------------------
    def rank_opportunities(self,
                           opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rank multiple opportunities.

        Args:
            opportunities: list of dicts with keys: symbol, df, action, spread_bps, etc.

        Returns:
            Dict with ranking, best, actionable, skipped
        """
        scores: List[OpportunityScore] = []
        for opp in opportunities:
            score = self.score_opportunity(
                symbol=opp["symbol"],
                df=opp.get("df"),
                action=opp.get("action", "BUY"),
                spread_bps=opp.get("spread_bps", 5.0),
                session=opp.get("session", "off_hours"),
                sl=opp.get("sl", 0),
                tp=opp.get("tp", 0),
                entry_price=opp.get("entry_price", 0),
                news_minutes=opp.get("news_minutes", 999),
            )
            scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)

        # Assign ranks
        for i, s in enumerate(scores):
            s.rank = i + 1

        # Classify
        actionable = [s.symbol for s in scores if s.tier in ("PRIORITY", "GOOD")]
        skipped = [s.symbol for s in scores if s.tier in ("MARGINAL", "SKIP")]
        priority = [s.symbol for s in scores if s.tier == "PRIORITY"]

        return {
            "best": scores[0].symbol if scores else None,
            "best_score": scores[0].score if scores else 0,
            "ranking": [s.to_dict() for s in scores],
            "actionable": actionable,
            "priority": priority,
            "skipped": skipped,
            "total": len(scores),
            "recommendation": (
                f"Best: {scores[0].symbol} ({scores[0].score:.0f}/100, {scores[0].tier})"
                if scores else "No opportunities"
            ),
        }
