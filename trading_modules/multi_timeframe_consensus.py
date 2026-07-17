"""trading_modules/multi_timeframe_consensus.py
=====================================================================
Multi-Timeframe Consensus Engine (Principle #125)
=====================================================================
Confirms trade signals across multiple timeframes before execution.

Rule (Livermore #125):
    "Don't trade M15 in isolation. Confirm with Weekly → Daily → H4 → H1 → M15 → M5."

    A signal on M15 is weak if it conflicts with H4 and Daily.
    A signal on M15 is STRONG if it aligns with H4, Daily, and Weekly.

Timeframe Hierarchy (institutional):
    Weekly  → macro trend (institutions)
    Daily   → swing trend (position traders)
    H4      → intermediate trend
    H1      → short-term trend (day traders)
    M15     → entry trigger (our working TF)
    M5      → fine-tuning entry

Consensus Scoring:
    Each TF votes: BUY, SELL, or NEUTRAL
    Higher TFs get more weight:
        Weekly:  25%
        Daily:   25%
        H4:      20%
        H1:      15%
        M15:     10%
        M5:       5%

    Consensus thresholds:
        score > +0.7  → STRONG BUY consensus
        score +0.3 to +0.7 → BUY consensus
        score -0.3 to +0.3 → NO CONSENSUS (skip)
        score -0.7 to -0.3 → SELL consensus
        score < -0.7  → STRONG SELL consensus

Usage:
    engine = MultiTimeframeConsensusEngine()

    consensus = engine.evaluate({
        "W1": df_weekly,
        "D1": df_daily,
        "H4": df_h4,
        "H1": df_h1,
        "M15": df_m15,
        "M5": df_m5,
    })
    if consensus.consensus == "STRONG_BUY":
        place_buy_order()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.multi_timeframe_consensus")


class TimeframeVote(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


class ConsensusLevel(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NO_CONSENSUS = "NO_CONSENSUS"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# Default weights for each timeframe (must sum to 1.0)
DEFAULT_TF_WEIGHTS: Dict[str, float] = {
    "W1": 0.25,
    "D1": 0.25,
    "H4": 0.20,
    "H1": 0.15,
    "M15": 0.10,
    "M5": 0.05,
}


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe: str
    vote: TimeframeVote = TimeframeVote.NEUTRAL
    strength: float = 0.0     # 0-1
    trend: str = "neutral"    # up, down, range
    ema_alignment: str = ""   # "bull_stack", "bear_stack", "mixed"
    rsi: float = 50.0
    adx: float = 0.0          # trend strength
    price_position: float = 0.5  # 0=bottom of range, 1=top
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "vote": self.vote.value,
            "strength": round(self.strength, 3),
            "trend": self.trend,
            "ema_alignment": self.ema_alignment,
            "rsi": round(self.rsi, 1),
            "adx": round(self.adx, 1),
            "price_position": round(self.price_position, 3),
            "detail": self.detail,
        }


@dataclass
class ConsensusResult:
    """Multi-timeframe consensus result."""
    consensus: ConsensusLevel = ConsensusLevel.NO_CONSENSUS
    score: float = 0.0           # -1 (strong sell) to +1 (strong buy)
    confidence: float = 0.0      # 0-1
    agreeing_tfs: List[str] = field(default_factory=list)
    disagreeing_tfs: List[str] = field(default_factory=list)
    timeframes: Dict[str, TimeframeAnalysis] = field(default_factory=dict)
    recommendation: str = ""
    trade_allowed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consensus": self.consensus.value,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "agreeing_tfs": self.agreeing_tfs,
            "disagreeing_tfs": self.disagreeing_tfs,
            "timeframes": {tf: a.to_dict() for tf, a in self.timeframes.items()},
            "recommendation": self.recommendation,
            "trade_allowed": self.trade_allowed,
        }


class MultiTimeframeConsensusEngine:
    """Evaluates trade signals across multiple timeframes."""

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 strong_threshold: float = 0.7,
                 weak_threshold: float = 0.3,
                 require_high_tf_agreement: bool = True):
        """Initialize engine.

        Args:
            weights: TF weights (default: W1=25%, D1=25%, H4=20%, H1=15%, M15=10%, M5=5%)
            strong_threshold: score above this = STRONG BUY/SELL
            weak_threshold: score above this = BUY/SELL
            require_high_tf_agreement: if True, high TFs (W1+D1) must agree
        """
        self.weights = weights or DEFAULT_TF_WEIGHTS
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.require_high_tf = require_high_tf_agreement

    def evaluate(self, dfs: Dict[str, pd.DataFrame]) -> ConsensusResult:
        """Evaluate consensus across timeframes.

        Args:
            dfs: {"W1": df_w, "D1": df_d, "H4": df_h4, "H1": df_h1, "M15": df_m15, "M5": df_m5}

        Returns:
            ConsensusResult with weighted vote
        """
        result = ConsensusResult()

        if not dfs:
            result.recommendation = "no timeframes provided"
            return result

        # Analyze each timeframe
        for tf, df in dfs.items():
            if df is None or df.empty or len(df) < 50:
                continue
            result.timeframes[tf] = self._analyze_timeframe(tf, df)

        if not result.timeframes:
            result.recommendation = "insufficient data on all timeframes"
            return result

        # Compute weighted score
        score = 0.0
        total_weight = 0.0
        for tf, analysis in result.timeframes.items():
            w = self.weights.get(tf, 0.05)
            # Vote → numeric: BUY=+1, SELL=-1, NEUTRAL=0
            vote_val = 1 if analysis.vote == TimeframeVote.BUY else \
                      -1 if analysis.vote == TimeframeVote.SELL else 0
            # Weight by vote strength
            score += w * vote_val * analysis.strength
            total_weight += w

        if total_weight > 0:
            result.score = score / total_weight

        # Determine consensus level
        if result.score > self.strong_threshold:
            result.consensus = ConsensusLevel.STRONG_BUY
        elif result.score > self.weak_threshold:
            result.consensus = ConsensusLevel.BUY
        elif result.score < -self.strong_threshold:
            result.consensus = ConsensusLevel.STRONG_SELL
        elif result.score < -self.weak_threshold:
            result.consensus = ConsensusLevel.SELL
        else:
            result.consensus = ConsensusLevel.NO_CONSENSUS

        # Find agreeing/disagreeing TFs
        main_direction = "BUY" if result.score > 0 else "SELL" if result.score < 0 else "NEUTRAL"
        for tf, analysis in result.timeframes.items():
            if analysis.vote.value == main_direction:
                result.agreeing_tfs.append(tf)
            elif analysis.vote != TimeframeVote.NEUTRAL:
                result.disagreeing_tfs.append(tf)

        # Confidence: based on agreement ratio
        total_tfs = len(result.timeframes)
        if total_tfs > 0:
            result.confidence = len(result.agreeing_tfs) / total_tfs

        # High TF agreement check
        high_tfs_agree = True
        if self.require_high_tf:
            for high_tf in ["W1", "D1"]:
                if high_tf in result.timeframes:
                    if (main_direction == "BUY" and
                        result.timeframes[high_tf].vote != TimeframeVote.BUY):
                        high_tfs_agree = False
                    elif (main_direction == "SELL" and
                          result.timeframes[high_tf].vote != TimeframeVote.SELL):
                        high_tfs_agree = False

        # Trade allowed?
        result.trade_allowed = (
            result.consensus in (ConsensusLevel.STRONG_BUY, ConsensusLevel.STRONG_SELL)
            and high_tfs_agree
            and result.confidence >= 0.6
        )

        # Recommendation
        result.recommendation = self._recommend(result, high_tfs_agree)

        return result

    # ------------------------------------------------------------------
    # Analyze single timeframe
    # ------------------------------------------------------------------
    def _analyze_timeframe(self, tf: str, df: pd.DataFrame) -> TimeframeAnalysis:
        """Analyze a single timeframe and vote."""
        analysis = TimeframeAnalysis(timeframe=tf)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # EMA alignment
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]
        ema21 = close.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(close.iloc[-1])

        bull_stack = ema9 > ema21 > ema50
        bear_stack = ema9 < ema21 < ema50
        if bull_stack:
            analysis.ema_alignment = "bull_stack"
            analysis.trend = "up"
        elif bear_stack:
            analysis.ema_alignment = "bear_stack"
            analysis.trend = "down"
        else:
            analysis.ema_alignment = "mixed"
            analysis.trend = "range"

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        analysis.rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # ADX (simplified)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_val = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        # ADX approximation: trend strength from slope
        slope = abs(float(ema21 - ema50) / max(ema50, 1e-10) * 1000)
        analysis.adx = min(100, slope * 10)

        # Price position in range
        range_high = float(high.tail(50).max())
        range_low = float(low.tail(50).min())
        analysis.price_position = (price - range_low) / max(range_high - range_low, 1e-10)

        # Vote
        bull_score = 0
        bear_score = 0

        if bull_stack:
            bull_score += 2
        if bear_stack:
            bear_score += 2
        if analysis.rsi > 55:
            bull_score += 1
        elif analysis.rsi < 45:
            bear_score += 1
        if price > ema21:
            bull_score += 1
        elif price < ema21:
            bear_score += 1

        if bull_score > bear_score + 1:
            analysis.vote = TimeframeVote.BUY
            analysis.strength = min(1.0, bull_score / 5)
            analysis.detail = f"Bullish: {bull_score} signals (RSI={analysis.rsi:.0f}, ADX={analysis.adx:.0f})"
        elif bear_score > bull_score + 1:
            analysis.vote = TimeframeVote.SELL
            analysis.strength = min(1.0, bear_score / 5)
            analysis.detail = f"Bearish: {bear_score} signals (RSI={analysis.rsi:.0f}, ADX={analysis.adx:.0f})"
        else:
            analysis.vote = TimeframeVote.NEUTRAL
            analysis.strength = 0.3
            analysis.detail = f"Neutral: mixed signals"

        return analysis

    def _recommend(self, result: ConsensusResult, high_tf_agree: bool) -> str:
        """Generate recommendation."""
        if not high_tf_agree and self.require_high_tf:
            return (f"REJECT — high timeframes (W1/D1) don't agree with M15. "
                    f"Score={result.score:.2f} but high TF misaligned.")
        if result.consensus == ConsensusLevel.STRONG_BUY:
            return f"STRONG BUY — score={result.score:.2f}, {len(result.agreeing_tfs)}/{len(result.timeframes)} TFs agree"
        if result.consensus == ConsensusLevel.BUY:
            return f"BUY — score={result.score:.2f}, {len(result.agreeing_tfs)}/{len(result.timeframes)} TFs agree"
        if result.consensus == ConsensusLevel.STRONG_SELL:
            return f"STRONG SELL — score={result.score:.2f}, {len(result.agreeing_tfs)}/{len(result.timeframes)} TFs agree"
        if result.consensus == ConsensusLevel.SELL:
            return f"SELL — score={result.score:.2f}, {len(result.agreeing_tfs)}/{len(result.timeframes)} TFs agree"
        return f"NO CONSENSUS — score={result.score:.2f}, wait for alignment"
