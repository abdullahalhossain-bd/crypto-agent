"""
NLP for Trading — lexicon-based news sentiment (no external API)
================================================================

Lightweight news sentiment analysis for trading signals. Uses a
financial-domain lexicon (Loughran-McDonald inspired) — no external
API call, no model download, no internet required.

Features:
    1. News classification  — bullish / bearish / neutral
    2. Sentiment score       -1..+1
    3. Event detection       — Fed/CPI/NFP/earnings mentions
    4. Central bank tone     — hawkish / dovish / neutral
    5. Multi-headline aggregation

Usage:
    from trading_modules.nlp_trading import NewsSentimentAnalyzer
    analyzer = NewsSentimentAnalyzer()
    result = analyzer.analyze("Fed signals more rate hikes possible if inflation persists")
    print(result.sentiment, result.score, result.events)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Lexicons
# ──────────────────────────────────────────────────────────────────────
BULLISH_WORDS = {
    # Generic positive
    "surge": 2, "soar": 2, "rally": 2, "gain": 1, "gains": 1, "jump": 2,
    "jumped": 2, "rising": 1, "rise": 1, "rose": 1, "bullish": 2, "optimism": 1,
    "optimistic": 1, "breakthrough": 2, "outperform": 2, "beat": 1, "beats": 1,
    "exceed": 1, "exceeds": 1, "strong": 1, "strength": 1, "boost": 2,
    "boosted": 2, "rises": 1, "higher": 1, "high": 1, "support": 1,
    "supported": 1, "recovery": 2, "recover": 2, "rebound": 2, "rebounded": 2,
    "upgrade": 2, "upgraded": 2, "buy": 1, "buying": 1, "accumulate": 1,
    "accumulate": 1, "long": 1, "bull": 2, "bulls": 2, "momentum": 1,
    "growth": 1, "growing": 1, "expand": 1, "expanding": 1, "expansion": 1,
    "profit": 1, "profits": 1, "profitable": 2, "record": 2, "all-time": 2,
    "ath": 2, "breakout": 2, "broke": 1, "above": 1, "climb": 1, "climbed": 1,
    "rally": 2, "rallied": 2,
    # Crypto-specific
    "adoption": 2, "institutional": 1, "etf": 1, "approved": 2, "approval": 2,
    "halving": 1, "scarcity": 1, "demand": 1,
}

BEARISH_WORDS = {
    # Generic negative
    "crash": -3, "crashed": -3, "plunge": -3, "plunged": -3, "dump": -3,
    "dumped": -3, "sell": -1, "selling": -2, "sold": -2, "selloff": -3,
    "loss": -1, "losses": -1, "bearish": -2, "pessimism": -1, "decline": -1,
    "declined": -1, "declines": -1, "drop": -2, "dropped": -2, "drops": -2,
    "fall": -1, "falls": -1, "fell": -1, "falling": -1, "lower": -1,
    "weak": -1, "weakness": -1, "underperform": -2, "miss": -1, "missed": -1,
    "below": -1, "downgrade": -2, "downgraded": -2, "bear": -2, "bears": -2,
    "short": -1, "resistance": -1, "rejected": -1, "rejection": -1,
    "correction": -2, "pullback": -1, "pullbacks": -1, "retreat": -1,
    "retreated": -1, "reverse": -1, "reversed": -1, "reversal": -2,
    "risk": -1, "risky": -1, "risk-off": -2, "fear": -2, "panic": -3,
    "capitulation": -3, "liquidation": -2, "liquidated": -2, "long": 0,
    # Crypto-specific
    "ban": -3, "banned": -3, "hack": -3, "hacked": -3, "exploit": -3,
    "regulation": -1, "regulatory": -1, "sec": -2, "lawsuit": -2,
    "fraud": -3, "ponzi": -3, "collapse": -3, "bankrupt": -3,
    "delisting": -2, "delist": -2,
}

HAWKISH_WORDS = {
    # Central bank hawkish (raises rates → currency bullish, equities bearish)
    "rate hike": 2, "rate hikes": 2, "hike rates": 2, "hiking": 1,
    "tightening": 2, "tighten": 2, "fight inflation": 2, "hawkish": 2,
    "aggressive": 1, "restrictive": 1, "higher for longer": 2,
    "inflation concern": 1, "price pressure": 1, "overheating": 1,
}

DOVISH_WORDS = {
    # Central bank dovish (cuts rates → currency bearish, equities bullish)
    "rate cut": -2, "rate cuts": -2, "cut rates": -2, "cutting": -1,
    "easing": -2, "ease": -2, "dovish": -2, "accommodative": -1,
    "support growth": -1, "stimulus": -1, "quantitative easing": -2,
    "qe": -2, "lower for longer": -2, "patient": -1, "pause": -1,
    "pausing": -1, "hold rates": -1,
}

# Event keywords → event type
EVENT_KEYWORDS: dict[str, list[str]] = {
    "fed": ["federal reserve", "fed ", "powell", "yellen", "fomc", "federal open market"],
    "ecb": ["ecb", "european central bank", "lagarde"],
    "boe": ["bank of england", "boe ", "bailey"],
    "boj": ["bank of japan", "boj ", "kuroda", "ueda"],
    "cpi": ["cpi", "consumer price index", "inflation data"],
    "ppi": ["ppi", "producer price index"],
    "nfp": ["nonfarm", "non-farm", "nfp", "payrolls", "payroll"],
    "gdp": ["gdp", "gross domestic product"],
    "earnings": ["earnings", "quarterly results", "q1 ", "q2 ", "q3 ", "q4 "],
    "unemployment": ["unemployment", "jobless claims", "employment data"],
    "retail_sales": ["retail sales", "consumer spending"],
    "pmi": ["pmi", "purchasing managers"],
}


@dataclass
class NewsAnalysisResult:
    headline: str
    sentiment: str               # "bullish" / "bearish" / "neutral"
    score: float                 # -1..+1
    confidence: float            # 0..1
    events: list[str]            # ["fed", "cpi", ...]
    central_bank_tone: str       # "hawkish" / "dovish" / "neutral"
    bullish_signals: list[str] = field(default_factory=list)
    bearish_signals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "sentiment": self.sentiment,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "events": self.events,
            "central_bank_tone": self.central_bank_tone,
            "bullish_signals": self.bullish_signals,
            "bearish_signals": self.bearish_signals,
            "notes": self.notes,
        }


@dataclass
class AggregatedNewsResult:
    n_headlines: int
    avg_score: float
    sentiment_distribution: dict[str, int]   # {"bullish": 5, "bearish": 3, "neutral": 2}
    dominant_sentiment: str
    events_detected: list[str]
    central_bank_tone: str
    confidence: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_headlines": self.n_headlines,
            "avg_score": round(self.avg_score, 3),
            "sentiment_distribution": self.sentiment_distribution,
            "dominant_sentiment": self.dominant_sentiment,
            "events_detected": self.events_detected,
            "central_bank_tone": self.central_bank_tone,
            "confidence": round(self.confidence, 3),
            "notes": self.notes,
        }


class NewsSentimentAnalyzer:
    """Lexicon-based news sentiment analyzer.

    Parameters:
        bullish_threshold: score above this → bullish (default 0.1)
        bearish_threshold: score below this → bearish (default -0.1)
    """

    def __init__(
        self, bullish_threshold: float = 0.1, bearish_threshold: float = -0.1,
    ) -> None:
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    def analyze(self, headline: str) -> NewsAnalysisResult:
        """Analyze a single news headline."""
        if not headline or not isinstance(headline, str):
            return NewsAnalysisResult(
                headline="", sentiment="neutral", score=0.0, confidence=0.0,
                events=[], central_bank_tone="neutral", notes=["empty headline"],
            )
        text = headline.lower()
        # Tokenize (split on non-alphanumeric, but keep multi-word phrases)
        # Score words
        bullish_score = 0
        bearish_score = 0
        bullish_signals: list[str] = []
        bearish_signals: list[str] = []
        for word, weight in BULLISH_WORDS.items():
            count = text.count(word)
            if count > 0:
                bullish_score += count * weight
                bullish_signals.append(f"{word}×{count}")
        for word, weight in BEARISH_WORDS.items():
            count = text.count(word)
            if count > 0:
                bearish_score += count * abs(weight)
                bearish_signals.append(f"{word}×{count}")
        # Central bank tone
        hawkish_score = sum(text.count(w) * abs(w_) for w, w_ in HAWKISH_WORDS.items() for _ in [1] if w in text)
        dovish_score = sum(text.count(w) * abs(w_) for w, w_ in DOVISH_WORDS.items() for _ in [1] if w in text)
        if hawkish_score > dovish_score:
            central_bank_tone = "hawkish"
        elif dovish_score > hawkish_score:
            central_bank_tone = "dovish"
        else:
            central_bank_tone = "neutral"
        # Final score (-1..+1)
        total = bullish_score + bearish_score
        if total == 0:
            score = 0.0
            sentiment = "neutral"
            confidence = 0.0
        else:
            # Normalize: score = (bullish - bearish) / (bullish + bearish), scaled
            raw = (bullish_score - bearish_score) / (bullish_score + bearish_score)
            # Scale to -1..+1 (already in that range, but dampen slightly)
            score = max(-1.0, min(1.0, raw))
            if score > self.bullish_threshold:
                sentiment = "bullish"
            elif score < self.bearish_threshold:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            # Confidence: based on total signal strength
            confidence = min(1.0, total / 10.0)
        # Event detection
        events: list[str] = []
        for event_type, keywords in EVENT_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    events.append(event_type)
                    break
        notes: list[str] = []
        if bullish_signals:
            notes.append(f"bullish: {', '.join(bullish_signals[:5])}")
        if bearish_signals:
            notes.append(f"bearish: {', '.join(bearish_signals[:5])}")
        if events:
            notes.append(f"events: {', '.join(events)}")
        if central_bank_tone != "neutral":
            notes.append(f"central_bank_tone: {central_bank_tone}")
        return NewsAnalysisResult(
            headline=headline, sentiment=sentiment, score=score,
            confidence=confidence, events=events,
            central_bank_tone=central_bank_tone,
            bullish_signals=bullish_signals,
            bearish_signals=bearish_signals,
            notes=notes,
        )

    def analyze_batch(self, headlines: list[str]) -> AggregatedNewsResult:
        """Analyze multiple headlines and aggregate."""
        if not headlines:
            return AggregatedNewsResult(
                n_headlines=0, avg_score=0.0,
                sentiment_distribution={"bullish": 0, "bearish": 0, "neutral": 0},
                dominant_sentiment="neutral", events_detected=[],
                central_bank_tone="neutral", confidence=0.0,
                notes=["no headlines"],
            )
        results = [self.analyze(h) for h in headlines]
        scores = [r.score for r in results]
        avg_score = float(sum(scores) / len(scores))
        # Distribution
        distribution = {"bullish": 0, "bearish": 0, "neutral": 0}
        for r in results:
            distribution[r.sentiment] += 1
        dominant = max(distribution, key=distribution.get)
        # Events
        all_events: set[str] = set()
        hawkish_count = 0
        dovish_count = 0
        for r in results:
            all_events.update(r.events)
            if r.central_bank_tone == "hawkish":
                hawkish_count += 1
            elif r.central_bank_tone == "dovish":
                dovish_count += 1
        if hawkish_count > dovish_count:
            cb_tone = "hawkish"
        elif dovish_count > hawkish_count:
            cb_tone = "dovish"
        else:
            cb_tone = "neutral"
        # Confidence = fraction agreeing with dominant sentiment
        confidence = distribution[dominant] / len(results)
        notes = [
            f"analyzed {len(headlines)} headlines",
            f"distribution: {distribution}",
            f"avg score: {avg_score:+.3f}",
        ]
        return AggregatedNewsResult(
            n_headlines=len(headlines), avg_score=avg_score,
            sentiment_distribution=distribution,
            dominant_sentiment=dominant,
            events_detected=sorted(all_events),
            central_bank_tone=cb_tone,
            confidence=confidence,
            notes=notes,
        )


__all__ = [
    "NewsSentimentAnalyzer", "NewsAnalysisResult", "AggregatedNewsResult",
    "BULLISH_WORDS", "BEARISH_WORDS", "HAWKISH_WORDS", "DOVISH_WORDS", "EVENT_KEYWORDS",
]
