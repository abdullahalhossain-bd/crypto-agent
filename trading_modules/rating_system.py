"""
5-Tier Rating System — Institutional Rating Scale
===================================================

Implements the institutional 5-tier rating scale used by sell-side research:
  Buy → Overweight → Hold → Underweight → Sell

This replaces the simplistic 3-tier (Buy/Hold/Sell) with a more nuanced
scale that captures gradations between "I like this" and "I'm neutral".

The parser is deterministic (no LLM call needed) — it extracts the rating
from the Portfolio Manager's decision text using a 2-pass heuristic:
  1. Look for explicit "Rating: X" label
  2. Fall back to first rating word found in text

Source: TradingAgents v0.3.1 (review #30)
License: Apache 2.0 (permissive)

Usage:
    from rating_system import RatingSystem, PortfolioRating

    rs = RatingSystem()

    # Parse rating from LLM output
    rating = rs.parse("Based on analysis... Rating: Overweight ...")
    print(rating)  # "Overweight"

    # Check if rating is actionable
    if rs.is_actionable(rating):
        print(f"Rating {rating} warrants a trade")

    # Convert to confidence score (0-1)
    confidence = rs.to_confidence(rating)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PortfolioRating(str, Enum):
    """5-tier institutional rating scale."""
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


# Ordered from most bullish to most bearish
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy",
    "Overweight",
    "Hold",
    "Underweight",
    "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


@dataclass(frozen=True)
class RatingResult:
    """Parsed rating with confidence and metadata."""
    rating: str
    confidence: float  # 0.0 to 1.0
    is_actionable: bool  # True for Buy/Overweight/Sell/Underweight
    direction: str  # "long" / "short" / "neutral"
    raw_text: str = ""


class RatingSystem:
    """
    5-tier rating system for trade decisions.

    Features:
    - Deterministic parsing (no LLM call)
    - Confidence mapping (Buy=0.9, Overweight=0.7, Hold=0.5, etc.)
    - Direction extraction (long/short/neutral)
    - Actionability check (Hold is NOT actionable)
    """

    # Rating → confidence score mapping
    CONFIDENCE_MAP = {
        "Buy": 0.90,
        "Overweight": 0.70,
        "Hold": 0.50,
        "Underweight": 0.30,
        "Sell": 0.10,
    }

    # Rating → trade direction
    DIRECTION_MAP = {
        "Buy": "long",
        "Overweight": "long",
        "Hold": "neutral",
        "Underweight": "short",
        "Sell": "short",
    }

    # Star rating → 5-tier rating mapping (from candlestick pattern guide)
    STAR_TO_RATING = {
        5: ("Buy", "Sell"),       # 5-star patterns → Buy or Sell
        4: ("Overweight", "Underweight"),  # 4-star → Overweight or Underweight
        3: ("Hold", "Hold"),      # 3-star → Hold
    }

    def parse(self, text: str, default: str = "Hold") -> str:
        """
        Heuristically extract a 5-tier rating from prose text.

        Two-pass strategy:
        1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
        2. Fall back to the first 5-tier rating word found anywhere in the text.

        Returns a Title-cased rating string, or ``default`` if no rating word appears.
        """
        if not text:
            return default

        # Pass 1: explicit "Rating: X" label
        for line in text.splitlines():
            m = _RATING_LABEL_RE.search(line)
            if m and m.group(1).lower() in _RATING_SET:
                return m.group(1).capitalize()

        # Pass 2: first rating word anywhere
        for line in text.splitlines():
            for word in line.lower().split():
                clean = word.strip("*:.,")
                if clean in _RATING_SET:
                    return clean.capitalize()

        return default

    def parse_full(self, text: str, default: str = "Hold") -> RatingResult:
        """
        Parse rating and return full result with confidence and direction.

        Args:
            text: LLM output text containing a rating
            default: Default rating if none found

        Returns:
            RatingResult with rating, confidence, direction, is_actionable
        """
        rating = self.parse(text, default)
        confidence = self.CONFIDENCE_MAP.get(rating, 0.5)
        direction = self.DIRECTION_MAP.get(rating, "neutral")
        actionable = rating not in ("Hold",)

        return RatingResult(
            rating=rating,
            confidence=confidence,
            is_actionable=actionable,
            direction=direction,
            raw_text=text,
        )

    def is_actionable(self, rating: str) -> bool:
        """Check if the rating warrants a trade (not Hold)."""
        return rating.capitalize() not in ("Hold",)

    def to_confidence(self, rating: str) -> float:
        """Convert rating to confidence score (0-1)."""
        return self.CONFIDENCE_MAP.get(rating.capitalize(), 0.5)

    def to_direction(self, rating: str) -> str:
        """Convert rating to trade direction (long/short/neutral)."""
        return self.DIRECTION_MAP.get(rating.capitalize(), "neutral")

    def from_star_rating(self, stars: int, direction: str) -> str:
        """
        Convert star rating (1-5) to 5-tier rating.

        Used for candlestick patterns:
          5 stars + bullish → Buy
          5 stars + bearish → Sell
          4 stars + bullish → Overweight
          4 stars + bearish → Underweight
          3 stars → Hold

        Args:
            stars: Star rating (1-5)
            direction: "bullish" or "bearish"

        Returns:
            5-tier rating string
        """
        mapping = self.STAR_TO_RATING.get(stars, ("Hold", "Hold"))
        if direction.lower() in ("bullish", "buy", "long"):
            return mapping[0]
        elif direction.lower() in ("bearish", "sell", "short"):
            return mapping[1]
        return "Hold"

    def format_rating_label(self, rating: str, confidence: float) -> str:
        """Format a rating label for display."""
        emoji = {
            "Buy": "🟢",
            "Overweight": "🟩",
            "Hold": "🟡",
            "Underweight": "🟥",
            "Sell": "🔴",
        }
        icon = emoji.get(rating, "⚪")
        return f"{icon} {rating} ({confidence:.0%} confidence)"
