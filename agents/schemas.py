"""agents.schemas
=====================================================================
Structured outputs for the multi-agent trading framework.

Inspired by TradingAgents' Pydantic schemas. We use Python dataclasses
for simplicity (no Pydantic dependency). Each decision-making agent
produces a typed result that downstream agents consume.

Rating scale (5-tier, from most bullish to most bearish):
  Buy         — strong conviction to enter or add
  Overweight  — favorable, gradually increase
  Hold        — maintain, no action
  Underweight — reduce, take partial profits
  Sell        — exit or avoid

FIXES (Batch 2 audit):
  - C11/H9: parse_rating now checks several label patterns
    ("rating:", "recommendation:", "action:", "decision:", "stance:")
    plus a bare-word fallback, so it no longer silently defaults to
    Hold just because the LLM phrased things slightly differently.
  - H17/M16: added parse_confidence() with support for "confidence: 70",
    "confidence: 70%", "confidence = high", etc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PortfolioRating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ----------------------------------------------------------------------
@dataclass
class ResearchPlan:
    """Structured investment plan from the Research Manager."""
    recommendation: PortfolioRating
    rationale: str
    strategic_actions: str
    confidence: float = 0.5       # 0-1, how confident the research manager is

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation.value,
            "rationale": self.rationale,
            "strategic_actions": self.strategic_actions,
            "confidence": self.confidence,
        }

    def to_markdown(self) -> str:
        return (
            f"**Recommendation**: {self.recommendation.value}\n\n"
            f"**Rationale**: {self.rationale}\n\n"
            f"**Strategic Actions**: {self.strategic_actions}\n\n"
            f"**Confidence**: {self.confidence:.0%}"
        )


@dataclass
class TraderProposal:
    """Concrete transaction proposal from the Trader."""
    action: TraderAction
    lots: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "lots": self.lots,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reasoning": self.reasoning,
        }

    def to_markdown(self) -> str:
        return (
            f"**Action**: {self.action.value}\n"
            f"**Lots**: {self.lots:.4f}\n"
            f"**Entry**: {self.entry_price:.5f}\n"
            f"**Stop Loss**: {self.stop_loss:.5f}\n"
            f"**Take Profit**: {self.take_profit:.5f}\n"
            f"**Reasoning**: {self.reasoning}"
        )


@dataclass
class PortfolioDecision:
    """Final decision from the Portfolio Manager."""
    rating: PortfolioRating
    approved: bool
    final_lots: float = 0.0
    rationale: str = ""
    risk_adjustments: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rating": self.rating.value,
            "approved": self.approved,
            "final_lots": self.final_lots,
            "rationale": self.rationale,
            "risk_adjustments": self.risk_adjustments,
        }

    def to_markdown(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        return (
            f"**Rating**: {self.rating.value}\n"
            f"**Status**: {status}\n"
            f"**Final Lots**: {self.final_lots:.4f}\n"
            f"**Rationale**: {self.rationale}\n"
            f"**Risk Adjustments**: {self.risk_adjustments}"
        )


# ----------------------------------------------------------------------
# Rating parser (ported from TradingAgents, hardened)
# ----------------------------------------------------------------------
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)
_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Several label variants an LLM might use, in priority order.
_RATING_LABEL_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\brating\b\s*[:\-]?\s*\**\s*(\w+)", re.IGNORECASE),
    re.compile(r"\brecommendation\b\s*[:\-]?\s*\**\s*(\w+)", re.IGNORECASE),
    re.compile(r"\bfinal\s+rating\b\s*[:\-]?\s*\**\s*(\w+)", re.IGNORECASE),
    re.compile(r"\bstance\b\s*[:\-]?\s*\**\s*(\w+)", re.IGNORECASE),
    re.compile(r"\baction\b\s*[:\-]?\s*\**\s*(\w+)", re.IGNORECASE),
    re.compile(r"\bdecision\b\s*[:\-]?\s*\**\s*(\w+)", re.IGNORECASE),
    # "my rating is Buy" / "I recommend a Sell" (no colon)
    re.compile(r"\brating\s+is\s+(\w+)", re.IGNORECASE),
    re.compile(r"\brecommend(?:ing)?\s+(?:a\s+|an\s+)?(\w+)", re.IGNORECASE),
)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Tries several explicit label patterns first (most reliable), then
    falls back to scanning every line for a bare rating word. Only
    falls back to `default` if nothing at all is found, rather than
    silently defaulting whenever the LLM's exact phrasing changes.
    """
    if not text:
        return default

    for pattern in _RATING_LABEL_PATTERNS:
        for line in text.splitlines():
            m = pattern.search(line)
            if m and m.group(1).lower() in _RATING_SET:
                return m.group(1).capitalize()

    # Bare-word fallback: scan every line for a standalone rating word.
    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,()[]")
            if clean in _RATING_SET:
                return clean.capitalize()

    return default


# ----------------------------------------------------------------------
# Confidence parser
# ----------------------------------------------------------------------
_CONF_NUMERIC_RE = re.compile(
    r"confidence\b[^0-9%]{0,15}?(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE,
)
_CONF_WORD_MAP = {
    "very high": 0.9, "high": 0.8, "moderate": 0.6, "medium": 0.55,
    "low": 0.3, "very low": 0.15,
}
_CONF_WORD_RE = re.compile(
    r"confidence\b\s*[:\-=]?\s*(very high|very low|high|moderate|medium|low)",
    re.IGNORECASE,
)


def parse_confidence(text: str, default: float = 0.5) -> float:
    """Extract a 0-1 confidence value from prose.

    Handles "confidence: 70", "confidence: 70%", "confidence = 0.7",
    and qualitative phrasing like "confidence: high".
    """
    if not text:
        return default

    m = _CONF_NUMERIC_RE.search(text)
    if m:
        try:
            val = float(m.group(1))
        except ValueError:
            val = None
        if val is not None:
            # Distinguish "0.7" (already a fraction) from "70" (percent)
            if val <= 1.0:
                return max(0.0, min(1.0, val))
            return max(0.0, min(1.0, val / 100.0))

    m = _CONF_WORD_RE.search(text)
    if m:
        return _CONF_WORD_MAP.get(m.group(1).lower(), default)

    return default