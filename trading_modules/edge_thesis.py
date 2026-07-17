"""
Edge Thesis Module — "No Theory of Edge" Enforcement
=====================================================

Forces the LLM to articulate WHY a trade has an edge. If it can't
explain the economic driver, confidence is automatically downgraded.

Philosophy (Longmore "no theory of edge" rule):
  "If you can't state the edge in one sentence, you don't have one."

A pattern appearing on a chart is NOT an edge. An edge requires:
  - A clear economic reason WHY this trade should work
  - A mechanism that forces price to move in your favor
  - A reason why the market hasn't already priced this in

Source: Orallexa (review #27) — edge_thesis required field
Pattern: Conviction auto-downgrades if model can't articulate driver

Usage:
    from edge_thesis import EdgeThesisGate, EdgeThesis

    gate = EdgeThesisGate()

    # LLM output with edge explanation
    thesis = EdgeThesis(
        statement="Funding rate at 54% annualized creates carry trade opportunity",
        category="funding_arb",
        mechanism="High funding → longs pay shorts → spot-perp basis narrows",
        expiration="Next funding settlement (8h)",
    )
    result = gate.evaluate(thesis, confidence=0.75)
    print(f"Adjusted confidence: {result.adjusted_confidence:.0%}")

    # LLM output WITHOUT edge explanation
    thesis2 = EdgeThesis(statement="", category="none")
    result2 = gate.evaluate(thesis2, confidence=0.75)
    print(f"Adjusted: {result2.adjusted_confidence:.0%}")  # 0.45 (downgraded)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EdgeCategory(str, Enum):
    """Categories of trading edges."""
    FUNDING_ARB = "funding_arb"           # Funding rate carry trade
    MEAN_REVERSION = "mean_reversion"     # Statistical reversion to mean
    MOMENTUM = "momentum"                 # Trend continuation
    LIQUIDITY_SWEEP = "liquidity_sweep"   # Stop hunt + reversal
    ORDER_BLOCK = "order_block"           # Institutional zone reaction
    FVG_REFILL = "fvg_refill"             # Fair value gap fill
    SENTIMENT_EXTREME = "sentiment_extreme"  # Contrarian sentiment
    NEWS_CATALYST = "news_catalyst"       # Event-driven
    STRUCTURAL_BREAK = "structural_break" # BOS/CHoCH
    CORRELATION_BREAK = "correlation_break"  # Cross-asset divergence
    VOLATILITY_EXPANSION = "volatility_expansion"  # Squeeze breakout
    NONE = "none"


@dataclass
class EdgeThesis:
    """
    A structured edge thesis — the economic driver behind a trade.

    All fields should be filled for a valid edge. Empty statement → downgrade.
    """
    statement: str = ""               # One-sentence edge explanation
    category: EdgeCategory = EdgeCategory.NONE
    mechanism: str = ""               # WHY this edge exists (the forcing function)
    expiration: str = ""              # When this edge expires (time-bound)
    evidence: list[str] = field(default_factory=list)  # Supporting data points

    def is_valid(self) -> bool:
        """Check if this thesis has a meaningful edge explanation."""
        return (
            bool(self.statement.strip())
            and len(self.statement.strip()) > 15
            and self.category != EdgeCategory.NONE
        )

    def quality_score(self) -> float:
        """Score thesis quality 0-1 based on completeness."""
        score = 0.0
        if self.statement.strip() and len(self.statement) > 15:
            score += 0.25
        if self.category != EdgeCategory.NONE:
            score += 0.25
        if self.mechanism.strip() and len(self.mechanism) > 10:
            score += 0.25
        if self.expiration.strip():
            score += 0.15
        if self.evidence:
            score += 0.10
        return min(1.0, score)


@dataclass
class EdgeEvaluation:
    """Result of edge thesis evaluation."""
    has_edge: bool
    quality_score: float          # 0-1 thesis completeness
    original_confidence: float    # Input confidence
    adjusted_confidence: float    # After edge-based adjustment
    downgrade_pct: float          # How much confidence was reduced
    reason: str                   # Why adjusted (or not)
    category: str                 # Edge category detected

    def to_dict(self) -> dict:
        return {
            "has_edge": self.has_edge,
            "quality_score": round(self.quality_score, 4),
            "original_confidence": round(self.original_confidence, 4),
            "adjusted_confidence": round(self.adjusted_confidence, 4),
            "downgrade_pct": round(self.downgrade_pct, 4),
            "reason": self.reason,
            "category": self.category,
        }


class EdgeThesisGate:
    """
    Evaluates whether an LLM's trade recommendation has a real edge.

    Rules:
    1. No edge thesis → confidence downgraded by 40% (0.75 → 0.45)
    2. Weak thesis (statement only, no mechanism) → downgraded by 20%
    3. Moderate thesis (statement + category) → downgraded by 10%
    4. Strong thesis (all fields) → no downgrade
    5. Vague statements ("technical analysis says", "pattern looks good")
       are detected and treated as no edge
    """

    # Confidence downgrade factors by thesis quality
    DOWNGRADE_NONE = 0.40       # No thesis → 60% reduction
    DOWNGRADE_WEAK = 0.20       # Statement only → 20% reduction
    DOWNGRADE_MODERATE = 0.10   # Statement + category → 10% reduction
    DOWNGRADE_STRONG = 0.0      # All fields → no reduction

    # Vague phrases that indicate no real edge
    VAGUE_PHRASES = [
        "technical analysis says",
        "pattern looks good",
        "momentum is strong",
        "looks bullish",
        "looks bearish",
        "chart looks good",
        "indicator shows",
        "the trend is up",
        "the trend is down",
        "price action suggests",
        "market seems",
        "feels like",
        "gut feeling",
        "looks like a good trade",
        "should go up",
        "should go down",
    ]

    def evaluate(
        self,
        thesis: EdgeThesis,
        confidence: float,
    ) -> EdgeEvaluation:
        """
        Evaluate edge thesis and adjust confidence.

        Args:
            thesis: The edge thesis to evaluate
            confidence: Original confidence (0-1)

        Returns:
            EdgeEvaluation with adjusted confidence
        """
        original = confidence
        category = thesis.category.value if isinstance(thesis.category, EdgeCategory) else str(thesis.category)

        # Check for vague statements
        if self._is_vague(thesis.statement):
            adjusted = confidence * (1.0 - self.DOWNGRADE_NONE)
            return EdgeEvaluation(
                has_edge=False,
                quality_score=0.0,
                original_confidence=original,
                adjusted_confidence=adjusted,
                downgrade_pct=self.DOWNGRADE_NONE,
                reason="Vague statement — no real edge articulated (Longmore rule)",
                category="vague",
            )

        # Evaluate thesis quality
        quality = thesis.quality_score()

        if not thesis.is_valid():
            adjusted = confidence * (1.0 - self.DOWNGRADE_NONE)
            return EdgeEvaluation(
                has_edge=False,
                quality_score=quality,
                original_confidence=original,
                adjusted_confidence=adjusted,
                downgrade_pct=self.DOWNGRADE_NONE,
                reason="No edge thesis provided — confidence downgraded 40%",
                category=category,
            )

        # Determine downgrade level
        if quality >= 0.85:
            # Strong thesis — all fields filled
            adjusted = confidence
            reason = "Strong edge thesis — all components present"
            downgrade = self.DOWNGRADE_STRONG
        elif quality >= 0.50:
            # Moderate thesis — statement + category
            adjusted = confidence * (1.0 - self.DOWNGRADE_MODERATE)
            reason = "Moderate edge thesis — downgraded 10% (add mechanism for full confidence)"
            downgrade = self.DOWNGRADE_MODERATE
        else:
            # Weak thesis — statement only
            adjusted = confidence * (1.0 - self.DOWNGRADE_WEAK)
            reason = "Weak edge thesis — downgraded 20% (add category + mechanism)"
            downgrade = self.DOWNGRADE_WEAK

        return EdgeEvaluation(
            has_edge=True,
            quality_score=quality,
            original_confidence=original,
            adjusted_confidence=adjusted,
            downgrade_pct=downgrade,
            reason=reason,
            category=category,
        )

    def _is_vague(self, statement: str) -> bool:
        """Check if a statement is vague (no real edge)."""
        if not statement:
            return False  # Empty is handled separately
        lower = statement.lower()
        return any(phrase in lower for phrase in self.VAGUE_PHRASES)

    def extract_from_text(self, text: str) -> EdgeThesis:
        """
        Extract edge thesis from LLM free-text output.

        Looks for keywords: edge, catalyst, driver, because, reason, alpha
        """
        if not text:
            return EdgeThesis()

        # Find edge-related sections
        patterns = [
            r"(?:edge|alpha\s*source|catalyst|driver)[:\s]+([^\n]{15,300})",
            r"(?:because|reason|rationale)[:\s]+([^\n]{15,300})",
        ]

        statement = ""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                statement = match.group(1).strip().rstrip(".")
                break

        # Detect category from keywords
        category = self._detect_category(text)

        # Detect mechanism
        mechanism = ""
        mech_match = re.search(
            r"(?:mechanism|why|how)[:\s]+([^\n]{15,300})",
            text, re.IGNORECASE,
        )
        if mech_match:
            mechanism = mech_match.group(1).strip()

        # Detect expiration
        expiration = ""
        exp_match = re.search(
            r"(?:expires?|expiration|until|timeframe|horizon)[:\s]+([^\n]{5,100})",
            text, re.IGNORECASE,
        )
        if exp_match:
            expiration = exp_match.group(1).strip()

        return EdgeThesis(
            statement=statement,
            category=category,
            mechanism=mechanism,
            expiration=expiration,
        )

    def _detect_category(self, text: str) -> EdgeCategory:
        """Detect edge category from text keywords."""
        lower = text.lower()

        category_keywords = {
            EdgeCategory.FUNDING_ARB: ["funding rate", "carry trade", "basis", "funding arb"],
            EdgeCategory.MEAN_REVERSION: ["mean reversion", "revert", "oversold", "overbought", "reversal"],
            EdgeCategory.MOMENTUM: ["momentum", "trend continuation", "breakout", "trend following"],
            EdgeCategory.LIQUIDITY_SWEEP: ["liquidity sweep", "stop hunt", "stop grab", "liquidity grab"],
            EdgeCategory.ORDER_BLOCK: ["order block", "ob", "institutional zone", "demand zone", "supply zone"],
            EdgeCategory.FVG_REFILL: ["fvg", "fair value gap", "imbalance", "gap fill"],
            EdgeCategory.SENTIMENT_EXTREME: ["sentiment", "fear greed", "contrarian", "crowded"],
            EdgeCategory.NEWS_CATALYST: ["news", "earnings", "catalyst", "event", "announcement"],
            EdgeCategory.STRUCTURAL_BREAK: ["bos", "choch", "break of structure", "change of character"],
            EdgeCategory.CORRELATION_BREAK: ["correlation", "divergence", "cross-asset", "smt"],
            EdgeCategory.VOLATILITY_EXPANSION: ["volatility", "squeeze", "bollinger squeeze", "compression"],
        }

        for cat, keywords in category_keywords.items():
            if any(kw in lower for kw in keywords):
                return cat

        return EdgeCategory.NONE

    def get_prompt_instruction(self) -> str:
        """
        Generate instruction text for LLM prompts.

        Add this to your analyst agent's system prompt to enforce
        edge thesis generation.
        """
        return """
## Edge Thesis Required

Every trade recommendation MUST include an edge thesis:

1. **Edge Statement**: One sentence explaining WHY this trade has an edge.
   - ❌ BAD: "Technical analysis looks bullish"
   - ✅ GOOD: "Funding rate at 54% annualized creates carry trade opportunity"

2. **Edge Category**: What type of edge is this?
   - funding_arb, mean_reversion, momentum, liquidity_sweep, order_block,
     fvg_refill, sentiment_extreme, news_catalyst, structural_break,
     correlation_break, volatility_expansion

3. **Mechanism**: WHY does this edge exist? What forces price to move?
   - Example: "High funding → longs pay shorts → spot-perp basis narrows"

4. **Expiration**: When does this edge expire?
   - Example: "Next funding settlement (8h)" or "Earnings announcement"

⚠️ If you cannot articulate a clear edge, your confidence will be
automatically downgraded by 40%. "The chart looks good" is NOT an edge.
""".strip()
