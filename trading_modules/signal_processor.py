"""
Signal Processor — Structured JSON Extraction from LLM Output
===============================================================

Extracts structured decision data from the trader agent's free-text output.
Converts natural language analysis into actionable JSON with:
  - action: BUY / SELL / HOLD
  - target_price: specific price target
  - confidence: 0-1 confidence score
  - risk_score: 0-1 risk assessment
  - reasoning: summary of decision rationale
  - edge_thesis: required economic driver (Orallexa pattern)

Also includes the 5-tier rating extraction from TradingAgents v0.3.1.

Source: TradingAgents-CN (review #24) + TradingAgents v0.3.1 (review #30)
Pattern: Orallexa edge_thesis required field

Usage:
    from signal_processor import SignalProcessor

    sp = SignalProcessor()

    llm_output = '''
    Based on comprehensive analysis, I recommend BUYING BTCUSDT.

    Target Price: $72,000
    Confidence: 75%
    Risk Score: 30%

    The bullish case is supported by strong momentum above EMA50,
    with institutional volume confirmation. The edge comes from
    funding rate divergence creating a carry opportunity.

    Rating: Overweight
    '''

    decision = sp.process(llm_output, symbol="BTCUSDT")
    print(decision.to_dict())
"""

from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradingDecision:
    """Structured trading decision extracted from LLM output."""
    action: str = "HOLD"  # BUY / SELL / HOLD
    target_price: Optional[float] = None
    confidence: float = 0.5  # 0-1
    risk_score: float = 0.5  # 0-1 (0=low risk, 1=high risk)
    reasoning: str = ""
    edge_thesis: str = ""
    rating: str = "Hold"  # 5-tier: Buy/Overweight/Hold/Underweight/Sell
    symbol: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target_price": self.target_price,
            "confidence": round(self.confidence, 4),
            "risk_score": round(self.risk_score, 4),
            "reasoning": self.reasoning,
            "edge_thesis": self.edge_thesis,
            "rating": self.rating,
            "symbol": self.symbol,
        }


class SignalProcessor:
    """
    Extracts structured trading decisions from LLM free-text output.

    Uses regex patterns to extract:
    - Action (BUY/SELL/HOLD) from keywords
    - Target price from "target", "tp", "target price" patterns
    - Confidence from "confidence" or percentage patterns
    - Risk score from "risk" patterns
    - Edge thesis from "edge" or "catalyst" patterns
    - 5-tier rating from "rating" labels
    """

    # Action keywords (bilingual EN/CN)
    BUY_KEYWORDS = [
        "buy", "long", "bull", "bullish", "overweight",
        "accumulate", "enter long", "go long",
        "买入", "做多", "看多", "增持",
    ]
    SELL_KEYWORDS = [
        "sell", "short", "bear", "bearish", "underweight",
        "exit", "close long", "go short",
        "卖出", "做空", "看空", "减持",
    ]
    HOLD_KEYWORDS = [
        "hold", "wait", "neutral", "stand aside", "no trade",
        "观望", "持有", "等待",
    ]

    # Regex patterns
    PRICE_RE = re.compile(
        r"(?:target|tp|目标价|price\s*target|entry)[^\d]{0,15}"
        r"[$\s]*(\d{1,}(?:\.\d+)?)",
        re.IGNORECASE,
    )
    CONFIDENCE_RE = re.compile(
        r"(?:confidence|conf|置信度)[^\d]{0,15}(\d{1,3}(?:\.\d+)?)",
        re.IGNORECASE,
    )
    RISK_RE = re.compile(
        r"(?:risk\s*score|risk\s*level|risk|风险)[^\d]{0,15}(\d{1,3}(?:\.\d+)?)",
        re.IGNORECASE,
    )
    PERCENTAGE_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
    RATING_RE = re.compile(
        r"rating.*?[:\-][\s*]*(\w+)",
        re.IGNORECASE,
    )

    VALID_RATINGS = {"buy", "overweight", "hold", "underweight", "sell"}

    def process(self, text: str, symbol: str = "") -> TradingDecision:
        """
        Process LLM output text and extract structured decision.

        Args:
            text: LLM agent's free-text output
            symbol: Trading symbol for context

        Returns:
            TradingDecision with all extracted fields
        """
        if not text or not text.strip():
            return TradingDecision(
                symbol=symbol,
                reasoning="Empty input — defaulting to HOLD",
            )

        text_lower = text.lower()

        decision = TradingDecision(symbol=symbol)

        # Extract action
        decision.action = self._extract_action(text_lower)

        # Extract target price
        decision.target_price = self._extract_target_price(text)

        # Extract confidence
        decision.confidence = self._extract_confidence(text)

        # Extract risk score
        decision.risk_score = self._extract_risk_score(text)

        # Extract edge thesis
        decision.edge_thesis = self._extract_edge_thesis(text)

        # Extract 5-tier rating
        decision.rating = self._extract_rating(text)

        # Extract reasoning (first paragraph or summary)
        decision.reasoning = self._extract_reasoning(text)

        return decision

    def _extract_action(self, text_lower: str) -> str:
        """Extract BUY/SELL/HOLD from text."""
        # Check for explicit rating first (highest priority)
        for rating in ("buy", "overweight", "sell", "underweight"):
            if rating in text_lower:
                if rating in ("buy", "overweight"):
                    return "BUY"
                else:
                    return "SELL"

        # Check keywords
        for kw in self.SELL_KEYWORDS:
            if kw in text_lower:
                return "SELL"

        for kw in self.BUY_KEYWORDS:
            if kw in text_lower:
                return "BUY"

        for kw in self.HOLD_KEYWORDS:
            if kw in text_lower:
                return "HOLD"

        return "HOLD"

    def _extract_target_price(self, text: str) -> Optional[float]:
        """Extract target price from text."""
        m = self.PRICE_RE.search(text)
        if m:
            try:
                price = float(m.group(1))
                if price > 0:
                    return price
            except (ValueError, IndexError):
                pass
        return None

    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score (0-1) from text."""
        m = self.CONFIDENCE_RE.search(text)
        if m:
            try:
                raw = float(m.group(1))
                # Normalize: if > 1, assume percentage (75 → 0.75)
                if raw > 1:
                    raw = raw / 100.0
                return max(0.0, min(1.0, raw))
            except (ValueError, IndexError):
                pass
        return 0.5  # Default neutral

    def _extract_risk_score(self, text: str) -> float:
        """Extract risk score (0-1) from text."""
        m = self.RISK_RE.search(text)
        if m:
            try:
                raw = float(m.group(1))
                if raw > 1:
                    raw = raw / 100.0
                return max(0.0, min(1.0, raw))
            except (ValueError, IndexError):
                pass
        return 0.5  # Default medium risk

    def _extract_edge_thesis(self, text: str) -> str:
        """
        Extract edge thesis — the economic driver behind the trade.

        Implements the Orallexa 'edge_thesis' required field pattern:
        "Conviction auto-downgrades if the model can't articulate an
        economic driver (Longmore 'no theory of edge' rule)."

        Looks for keywords: "edge", "catalyst", "driver", "because", "reason"
        """
        patterns = [
            r"(?:edge|catalyst|driver|alpha\s*source)[^\n]{0,200}",
            r"(?:because|reason|rationale|因为|原因)[^\n]{0,200}",
        ]

        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                # Clean up and truncate
                thesis = m.group(0).strip()
                # Remove the leading keyword for cleaner output
                for prefix in ("edge:", "catalyst:", "driver:", "because:", "reason:", "rationale:"):
                    if thesis.lower().startswith(prefix):
                        thesis = thesis[len(prefix):].strip()
                if len(thesis) > 10:  # Minimum meaningful length
                    return thesis[:300]  # Cap at 300 chars
        return ""

    def _extract_rating(self, text: str) -> str:
        """Extract 5-tier rating from text."""
        m = self.RATING_RE.search(text)
        if m:
            rating = m.group(1).lower()
            if rating in self.VALID_RATINGS:
                return rating.capitalize()

        # Fallback: find first rating word
        for word in text.lower().split():
            clean = word.strip("*:.,")
            if clean in self.VALID_RATINGS:
                return clean.capitalize()

        # Default based on action
        action = self._extract_action(text.lower())
        if action == "BUY":
            return "Overweight"
        elif action == "SELL":
            return "Underweight"
        return "Hold"

    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning summary (first meaningful paragraph)."""
        paragraphs = text.strip().split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Skip short lines
                # Remove markdown headers
                para = re.sub(r"^#+\s*", "", para)
                return para[:500]  # Cap at 500 chars
        return text[:500]
