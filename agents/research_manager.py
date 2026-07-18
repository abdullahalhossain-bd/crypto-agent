"""agents.research_manager
=====================================================================
Research Manager — synthesises the bull/bear debate into a structured
investment plan for the Trader.

Input: debate history + analyst reports
Output: ResearchPlan (recommendation + rationale + strategic actions)

FIXES (Batch 2 audit):
  - C2/C17/X3 (CRITICAL): the prompt previously only included the
    technical report; fundamentals, news, and sentiment were passed
    in as arguments but never used, so the plan was based on
    incomplete data. All four analyst reports are now included.
  - H17/M16: confidence is now parsed with schemas.parse_confidence,
    which handles percentages, fractions, and qualitative words
    ("confidence: high") instead of only "confidence: <digits>".
  - Wrapped the LLM call so a raised exception doesn't propagate as
    an unhandled error.
"""
from __future__ import annotations

from typing import Optional

from agents.schemas import ResearchPlan, PortfolioRating, parse_rating, parse_confidence
from external.llm_provider import LLMProvider, LLMMessage
from utils.logger import get_logger

log = get_logger("agents.research_manager")

_REPORT_TRUNCATE = 800
_DEBATE_TRUNCATE = 2000


class ResearchManager:
    """Turns bull/bear debate into a structured investment plan."""

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm or LLMProvider()

    # ----------------------------------------------------------------
    def create_plan(self, symbol: str,
                      debate_history: str,
                      fundamentals_report: str = "",
                      news_report: str = "",
                      sentiment_report: str = "",
                      technical_report: str = "") -> ResearchPlan:
        """Synthesize the debate AND all analyst reports into a ResearchPlan."""
        system_prompt = (
            "You are the Research Manager and debate facilitator. Critically "
            "evaluate the bull/bear debate and the underlying analyst reports, "
            "then deliver a clear, actionable investment plan for the trader.\n\n"
            "Rating Scale (use exactly one):\n"
            "- Buy: Strong conviction in the bull thesis\n"
            "- Overweight: Constructive view, gradually increase\n"
            "- Hold: Balanced view, maintain current position\n"
            "- Underweight: Cautious view, trim exposure\n"
            "- Sell: Strong conviction in the bear thesis\n\n"
            "Commit to a clear stance whenever the debate's strongest arguments "
            "warrant one; reserve Hold for genuinely balanced evidence. "
            "Base your plan on ALL analyst reports provided below, not just one."
        )
        user_prompt = (
            f"Symbol: {symbol}\n\n"
            f"Debate history:\n{debate_history[:_DEBATE_TRUNCATE]}\n\n"
            f"Fundamentals report summary: {fundamentals_report[:_REPORT_TRUNCATE]}\n\n"
            f"News report summary: {news_report[:_REPORT_TRUNCATE]}\n\n"
            f"Sentiment report summary: {sentiment_report[:_REPORT_TRUNCATE]}\n\n"
            f"Technical report summary: {technical_report[:_REPORT_TRUNCATE]}\n\n"
            "Deliver your investment plan. Include:\n"
            "1. Recommendation: (Buy/Overweight/Hold/Underweight/Sell)\n"
            "2. Rationale: (2-3 sentences on which side won the debate, "
            "referencing fundamentals/news/sentiment/technical where relevant)\n"
            "3. Strategic actions: (concrete steps for the trader)\n"
            "4. Confidence: (0-100%)"
        )
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=700,
                temperature=0.3,
            )
            text = resp.text if resp.success else ""  # M2 FIX: empty, not error text
        except Exception as e:  # noqa: BLE001
            log.warning("Research manager LLM call raised: %r", e)
            text = ""  # M2 FIX: empty, not error text

        rating_str = parse_rating(text, default="Hold")
        try:
            rating = PortfolioRating(rating_str)
        except ValueError:
            rating = PortfolioRating.HOLD

        confidence = parse_confidence(text, default=0.5)

        return ResearchPlan(
            recommendation=rating,
            rationale=text[:1000],
            strategic_actions=text[:500],
            confidence=float(min(1.0, max(0.0, confidence))),
        )