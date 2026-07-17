"""agents.portfolio_manager
=====================================================================
Portfolio Manager — synthesises the risk debate into the final decision.

The PM is the FINAL authority. They take:
  - Research Manager's investment plan
  - Trader's transaction proposal
  - Risk debate history (aggressive/conservative/neutral)
  - Past decisions + lessons (from memory log)
  - Current portfolio state (equity, open positions), if available

And produce a PortfolioDecision:
  - rating (5-tier)
  - approved (bool)
  - final_lots (may be adjusted down from trader's proposal)
  - rationale
  - risk_adjustments

FIXES (Batch 2 audit):
  - C15/H10 (CRITICAL): previously only Buy/Overweight ratings could
    ever be `approved`, so a Sell/Underweight call from a rating that
    matched a Sell trader proposal was always rejected. Approval now
    checks that the rating's direction agrees with the trader's
    proposed action (both "up" or both "down"), not that the rating
    happens to be bullish.
  - C6/X4: added optional `portfolio_state` (equity, open positions,
    exposure) which, when supplied, is included in the prompt so the
    PM isn't deciding in a vacuum.
  - C12-style fix applied to lots parsing: several phrasings accepted.
  - M5: `risk_adjustments` is now derived from the conservative
    analyst's argument in the debate history instead of a static
    placeholder string.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from agents.schemas import PortfolioDecision, PortfolioRating, TraderAction, parse_rating
from external.llm_provider import LLMProvider, LLMMessage
from utils.logger import get_logger

log = get_logger("agents.portfolio_manager")

_LOTS_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"final\s+lots\s*[:\-=]\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"(?:lots?|size)\s*[:\-=]\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*lots?\b", re.IGNORECASE),
)

# Which ratings represent the "same direction" as a Buy vs a Sell trade.
_BUY_ALIGNED = {PortfolioRating.BUY, PortfolioRating.OVERWEIGHT}
_SELL_ALIGNED = {PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT}


def _extract_conservative_concern(risk_debate_history: str) -> str:
    """Pull a short excerpt of the conservative analyst's argument to
    use as a human-readable risk-adjustment note (M5)."""
    if not risk_debate_history:
        return "None specified."
    marker = "Conservative Analyst:"
    idx = risk_debate_history.find(marker)
    if idx == -1:
        return "None specified."
    excerpt = risk_debate_history[idx + len(marker):idx + len(marker) + 300].strip()
    return excerpt or "None specified."


class PortfolioManager:
    """Final decision-maker."""

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm or LLMProvider()

    # ----------------------------------------------------------------
    def decide(self, symbol: str,
                research_plan: Any,  # ResearchPlan
                trader_proposal: Any,  # TraderProposal
                risk_debate_history: str = "",
                past_context: str = "",
                portfolio_state: Optional[dict] = None) -> PortfolioDecision:
        """Make the final approve/reject decision."""
        system_prompt = (
            "You are the Portfolio Manager. Synthesize the risk analysts' debate "
            "and deliver the final trading decision.\n\n"
            "Rating Scale (use exactly one):\n"
            "- Buy: Strong conviction to enter or add\n"
            "- Overweight: Favorable, gradually increase\n"
            "- Hold: Maintain, no action\n"
            "- Underweight: Reduce, take partial profits\n"
            "- Sell: Exit or avoid\n\n"
            "You can REJECT the trader's proposal if the risk debate reveals "
            "unacceptable downside. You can also ADJUST the lot size down. "
            "Consider account equity and existing open positions if provided — "
            "avoid over-concentration."
        )
        user_prompt = (
            f"Symbol: {symbol}\n\n"
            f"Research Manager's plan:\n{research_plan.to_markdown()}\n\n"
            f"Trader's proposal:\n{trader_proposal.to_markdown()}\n\n"
        )
        if portfolio_state:
            state_lines = "\n".join(f"- {k}: {v}" for k, v in portfolio_state.items())
            user_prompt += f"Current portfolio state:\n{state_lines}\n\n"
        if past_context:
            user_prompt += f"Lessons from prior decisions:\n{past_context[:500]}\n\n"
        user_prompt += f"Risk Analysts Debate:\n{risk_debate_history[:2000]}\n\n"
        user_prompt += (
            "Make your final decision. Include:\n"
            "1. Rating: (Buy/Overweight/Hold/Underweight/Sell)\n"
            "2. Approved: (yes/no)\n"
            "3. Final lots: (adjust if needed)\n"
            "4. Rationale: (2-3 sentences)\n"
            "5. Risk adjustments: (if any)"
        )
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=800,
                temperature=0.3,
            )
            text = resp.text if resp.success else ""  # M2 FIX: empty, not error text
            llm_ok = resp.success
        except Exception as e:  # noqa: BLE001
            log.warning("Portfolio manager LLM call raised: %r", e)
            text = ""  # M2 FIX: empty, not error text
            llm_ok = False

        rating_str = parse_rating(text, default=research_plan.recommendation.value)
        try:
            rating = PortfolioRating(rating_str)
        except ValueError:
            rating = PortfolioRating.HOLD

        explicit_reject = "reject" in text.lower()[:300] or "not approved" in text.lower()[:300]
        explicit_approve = re.search(r"approved\s*[:\-]?\s*yes", text, re.IGNORECASE) is not None

        # Direction-aware approval (C15/H10): a Sell/Underweight rating
        # that agrees with a Sell trader proposal is a legitimate,
        # approvable decision — it should not be rejected just because
        # it isn't bullish.
        if trader_proposal.action == TraderAction.BUY:
            direction_aligned = rating in _BUY_ALIGNED
        elif trader_proposal.action == TraderAction.SELL:
            direction_aligned = rating in _SELL_ALIGNED
        else:
            direction_aligned = False

        if not llm_ok:
            approved = False
        elif explicit_reject:
            approved = False
        elif explicit_approve:
            approved = direction_aligned
        else:
            approved = direction_aligned

        final_lots = trader_proposal.lots
        for pattern in _LOTS_PATTERNS:
            m = pattern.search(text)
            if m:
                try:
                    final_lots = float(m.group(1))
                    break
                except ValueError:
                    continue
        # Cap at trader's proposed lots — PM can only reduce, not increase.
        final_lots = max(0.0, min(final_lots, trader_proposal.lots))
        if not approved:
            final_lots = 0.0

        risk_adjustments = _extract_conservative_concern(risk_debate_history)

        return PortfolioDecision(
            rating=rating,
            approved=approved,
            final_lots=float(final_lots),
            rationale=text[:1500],
            risk_adjustments=risk_adjustments,
        )