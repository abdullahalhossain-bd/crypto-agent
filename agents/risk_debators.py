"""agents.risk_debators
=====================================================================
Risk Management debate — aggressive vs conservative vs neutral.

After the Trader proposes a transaction, three risk analysts debate
the proposal from different risk perspectives:
  - Aggressive  : champions high-reward, high-risk opportunities
  - Conservative : emphasizes capital protection, downside risk
  - Neutral     : balanced view, weighs both sides

They take turns for N rounds, then the Portfolio Manager synthesizes
the debate into the final approve/reject decision.

FIXES (Batch 2 audit):
  - C13: truncation of the trader proposal / prior arguments raised
    from 300-500 to 500-1000 chars so the risk debate has enough
    context to be meaningful.
  - H6/X6: `RiskDebate.run` now accepts an optional `risk_metrics`
    dict (e.g. current exposure, VaR, account equity, open
    positions) which is injected into every risk analyst's prompt
    instead of the debate happening with no quantitative grounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from external.llm_provider import LLMProvider, LLMMessage
from utils.logger import get_logger

log = get_logger("agents.risk_debators")

_PROPOSAL_TRUNCATE = 1000
_LAST_ARG_TRUNCATE = 500
# Critical #1 fix: analyst reports are now included in risk debate prompts.
# Truncated to keep prompt size reasonable (4 reports × 500 chars = 2000 chars).
_REPORT_TRUNCATE = 500


def _format_risk_metrics(risk_metrics: Optional[dict]) -> str:
    if not risk_metrics:
        return "No portfolio risk metrics provided."
    return "\n".join(f"- {k}: {v}" for k, v in risk_metrics.items())


@dataclass
class RiskDebateResult:
    history: str = ""
    aggressive_history: str = ""
    conservative_history: str = ""
    neutral_history: str = ""
    rounds: int = 0
    final_aggressive: str = ""
    final_conservative: str = ""
    final_neutral: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "aggressive_history": self.aggressive_history,
            "conservative_history": self.conservative_history,
            "neutral_history": self.neutral_history,
            "rounds": self.rounds,
            "final_aggressive": self.final_aggressive,
            "final_conservative": self.final_conservative,
            "final_neutral": self.final_neutral,
        }


# ----------------------------------------------------------------------
class RiskDebate:
    """Orchestrates the 3-way risk debate."""

    def __init__(self, llm: Optional[LLMProvider] = None,
                 max_rounds: int = 1) -> None:
        self.llm = llm or LLMProvider()
        self.max_rounds = int(max_rounds)

    # ----------------------------------------------------------------
    def run(self, symbol: str, trader_proposal: str,
              fundamentals_report: str = "",
              news_report: str = "",
              sentiment_report: str = "",
              technical_report: str = "",
              risk_metrics: Optional[dict] = None) -> RiskDebateResult:
        metrics_text = _format_risk_metrics(risk_metrics)
        result = RiskDebateResult()
        for round_num in range(self.max_rounds):
            agg = self._aggressive(
                symbol, trader_proposal, result.history, metrics_text,
                fundamentals_report, news_report, sentiment_report, technical_report,
                result.final_conservative, result.final_neutral,
            )
            result.aggressive_history += "\n" + agg
            result.history += "\n" + agg
            result.final_aggressive = agg

            con = self._conservative(
                symbol, trader_proposal, result.history, metrics_text,
                fundamentals_report, news_report, sentiment_report, technical_report,
                result.final_aggressive, result.final_neutral,
            )
            result.conservative_history += "\n" + con
            result.history += "\n" + con
            result.final_conservative = con

            neu = self._neutral(
                symbol, trader_proposal, result.history, metrics_text,
                fundamentals_report, news_report, sentiment_report, technical_report,
                result.final_aggressive, result.final_conservative,
            )
            result.neutral_history += "\n" + neu
            result.history += "\n" + neu
            result.final_neutral = neu
            result.rounds += 1
        return result

    # ----------------------------------------------------------------
    def _call(self, system_prompt: str, user_prompt: str, label: str,
               max_tokens: int = 500, temperature: float = 0.5) -> str:
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=max_tokens, temperature=temperature,
            )
            text = resp.text if resp.success else ""  # M2 FIX: empty, not error text
        except Exception as e:  # noqa: BLE001
            log.warning("%s LLM call raised: %r", label, e)
            text = ""  # M2 FIX: empty, not error text
        return text

    def _aggressive(self, symbol, proposal, history, metrics_text,
                     fund, news, sent, tech,
                     last_con, last_neu) -> str:
        system_prompt = (
            "You are an Aggressive Risk Analyst championing high-reward, high-risk "
            "opportunities. Focus on upside potential, growth, and competitive advantage. "
            "Challenge conservative and neutral stances with data-driven rebuttals. "
            "Consider ALL analyst reports — fundamentals, news, sentiment, and technical."
        )
        user_prompt = (
            f"Trader's proposal for {symbol}:\n{proposal[:_PROPOSAL_TRUNCATE]}\n\n"
            f"Portfolio risk metrics:\n{metrics_text}\n\n"
            f"Fundamentals report:\n{fund[:_REPORT_TRUNCATE]}\n\n"
            f"News report:\n{news[:_REPORT_TRUNCATE]}\n\n"
            f"Sentiment report:\n{sent[:_REPORT_TRUNCATE]}\n\n"
            f"Technical report:\n{tech[:_REPORT_TRUNCATE]}\n\n"
        )
        if last_con:
            user_prompt += f"Conservative's argument:\n{last_con[:_LAST_ARG_TRUNCATE]}\n\n"
        if last_neu:
            user_prompt += f"Neutral's argument:\n{last_neu[:_LAST_ARG_TRUNCATE]}\n\n"
        user_prompt += "Make the aggressive case (2-3 paragraphs). Be bold but evidence-based. "
        user_prompt += "Reference specific points from the analyst reports."
        text = self._call(system_prompt, user_prompt, "Aggressive argument")
        return f"Aggressive Analyst: {text}"

    def _conservative(self, symbol, proposal, history, metrics_text,
                       fund, news, sent, tech,
                       last_agg, last_neu) -> str:
        system_prompt = (
            "You are a Conservative Risk Analyst emphasizing capital protection and "
            "downside risk. Focus on what could go wrong, worst-case scenarios, and "
            "why caution is warranted. Challenge aggressive optimism. "
            "Consider ALL analyst reports — fundamentals, news, sentiment, and technical."
        )
        user_prompt = (
            f"Trader's proposal for {symbol}:\n{proposal[:_PROPOSAL_TRUNCATE]}\n\n"
            f"Portfolio risk metrics:\n{metrics_text}\n\n"
            f"Fundamentals report:\n{fund[:_REPORT_TRUNCATE]}\n\n"
            f"News report:\n{news[:_REPORT_TRUNCATE]}\n\n"
            f"Sentiment report:\n{sent[:_REPORT_TRUNCATE]}\n\n"
            f"Technical report:\n{tech[:_REPORT_TRUNCATE]}\n\n"
        )
        if last_agg:
            user_prompt += f"Aggressive's argument:\n{last_agg[:_LAST_ARG_TRUNCATE]}\n\n"
        if last_neu:
            user_prompt += f"Neutral's argument:\n{last_neu[:_LAST_ARG_TRUNCATE]}\n\n"
        user_prompt += "Make the conservative case (2-3 paragraphs). Be cautious but evidence-based. "
        user_prompt += "Reference specific points from the analyst reports."
        text = self._call(system_prompt, user_prompt, "Conservative argument")
        return f"Conservative Analyst: {text}"

    def _neutral(self, symbol, proposal, history, metrics_text,
                  fund, news, sent, tech,
                  last_agg, last_con) -> str:
        system_prompt = (
            "You are a Neutral Risk Analyst providing a balanced view. Weigh both "
            "the upside and downside objectively. Do not favor risk-taking or caution "
            "— present the trade-offs clearly. "
            "Consider ALL analyst reports — fundamentals, news, sentiment, and technical."
        )
        user_prompt = (
            f"Trader's proposal for {symbol}:\n{proposal[:_PROPOSAL_TRUNCATE]}\n\n"
            f"Portfolio risk metrics:\n{metrics_text}\n\n"
            f"Fundamentals report:\n{fund[:_REPORT_TRUNCATE]}\n\n"
            f"News report:\n{news[:_REPORT_TRUNCATE]}\n\n"
            f"Sentiment report:\n{sent[:_REPORT_TRUNCATE]}\n\n"
            f"Technical report:\n{tech[:_REPORT_TRUNCATE]}\n\n"
        )
        if last_agg:
            user_prompt += f"Aggressive's argument:\n{last_agg[:_LAST_ARG_TRUNCATE]}\n\n"
        if last_con:
            user_prompt += f"Conservative's argument:\n{last_con[:_LAST_ARG_TRUNCATE]}\n\n"
        user_prompt += "Provide the neutral assessment (2-3 paragraphs). Be balanced and objective. "
        user_prompt += "Reference specific points from the analyst reports."
        text = self._call(system_prompt, user_prompt, "Neutral argument", temperature=0.4)
        return f"Neutral Analyst: {text}"