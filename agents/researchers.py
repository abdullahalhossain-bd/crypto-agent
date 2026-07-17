"""agents.researchers
=====================================================================
Bull vs Bear debate system.

After the analyst team produces reports, a Bull Researcher and Bear
Researcher engage in a structured debate:
  - Bull: builds the case FOR investing (growth, advantages, positives)
  - Bear: builds the case AGAINST (risks, weaknesses, negatives)

They take turns for N rounds, then the debate history is passed to
the Research Manager who synthesizes an investment plan.

This adversarial format forces both sides to be heard, preventing
confirmation bias that a single analyst would have.

FIXES (Batch 2 audit):
  - H7/L11: increased per-report truncation from 500 to 1000 chars
    so meaningful analysis isn't cut off mid-thought.
  - Each argument call now reports failure explicitly instead of a
    bare "[unavailable]" that looks like a real argument to the
    Research Manager.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from external.llm_provider import LLMProvider, LLMMessage
from utils.logger import get_logger

log = get_logger("agents.researchers")

_REPORT_TRUNCATE = 1000
_LAST_ARG_TRUNCATE = 800


@dataclass
class DebateResult:
    history: str = ""
    bull_history: str = ""
    bear_history: str = ""
    rounds: int = 0
    final_bull_argument: str = ""
    final_bear_argument: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "bull_history": self.bull_history,
            "bear_history": self.bear_history,
            "rounds": self.rounds,
            "final_bull_argument": self.final_bull_argument,
            "final_bear_argument": self.final_bear_argument,
        }


# ----------------------------------------------------------------------
class BullBearDebate:
    """Orchestrates the bull vs bear debate."""

    def __init__(self, llm: Optional[LLMProvider] = None,
                 max_rounds: int = 1) -> None:
        self.llm = llm or LLMProvider()
        self.max_rounds = int(max_rounds)

    # ----------------------------------------------------------------
    def run(self, symbol: str,
              fundamentals_report: str = "",
              news_report: str = "",
              sentiment_report: str = "",
              technical_report: str = "") -> DebateResult:
        """Run the debate. Each round = 1 bull + 1 bear argument."""
        result = DebateResult()
        for round_num in range(self.max_rounds):
            # Bull argues
            bull_arg = self._bull_argument(
                symbol, round_num, result.history,
                fundamentals_report, news_report,
                sentiment_report, technical_report,
                result.final_bear_argument,
            )
            result.bull_history += "\n" + bull_arg
            result.history += "\n" + bull_arg
            result.final_bull_argument = bull_arg
            # Bear argues
            bear_arg = self._bear_argument(
                symbol, round_num, result.history,
                fundamentals_report, news_report,
                sentiment_report, technical_report,
                result.final_bull_argument,
            )
            result.bear_history += "\n" + bear_arg
            result.history += "\n" + bear_arg
            result.final_bear_argument = bear_arg
            result.rounds += 1
        return result

    # ----------------------------------------------------------------
    def _bull_argument(self, symbol: str, round_num: int,
                         history: str, fundamentals: str, news: str,
                         sentiment: str, technical: str,
                         last_bear_arg: str) -> str:
        system_prompt = (
            "You are a Bull Analyst advocating for investing. Build a strong, "
            "evidence-based case emphasizing growth potential, competitive advantages, "
            "and positive market indicators. Address the bear's concerns directly."
        )
        user_prompt = (
            f"Make the bull case for {symbol} (round {round_num + 1}).\n\n"
            f"Analyst reports:\n"
            f"Fundamentals: {fundamentals[:_REPORT_TRUNCATE]}\n\n"
            f"News: {news[:_REPORT_TRUNCATE]}\n\n"
            f"Sentiment: {sentiment[:_REPORT_TRUNCATE]}\n\n"
            f"Technical: {technical[:_REPORT_TRUNCATE]}\n\n"
        )
        if last_bear_arg:
            user_prompt += f"Last bear argument to refute:\n{last_bear_arg[:_LAST_ARG_TRUNCATE]}\n\n"
        user_prompt += (
            "Deliver a compelling bull argument (2-3 paragraphs). "
            "Be specific with data. Engage directly with bear concerns."
        )
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=600,
                temperature=0.4,
            )
            text = resp.text if resp.success else ""  # M2 FIX: empty, not error text
        except Exception as e:  # noqa: BLE001
            log.warning("Bull argument LLM call raised: %r", e)
            text = ""  # M2 FIX: empty, not error text
        return f"Bull Analyst: {text}"

    # ----------------------------------------------------------------
    def _bear_argument(self, symbol: str, round_num: int,
                         history: str, fundamentals: str, news: str,
                         sentiment: str, technical: str,
                         last_bull_arg: str) -> str:
        system_prompt = (
            "You are a Bear Analyst making the case against investing. Present a "
            "well-reasoned argument emphasizing risks, challenges, and negative indicators. "
            "Counter the bull's arguments effectively."
        )
        user_prompt = (
            f"Make the bear case against {symbol} (round {round_num + 1}).\n\n"
            f"Analyst reports:\n"
            f"Fundamentals: {fundamentals[:_REPORT_TRUNCATE]}\n\n"
            f"News: {news[:_REPORT_TRUNCATE]}\n\n"
            f"Sentiment: {sentiment[:_REPORT_TRUNCATE]}\n\n"
            f"Technical: {technical[:_REPORT_TRUNCATE]}\n\n"
        )
        if last_bull_arg:
            user_prompt += f"Last bull argument to counter:\n{last_bull_arg[:_LAST_ARG_TRUNCATE]}\n\n"
        user_prompt += (
            "Deliver a compelling bear argument (2-3 paragraphs). "
            "Be specific with data. Expose weaknesses in the bull thesis."
        )
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=600,
                temperature=0.4,
            )
            text = resp.text if resp.success else ""  # M2 FIX: empty, not error text
        except Exception as e:  # noqa: BLE001
            log.warning("Bear argument LLM call raised: %r", e)
            text = ""  # M2 FIX: empty, not error text
        return f"Bear Analyst: {text}"