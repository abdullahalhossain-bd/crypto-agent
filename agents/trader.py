"""agents.trader
=====================================================================
Trader agent — turns the Research Manager's investment plan into a
concrete transaction proposal.

Input: ResearchPlan (recommendation + rationale + strategic actions)
       + analyst reports + current market data
Output: TraderProposal (action, lots, entry, SL, TP, reasoning)

FIXES (Batch 2 audit):
  - C12/L9: `_parse_lots` now tries several phrasings ("lots: X",
    "size: X", "position size of X lots", "X lots") and the `re`
    import is at module scope.
  - C16/H18: SL/TP are now validated — a minimum stop distance
    (0.5% of price) is enforced so a tiny/NaN-derived ATR can't
    produce an unrealistically tight stop, and SL is clamped so it
    can never go to/below zero.
  - Minor: `atr(df, 14)` was being computed twice per call; now
    computed once.
  - `resp.success` is checked before treating `resp.text` as valid
    reasoning; on failure a clear fallback reasoning string is used
    instead of an LLM error message posing as trade reasoning.
"""
from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from agents.schemas import (
    TraderAction, TraderProposal, PortfolioRating,
)
from external.llm_provider import LLMProvider, LLMMessage
from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("agents.trader")

_MIN_STOP_PCT = 0.005  # minimum SL distance as a fraction of price (0.5%)

_LOTS_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"(?:final\s+)?(?:lots?|size)\s*[:\-=]\s*([\d.]+)", re.IGNORECASE),
    re.compile(r"position\s+size\s+of\s+([\d.]+)\s*lots?", re.IGNORECASE),
    re.compile(r"([\d.]+)\s*lots?\b", re.IGNORECASE),
)


class Trader:
    """Turns research plans into transaction proposals."""

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm or LLMProvider()

    # ----------------------------------------------------------------
    def propose(self, symbol: str, df: pd.DataFrame,
                 research_plan: Any,  # ResearchPlan
                 fundamentals_report: str = "",
                 news_report: str = "",
                 sentiment_report: str = "",
                 technical_report: str = "",
                 max_lot: float = 0.1) -> TraderProposal:
        """Generate a transaction proposal from the research plan.

        C3/R3 FIX (Chief AI Architect Audit): Trader no longer computes
        lots/SL/TP via LLM — these are ALWAYS overwritten by SizingGate
        and SLTPGate in the risk pipeline. Trader now focuses on:
          - action (BUY/SELL/HOLD based on research plan)
          - reasoning (LLM justification for the trade)
          - entry_price (current market price)

        This saves 1 LLM call's worth of parsing work + eliminates the
        misleading audit trail where Trader's lots were logged but never
        used.
        """
        rec = research_plan.recommendation
        if rec in (PortfolioRating.BUY, PortfolioRating.OVERWEIGHT):
            action = TraderAction.BUY
        elif rec in (PortfolioRating.SELL, PortfolioRating.UNDERWEIGHT):
            action = TraderAction.SELL
        else:
            action = TraderAction.HOLD

        if df is None or df.empty or "close" not in df.columns:
            return TraderProposal(
                action=TraderAction.HOLD, lots=0.0,
                reasoning="No usable market data (empty DataFrame or missing 'close' column); defaulting to Hold.",
            )

        current_price = float(df["close"].iloc[-1])

        if action == TraderAction.HOLD:
            return TraderProposal(
                action=action, lots=0.0, entry_price=current_price,
                reasoning=f"Hold per research plan: {research_plan.rationale[:200]}",
            )

        # R3 FIX: lots/SL/TP are set to 0 — they will be computed by
        # SizingGate and SLTPGate in the risk pipeline. Trader only
        # provides action + reasoning + entry price.
        # Generate LLM reasoning for the trade decision.
        system_prompt = (
            "You are a trading agent. Based on the research plan and ALL analyst reports, "
            "provide a brief justification for this trade. Do NOT specify lots, SL, or TP — "
            "those are computed by the risk engine. Focus on WHY this trade makes sense "
            "given the fundamentals, news, sentiment, and technical analysis."
        )
        user_prompt = (
            f"Symbol: {symbol}\n"
            f"Current price: {current_price:.5f}\n\n"
            f"Research recommendation: {rec.value}\n"
            f"Research rationale: {research_plan.rationale[:500]}\n"
            f"Strategic actions: {research_plan.strategic_actions[:300]}\n\n"
            f"Fundamentals report:\n{fundamentals_report[:600]}\n\n"
            f"News report:\n{news_report[:600]}\n\n"
            f"Sentiment report:\n{sentiment_report[:600]}\n\n"
            f"Technical report:\n{technical_report[:600]}\n\n"
            f"Write a 2-3 sentence justification for this {action.value} trade."
        )
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=300,
                temperature=0.3,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Trader LLM call raised: %r", e)
            resp = None

        if resp is not None and resp.success:
            reasoning = resp.text
        else:
            err = resp.error if resp is not None else "no response"
            reasoning = f"[LLM unavailable ({err})] Auto: {action.value} per {rec.value} recommendation."
            log.warning("Trader falling back to auto reasoning for %s: %s", symbol, err)

        return TraderProposal(
            action=action,
            lots=0.0,  # R3 FIX: computed by SizingGate, not Trader
            entry_price=float(current_price),
            stop_loss=0.0,  # R3 FIX: computed by SLTPGate, not Trader
            take_profit=0.0,  # R3 FIX: computed by SLTPGate, not Trader
            reasoning=reasoning[:500],
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _compute_sl_tp(action: TraderAction, price: float, atr_val: float) -> tuple[float, float]:
        """Compute SL/TP with a minimum stop distance and sane bounds (C16/H18).

        Major #6 fix: SL is floored at 0.1% of price (not 0.0) so a zero
        stop is never produced. A zero stop = no stop = unlimited loss.
        """
        min_dist = max(price * _MIN_STOP_PCT, 1e-8)
        stop_dist = max(2.0 * atr_val, min_dist)
        tp_dist = max(3.0 * atr_val, min_dist * 1.5)
        # Major #6 fix: floor at 0.1% of price, not 0.0.
        sl_floor = price * 0.001

        if action == TraderAction.BUY:
            sl = price - stop_dist
            tp = price + tp_dist
            # Floor SL at 0.1% of price — never zero or negative.
            sl = max(sl, sl_floor)
            # TP must be above entry for a BUY.
            tp = max(tp, price + sl_floor)
        else:  # SELL
            sl = price + stop_dist
            tp = price - tp_dist
            # For SELL, SL is above entry — always positive, no floor needed,
            # but TP must be below entry and positive.
            tp = max(tp, sl_floor)
        return sl, tp

    # ----------------------------------------------------------------
    @staticmethod
    def _parse_lots(text: str, max_lot: float, rec: PortfolioRating) -> float:
        """Extract lot size from LLM text, with sensible defaults (C12/L9)."""
        if text:
            for pattern in _LOTS_PATTERNS:
                m = pattern.search(text)
                if m:
                    try:
                        lots = float(m.group(1))
                        if lots >= 0:
                            return min(lots, max_lot)
                    except ValueError:
                        continue
        # Default: scale by recommendation
        if rec == PortfolioRating.BUY:
            return max_lot
        if rec == PortfolioRating.OVERWEIGHT:
            return max_lot * 0.75
        if rec == PortfolioRating.SELL:
            return max_lot
        if rec == PortfolioRating.UNDERWEIGHT:
            return max_lot * 0.75
        return 0.0