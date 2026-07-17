"""
Explainable AI (XAI) — feature contribution breakdown
=====================================================

Institutional desks don't trust black-box signals. They want to know WHY
a trade was recommended. This module takes the InstitutionalEntryGate's
score breakdown + all confluence module outputs and produces a
human-readable explanation:

    Trend alignment:        +15
    HTF alignment:          +15
    Key level (swing_low):  +20
    Liquidity (eq_highs):   +15
    Confirmation (BOS):     +15
    Volume (1.8x avg):      +8
    Session (london):       +5
    News (>60min clear):    +5
    ────────────────────────────
    Candle quality:         -3   (weak body)
    Regime (trending):       0   (no penalty)
    Cross-asset (DXY down): +2   (bonus, not in base score)
    ────────────────────────────
    Final Score:             97/100  → A+  → BUY

Usage:
    from trading_modules.explainable_ai import Explainer, ExplanationInput
    explainer = Explainer()
    explanation = explainer.explain(decision, confluence_results)
    print(explanation.text_summary)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExplanationInput:
    """Bag of confluence results to feed into the explainer."""
    # Score breakdown from InstitutionalEntryGate
    score_breakdown: dict[str, float] = field(default_factory=dict)
    # Bonus/penalty signals from confluence modules
    bonus_signals: list[tuple[str, float, str]] = field(default_factory=list)
    # (name, contribution, note)
    penalty_signals: list[tuple[str, float, str]] = field(default_factory=list)
    # Critical failures
    critical_failures: list[str] = field(default_factory=list)


@dataclass
class Explanation:
    text_summary: str              # human-readable multi-line explanation
    contributions: list[dict]      # machine-readable list of contributions
    final_score: float
    final_grade: str
    final_action: str
    top_positive: Optional[dict]   # largest positive contributor
    top_negative: Optional[dict]   # largest negative contributor

    def to_dict(self) -> dict:
        return {
            "text_summary": self.text_summary,
            "contributions": self.contributions,
            "final_score": round(self.final_score, 1),
            "final_grade": self.final_grade,
            "final_action": self.final_action,
            "top_positive": self.top_positive,
            "top_negative": self.top_negative,
        }


class Explainer:
    """Generate human-readable explanations from gate + confluence results."""

    # Friendly names for each score dimension
    LABELS: dict[str, str] = {
        "trend": "Trend alignment",
        "htf_alignment": "HTF (D1+H4+H1) alignment",
        "key_level": "Key level proximity",
        "liquidity": "Liquidity sweep",
        "confirmation": "Confirmation (BOS/CHoCH/Engulfing)",
        "volume": "Volume confirmation",
        "session": "Session quality",
        "news": "News clearance",
        "momentum": "Momentum",
        "rr": "Risk:Reward",
    }

    def explain(
        self,
        decision,                # InstitutionalEntryGate.EntryDecision
        confluence_input: Optional[ExplanationInput] = None,
    ) -> Explanation:
        """Build an explanation from a gate decision + optional confluence signals."""
        contributions: list[dict] = []
        breakdown = decision.score_breakdown or {}

        # Add base score breakdown
        for key, value in breakdown.items():
            label = self.LABELS.get(key, key.replace("_", " ").title())
            contributions.append({
                "name": key,
                "label": label,
                "value": round(float(value), 1),
                "type": "base",
                "note": "",
            })

        # Add bonus signals from confluence modules
        if confluence_input is not None:
            for name, value, note in confluence_input.bonus_signals:
                contributions.append({
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "value": round(float(value), 1),
                    "type": "bonus",
                    "note": note,
                })
            for name, value, note in confluence_input.penalty_signals:
                contributions.append({
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "value": round(float(value), 1),
                    "type": "penalty",
                    "note": note,
                })

        # Sort by value descending
        contributions.sort(key=lambda c: c["value"], reverse=True)

        # Find top positive and top negative
        positives = [c for c in contributions if c["value"] > 0]
        negatives = [c for c in contributions if c["value"] < 0]
        top_positive = max(positives, key=lambda c: c["value"]) if positives else None
        top_negative = min(negatives, key=lambda c: c["value"]) if negatives else None

        # Build text summary
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append(f"  TRADE EXPLANATION — {decision.symbol} {decision.direction}")
        lines.append("=" * 60)
        lines.append(f"  Action:  {decision.action}")
        lines.append(f"  Score:   {decision.score:.1f}/100  (Grade: {decision.grade})")
        if decision.confidence_pct:
            lines.append(f"  Confidence: {decision.confidence_pct:.1f}%  "
                         f"(Win prob: {decision.win_probability:.1%})")
        if decision.rr_ratio:
            lines.append(f"  R:R:     {decision.rr_ratio:.2f}:1")
        lines.append("")
        lines.append("  ── CONTRIBUTIONS ──")
        for c in contributions:
            sign = "+" if c["value"] >= 0 else ""
            tag = ""
            if c["type"] == "bonus":
                tag = " [bonus]"
            elif c["type"] == "penalty":
                tag = " [penalty]"
            note_str = f"  ({c['note']})" if c["note"] else ""
            lines.append(f"  {c['label']:<35} {sign}{c['value']:>5.1f}{tag}{note_str}")

        if decision.failed_checks:
            lines.append("")
            lines.append("  ── FAILED CHECKS ──")
            for f in decision.failed_checks:
                lines.append(f"  ✗ {f}")

        if decision.skip_reason:
            lines.append("")
            lines.append(f"  ── SKIP REASON ──")
            lines.append(f"  {decision.skip_reason}")

        lines.append("")
        lines.append("  ── CONTEXT ──")
        if decision.regime:
            lines.append(f"  Regime:            {decision.regime}")
        if decision.trend_ltf:
            lines.append(f"  Trend (LTF):       {decision.trend_ltf}")
        if decision.trend_htf:
            lines.append(f"  Trend (HTF):       {decision.trend_htf}")
        lines.append(f"  HTF alignment:     {decision.htf_alignment}")
        if decision.candle_quality_label:
            lines.append(f"  Candle quality:    {decision.candle_quality_label} "
                         f"({decision.candle_quality_score:.2f})")
        if decision.confirmation:
            lines.append(f"  Confirmation:      {decision.confirmation}")
        if decision.notes:
            lines.append(f"  Notes:             {', '.join(decision.notes)}")

        lines.append("")
        lines.append("=" * 60)

        return Explanation(
            text_summary="\n".join(lines),
            contributions=contributions,
            final_score=float(decision.score),
            final_grade=decision.grade,
            final_action=decision.action,
            top_positive=top_positive,
            top_negative=top_negative,
        )


__all__ = ["Explainer", "Explanation", "ExplanationInput"]
