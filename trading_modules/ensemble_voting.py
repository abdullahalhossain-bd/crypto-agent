"""
Ensemble Voting System — multi-model decision fusion
====================================================

No single model is right all the time. An ensemble of independent models,
each with its own bias, votes on every trade. The final decision is a
weighted vote — models with better recent accuracy get more weight.

Models supported (each is a callable returning a vote):
    - TrendModel       — EMA/ADX-based trend follower
    - LiquidityModel   — liquidity sweep / order block based
    - PatternModel     — candlestick / structure pattern
    - RiskModel        — R:R + position sizing
    - NewsModel        — news sentiment / blackout
    - RegimeModel      — market regime filter
    - (any user-supplied model)

Each model returns a Vote(action, confidence, reason). The ensemble
combines them into a final EnsembleDecision.

Usage:
    from trading_modules.ensemble_voting import EnsembleVoter, Vote, Model
    ensemble = EnsembleVoter()
    ensemble.register("trend", trend_model, weight=1.0)
    ensemble.register("liquidity", liq_model, weight=1.2)
    decision = ensemble.vote(symbol="BTCUSD", context={...})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Vote:
    action: str               # "BUY" / "SELL" / "WAIT" / "SKIP"
    confidence: float         # 0..1
    reason: str = ""


@dataclass
class EnsembleDecision:
    action: str               # final fused action
    confidence: float         # 0..1
    score: float              # -100..+100 (positive = bullish, negative = bearish)
    votes: list[dict] = field(default_factory=list)
    # each vote dict: {model, action, confidence, weight, contribution, reason}
    disagreements: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def should_execute(self) -> bool:
        return self.action in ("BUY", "SELL")

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "score": round(self.score, 1),
            "votes": self.votes,
            "disagreements": self.disagreements,
            "notes": self.notes,
        }


# Type for a model: takes (symbol, context) → Vote
Model = Callable[[str, dict], Vote]


class EnsembleVoter:
    """Multi-model ensemble voter.

    Models are registered with a name + initial weight. Weights are
    dynamically adjusted based on recent accuracy (if `track_accuracy=True`).

    Parameters:
        agreement_threshold: fraction of weighted votes that must agree (default 0.6)
        min_confidence: minimum fused confidence to act (default 0.55)
        track_accuracy: if True, weights adapt based on recent outcomes (default True)
        accuracy_window: # of recent votes to track per model (default 50)
    """

    def __init__(
        self, agreement_threshold: float = 0.6,
        min_confidence: float = 0.55,
        track_accuracy: bool = True,
        accuracy_window: int = 50,
    ) -> None:
        self.agreement_threshold = float(agreement_threshold)
        self.min_confidence = float(min_confidence)
        self.track_accuracy = bool(track_accuracy)
        self.accuracy_window = int(accuracy_window)
        self.models: dict[str, dict] = {}  # name → {callable, weight, history}

    def register(self, name: str, model: Model, weight: float = 1.0) -> None:
        """Register a voting model."""
        if weight <= 0:
            raise ValueError(f"weight must be > 0, got {weight}")
        self.models[name] = {
            "callable": model,
            "weight": float(weight),
            "history": [],  # list of (predicted_action, was_correct)
        }
        logger.info(f"Ensemble: registered model '{name}' with weight {weight}")

    def update_weight(self, name: str, weight: float) -> None:
        if name in self.models and weight > 0:
            self.models[name]["weight"] = float(weight)

    def record_outcome(self, name: str, predicted_action: str, was_correct: bool) -> None:
        """Record whether a model's vote was correct (for adaptive weighting)."""
        if name not in self.models or not self.track_accuracy:
            return
        self.models[name]["history"].append((predicted_action, was_correct))
        if len(self.models[name]["history"]) > self.accuracy_window:
            self.models[name]["history"] = self.models[name]["history"][-self.accuracy_window:]
        # Adjust weight: weight = base_accuracy × 2 (cap at 2.0)
        history = self.models[name]["history"]
        if len(history) >= 10:
            correct = sum(1 for _, ok in history if ok)
            accuracy = correct / len(history)
            self.models[name]["weight"] = max(0.1, min(2.0, accuracy * 2.0))

    def vote(self, symbol: str, context: dict) -> EnsembleDecision:
        """Collect votes from all registered models and fuse into a decision."""
        if not self.models:
            return EnsembleDecision(
                action="SKIP", confidence=0.0, score=0.0,
                notes=["no models registered"],
            )

        votes: list[dict] = []
        buy_score = 0.0
        sell_score = 0.0
        wait_score = 0.0
        skip_score = 0.0
        total_weight = 0.0

        for name, info in self.models.items():
            try:
                v: Vote = info["callable"](symbol, context)
            except Exception as e:
                logger.warning(f"Model '{name}' raised: {e}")
                v = Vote(action="SKIP", confidence=0.0, reason=f"error: {e}")
            weight = info["weight"]
            contribution = v.confidence * weight
            votes.append({
                "model": name,
                "action": v.action,
                "confidence": round(v.confidence, 3),
                "weight": round(weight, 3),
                "contribution": round(contribution, 3),
                "reason": v.reason,
            })
            total_weight += weight
            if v.action == "BUY":
                buy_score += contribution
            elif v.action == "SELL":
                sell_score += contribution
            elif v.action == "WAIT":
                wait_score += contribution
            else:  # SKIP
                skip_score += contribution

        if total_weight <= 0:
            return EnsembleDecision(action="SKIP", confidence=0.0, score=0.0,
                                     votes=votes, notes=["total weight is zero"])

        # Normalize scores
        buy_norm = buy_score / total_weight
        sell_norm = sell_score / total_weight
        wait_norm = wait_score / total_weight
        skip_norm = skip_score / total_weight

        # Final score: +100 = strong buy, -100 = strong sell, 0 = neutral
        score = (buy_norm - sell_norm) * 100

        # Determine final action
        # If skip_norm > 0.4 → SKIP (a strong "skip" vote overrides)
        # Else if buy_norm >= agreement_threshold → BUY
        # Else if sell_norm >= agreement_threshold → SELL
        # Else if wait_norm > 0.3 → WAIT
        # Else SKIP
        disagreements: list[str] = []
        buy_voters = [v for v in votes if v["action"] == "BUY"]
        sell_voters = [v for v in votes if v["action"] == "SELL"]
        if buy_voters and sell_voters:
            disagreements.append(
                f"Conflicting signals: {len(buy_voters)} BUY vs {len(sell_voters)} SELL"
            )

        if skip_norm > 0.4:
            action = "SKIP"
            confidence = skip_norm
        elif buy_norm >= self.agreement_threshold and buy_norm > sell_norm:
            action = "BUY"
            confidence = buy_norm
        elif sell_norm >= self.agreement_threshold and sell_norm > buy_norm:
            action = "SELL"
            confidence = sell_norm
        elif wait_norm > 0.3:
            action = "WAIT"
            confidence = wait_norm
        else:
            action = "SKIP"
            confidence = 1.0 - max(buy_norm, sell_norm)

        # Apply minimum confidence threshold
        if action in ("BUY", "SELL") and confidence < self.min_confidence:
            action = "WAIT"
            notes = [f"Confidence {confidence:.2f} < min {self.min_confidence:.2f} → WAIT"]
        else:
            notes = []

        return EnsembleDecision(
            action=action,
            confidence=float(confidence),
            score=float(score),
            votes=votes,
            disagreements=disagreements,
            notes=notes,
        )


# ──────────────────────────────────────────────────────────────────────
# Sample built-in models (can be used as starting points)
# ──────────────────────────────────────────────────────────────────────
def trend_model(symbol: str, context: dict) -> Vote:
    """Simple trend-following model based on EMA stack + ADX."""
    trend = context.get("trend_ltf", "unknown")
    htf_aligned = context.get("htf_alignment", False)
    if trend == "bullish" and htf_aligned:
        return Vote("BUY", 0.7, "bullish trend + HTF aligned")
    if trend == "bearish" and htf_aligned:
        return Vote("SELL", 0.7, "bearish trend + HTF aligned")
    if trend in ("bullish", "bearish"):
        return Vote("WAIT", 0.5, f"trend={trend} but HTF not aligned")
    return Vote("SKIP", 0.6, "no clear trend")


def liquidity_model(symbol: str, context: dict) -> Vote:
    """Model based on liquidity sweeps."""
    direction = context.get("direction", "BUY")
    liquidity_taken = context.get("liquidity_taken", False)
    if liquidity_taken and direction == "BUY":
        return Vote("BUY", 0.65, "liquidity sweep below → bullish reversal")
    if liquidity_taken and direction == "SELL":
        return Vote("SELL", 0.65, "liquidity sweep above → bearish reversal")
    return Vote("WAIT", 0.4, "no liquidity sweep yet")


def risk_model(symbol: str, context: dict) -> Vote:
    """Model that votes based on R:R."""
    rr = context.get("rr_ratio", 0.0)
    if rr >= 3.0:
        return Vote(context.get("direction", "BUY"), 0.8, f"R:R={rr:.1f}")
    if rr >= 2.0:
        return Vote(context.get("direction", "BUY"), 0.55, f"R:R={rr:.1f}")
    return Vote("SKIP", 0.7, f"R:R={rr:.1f} too low")


def regime_model(symbol: str, context: dict) -> Vote:
    """Model that filters by market regime."""
    regime = context.get("regime", "unknown")
    if regime in ("trending_up", "trending_down", "high_vol_breakout"):
        return Vote(context.get("direction", "BUY"), 0.6, f"regime={regime}")
    if regime in ("ranging",):
        return Vote("WAIT", 0.5, "ranging market — wait for breakout")
    if regime in ("low_vol_dead", "choppy"):
        return Vote("SKIP", 0.8, f"regime={regime} — too dangerous")
    return Vote("WAIT", 0.4, f"regime={regime} unclear")


__all__ = [
    "EnsembleVoter", "EnsembleDecision", "Vote", "Model",
    "trend_model", "liquidity_model", "risk_model", "regime_model",
]
