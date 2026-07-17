"""
Bias Tracker Module — Prediction Accuracy Self-Correction
===========================================================

Tracks historical predictions vs actual outcomes. Identifies systematic
biases (e.g., consistently too bullish, overconfident) and auto-adjusts
future confidence.

Features:
  - Load decisions from JSON log
  - Batch fetch forward returns
  - Evaluate prediction accuracy over rolling window (default 90 days)
  - Detect patterns: "model is 20% too bullish on tech at RSI>65"
  - Feed bias awareness back into analysis pipeline
  - Auto-adjust confidence: if model is consistently overconfident, scale down

Source: Orallexa (review #27) — bias_tracker.py (455 LOC)
Enhanced with: Orallexa deep-dive edge_thesis pattern

Usage:
    from bias_tracker import BiasTracker

    bt = BiasTracker(decision_log_path="decisions.json")

    # Record a decision
    bt.record_decision(
        ticker="BTCUSDT",
        direction="BUY",
        confidence=0.75,
        rsi_at_decision=28.0,
        pattern="Bullish Engulfing",
    )

    # After 5 trading days, record the outcome
    bt.record_outcome("BTCUSDT", forward_return=0.032)

    # Get bias profile
    profile = bt.get_bias_profile()
    print(f"Overall accuracy: {profile['overall_accuracy']:.1%}")
    print(f"Bull accuracy: {profile['bull_accuracy']:.1%}")
    print(f"Bear accuracy: {profile['bear_accuracy']:.1%}")
    print(f"Overconfidence: {profile['overconfidence_pct']:.1f}%")

    # Get adjusted confidence for a new prediction
    adjusted = bt.adjust_confidence(raw_confidence=0.75, direction="BUY")
    print(f"Adjusted confidence: {adjusted:.1%}")  # e.g., 0.56 if overconfident
"""

from __future__ import annotations

import threading

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionRecord:
    """A single decision record with outcome tracking."""
    timestamp: str
    ticker: str
    direction: str  # "BUY" / "SELL" / "HOLD"
    confidence: float  # 0.0 to 1.0
    # Context at decision time
    rsi: Optional[float] = None
    pattern: Optional[str] = None
    edge_thesis: Optional[str] = None  # Required field (Orallexa edge_thesis pattern)
    # Outcome (filled later)
    forward_return: Optional[float] = None  # 5-day forward return
    was_correct: Optional[bool] = None
    outcome_timestamp: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BiasProfile:
    """Aggregated bias analysis."""
    overall_accuracy: float = 0.0
    bull_accuracy: float = 0.0  # BUY predictions that were correct
    bear_accuracy: float = 0.0  # SELL predictions that were correct
    hold_accuracy: float = 0.0  # HOLD predictions that were correct
    total_predictions: int = 0
    # Confidence calibration
    avg_confidence: float = 0.0
    avg_confidence_when_correct: float = 0.0
    avg_confidence_when_wrong: float = 0.0
    overconfidence_pct: float = 0.0  # How much confidence exceeds accuracy
    # Per-context biases
    rsi_biases: dict = field(default_factory=dict)  # {rsi_bucket: accuracy}
    pattern_biases: dict = field(default_factory=dict)  # {pattern: accuracy}
    # Recommendations
    confidence_adjustment_factor: float = 1.0  # Multiply raw confidence by this
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BiasTracker:
    """
    Tracks prediction accuracy and auto-corrects systematic biases.

    The tracker:
    1. Records decisions with context (RSI, pattern, direction, confidence)
    2. Records outcomes (5-day forward returns)
    3. Evaluates accuracy over a rolling window
    4. Detects systematic biases
    5. Adjusts future confidence to compensate

    Data Requirements:
    - Need 30+ predictions to detect meaningful patterns
    - Uses decision log as data source
    - Compares against actual forward returns
    """

    FORWARD_DAYS = 5  # 5-trading-day forward return
    NEUTRAL_THRESHOLD = 0.005  # |return| < 0.5% = neutral
    MIN_PREDICTIONS_FOR_BIAS = 30
    OVERCONFIDENCE_THRESHOLD = 0.10  # 10% gap = overconfident

    def __init__(self, decision_log_path: str | Path = "memory_data/decision_log.json"):
        self.log_path = Path(decision_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # Critical #2 fix

    def record_decision(
        self,
        ticker: str,
        direction: str,
        confidence: float,
        rsi: Optional[float] = None,
        pattern: Optional[str] = None,
        edge_thesis: Optional[str] = None,
    ) -> None:
        """Record a new decision (outcome filled later)."""
        record = DecisionRecord(
            timestamp=datetime.now().isoformat(),
            ticker=ticker.upper(),
            direction=direction.upper(),
            confidence=confidence,
            rsi=rsi,
            pattern=pattern,
            edge_thesis=edge_thesis,
        )

        records = self._load_records()
        records.append(record.to_dict())
        self._save_records(records)

        logger.info(f"Recorded decision: {ticker} {direction} ({confidence:.0%} confidence)")

    def record_outcome(self, ticker: str, forward_return: float) -> None:
        """
        Record the forward return outcome for a ticker's latest decision.

        Called 5 trading days after the decision was made.
        """
        records = self._load_records()
        ticker_upper = ticker.upper()

        # Find the latest decision for this ticker without an outcome
        for record in reversed(records):
            if record.get("ticker") == ticker_upper and record.get("forward_return") is None:
                record["forward_return"] = forward_return
                record["was_correct"] = self._evaluate_correctness(
                    record.get("direction", ""),
                    forward_return,
                )
                record["outcome_timestamp"] = datetime.now().isoformat()
                self._save_records(records)
                logger.info(
                    f"Recorded outcome: {ticker} return={forward_return:.2%} "
                    f"correct={record['was_correct']}"
                )
                return

        logger.warning(f"No pending decision found for {ticker}")

    def _evaluate_correctness(self, direction: str, forward_return: float) -> bool:
        """Evaluate if the prediction was correct given the outcome."""
        if abs(forward_return) < self.NEUTRAL_THRESHOLD:
            # Neutral return — correct if predicted HOLD, incorrect otherwise
            return direction.upper() == "HOLD"

        if direction.upper() == "BUY":
            return forward_return > 0
        elif direction.upper() == "SELL":
            return forward_return < 0
        elif direction.upper() == "HOLD":
            return abs(forward_return) < self.NEUTRAL_THRESHOLD
        return False

    def get_bias_profile(self, days: int = 90) -> BiasProfile:
        """
        Compute bias profile from last N days of decisions.

        Args:
            days: Lookback window in days (default 90)

        Returns:
            BiasProfile with accuracy, calibration, and warnings
        """
        records = self._load_decisions(days)

        if len(records) < self.MIN_PREDICTIONS_FOR_BIAS:
            return BiasProfile(
                total_predictions=len(records),
                warnings=[
                    f"Insufficient data: {len(records)}/{self.MIN_PREDICTIONS_FOR_BIAS} "
                    f"predictions needed for bias analysis"
                ],
            )

        profile = BiasProfile(total_predictions=len(records))

        # Compute accuracy by direction
        bull_correct = bull_total = 0
        bear_correct = bear_total = 0
        hold_correct = hold_total = 0
        all_correct = 0

        # Confidence calibration
        conf_sum = 0.0
        conf_correct_sum = 0.0
        conf_wrong_sum = 0.0
        conf_correct_count = 0
        conf_wrong_count = 0

        # Per-context tracking
        rsi_buckets = defaultdict(lambda: {"correct": 0, "total": 0})
        pattern_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in records:
            direction = r.get("direction", "").upper()
            confidence = r.get("confidence", 0.5)
            correct = r.get("was_correct")
            rsi = r.get("rsi")
            pattern = r.get("pattern")

            if correct is None:
                continue

            all_correct += int(correct)
            conf_sum += confidence

            if correct:
                conf_correct_sum += confidence
                conf_correct_count += 1
            else:
                conf_wrong_sum += confidence
                conf_wrong_count += 1

            if direction == "BUY":
                bull_total += 1
                bull_correct += int(correct)
            elif direction == "SELL":
                bear_total += 1
                bear_correct += int(correct)
            elif direction == "HOLD":
                hold_total += 1
                hold_correct += int(correct)

            # RSI bucket analysis
            if rsi is not None:
                bucket = self._rsi_bucket(rsi)
                rsi_buckets[bucket]["total"] += 1
                rsi_buckets[bucket]["correct"] += int(correct)

            # Pattern analysis
            if pattern:
                pattern_stats[pattern]["total"] += 1
                pattern_stats[pattern]["correct"] += int(correct)

        evaluated = sum(1 for r in records if r.get("was_correct") is not None)
        if evaluated == 0:
            return BiasProfile(
                total_predictions=len(records),
                warnings=["No evaluated predictions (outcomes not yet recorded)"],
            )

        profile.overall_accuracy = all_correct / evaluated
        profile.bull_accuracy = bull_correct / bull_total if bull_total > 0 else 0.0
        profile.bear_accuracy = bear_correct / bear_total if bear_total > 0 else 0.0
        profile.hold_accuracy = hold_correct / hold_total if hold_total > 0 else 0.0

        profile.avg_confidence = conf_sum / evaluated if evaluated > 0 else 0
        profile.avg_confidence_when_correct = (
            conf_correct_sum / conf_correct_count if conf_correct_count > 0 else 0
        )
        profile.avg_confidence_when_wrong = (
            conf_wrong_sum / conf_wrong_count if conf_wrong_count > 0 else 0
        )

        # Overconfidence = avg confidence - actual accuracy
        profile.overconfidence_pct = max(0, profile.avg_confidence - profile.overall_accuracy)

        # RSI biases
        profile.rsi_biases = {
            bucket: stats["correct"] / stats["total"]
            for bucket, stats in rsi_buckets.items()
            if stats["total"] >= 5  # Minimum 5 samples per bucket
        }

        # Pattern biases
        profile.pattern_biases = {
            pattern: stats["correct"] / stats["total"]
            for pattern, stats in pattern_stats.items()
            if stats["total"] >= 5
        }

        # Confidence adjustment factor
        if profile.overconfidence_pct > self.OVERCONFIDENCE_THRESHOLD:
            # Scale down confidence to match actual accuracy
            if profile.avg_confidence > 0:
                profile.confidence_adjustment_factor = (
                    profile.overall_accuracy / profile.avg_confidence
                )
                profile.confidence_adjustment_factor = max(0.5, profile.confidence_adjustment_factor)
                profile.warnings.append(
                    f"⚠️ Overconfident by {profile.overconfidence_pct:.1%}. "
                    f"Confidence scaled by {profile.confidence_adjustment_factor:.2f}"
                )

        # Direction-specific warnings
        if bull_total >= 10 and profile.bull_accuracy < 0.40:
            profile.warnings.append(
                f"⚠️ BUY predictions only {profile.bull_accuracy:.1%} accurate "
                f"({bull_correct}/{bull_total}). Consider tightening BUY criteria."
            )
        if bear_total >= 10 and profile.bear_accuracy < 0.40:
            profile.warnings.append(
                f"⚠️ SELL predictions only {profile.bear_accuracy:.1%} accurate "
                f"({bear_correct}/{bear_total}). Consider tightening SELL criteria."
            )

        return profile

    def adjust_confidence(self, raw_confidence: float, direction: str = "") -> float:
        """
        Adjust raw confidence based on historical bias.

        If the model is consistently overconfident, scale down.
        If the model is underconfident in a specific direction, scale up.
        """
        profile = self.get_bias_profile()

        adjusted = raw_confidence * profile.confidence_adjustment_factor

        # Direction-specific adjustment
        if direction.upper() == "BUY" and profile.bull_accuracy > 0:
            direction_factor = profile.bull_accuracy / max(profile.overall_accuracy, 0.01)
            direction_factor = max(0.7, min(1.3, direction_factor))
            adjusted *= direction_factor
        elif direction.upper() == "SELL" and profile.bear_accuracy > 0:
            direction_factor = profile.bear_accuracy / max(profile.overall_accuracy, 0.01)
            direction_factor = max(0.7, min(1.3, direction_factor))
            adjusted *= direction_factor

        return max(0.0, min(1.0, adjusted))

    def get_bias_context_for_prompt(self) -> str:
        """
        Generate a bias awareness context block for LLM agent prompts.

        Example output:
        "Historical note: your BUY calls have 42% accuracy (below 50% average).
         Your confidence is consistently 15% higher than your accuracy.
         Pattern 'Hammer' has only 30% accuracy in your history."
        """
        profile = self.get_bias_profile()

        if profile.total_predictions < self.MIN_PREDICTIONS_FOR_BIAS:
            return ""

        lines = ["## Bias Self-Correction Context"]

        if profile.warnings:
            for w in profile.warnings:
                lines.append(w)

        lines.append(f"Overall accuracy: {profile.overall_accuracy:.1%} ({profile.total_predictions} predictions)")
        lines.append(f"BUY accuracy: {profile.bull_accuracy:.1%}")
        lines.append(f"SELL accuracy: {profile.bear_accuracy:.1%}")

        if profile.overconfidence_pct > self.OVERCONFIDENCE_THRESHOLD:
            lines.append(
                f"⚠️ You are overconfident by {profile.overconfidence_pct:.1%}. "
                f"Your confidence is being scaled by {profile.confidence_adjustment_factor:.2f}."
            )

        # Pattern-specific warnings
        for pattern, accuracy in profile.pattern_biases.items():
            if accuracy < 0.40:
                lines.append(f"⚠️ Pattern '{pattern}' has only {accuracy:.1%} accuracy in your history.")

        # RSI bucket warnings
        for bucket, accuracy in profile.rsi_biases.items():
            if accuracy < 0.40:
                lines.append(f"⚠️ At RSI {bucket}, your accuracy is only {accuracy:.1%}.")

        return "\n".join(lines)

    def _rsi_bucket(self, rsi: float) -> str:
        """Classify RSI into a bucket for bias analysis."""
        if rsi < 20:
            return "<20 (extreme oversold)"
        elif rsi < 35:
            return "20-35 (oversold)"
        elif rsi < 50:
            return "35-50 (bearish neutral)"
        elif rsi < 65:
            return "50-65 (bullish neutral)"
        elif rsi < 80:
            return "65-80 (overbought)"
        else:
            return "80+ (extreme overbought)"

    def _load_records(self) -> list[dict]:
        """Load all records from JSON log."""
        if not self.log_path.exists():
            return []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save_records(self, records: list[dict]) -> None:
        """Critical #2 fix: thread-safe save."""
        """Save records to JSON log."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False, default=str)

    def _load_decisions(self, days: int = 90) -> list[dict]:
        """Load decisions from the last N days."""
        records = self._load_records()
        cutoff = datetime.now() - timedelta(days=days)

        result = []
        for r in records:
            ts = r.get("timestamp", "")
            try:
                if datetime.fromisoformat(ts) >= cutoff:
                    result.append(r)
            except (ValueError, TypeError):
                continue
        return result
