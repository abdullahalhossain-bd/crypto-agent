"""
Platt Calibration Module — LLM Confidence Miscalibration Fix
==============================================================

LLMs are systematically overconfident on ~50% probability predictions.
A "70% confidence" prediction probably doesn't actually hit 70% of the time.

Platt calibration learns a sigmoid mapping from raw scores to calibrated
probabilities using historical prediction-outcome pairs:

    calibrated = sigmoid(a * raw + b)

where (a, b) are fitted via logistic regression on (raw_confidence, was_correct)
pairs.

Problem (Prophet Arena middle-band pathology, arxiv 2510.17638):
  - Models say "60% confident" but actual accuracy is 45%
  - Models say "90% confident" but actual accuracy is 75%
  - Middle-range predictions (40-60%) are the most miscalibrated

Solution:
  - Collect 30+ (prediction, outcome) pairs
  - Fit sigmoid parameters
  - Apply calibration to future predictions

Source: Orallexa (review #27) — platt_calibration.py
Pattern: Bridgewater AIA Forecaster (arxiv 2511.07678)

Usage:
    from platt_calibration import PlattCalibrator

    cal = PlattCalibrator()

    # Record predictions and outcomes
    cal.record(0.75, correct=True)
    cal.record(0.80, correct=False)
    cal.record(0.60, correct=True)
    # ... 30+ records

    # Fit calibration
    cal.fit()

    # Calibrate new predictions
    raw = 0.75
    calibrated = cal.calibrate(raw)
    print(f"Raw: {raw:.0%} → Calibrated: {calibrated:.0%}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Calibration fit result."""
    a: float  # Sigmoid slope
    b: float  # Sigmoid intercept
    n_samples: int
    before_brier: float  # Brier score before calibration
    after_brier: float   # Brier score after calibration
    improvement: float   # Brier reduction
    is_fitted: bool = False

    def to_dict(self) -> dict:
        return {
            "a": round(self.a, 6),
            "b": round(self.b, 6),
            "n_samples": self.n_samples,
            "before_brier": round(self.before_brier, 6),
            "after_brier": round(self.after_brier, 6),
            "improvement": round(self.improvement, 6),
            "is_fitted": self.is_fitted,
        }


class PlattCalibrator:
    """
    Platt sigmoid calibration for LLM confidence scores.

    Pipeline:
    1. Record (raw_confidence, was_correct) pairs
    2. Fit sigmoid: calibrated = 1 / (1 + exp(a * raw + b))
    3. Apply to future predictions

    Cold-start: returns identity (no calibration) until 30+ samples collected.
    """

    MIN_SAMPLES = 30
    REFIT_INTERVAL_DAYS = 7

    def __init__(self, storage_path: str | Path = "memory_data/platt_calibrations.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = self._load()
        self._params: Optional[tuple[float, float]] = None
        self._last_fit_time: Optional[datetime] = None

    def record(self, raw_confidence: float, correct: bool) -> None:
        """
        Record a prediction-outcome pair.

        Args:
            raw_confidence: LLM's raw confidence (0-1)
            correct: Whether the prediction was correct
        """
        self._records.append({
            "raw": float(raw_confidence),
            "correct": bool(correct),
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def fit(self) -> CalibrationResult:
        """
        Fit sigmoid calibration parameters.

        Uses gradient descent on logistic loss:
            minimize -sum(y * log(sigmoid(a*x+b)) + (1-y) * log(1-sigmoid(a*x+b)))

        Returns CalibrationResult with fit quality metrics.
        """
        if len(self._records) < self.MIN_SAMPLES:
            return CalibrationResult(
                a=1.0, b=0.0, n_samples=len(self._records),
                before_brier=0, after_brier=0, improvement=0,
                is_fitted=False,
            )

        # Prepare data
        raws = np.array([r["raw"] for r in self._records])
        labels = np.array([1.0 if r["correct"] else 0.0 for r in self._records])

        # Compute Brier score before calibration
        before_brier = self._brier_score(raws, labels)

        # Gradient descent to fit (a, b)
        a, b = self._gradient_descent(raws, labels, lr=0.01, iterations=2000)

        # Compute Brier score after calibration
        calibrated = self._sigmoid(a * raws + b)
        after_brier = self._brier_score(calibrated, labels)

        self._params = (a, b)
        self._last_fit_time = datetime.now()

        result = CalibrationResult(
            a=a, b=b,
            n_samples=len(self._records),
            before_brier=before_brier,
            after_brier=after_brier,
            improvement=before_brier - after_brier,
            is_fitted=True,
        )

        logger.info(
            f"Platt calibration fitted: a={a:.4f}, b={b:.4f}, "
            f"Brier {before_brier:.4f} → {after_brier:.4f} "
            f"(improvement: {result.improvement:.4f})"
        )

        return result

    def calibrate(self, raw_confidence: float) -> float:
        """
        Calibrate a raw confidence score.

        If not fitted (cold-start), returns the raw confidence unchanged.
        """
        if self._params is None:
            # Check if refit is needed
            if self._should_refit():
                self.fit()
            if self._params is None:
                return raw_confidence  # Identity (cold-start)

        a, b = self._params
        calibrated = self._sigmoid(a * raw_confidence + b)
        return float(calibrated)

    def _should_refit(self) -> bool:
        """Check if refit is needed (enough new data + time elapsed)."""
        if len(self._records) < self.MIN_SAMPLES:
            return False
        if self._last_fit_time is None:
            return True
        elapsed = (datetime.now() - self._last_fit_time).days
        return elapsed >= self.REFIT_INTERVAL_DAYS

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x)),
        )

    def _gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        iterations: int = 2000,
    ) -> tuple[float, float]:
        """
        Gradient descent for Platt scaling.

        Minimizes logistic loss: -sum(y*log(sig(a*x+b)) + (1-y)*log(1-sig(a*x+b)))
        """
        a, b = 1.0, 0.0  # Start with identity

        for _ in range(iterations):
            z = a * X + b
            preds = self._sigmoid(z)

            # Gradients
            da = np.mean((preds - y) * X)
            db = np.mean(preds - y)

            # Update
            a -= lr * da
            b -= lr * db

        return float(a), float(b)

    @staticmethod
    def _brier_score(preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute Brier score (lower = better calibration)."""
        return float(np.mean((preds - labels) ** 2))

    def get_calibration_table(self, n_points: int = 11) -> list[dict]:
        """
        Generate a calibration lookup table.

        Shows what raw confidence maps to what calibrated confidence.
        """
        table = []
        for i in range(n_points):
            raw = i / (n_points - 1)  # 0.0, 0.1, ..., 1.0
            calibrated = self.calibrate(raw)
            table.append({
                "raw": round(raw, 2),
                "calibrated": round(calibrated, 4),
                "adjustment": round(calibrated - raw, 4),
            })
        return table

    def get_status(self) -> dict:
        """Get calibrator status for monitoring."""
        return {
            "n_records": len(self._records),
            "is_fitted": self._params is not None,
            "last_fit": self._last_fit_time.isoformat() if self._last_fit_time else None,
            "needs_refit": self._should_refit(),
            "params": {"a": self._params[0], "b": self._params[1]} if self._params else None,
        }

    def _load(self) -> list[dict]:
        """Critical #10 fix: validate JSON structure before returning."""
        if not self.storage_path.exists():
            return []
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            # Validate structure
            if not isinstance(data, dict) or "records" not in data:
                return []
            records = data["records"]
            if not isinstance(records, list):
                return []
            # Validate each record has required fields
            valid_records = []
            for r in records:
                if isinstance(r, dict) and "raw" in r and "correct" in r:
                    valid_records.append(r)
            return valid_records
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            return []

    def _save(self) -> None:
        try:
            data = {
                "records": self._records,
                "params": list(self._params) if self._params else None,
                "last_fit": self._last_fit_time.isoformat() if self._last_fit_time else None,
            }
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except OSError as e:
            logger.warning(f"Failed to save Platt calibration: {e}")