"""
Drift Detection — Data + Model Performance Monitoring
======================================================

Detects when market conditions change (data drift) or model
performance degrades (model drift), triggering retraining.

Two types:
  1. Data Drift: Feature distribution has shifted (market regime change)
  2. Model Drift: Model accuracy/Sharpe declining (edge decay)

Source: ml4t-3e (review #18) — MLOps drift detection
        Orallexa (review #27) — model health monitoring

Usage:
    from trading_modules.drift_detection import DriftDetector

    detector = DriftDetector()

    # Establish baseline (reference distribution)
    detector.set_baseline(reference_features)

    # Check for drift
    drift = detector.check_drift(current_features)
    if drift.is_drifted:
        print(f"Drift detected: {drift.drift_type}")
        print(f"Affected features: {drift.affected_features}")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
# Critical #3 fix: scipy is optional — wrap import with fallback.
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    _scipy_stats = None


@dataclass
class DriftResult:
    """Result of drift detection check."""
    is_drifted: bool = False
    drift_type: str = "none"  # "data_drift" / "model_drift" / "none"
    drift_score: float = 0.0  # 0-1, higher = more drift
    affected_features: list = field(default_factory=list)
    description: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "is_drifted": self.is_drifted,
            "drift_type": self.drift_type,
            "drift_score": round(self.drift_score, 4),
            "affected_features": self.affected_features,
            "description": self.description,
            "recommendation": self.recommendation,
        }


class DriftDetector:
    """
    Detects data and model drift.

    Data Drift:
      - Uses Kolmogorov-Smirnov test per feature
      - If >30% of features show significant shift → data drift

    Model Drift:
      - Tracks rolling accuracy/Sharpe
      - If performance drops >20% from baseline → model drift
    """

    DRIFT_THRESHOLD = 0.05    # KS p-value threshold
    FEATURE_PCT_THRESHOLD = 0.30  # 30% of features must drift
    PERF_DROP_THRESHOLD = 0.20    # 20% performance drop = model drift

    def __init__(self):
        self._baseline: Optional[pd.DataFrame] = None
        self._baseline_stats: dict = {}
        self._baseline_accuracy: float = 0.0
        self._baseline_sharpe: float = 0.0
        self._performance_history: list = []

    @staticmethod
    def _ks_statistic(data1: np.ndarray, data2: np.ndarray) -> float:
        """Critical #3 fix: pure-numpy KS statistic (fallback when scipy missing)."""
        data1_sorted = np.sort(data1)
        data2_sorted = np.sort(data2)
        all_vals = np.concatenate([data1_sorted, data2_sorted])
        cdf1 = np.searchsorted(data1_sorted, all_vals, side='right') / len(data1_sorted)
        cdf2 = np.searchsorted(data2_sorted, all_vals, side='right') / len(data2_sorted)
        return float(np.max(np.abs(cdf1 - cdf2)))

    def set_baseline(self, reference_data: pd.DataFrame) -> None:
        """Set baseline (reference) data distribution."""
        self._baseline = reference_data.copy()
        self._baseline_stats = {}
        for col in reference_data.columns:
            if reference_data[col].dtype in [np.float64, np.int64, float, int]:
                self._baseline_stats[col] = {
                    "mean": float(reference_data[col].mean()),
                    "std": float(reference_data[col].std()),
                    "values": reference_data[col].dropna().values,
                }

    def set_model_baseline(self, accuracy: float, sharpe: float) -> None:
        """Set baseline model performance."""
        self._baseline_accuracy = accuracy
        self._baseline_sharpe = sharpe

    def check_data_drift(self, current_data: pd.DataFrame) -> DriftResult:
        """Check for data distribution drift."""
        if self._baseline is None or self._baseline_stats is None:
            return DriftResult(description="No baseline set")

        drifted_features = []
        total_features = 0

        for col, baseline in self._baseline_stats.items():
            if col not in current_data.columns:
                continue
            total_features += 1

            current_values = current_data[col].dropna().values
            if len(current_values) < 10:
                continue

            # Kolmogorov-Smirnov test
            # Critical #3 fix: use scipy if available, else fallback to
            # a numpy-based KS statistic approximation (p-value approximated
            # using the asymptotic distribution).
            try:
                if _HAS_SCIPY:
                    ks_stat, p_value = _scipy_stats.ks_2samp(
                        baseline["values"], current_values)
                else:
                    # Fallback: compute KS statistic manually and approximate p-value.
                    ks_stat = self._ks_statistic(baseline["values"], current_values)
                    n1, n2 = len(baseline["values"]), len(current_values)
                    en = np.sqrt(n1 * n2 / (n1 + n2))
                    p_value = 2.0 * np.exp(-2.0 * (en * ks_stat) ** 2)
                    p_value = min(1.0, p_value)
                if p_value < self.DRIFT_THRESHOLD:
                    drifted_features.append(col)
            except Exception:
                continue

        drift_pct = len(drifted_features) / max(total_features, 1)
        is_drifted = drift_pct >= self.FEATURE_PCT_THRESHOLD

        result = DriftResult(
            is_drifted=is_drifted,
            drift_type="data_drift" if is_drifted else "none",
            drift_score=float(drift_pct),
            affected_features=drifted_features,
        )

        if is_drifted:
            result.description = f"Data drift: {drift_pct:.0%} of features shifted ({len(drifted_features)}/{total_features})"
            result.recommendation = "Market regime change detected — retrain models with recent data"
        else:
            result.description = f"No data drift ({drift_pct:.0%} of features shifted)"

        return result

    def check_model_drift(
        self,
        current_accuracy: float,
        current_sharpe: float,
    ) -> DriftResult:
        """Check for model performance drift."""
        if self._baseline_accuracy == 0 and self._baseline_sharpe == 0:
            return DriftResult(description="No model baseline set")

        self._performance_history.append({
            "accuracy": current_accuracy,
            "sharpe": current_sharpe,
            "timestamp": pd.Timestamp.now().isoformat(),
        })

        acc_drop = (self._baseline_accuracy - current_accuracy) / max(self._baseline_accuracy, 0.01)
        sharpe_drop = (self._baseline_sharpe - current_sharpe) / max(abs(self._baseline_sharpe), 0.01)

        max_drop = max(acc_drop, sharpe_drop)
        is_drifted = max_drop >= self.PERF_DROP_THRESHOLD

        result = DriftResult(
            is_drifted=is_drifted,
            drift_type="model_drift" if is_drifted else "none",
            drift_score=float(max_drop),
        )

        if is_drifted:
            result.description = (
                f"Model drift: accuracy dropped {acc_drop:.0%} "
                f"(base={self._baseline_accuracy:.1%} → now={current_accuracy:.1%}), "
                f"Sharpe dropped {sharpe_drop:.0%}"
            )
            result.recommendation = "Edge decaying — retrain model or switch to challenger"
            result.affected_features = ["accuracy", "sharpe"]

        return result

    def check_drift(
        self,
        current_data: Optional[pd.DataFrame] = None,
        current_accuracy: float = 0.0,
        current_sharpe: float = 0.0,
    ) -> DriftResult:
        """Check both data and model drift."""
        results = []

        if current_data is not None:
            results.append(self.check_data_drift(current_data))

        if current_accuracy > 0 or current_sharpe != 0:
            results.append(self.check_model_drift(current_accuracy, current_sharpe))

        if not results:
            return DriftResult(description="No checks performed — missing data")

        # Return worst case
        worst = max(results, key=lambda r: r.drift_score)
        return worst

    def get_status(self) -> dict:
        """Get drift detector status."""
        return {
            "has_baseline": self._baseline is not None,
            "baseline_accuracy": self._baseline_accuracy,
            "baseline_sharpe": self._baseline_sharpe,
            "performance_history_len": len(self._performance_history),
            "last_check": self._performance_history[-1] if self._performance_history else None,
        }
