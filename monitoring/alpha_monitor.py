"""monitoring.alpha_monitor
=====================================================================
Day 74 — Alpha-layer monitor.

Tracks the health of strategies + ML models:
  - Signal decay (per-strategy hit rate over time)
  - Feature drift (KL divergence between training + live distributions)
  - Regime mismatch (current regime vs. strategy affinity)
  - ML model calibration (Brier score)
  - Prediction latency

Catches the silent killers: strategies that *appear* to work but have
lost their edge, ML models that have drifted from training distribution.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("monitoring.alpha")


@dataclass
class AlphaHealth:
    status: str
    signal_decay_detected: bool
    feature_drift_detected: bool
    regime_mismatch_detected: bool
    ml_calibration_score: float          # Brier score (lower = better)
    avg_prediction_latency_ms: float
    per_strategy_signal_rate: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "signal_decay_detected": self.signal_decay_detected,
            "feature_drift_detected": self.feature_drift_detected,
            "regime_mismatch_detected": self.regime_mismatch_detected,
            "ml_calibration_score": self.ml_calibration_score,
            "avg_prediction_latency_ms": self.avg_prediction_latency_ms,
            "per_strategy_signal_rate": dict(self.per_strategy_signal_rate),
            "drifted_features": list(self.drifted_features),
            "issues": list(self.issues),
        }


# ----------------------------------------------------------------------
class AlphaMonitor:
    def __init__(self,
                 drift_threshold: float = 0.3,
                 signal_decay_window: int = 200,
                 baseline_window: int = 500,
                 ml_calibration_window: int = 200) -> None:
        self.drift_threshold = float(drift_threshold)
        self.signal_decay_window = int(signal_decay_window)
        self.baseline_window = int(baseline_window)
        self.ml_calibration_window = int(ml_calibration_window)
        # Per-strategy signal fires (1.0 / 0.0)
        self._signal_fires: dict[str, deque] = {}
        # Feature baselines + recent distributions
        self._feature_baselines: dict[str, np.ndarray] = {}
        self._feature_recent: dict[str, deque] = {}
        # ML predictions + actuals (for Brier score)
        self._ml_predictions: deque = deque(maxlen=ml_calibration_window)
        self._ml_actuals: deque = deque(maxlen=ml_calibration_window)
        # Prediction latencies
        self._latencies: deque = deque(maxlen=200)
        # Current regime
        self._current_regime: str = ""
        # Per-strategy regime affinity
        self._regime_affinities: dict[str, dict[str, float]] = {}

    # ----------------------------------------------------------------
    def record_signal(self, strategy_name: str, fired: bool) -> None:
        d = self._signal_fires.setdefault(strategy_name,
                                          deque(maxlen=self.baseline_window))
        d.append(1.0 if fired else 0.0)

    def set_feature_baseline(self, feature_name: str,
                             values: np.ndarray) -> None:
        self._feature_baselines[feature_name] = np.asarray(values, dtype=float)

    def record_feature_value(self, feature_name: str, value: float) -> None:
        d = self._feature_recent.setdefault(feature_name,
                                            deque(maxlen=100))
        d.append(float(value))

    def record_ml_prediction(self, proba: float, actual: int) -> None:
        self._ml_predictions.append(float(proba))
        self._ml_actuals.append(int(actual))

    def record_prediction_latency(self, latency_ms: float) -> None:
        self._latencies.append(float(latency_ms))

    def set_regime(self, regime: str,
                   affinities: Optional[dict[str, dict[str, float]]] = None) -> None:
        self._current_regime = regime
        if affinities:
            self._regime_affinities = affinities

    # ----------------------------------------------------------------
    def health(self) -> AlphaHealth:
        issues: list[str] = []
        signal_decay = False
        feature_drift = False
        regime_mismatch = False
        drifted_features: list[str] = []

        # Signal rate per strategy
        per_strategy: dict[str, float] = {}
        for name, fires in self._signal_fires.items():
            if len(fires) < self.signal_decay_window:
                continue
            arr = np.array(fires)
            baseline_rate = float(arr[:-self.signal_decay_window].mean()) if len(arr) > self.signal_decay_window else float(arr.mean())
            recent_rate = float(arr[-self.signal_decay_window:].mean())
            per_strategy[name] = recent_rate
            if baseline_rate > 0 and recent_rate / baseline_rate < 0.5:
                signal_decay = True
                issues.append(f"{name}: signal rate decayed {recent_rate:.2f}/{baseline_rate:.2f}")

        # Feature drift (mean + std comparison)
        for feat, baseline in self._feature_baselines.items():
            recent = self._feature_recent.get(feat)
            if recent is None or len(recent) < 50:
                continue
            recent_arr = np.array(recent)
            # KL divergence proxy: |mean_diff| / std
            mean_diff = abs(float(baseline.mean()) - float(recent_arr.mean()))
            baseline_std = float(baseline.std()) or 1.0
            drift_score = mean_diff / baseline_std
            if drift_score > self.drift_threshold:
                feature_drift = True
                drifted_features.append(feat)
                issues.append(f"feature drift: {feat} score={drift_score:.2f}")

        # Regime mismatch
        if self._current_regime and self._regime_affinities:
            for name, affinities in self._regime_affinities.items():
                affinity = affinities.get(self._current_regime, 0.5)
                if affinity < 0.2:
                    regime_mismatch = True
                    issues.append(f"{name}: low affinity {affinity:.2f} for regime {self._current_regime}")

        # ML calibration (Brier score)
        brier = 0.0
        if len(self._ml_predictions) >= 50:
            preds = np.array(self._ml_predictions)
            acts = np.array(self._ml_actuals)
            brier = float(np.mean((preds - acts) ** 2))
            if brier > 0.30:
                issues.append(f"ML calibration poor: Brier={brier:.3f}")

        # Latency
        avg_lat = float(np.mean(self._latencies)) if self._latencies else 0.0

        status = "ok"
        if issues:
            status = "degraded" if len(issues) <= 2 else "critical"

        return AlphaHealth(
            status=status,
            signal_decay_detected=signal_decay,
            feature_drift_detected=feature_drift,
            regime_mismatch_detected=regime_mismatch,
            ml_calibration_score=brier,
            avg_prediction_latency_ms=avg_lat,
            per_strategy_signal_rate=per_strategy,
            drifted_features=drifted_features,
            issues=issues,
        )
