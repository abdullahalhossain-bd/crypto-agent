"""ml.signal_classifier
=====================================================================
Day 21 — Signal classifier.

ML is a GATEKEEPER, NOT a trader. The classifier takes a strategy's
signal + a feature vector and returns a confidence score in [0, 1]
that the signal will be profitable. The portfolio manager uses this
score to scale position size or veto the signal.

Backends:
  - "logistic"  : sklearn LogisticRegression (baseline, always works)
  - "lightgbm"  : LightGBM classifier (preferred if installed)
  - "xgboost"   : XGBoost classifier (alternative)

All backends are wrapped behind a single `.predict_proba` interface
so the rest of the system doesn't care which model is loaded.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("ml.classifier")


@dataclass
class ClassificationResult:
    confidence: float           # probability the signal is profitable
    predicted_class: int        # 1 / -1 / 0
    raw_proba: dict[str, float] = field(default_factory=dict)
    features_used: list[str] = field(default_factory=list)
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "predicted_class": self.predicted_class,
            "raw_proba": dict(self.raw_proba),
            "features_used": list(self.features_used),
            "model_version": self.model_version,
        }


# ----------------------------------------------------------------------
class SignalClassifier:
    """Pluggable binary classifier over feature vectors."""

    def __init__(self, backend: str = "lightgbm",
                 model_path: Optional[str] = None,
                 min_confidence: float = 0.55,
                 max_confidence: float = 0.95) -> None:
        self.backend = backend.lower()
        self.model_path = model_path
        self.min_confidence = float(min_confidence)
        self.max_confidence = float(max_confidence)
        self._model: Any = None
        self._feature_names: list[str] = []
        self._version: str = "untrained"
        if model_path and os.path.isfile(model_path):
            self.load(model_path)

    # ----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Train the model on (X, y). Returns metrics."""
        # Drop rows with NaN
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X_clean = X[mask].copy()
        y_clean = y[mask].copy()
        self._feature_names = list(X_clean.columns)
        if len(X_clean) < 50:
            raise ValueError(f"too few rows to train: {len(X_clean)}")

        # Map labels to {0, 1} for binary classifiers
        y_bin = (y_clean > 0).astype(int)

        if self.backend == "lightgbm":
            from lightgbm import LGBMClassifier
            self._model = LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=-1,
            )
        elif self.backend == "xgboost":
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="logloss",
                use_label_encoder=False,
            )
        elif self.backend == "logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            self._model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
            ])
        else:
            raise ValueError(f"unknown backend: {self.backend}")

        self._model.fit(X_clean.values, y_bin.values)
        self._version = f"{self.backend}-fit-{len(X_clean)}"

        # Compute training metrics
        from sklearn.metrics import accuracy_score, roc_auc_score
        preds = self._model.predict(X_clean.values)
        proba = self._predict_proba(X_clean.values)
        metrics = {
            "n_samples": int(len(X_clean)),
            "accuracy": float(accuracy_score(y_bin, preds)),
            "auc": float(roc_auc_score(y_bin, proba)) if len(set(y_bin)) == 2 else 0.5,
            "features": list(self._feature_names),
            "version": self._version,
        }
        log.info("ML trained backend=%s n=%d acc=%.3f auc=%.3f",
                 self.backend, metrics["n_samples"],
                 metrics["accuracy"], metrics["auc"])
        return metrics

    # ----------------------------------------------------------------
    def predict(self, feature_vector: dict[str, float],
                signal_action: str = "BUY") -> ClassificationResult:
        """Score a single feature vector.

        Returns confidence = P(signal is profitable). If the model
        disagrees with the signal's direction (e.g. model says DOWN
        but signal is BUY), confidence is capped at min_confidence.
        """
        if self._model is None or not self._feature_names:
            # Untrained — permissive (returns 0.5 = neutral)
            return ClassificationResult(
                confidence=0.5, predicted_class=0,
                features_used=[], model_version="untrained",
            )
        row = np.array([[float(feature_vector.get(f, 0.0) or 0.0)
                         for f in self._feature_names]])
        proba = float(self._predict_proba(row)[0])
        predicted_class = 1 if proba >= 0.5 else -1

        # Penalise disagreement
        signal_dir = 1 if signal_action.upper() == "BUY" else -1
        if predicted_class != signal_dir:
            proba = min(proba, self.min_confidence)

        proba = max(self.min_confidence, min(self.max_confidence, proba))
        return ClassificationResult(
            confidence=proba,
            predicted_class=predicted_class,
            raw_proba={"up": proba, "down": 1.0 - proba},
            features_used=list(self._feature_names),
            model_version=self._version,
        )

    # ----------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "version": self._version,
                "backend": self.backend,
            }, f)
        log.info("ML model saved to %s", path)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            blob = pickle.load(f)
        self._model = blob["model"]
        self._feature_names = blob["feature_names"]
        self._version = blob["version"]
        self.backend = blob.get("backend", self.backend)
        log.info("ML model loaded from %s version=%s", path, self._version)

    # ----------------------------------------------------------------
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each row."""
        if self._model is None:
            return np.full(len(X), 0.5)
        if self.backend == "logistic":
            return self._model.predict_proba(X)[:, 1]
        # LightGBM / XGBoost expose predict_proba
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(X)[:, 1]
        # Fallback
        return self._model.predict(X)
