"""ml.trainer
=====================================================================
Day 22 — Walk-forward trainer.

Trains the SignalClassifier using a strict walk-forward protocol:
  1. Split data into N folds (train [0..k], test [k..k+h])
  2. Train on fold k, evaluate on fold k+1
  3. Slide forward; never test on data the model has seen
  4. Aggregate metrics across folds

This mirrors how the model will be used in production (train on past,
predict future) and surfaces overfitting early.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from ml.feature_store import FeatureStore, label_forward_return
from ml.signal_classifier import SignalClassifier
from utils.logger import get_logger

log = get_logger("ml.trainer")


# ----------------------------------------------------------------------
@dataclass
class TrainingResult:
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    avg_accuracy: float = 0.0
    avg_auc: float = 0.0
    n_total_train: int = 0
    n_total_test: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    leakage_check: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_metrics": list(self.fold_metrics),
            "avg_accuracy": self.avg_accuracy,
            "avg_auc": self.avg_auc,
            "n_total_train": self.n_total_train,
            "n_total_test": self.n_total_test,
            "feature_importance": dict(self.feature_importance),
            "leakage_check": dict(self.leakage_check),
        }


# ----------------------------------------------------------------------
class WalkForwardTrainer:
    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        n_folds: int = 5,
        train_ratio: float = 0.7,
        forward_horizon: int = 5,
        forward_threshold: float = 0.001,
    ) -> None:
        self.fs = feature_store or FeatureStore()
        self.n_folds = int(n_folds)
        self.train_ratio = float(train_ratio)
        self.forward_horizon = int(forward_horizon)
        self.forward_threshold = float(forward_threshold)

    # ----------------------------------------------------------------
    def train(self, df: pd.DataFrame,
              classifier: SignalClassifier) -> TrainingResult:
        """Run walk-forward training over `df`.

        The LAST fold's model is left trained inside `classifier` and
        can be saved via `classifier.save()`.
        """
        features = self.fs.build(df, include_time=True)
        labels = label_forward_return(df, self.forward_horizon,
                                      self.forward_threshold)
        # Align + drop NaN
        full = features.copy()
        full["label"] = labels
        full = full.dropna()
        if len(full) < 200:
            raise ValueError(f"too few rows after dropna: {len(full)}")

        # Leakage check: confirm no forward-looking columns
        leakage = self._leakage_check(features)

        fold_size = len(full) // (self.n_folds + 1)
        if fold_size < 50:
            raise ValueError(f"fold_size={fold_size} too small; reduce n_folds")

        fold_metrics: list[dict[str, Any]] = []
        accs, aucs = [], []
        feature_importance: dict[str, float] = {}

        for k in range(self.n_folds):
            train_start = k * fold_size
            train_end = train_start + int(fold_size * self.train_ratio)
            test_start = train_end
            test_end = min(len(full), (k + 1) * fold_size + int(fold_size * (1 - self.train_ratio)))
            if test_end <= test_start:
                continue
            train_df = full.iloc[train_start:train_end]
            test_df = full.iloc[test_start:test_end]
            X_train = train_df.drop(columns=["label"])
            y_train = train_df["label"]
            X_test = test_df.drop(columns=["label"])
            y_test = test_df["label"]

            try:
                classifier.fit(X_train, y_train)
            except Exception as e:  # noqa: BLE001
                log.warning("fold %d fit failed: %r", k, e)
                continue

            # Evaluate
            from sklearn.metrics import accuracy_score, roc_auc_score
            proba = np.array([
                classifier.predict(row.to_dict()).confidence
                for _, row in X_test.iterrows()
            ])
            preds = (proba >= 0.5).astype(int)
            y_bin = (y_test > 0).astype(int)
            acc = float(accuracy_score(y_bin, preds))
            auc = float(roc_auc_score(y_bin, proba)) if len(set(y_bin)) == 2 else 0.5
            accs.append(acc)
            aucs.append(auc)
            fold_metrics.append({
                "fold": k,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "accuracy": acc,
                "auc": auc,
            })

            # Accumulate feature importance
            fi = self._extract_feature_importance(classifier)
            for fname, imp in fi.items():
                feature_importance[fname] = feature_importance.get(fname, 0.0) + imp

        # Average importance across folds
        if feature_importance:
            factor = 1.0 / max(1, len(fold_metrics))
            feature_importance = {k: v * factor for k, v in feature_importance.items()}
            feature_importance = dict(sorted(feature_importance.items(),
                                             key=lambda kv: -kv[1]))

        log.info("walk-forward done folds=%d avg_acc=%.3f avg_auc=%.3f",
                 len(fold_metrics),
                 np.mean(accs) if accs else 0.0,
                 np.mean(aucs) if aucs else 0.0)

        return TrainingResult(
            fold_metrics=fold_metrics,
            avg_accuracy=float(np.mean(accs)) if accs else 0.0,
            avg_auc=float(np.mean(aucs)) if aucs else 0.0,
            n_total_train=sum(m["train_size"] for m in fold_metrics),
            n_total_test=sum(m["test_size"] for m in fold_metrics),
            feature_importance=feature_importance,
            leakage_check=leakage,
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _extract_feature_importance(classifier: SignalClassifier) -> dict[str, float]:
        model = classifier._model
        names = classifier._feature_names
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                return {n: float(i) for n, i in zip(names, imp)}
            # Pipeline (logistic) — pull from named_steps
            if hasattr(model, "named_steps") and "clf" in model.named_steps:
                clf = model.named_steps["clf"]
                if hasattr(clf, "coef_"):
                    imp = np.abs(clf.coef_[0])
                    return {n: float(i) for n, i in zip(names, imp)}
        except Exception:  # noqa: BLE001
            pass
        return {}

    # ----------------------------------------------------------------
    def _leakage_check(self, features: pd.DataFrame) -> dict[str, Any]:
        """Cheap heuristic: any feature column with a future-shifted twin?"""
        suspicious: list[str] = []
        for c in features.columns:
            # If a column has NaN at the END (instead of the beginning),
            # it likely used forward-shifted data.
            tail = features[c].tail(50)
            head = features[c].head(50)
            if tail.isna().sum() > 5 and head.isna().sum() < 5:
                suspicious.append(c)
        return {
            "suspicious_columns": suspicious,
            "ok": not suspicious,
        }
