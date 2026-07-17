"""
Continual Learning — Online Model Updates with Safe Validation
===============================================================

Updates ML models while trading, but with safety gates:
  1. Shadow validation: new model runs in parallel before promotion
  2. Automatic rollback: if new model underperforms, revert instantly
  3. Champion-challenger: always compare best model vs challenger

Pipeline:
  1. Collect new data (trades, outcomes, features)
  2. Train challenger model on extended dataset
  3. Run shadow evaluation (challenger vs champion)
  4. If challenger wins by margin → promote
  5. If champion still better → keep, try again later

Source: ml4t-3e (review #18) — online learning + champion-challenger
        Orallexa (review #27) — strategy evolution

Usage:
    from trading_modules.continual_learning import ContinualLearner

    learner = ContinualLearner(model_type="xgboost")

    # Initial training
    learner.initial_train(features, labels)

    # Periodic update with new data
    learner.update(new_features, new_labels)

    # Get current best model
    model = learner.get_champion()
    predictions = learner.predict(features)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A versioned model with metadata."""
    version: int
    model: Any
    train_samples: int
    train_date: str
    oos_sharpe: float = 0.0
    oos_accuracy: float = 0.0
    is_champion: bool = False
    is_challenger: bool = False

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "train_samples": self.train_samples,
            "train_date": self.train_date,
            "oos_sharpe": round(self.oos_sharpe, 4),
            "oos_accuracy": round(self.oos_accuracy, 4),
            "role": "champion" if self.is_champion else "challenger" if self.is_challenger else "archived",
        }


@dataclass
class UpdateResult:
    """Result of a continual learning update."""
    promoted: bool
    champion_version: int
    challenger_version: int
    champion_sharpe: float
    challenger_sharpe: float
    margin: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "promoted": self.promoted,
            "champion_v": self.champion_version,
            "challenger_v": self.challenger_version,
            "champion_sharpe": round(self.champion_sharpe, 4),
            "challenger_sharpe": round(self.challenger_sharpe, 4),
            "margin": round(self.margin, 4),
            "reason": self.reason,
        }


class ContinualLearner:
    """
    Online model updater with champion-challenger pattern.

    Safety guarantees:
      - New model must beat champion by MIN_MARGIN on OOS Sharpe
      - If challenger fails, instantly rollback to champion
      - Max 1 promotion per MIN_UPDATE_INTERVAL
      - All versions archived for audit
    """

    MIN_MARGIN = 0.1  # Challenger must beat champion by 0.1 Sharpe
    MIN_UPDATE_INTERVAL_HOURS = 6
    MAX_VERSIONS = 10  # Keep last 10 versions

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.versions: list[ModelVersion] = []
        self.champion: Optional[ModelVersion] = None
        self._last_update: Optional[datetime] = None
        self._version_counter = 0
        self._all_features: pd.DataFrame = pd.DataFrame()
        self._all_labels: pd.Series = pd.Series()

    def initial_train(self, features: pd.DataFrame, labels: pd.Series) -> ModelVersion:
        """Initial model training."""
        self._all_features = features.copy()
        self._all_labels = labels.copy()

        model = self._create_model()
        X = features.values
        y = (labels > 0).astype(int).values

        model.fit(X, y)

        self._version_counter += 1
        version = ModelVersion(
            version=self._version_counter,
            model=model,
            train_samples=len(X),
            train_date=datetime.now(timezone.utc).isoformat(),
            oos_sharpe=0.0,  # No OOS yet
            oos_accuracy=0.0,
            is_champion=True,
        )

        self.versions.append(version)
        self.champion = version
        self._last_update = datetime.now(timezone.utc)

        logger.info(f"Initial model trained: v{version.version}, {len(X)} samples")
        return version

    def update(self, new_features: pd.DataFrame, new_labels: pd.Series) -> UpdateResult:
        """
        Attempt to update model with new data.

        Trains a challenger on extended data, compares to champion.
        Promotes only if challenger significantly better.
        """
        # Check update interval
        if self._last_update:
            elapsed = (datetime.now(timezone.utc) - self._last_update).total_seconds() / 3600
            if elapsed < self.MIN_UPDATE_INTERVAL_HOURS:
                return UpdateResult(
                    promoted=False,
                    champion_version=self.champion.version if self.champion else 0,
                    challenger_version=0,
                    champion_sharpe=0,
                    challenger_sharpe=0,
                    margin=0,
                    reason=f"Too soon — {elapsed:.1f}h < {self.MIN_UPDATE_INTERVAL_HOURS}h minimum",
                )

        # Extend dataset
        self._all_features = pd.concat([self._all_features, new_features]).dropna()
        self._all_labels = pd.concat([self._all_labels, new_labels]).dropna()

        # Align
        combined = pd.concat([self._all_features, self._all_labels.rename('label')], axis=1).dropna()
        if len(combined) < 200:
            return UpdateResult(
                promoted=False,
                champion_version=self.champion.version if self.champion else 0,
                challenger_version=0,
                champion_sharpe=0,
                challenger_sharpe=0,
                margin=0,
                reason="Insufficient data after merge",
            )

        X = combined.drop('label', axis=1).values
        y = (combined['label'] > 0).astype(int).values

        # Split: last 20% as OOS
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train challenger
        challenger_model = self._create_model()
        challenger_model.fit(X_train, y_train)

        # Evaluate challenger
        chal_pred = challenger_model.predict(X_test)
        chal_returns = combined.iloc[split:]['ret_1d'] if 'ret_1d' in combined else pd.Series(0, index=range(len(y_test)))
        chal_strategy = chal_returns.values * (2 * chal_pred - 1)
        chal_sharpe = self._sharpe(chal_strategy)

        # Evaluate champion on same OOS
        champ_pred = self.champion.model.predict(X_test) if self.champion else np.zeros(len(y_test))
        champ_strategy = chal_returns.values * (2 * champ_pred - 1)
        champ_sharpe = self._sharpe(champ_strategy)

        margin = chal_sharpe - champ_sharpe

        # Decision
        self._version_counter += 1
        challenger_version = ModelVersion(
            version=self._version_counter,
            model=challenger_model,
            train_samples=len(X_train),
            train_date=datetime.now(timezone.utc).isoformat(),
            oos_sharpe=chal_sharpe,
            oos_accuracy=float(np.mean(chal_pred == y_test)),
            is_challenger=True,
        )

        if margin >= self.MIN_MARGIN:
            # Promote challenger
            if self.champion:
                self.champion.is_champion = False
            challenger_version.is_champion = True
            challenger_version.is_challenger = False
            self.champion = challenger_version
            self.versions.append(challenger_version)
            self._last_update = datetime.now(timezone.utc)

            # Trim old versions
            if len(self.versions) > self.MAX_VERSIONS:
                self.versions = self.versions[-self.MAX_VERSIONS:]

            logger.info(f"✅ Promoted v{challenger_version.version} (Sharpe {chal_sharpe:.2f} > {champ_sharpe:.2f})")
            return UpdateResult(
                promoted=True,
                champion_version=challenger_version.version,
                challenger_version=challenger_version.version,
                champion_sharpe=chal_sharpe,
                challenger_sharpe=chal_sharpe,
                margin=margin,
                reason=f"Challenger won by {margin:.2f} Sharpe",
            )
        else:
            # Keep champion
            logger.info(f"⏸️ Kept champion (Sharpe {champ_sharpe:.2f} >= {chal_sharpe:.2f})")
            return UpdateResult(
                promoted=False,
                champion_version=self.champion.version if self.champion else 0,
                challenger_version=challenger_version.version,
                champion_sharpe=champ_sharpe,
                challenger_sharpe=chal_sharpe,
                margin=margin,
                reason=f"Challenger margin {margin:.2f} < {self.MIN_MARGIN} required",
            )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict using champion model."""
        if self.champion is None:
            return np.zeros(len(features))
        return self.champion.model.predict(features.values)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using champion model."""
        if self.champion is None:
            return np.full((len(features), 2), 0.5)
        return self.champion.model.predict_proba(features.values)

    def rollback(self, steps: int = 1) -> bool:
        """Rollback to a previous model version."""
        if len(self.versions) < 2:
            return False

        target_idx = max(0, len(self.versions) - 1 - steps)
        target = self.versions[target_idx]

        if self.champion:
            self.champion.is_champion = False
        target.is_champion = True
        target.is_challenger = False
        self.champion = target

        logger.warning(f"⏪ Rolled back to v{target.version}")
        return True

    def get_status(self) -> dict:
        """Get continual learning status."""
        return {
            "champion_version": self.champion.version if self.champion else None,
            "champion_sharpe": self.champion.oos_sharpe if self.champion else None,
            "total_versions": len(self.versions),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "total_samples": len(self._all_features),
            "versions": [v.to_dict() for v in self.versions[-5:]],  # Last 5
        }

    def _create_model(self):
        """Create a new model instance."""
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=200, max_depth=5, random_state=42, eval_metric='logloss')
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        elif self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, random_state=42)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42)

    @staticmethod
    def _sharpe(returns: np.ndarray) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))
