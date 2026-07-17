"""
AutoML Module — Automatic Model Selection + Hyperparameter Tuning
===================================================================

Automatically compares multiple ML models with different hyperparameters
and selects the best one based on out-of-sample Sharpe ratio.

Models compared:
  - Logistic Regression (baseline)
  - Random Forest (multiple configs)
  - XGBoost (multiple configs)
  - LightGBM (multiple configs)
  - CatBoost (if available)

Selection metric: OOS Sharpe ratio (not accuracy — Sharpe matters more)

Source: Orallexa (review #27) — ML model comparison framework
        ml4t-3e (review #18) — AutoML best practices

Usage:
    from trading_modules.automl import AutoML

    automl = AutoML(n_trials=20, metric="sharpe")
    results = automl.run(features_df, labels)

    print(automl.leaderboard())
    # Returns DataFrame sorted by Sharpe

    best_model = automl.get_best_model()
    predictions = automl.predict(new_features)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelTrial:
    """Result of a single AutoML trial."""
    model_name: str
    params: dict
    accuracy: float
    sharpe: float
    max_drawdown: float
    auc_roc: Optional[float]
    train_time_sec: float
    model: Any = None

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "params": self.params,
            "accuracy": round(self.accuracy, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "auc_roc": round(self.auc_roc, 4) if self.auc_roc else None,
            "train_time_sec": round(self.train_time_sec, 2),
        }


class AutoML:
    """
    Automatic ML model selection and hyperparameter tuning.

    Tries multiple models with different hyperparameters, evaluates
    each on out-of-sample data, and selects the best by Sharpe ratio.
    """

    # Model search spaces
    SEARCH_SPACES = {
        "random_forest": [
            {"n_estimators": 100, "max_depth": 5, "min_samples_leaf": 20},
            {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 10},
            {"n_estimators": 300, "max_depth": 15, "min_samples_leaf": 5},
            {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 20},
        ],
        "xgboost": [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
            {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
            {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1},
        ],
        "lightgbm": [
            {"n_estimators": 100, "num_leaves": 31, "learning_rate": 0.1},
            {"n_estimators": 200, "num_leaves": 63, "learning_rate": 0.05},
            {"n_estimators": 300, "num_leaves": 127, "learning_rate": 0.05},
        ],
        "logistic_regression": [
            {"C": 0.1, "max_iter": 500},
            {"C": 1.0, "max_iter": 1000},
            {"C": 10.0, "max_iter": 1000},
        ],
    }

    def __init__(self, metric: str = "sharpe", test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            metric: Selection metric ("sharpe", "accuracy", "auc")
            test_size: Fraction of data for out-of-sample evaluation
            random_state: Reproducibility
        """
        self.metric = metric
        self.test_size = test_size
        self.random_state = random_state
        self.trials: list[ModelTrial] = []
        self.best_model: Optional[Any] = None
        self.best_trial: Optional[ModelTrial] = None

    def run(self, features: pd.DataFrame, labels: pd.Series, verbose: bool = True) -> list[ModelTrial]:
        """
        Run AutoML: try all models, return sorted results.

        Args:
            features: Feature DataFrame
            labels: Label Series (+1/-1/0 or 0/1)
            verbose: Print progress

        Returns:
            List of ModelTrial sorted by metric (best first)
        """
        # Prepare data
        combined = pd.concat([features, labels.rename('label')], axis=1).dropna()
        if len(combined) < 200:
            logger.warning(f"Only {len(combined)} samples — need 200+")
            return []

        y = (combined['label'] > 0).astype(int)
        X = combined.drop('label', axis=1)

        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        returns = X_test['ret_1d'] if 'ret_1d' in X_test else pd.Series(0, index=X_test.index)

        self.trials = []

        # Try each model config
        for model_name, configs in self.SEARCH_SPACES.items():
            for i, params in enumerate(configs):
                if verbose:
                    print(f"  Trying {model_name} config {i+1}/{len(configs)}...")

                start_time = time.time()
                try:
                    trial = self._train_and_evaluate(
                        model_name, params, X_train, y_train, X_test, y_test, returns
                    )
                    trial.train_time_sec = time.time() - start_time
                    self.trials.append(trial)

                    if verbose:
                        print(f"    Sharpe={trial.sharpe:.2f} Acc={trial.accuracy:.1%} "
                              f"DD={trial.max_drawdown:.2%} ({trial.train_time_sec:.1f}s)")
                except Exception as e:
                    logger.warning(f"    {model_name} config {i} failed: {e}")

        # Sort by metric
        reverse = self.metric != "max_drawdown"  # Lower DD is better
        self.trials.sort(key=lambda t: getattr(t, self.metric), reverse=reverse)

        if self.trials:
            self.best_trial = self.trials[0]
            self.best_model = self.best_trial.model

        return self.trials

    def _train_and_evaluate(
        self, model_name: str, params: dict,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        returns: pd.Series,
    ) -> ModelTrial:
        """Train one model config and evaluate."""
        from sklearn.metrics import accuracy_score, roc_auc_score

        model = self._create_model(model_name, params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else None
        except Exception:
            auc = None

        # Strategy returns
        strategy_returns = returns * (2 * y_pred - 1)

        return ModelTrial(
            model_name=model_name,
            params=params,
            accuracy=accuracy_score(y_test, y_pred),
            sharpe=self._sharpe(strategy_returns),
            max_drawdown=self._max_dd(strategy_returns),
            auc_roc=auc,
            train_time_sec=0,
            model=model,
        )

    def _create_model(self, name: str, params: dict):
        """Create model instance."""
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1, **params)
        elif name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(random_state=self.random_state, eval_metric='logloss', **params)
        elif name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1, **params)
        elif name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            scaler = StandardScaler()
            lr = LogisticRegression(random_state=self.random_state, **params)
            model = Pipeline([('scaler', scaler), ('lr', lr)])
            model.fit(self._last_X_train, self._last_y_train)
            return model
        else:
            raise ValueError(f"Unknown model: {name}")

    _last_X_train = None
    _last_y_train = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict using best model."""
        if self.best_model is None:
            return np.zeros(len(features))

        if hasattr(self.best_model, 'named_steps') and 'scaler' in self.best_model.named_steps:
            return self.best_model.predict_proba(features)[:, 1]
        return self.best_model.predict_proba(features)[:, 1]

    def leaderboard(self) -> pd.DataFrame:
        """Get model comparison leaderboard."""
        if not self.trials:
            return pd.DataFrame()
        rows = [t.to_dict() for t in self.trials]
        df = pd.DataFrame(rows)
        return df.drop(columns=['params'] if 'params' in df.columns else [])

    def get_best_model(self):
        """Get the best trained model."""
        return self.best_model

    def get_summary(self) -> dict:
        """Get AutoML summary."""
        if not self.trials:
            return {"status": "not_run"}
        return {
            "n_trials": len(self.trials),
            "best_model": self.best_trial.model_name if self.best_trial else None,
            "best_sharpe": self.best_trial.sharpe if self.best_trial else None,
            "best_accuracy": self.best_trial.accuracy if self.best_trial else None,
            "metric": self.metric,
            "models_tested": list(set(t.model_name for t in self.trials)),
        }

    @staticmethod
    def _sharpe(returns: pd.Series) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    @staticmethod
    def _max_dd(returns: pd.Series) -> float:
        if len(returns) == 0:
            return 0.0
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())
