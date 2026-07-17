"""
ML Models Module — Baseline Predictive Models
==============================================

Three baseline ML models for return direction prediction:
  1. Random Forest (scikit-learn)
  2. XGBoost (gradient boosting)
  3. Logistic Regression (regularized linear baseline)

Every later model must beat the LR baseline to justify its complexity.

Features: technical indicators computed from OHLCV
Labels: triple-barrier labels (+1/-1/0) or forward returns
Output: probability of upward move (0-1)

Usage:
    from ml_models import MLModelTrainer, build_features

    # Build features from OHLCV
    features = build_features(df)

    # Build labels (triple-barrier)
    from triple_barrier import compute_labels
    labels = compute_labels(df)

    # Train and evaluate
    trainer = MLModelTrainer()
    results = trainer.train_all(features, labels)

    # Get predictions
    pred = trainer.predict("random_forest", new_features)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical features from OHLCV data.

    Generates 20+ features including:
    - Returns (1d, 5d, 10d, 20d)
    - Moving averages (5, 10, 20, 50)
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2σ)
    - ATR (14)
    - Volume ratios
    - Volatility

    All features are point-in-time correct (no lookahead).
    """
    df = df.copy()

    # Normalize columns
    for col in df.columns:
        lower = col.lower()
        if lower in ('open', 'high', 'low', 'close', 'volume'):
            df.rename(columns={col: lower}, inplace=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

    features = pd.DataFrame(index=df.index)

    # Returns
    features['ret_1d'] = close.pct_change(1)
    features['ret_5d'] = close.pct_change(5)
    features['ret_10d'] = close.pct_change(10)
    features['ret_20d'] = close.pct_change(20)

    # Moving averages
    for period in [5, 10, 20, 50]:
        ma = close.rolling(period).mean()
        features[f'ma_{period}'] = ma
        features[f'price_ma_{period}_ratio'] = close / ma - 1.0

    # RSI (14)
    delta_close = close.diff()
    gain = delta_close.where(delta_close > 0, 0.0)
    loss = (-delta_close.where(delta_close < 0, 0.0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + avg_gain / avg_loss))

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    features['macd'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_hist'] = macd_line - signal_line

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    features['bb_pct'] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, 1e-10)
    features['bb_width'] = (bb_upper - bb_lower) / bb_mid.replace(0, 1e-10)

    # ATR (14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    features['atr_14'] = tr.rolling(14).mean()
    features['atr_pct'] = features['atr_14'] / close

    # Volume features
    vol_ma = volume.rolling(20).mean().replace(0, 1e-10)
    features['volume_ratio'] = volume / vol_ma
    features['volume_change'] = volume.pct_change()

    # Volatility
    features['volatility_10d'] = close.pct_change().rolling(10).std()
    features['volatility_20d'] = close.pct_change().rolling(20).std()

    # High-Low spread
    features['hl_spread'] = (high - low) / close

    # Candle body
    features['body'] = (close - df['open']) / close

    return features


# ═══════════════════════════════════════════════════════════════
# Model Results
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelResult:
    """Result of training a single model."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float]
    sharpe: float          # Sharpe of strategy using this model
    max_drawdown: float    # Max DD of strategy
    n_train: int
    n_test: int
    feature_importance: Optional[dict] = None
    model: object = None   # The trained model object

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1_score, 4),
            "auc_roc": round(self.auc_roc, 4) if self.auc_roc else None,
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "top_features": dict(sorted(
                (self.feature_importance or {}).items(),
                key=lambda x: x[1], reverse=True,
            )[:10]),
        }


# ═══════════════════════════════════════════════════════════════
# Model Trainer
# ═══════════════════════════════════════════════════════════════

class MLModelTrainer:
    """
    Train and evaluate multiple ML models for return prediction.

    Models:
      1. Logistic Regression (baseline — every model must beat this)
      2. Random Forest
      3. XGBoost (if available)

    Uses walk-forward split (last 20% as out-of-sample test).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self._models: dict[str, object] = {}
        self._results: dict[str, ModelResult] = {}

    def train_all(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
    ) -> dict[str, ModelResult]:
        """
        Train all available models and return comparison results.

        Args:
            features: Feature DataFrame (from build_features)
            labels: Label Series (+1/-1/0 from triple_barrier, or 0/1)

        Returns:
            Dict of {model_name: ModelResult}
        """
        # Align and clean
        combined = pd.concat([features, labels.rename('label')], axis=1).dropna()
        if len(combined) < 100:
            logger.warning(f"Only {len(combined)} samples after dropna — need 100+")
            return {}

        # Convert labels to binary: +1 → 1 (up), else → 0 (down/neutral)
        y = (combined['label'] > 0).astype(int)
        X = combined.drop('label', axis=1)

        # Walk-forward split (temporal)
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        results = {}

        # 1. Logistic Regression (baseline)
        try:
            results['logistic_regression'] = self._train_lr(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.error(f"LR training failed: {e}")

        # 2. Random Forest
        try:
            results['random_forest'] = self._train_rf(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.error(f"RF training failed: {e}")

        # 3. XGBoost
        try:
            results['xgboost'] = self._train_xgb(X_train, y_train, X_test, y_test)
        except Exception as e:
            logger.warning(f"XGBoost not available or failed: {e}")

        self._results = results
        return results

    def _train_lr(self, X_train, y_train, X_test, y_test) -> ModelResult:
        """Train Logistic Regression baseline."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            C=1.0,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        # Strategy returns (long when predict up)
        returns = X_test['ret_1d'] if 'ret_1d' in X_test else pd.Series(0, index=X_test.index)
        strategy_returns = returns * (2 * y_pred - 1)  # +1 if predict up, -1 if down

        self._models['logistic_regression'] = (model, scaler)

        return ModelResult(
            model_name='logistic_regression',
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else None,
            sharpe=self._compute_sharpe(strategy_returns),
            max_drawdown=self._compute_max_dd(strategy_returns),
            n_train=len(y_train),
            n_test=len(y_test),
            feature_importance=dict(zip(X_train.columns, abs(model.coef_[0]))),
            model=model,
        )

    def _train_rf(self, X_train, y_train, X_test, y_test) -> ModelResult:
        """Train Random Forest."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        returns = X_test['ret_1d'] if 'ret_1d' in X_test else pd.Series(0, index=X_test.index)
        strategy_returns = returns * (2 * y_pred - 1)

        self._models['random_forest'] = model

        return ModelResult(
            model_name='random_forest',
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else None,
            sharpe=self._compute_sharpe(strategy_returns),
            max_drawdown=self._compute_max_dd(strategy_returns),
            n_train=len(y_train),
            n_test=len(y_test),
            feature_importance=dict(zip(X_train.columns, model.feature_importances_)),
            model=model,
        )

    def _train_xgb(self, X_train, y_train, X_test, y_test) -> ModelResult:
        """Train XGBoost."""
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        returns = X_test['ret_1d'] if 'ret_1d' in X_test else pd.Series(0, index=X_test.index)
        strategy_returns = returns * (2 * y_pred - 1)

        self._models['xgboost'] = model

        return ModelResult(
            model_name='xgboost',
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1_score=f1_score(y_test, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else None,
            sharpe=self._compute_sharpe(strategy_returns),
            max_drawdown=self._compute_max_dd(strategy_returns),
            n_train=len(y_train),
            n_test=len(y_test),
            feature_importance=dict(zip(X_train.columns, model.feature_importances_)),
            model=model,
        )

    def predict(self, model_name: str, features: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from a trained model."""
        if model_name not in self._models:
            raise KeyError(f"Model {model_name} not trained. Available: {list(self._models)}")

        model_entry = self._models[model_name]

        # LR needs scaler
        if model_name == 'logistic_regression':
            model, scaler = model_entry
            features_s = scaler.transform(features)
            return model.predict_proba(features_s)[:, 1]
        else:
            model = model_entry
            return model.predict_proba(features)[:, 1]

    def get_leaderboard(self) -> pd.DataFrame:
        """Get model comparison leaderboard."""
        if not self._results:
            return pd.DataFrame()

        rows = [r.to_dict() for r in self._results.values()]
        df = pd.DataFrame(rows)
        df = df.sort_values('sharpe', ascending=False)
        return df

    @staticmethod
    def _compute_sharpe(returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    @staticmethod
    def _compute_max_dd(returns: pd.Series) -> float:
        """Maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return float(drawdown.min())
