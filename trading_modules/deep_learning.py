"""
Deep Learning Module — LSTM + Transformer for Time Series Forecasting
======================================================================

Two architectures:
  1. LSTM — Long Short-Term Memory for sequential pattern recognition
  2. Transformer — Self-attention for multi-horizon forecasting

Both predict future returns (not prices directly) to avoid non-stationarity.

Source: ml4t-3e (review #18) ch.13 — Deep Learning for Time Series
        Orallexa (review #27) — EMAformer (iTransformer variant)

Usage:
    from trading_modules.deep_learning import LSTMForecaster, TransformerForecaster

    # Train LSTM
    lstm = LSTMForecaster(input_dim=26, hidden_dim=128, n_layers=2)
    lstm.train(features_df, horizon=5, n_epochs=50)

    # Predict
    pred, confidence = lstm.predict(latest_features)
    # pred: predicted 5-bar return
    # confidence: model uncertainty (0-1)

    # Train Transformer
    tft = TransformerForecaster(input_dim=26, d_model=64, n_heads=4)
    tft.train(features_df, horizon=5, n_epochs=50)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Config & Result (always available, no torch dependency)
# ═══════════════════════════════════════════════════════════════

@dataclass
class DeepLearningConfig:
    """Training configuration."""
    window: int = 20
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    n_epochs: int = 50
    batch_size: int = 32
    val_split: float = 0.2
    patience: int = 10


@dataclass
class ForecastResult:
    """Forecast prediction result."""
    prediction: float
    direction: int
    confidence: float
    uncertainty: float
    mc_mean: float = 0.0
    mc_std: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prediction": round(self.prediction, 6),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "mc_mean": round(self.mc_mean, 6),
            "mc_std": round(self.mc_std, 6),
        }


if TORCH_AVAILABLE:

    # ═══════════════════════════════════════════════════════════════
    # Dataset
    # ═══════════════════════════════════════════════════════════════

    class TimeSeriesDataset(Dataset):
        """Sliding window dataset for time series."""

        def __init__(self, features: np.ndarray, targets: np.ndarray, window: int = 20):
            self.features = features
            self.targets = targets
            self.window = window

        def __len__(self):
            return max(0, len(self.features) - self.window)

        def __getitem__(self, idx):
            x = self.features[idx:idx + self.window]
            y = self.targets[idx + self.window]
            return torch.FloatTensor(x), torch.FloatTensor([y])


    # ═══════════════════════════════════════════════════════════════
    # LSTM Forecaster
    # ═══════════════════════════════════════════════════════════════

    class LSTMModel(nn.Module):
        """LSTM with dropout for MC Dropout uncertainty estimation."""

        def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            out = self.dropout(last_hidden)
            return self.fc(out)


    @dataclass
    class _DeepLearningConfigInternal:  # Kept for backward compat
        pass


    @dataclass
    class _ForecastResultInternal:  # Kept for backward compat
        pass


    class LSTMForecaster:
        """
        LSTM-based time series forecaster.

        Features:
          - Predicts future returns (not prices)
          - MC Dropout for uncertainty estimation
          - Early stopping with validation loss
          - Direction + confidence + uncertainty output

        Usage:
            lstm = LSTMForecaster(input_dim=26)
            lstm.train(features_df, horizon=5)
            result = lstm.predict(latest_window)
        """

        def __init__(
            self,
            input_dim: int = 26,
            config: Optional[DeepLearningConfig] = None,
            device: str = "cpu",
        ):
            self.config = config or DeepLearningConfig()
            self.device = torch.device(device)
            self.input_dim = input_dim
            self.model: Optional[LSTMModel] = None
            self.is_trained = False
            self.train_history: list = []

        def train(
            self,
            features: pd.DataFrame,
            horizon: int = 5,
            n_epochs: Optional[int] = None,
            verbose: bool = True,
        ) -> dict:
            """Train the LSTM on feature data."""
            if n_epochs is not None:
                self.config.n_epochs = n_epochs
            # Prepare targets: forward return at horizon
            # Major #8 fix: the old code silently used the first column as
            # a proxy for 'close' when 'close' was missing — this could
            # train the model on meaningless targets. Now we raise a clear
            # ValueError if neither 'close' nor a target_column is available.
            close_col = 'close' if 'close' in features.columns else None
            if close_col is None:
                raise ValueError(
                    "LSTMForecaster.train: 'close' column not found in features. "
                    "Either include a 'close' column or pass target_column explicitly. "
                    f"Available columns: {list(features.columns)}"
                )
            targets = features[close_col].pct_change(horizon).shift(-horizon)

            # Drop NaN
            valid = ~targets.isna()
            feat_array = features[valid].values.astype(np.float32)
            feat_array = np.nan_to_num(feat_array, nan=0.0, posinf=1.0, neginf=-1.0)
            target_array = targets[valid].values.astype(np.float32)

            if len(feat_array) < 200:
                logger.warning(f"Only {len(feat_array)} samples — need 200+")
                return {"error": "insufficient data"}

            # Train/val split
            split_idx = int(len(feat_array) * (1 - self.config.val_split))
            train_feat = feat_array[:split_idx]
            train_tgt = target_array[:split_idx]
            val_feat = feat_array[split_idx:]
            val_tgt = target_array[split_idx:]

            # Datasets
            train_ds = TimeSeriesDataset(train_feat, train_tgt, self.config.window)
            val_ds = TimeSeriesDataset(val_feat, val_tgt, self.config.window)
            train_dl = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
            val_dl = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

            # Model
            self.model = LSTMModel(
                input_dim=self.input_dim,
                hidden_dim=self.config.hidden_dim,
                n_layers=self.config.n_layers,
                dropout=self.config.dropout,
            ).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.n_epochs):
                # Train
                self.model.train()
                train_loss = 0.0
                for x, y in train_dl:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    pred = self.model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validate
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x, y in val_dl:
                        x, y = x.to(self.device), y.to(self.device)
                        pred = self.model(x)
                        val_loss += criterion(pred, y).item()

                train_loss /= max(len(train_dl), 1)
                val_loss /= max(len(val_dl), 1)
                self.train_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and (epoch % 10 == 0 or epoch == self.config.n_epochs - 1):
                    print(f"  Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}")
                    break

            self.is_trained = True
            return {"final_train_loss": train_loss, "final_val_loss": val_loss, "best_val_loss": best_val_loss}

        def predict(self, window_features: np.ndarray, n_mc_samples: int = 30) -> ForecastResult:
            """
            Predict with MC Dropout uncertainty estimation.

            Args:
                window_features: Shape (window, input_dim) — last N bars of features
                n_mc_samples: Number of MC Dropout forward passes

            Returns:
                ForecastResult with prediction + uncertainty
            """
            if not self.is_trained or self.model is None:
                return ForecastResult(prediction=0.0, direction=0, confidence=0.5, uncertainty=1.0)

            x = torch.FloatTensor(window_features).unsqueeze(0).to(self.device)

            # MC Dropout: enable dropout at inference
            self.model.train()  # Enables dropout
            mc_preds = []
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    pred = self.model(x).item()
                    mc_preds.append(pred)

            mc_mean = float(np.mean(mc_preds))
            mc_std = float(np.std(mc_preds))

            # Direction
            if mc_mean > 0.001:
                direction = 1
            elif mc_mean < -0.001:
                direction = -1
            else:
                direction = 0

            # Confidence: how certain is the model?
            # High std = low confidence
            confidence = max(0.0, min(1.0, 1.0 - mc_std * 10))
            uncertainty = min(1.0, mc_std * 10)

            return ForecastResult(
                prediction=mc_mean,
                direction=direction,
                confidence=confidence,
                uncertainty=uncertainty,
                mc_mean=mc_mean,
                mc_std=mc_std,
            )

        def save(self, path: str) -> None:
            torch.save({"model_state": self.model.state_dict(), "config": self.config.__dict__}, path)

        def load(self, path: str) -> None:
            ckpt = torch.load(path, map_location=self.device)
            self.config = DeepLearningConfig(**ckpt["config"])
            self.model = LSTMModel(self.input_dim, self.config.hidden_dim, self.config.n_layers, self.config.dropout).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.is_trained = True


    # ═══════════════════════════════════════════════════════════════
    # Transformer Forecaster
    # ═══════════════════════════════════════════════════════════════

    class TransformerModel(nn.Module):
        """Transformer encoder for time series forecasting."""

        def __init__(self, input_dim: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.size(1)
            x = self.input_proj(x)
            x = x + self.pos_encoding[:, :seq_len, :]
            x = self.encoder(x)
            x = x[:, -1, :]  # Last token
            x = self.dropout(x)
            return self.fc(x)


    class TransformerForecaster(LSTMForecaster):
        """
        Transformer-based forecaster (same interface as LSTMForecaster).

        Uses self-attention instead of recurrence. Better at capturing
        long-range dependencies in time series.
        """

        def __init__(
            self,
            input_dim: int = 26,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            config: Optional[DeepLearningConfig] = None,
            device: str = "cpu",
        ):
            self.config = config or DeepLearningConfig()
            self.device = torch.device(device)
            self.input_dim = input_dim
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.model = TransformerModel(
                input_dim=input_dim,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=self.config.dropout,
            ).to(self.device)
            self.is_trained = False
            self.train_history = []

else:
    # Stubs when torch not available
    class LSTMForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install: pip install torch")

    class TransformerForecaster:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not installed. Install: pip install torch")
