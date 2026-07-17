"""engine.volatility_predictor
=====================================================================
Simplified GARCH(1,1) volatility forecasting for the next 4 hours.

Predicts future volatility from historical returns, then:
  - Adjusts SL multiplier (wider in high vol, tighter in low vol)
  - Adjusts sizing multiplier (smaller in high vol)
  - Vetoes if volatility is extreme (> 2× baseline)

GARCH(1,1): σ²(t+1) = ω + α × r²(t) + β × σ²(t)
Simplified: we use fixed ω=0.0001, α=0.10, β=0.85 (sum ≈ 1)

Inspired by Centina-Quant's VolatilityPredictor. Adapted to fit
our risk_v2 engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("engine.volatility_predictor")

ATR_PERIOD = 14
LOOKBACK = 50

# GARCH(1,1) parameters (simplified, stationary)
GARCH_OMEGA = 0.0001
GARCH_ALPHA = 0.10
GARCH_BETA = 0.85


@dataclass
class VolatilityForecast:
    atr_predicted: float = 0.0       # predicted ATR for next 4h
    atr_ratio: float = 1.0           # predicted / 20-bar mean
    volatility_label: str = "NORMAL" # LOW / NORMAL / HIGH / EXTREME
    sl_multiplier: float = 1.0       # multiplier for SL distance
    sizing_multiplier: float = 1.0   # multiplier for position size
    veto: bool = False               # True if too volatile to trade
    bonus: int = 0                   # pts adjustment for confluence score
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atr_predicted": self.atr_predicted,
            "atr_ratio": self.atr_ratio,
            "volatility_label": self.volatility_label,
            "sl_multiplier": self.sl_multiplier,
            "sizing_multiplier": self.sizing_multiplier,
            "veto": self.veto,
            "bonus": self.bonus,
            "details": dict(self.details),
        }


# ----------------------------------------------------------------------
class VolatilityPredictor:
    """Predicts next-period volatility using simplified GARCH(1,1)."""

    def __init__(self,
                 lookback: int = LOOKBACK,
                 atr_period: int = ATR_PERIOD) -> None:
        self.lookback = int(lookback)
        self.atr_period = int(atr_period)

    # ----------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> VolatilityForecast:
        """Predict volatility for the next 4 bars.

        Args:
            df: OHLCV DataFrame

        Returns:
            VolatilityForecast with predicted ATR + multipliers
        """
        if len(df) < self.lookback:
            return VolatilityForecast(details={"reason": "insufficient data"})

        try:
            high = df["high"].astype(float).values
            low = df["low"].astype(float).values
            close = df["close"].astype(float).values

            # Historical ATR
            atr_series = self._compute_atr(high, low, close, self.atr_period)
            if len(atr_series) < 20:
                return VolatilityForecast(details={"reason": "ATR warmup"})
            atr_current = float(atr_series[-1])
            atr_mean20 = float(np.mean(atr_series[-20:]))
            atr_std20 = float(np.std(atr_series[-20:]))

            # GARCH(1,1) simplified on log returns
            returns = np.diff(np.log(close[-self.lookback:] + 1e-10))
            if len(returns) < 10:
                return VolatilityForecast(details={"reason": "returns warmup"})

            # Initialize variance = sample variance
            var_t = float(np.var(returns[:20])) if len(returns) >= 20 else float(np.var(returns))
            for r in returns[20:]:
                var_t = GARCH_OMEGA + GARCH_ALPHA * r ** 2 + GARCH_BETA * var_t

            # Predicted std dev (next-period return volatility, dimensionless)
            predicted_std = float(np.sqrt(var_t))

            # ---- Scale ATR by the forecast/baseline VOLATILITY RATIO ----
            # NOTE: predicted_std (return std-dev, dimensionless fraction) and
            # atr_current (a price-range quantity in the same units as the
            # instrument's price) are NOT interchangeable. Multiplying
            # predicted_std directly by price silently conflates two
            # different statistics and produces an atr_predicted with no
            # calibrated relationship to the real ATR series it's compared
            # against below. Instead we forecast the *ratio* of future to
            # current volatility from GARCH, and apply that ratio to the
            # already-correctly-scaled current ATR. Ratios are unit-free,
            # so this stays dimensionally consistent without requiring an
            # offline-fitted calibration constant.
            baseline_std = float(np.std(returns[-20:])) if len(returns) >= 20 else float(np.std(returns))
            if baseline_std > 1e-12:
                vol_ratio = predicted_std / baseline_std
            else:
                vol_ratio = 1.0
            # Guard against pathological GARCH blow-up from a single call
            vol_ratio = float(np.clip(vol_ratio, 0.1, 5.0))
            atr_predicted = atr_current * vol_ratio

            # Fallback: if GARCH produces unreasonable value, use current ATR
            if atr_predicted <= 0 or not np.isfinite(atr_predicted):
                atr_predicted = atr_current

            # Ratio
            if atr_mean20 > 0:
                atr_ratio = atr_predicted / atr_mean20
            else:
                atr_ratio = 1.0

            # Classify
            if atr_ratio >= 2.0:
                label = "EXTREME"
                sl_mult = 1.5
                sizing_mult = 0.3
                veto = True
                bonus = -20
            elif atr_ratio >= 1.5:
                label = "HIGH"
                sl_mult = 1.3
                sizing_mult = 0.5
                veto = False
                bonus = -10
            elif atr_ratio <= 0.7:
                label = "LOW"
                sl_mult = 0.8
                sizing_mult = 1.0
                veto = False
                bonus = 5
            else:
                label = "NORMAL"
                sl_mult = 1.0
                sizing_mult = 1.0
                veto = False
                bonus = 0

            return VolatilityForecast(
                atr_predicted=round(atr_predicted, 8),
                atr_ratio=round(atr_ratio, 4),
                volatility_label=label,
                sl_multiplier=sl_mult,
                sizing_multiplier=sizing_mult,
                veto=veto,
                bonus=bonus,
                details={
                    "garch_variance": round(var_t, 8),
                    "predicted_std": round(predicted_std, 8),
                    "atr_current": round(atr_current, 8),
                    "atr_mean20": round(atr_mean20, 8),
                    "atr_std20": round(atr_std20, 8),
                },
            )
        except Exception as e:  # noqa: BLE001
            # FAIL-CLOSED, not fail-open: a crashed volatility forecast must
            # never be treated as "normal volatility, safe to trade" by
            # downstream sizing/veto logic. VolatilityForecast()'s dataclass
            # defaults (veto=False, sizing_multiplier=1.0) are safe as
            # *defaults for a healthy call*, but are the wrong defaults for
            # an *error path* — silently trading full-size through an
            # unknown volatility state is a live-capital risk. Explicitly
            # veto and zero out sizing here so a broken predictor blocks
            # trades rather than passing them through unfiltered.
            log.error("VolatilityPredictor failed, forecast unavailable — "
                      "vetoing trade defensively: %s", e, exc_info=True)
            return VolatilityForecast(
                volatility_label="UNKNOWN",
                veto=True,
                sizing_multiplier=0.0,
                bonus=0,
                details={"error": str(e), "fail_closed": True},
            )

    # ----------------------------------------------------------------
    @staticmethod
    def _compute_atr(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, period: int) -> np.ndarray:
        """Compute ATR as numpy array."""
        if len(highs) < period + 1:
            return np.array([0.0])
        tr = np.zeros(len(highs))
        tr[0] = highs[0] - lows[0]
        for i in range(1, len(highs)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        # Wilder's smoothing
        atr = np.zeros(len(tr))
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        return atr[period - 1:]