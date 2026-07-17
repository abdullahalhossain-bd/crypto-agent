"""engine.candlestick.multi_timeframe
=====================================================================
Day 137 — Multi-timeframe confirmation.

The book insists: never trade against the higher timeframe trend.
Classical setup:
  - Daily:    trend direction (the dominant bias)
  - H4:       trend direction (must agree with Daily)
  - H1:       pullback / pattern (entry trigger timeframe)
  - M15:      execution (precise entry, tight stop)

We score alignment: 0-100 based on how many timeframes agree.
If any higher TF disagrees, the score drops sharply (veto-like).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("candlestick.mtf")


@dataclass
class TimeframeView:
    timeframe: str
    direction: str          # "up" / "down" / "flat"
    slope: float
    weight: float           # higher TFs get more weight


@dataclass
class MTFResult:
    aligned: bool
    score: float            # 0-100
    dominant_direction: str # "up" / "down" / "mixed"
    timeframes: list[TimeframeView] = field(default_factory=list)
    veto: bool = False
    veto_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "aligned": self.aligned,
            "score": self.score,
            "dominant_direction": self.dominant_direction,
            "timeframes": [
                {"timeframe": t.timeframe, "direction": t.direction,
                 "slope": t.slope, "weight": t.weight}
                for t in self.timeframes
            ],
            "veto": self.veto,
            "veto_reason": self.veto_reason,
        }


# ----------------------------------------------------------------------
class MultiTimeframeConfirmator:
    """Takes one DataFrame + resamples to multiple timeframes, then
    checks trend alignment.

    The classical weight scheme: higher TFs dominate.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "D1": 0.40,
        "H4": 0.30,
        "H1": 0.20,
        "M15": 0.10,
    }

    RESAMPLE_MAP: dict[str, str] = {
        "D1": "1D",
        "H4": "4h",
        "H1": "1h",
        "M15": "15min",
    }

    def __init__(self,
                 weights: Optional[dict[str, float]] = None,
                 base_timeframe: str = "M15",
                 trend_window: int = 20,
                 min_slope_atr: float = 0.1) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.base_timeframe = base_timeframe
        self.trend_window = int(trend_window)
        self.min_slope_atr = float(min_slope_atr)

    # ----------------------------------------------------------------
    def confirm(self, df: pd.DataFrame) -> MTFResult:
        """Resample `df` to each timeframe and check alignment."""
        if len(df) < 100:
            return MTFResult(aligned=False, score=0.0,
                              dominant_direction="mixed",
                              timeframes=[], veto=False,
                              veto_reason="insufficient data")
        # Sanity check only (no behavior change): resampling assumes the
        # base df is at or finer than M15. If it's coarser, `_resample`
        # to M15 just reproduces the input 1:1 without erroring, which
        # silently fakes resolution rather than failing loudly.
        if "time" in df.columns and len(df) >= 3:
            try:
                deltas = pd.to_datetime(df["time"].tail(20), utc=True).diff().dropna()
                if not deltas.empty:
                    median_minutes = deltas.median().total_seconds() / 60.0
                    if median_minutes > 15.0:
                        log.warning(
                            "MTF base timeframe (~%.0f min/bar) is coarser than "
                            "the M15 bucket — resampled M15 view will not add "
                            "real resolution.", median_minutes,
                        )
            except Exception:  # noqa: BLE001
                pass  # sanity check only, never block confirmation on this
        from utils.indicators import atr as atr_indicator
        views: list[TimeframeView] = []
        for tf, weight in self.weights.items():
            resampled = self._resample(df, tf)
            if resampled is None or len(resampled) < 10:
                continue
            close = resampled["close"]
            window = close.tail(min(self.trend_window, len(close))).values
            if len(window) < 3:
                continue
            x = np.arange(len(window), dtype=float)
            denom = ((x - x.mean()) ** 2).sum()
            slope = float(((x - x.mean()) * (window - window.mean())).sum()
                          / denom) if denom > 0 else 0.0
            atr_series = atr_indicator(resampled, 14)
            atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 1.0
            slope_atr = slope / atr_val if atr_val > 0 else 0.0
            if slope_atr > self.min_slope_atr:
                direction = "up"
            elif slope_atr < -self.min_slope_atr:
                direction = "down"
            else:
                direction = "flat"
            views.append(TimeframeView(
                timeframe=tf, direction=direction,
                slope=float(slope_atr), weight=float(weight),
            ))

        if not views:
            return MTFResult(aligned=False, score=0.0,
                              dominant_direction="mixed")

        # Determine dominant direction
        up_weight = sum(v.weight for v in views if v.direction == "up")
        down_weight = sum(v.weight for v in views if v.direction == "down")
        flat_weight = sum(v.weight for v in views if v.direction == "flat")
        if up_weight > down_weight and up_weight > flat_weight:
            dominant = "up"
        elif down_weight > up_weight and down_weight > flat_weight:
            dominant = "down"
        else:
            dominant = "mixed"

        # Score: how aligned are the timeframes?
        if dominant == "mixed":
            score = 30.0
            aligned = False
        else:
            agreement_weight = up_weight if dominant == "up" else down_weight
            score = float(100.0 * agreement_weight)
            aligned = agreement_weight >= 0.6

        # Veto: if highest-weight TF (D1) is opposite to the next (H4),
        # we veto — this is a "trading against the trend" scenario.
        veto = False
        veto_reason = ""
        if len(views) >= 2:
            top_two = sorted(views, key=lambda v: v.weight, reverse=True)[:2]
            directions = {v.direction for v in top_two}
            if "up" in directions and "down" in directions:
                veto = True
                veto_reason = (f"{top_two[0].timeframe}={top_two[0].direction} "
                                f"conflicts with {top_two[1].timeframe}={top_two[1].direction}")
                score = 0.0
                aligned = False

        return MTFResult(
            aligned=aligned, score=score,
            dominant_direction=dominant,
            timeframes=views,
            veto=veto, veto_reason=veto_reason,
        )

    # ----------------------------------------------------------------
    def _resample(self, df: pd.DataFrame, target_tf: str) -> Optional[pd.DataFrame]:
        """Resample base df to target timeframe."""
        if "time" not in df.columns:
            return None
        try:
            df_copy = df.copy()
            df_copy["time"] = pd.to_datetime(df_copy["time"], utc=True)
            df_copy = df_copy.set_index("time")
            rule = self.RESAMPLE_MAP.get(target_tf)
            if rule is None:
                return None
            # FIX: conditionally include volume in agg dict — the old code
            # always included "volume" key even when the column didn't exist,
            # causing a KeyError that silently killed all MTF resampling.
            agg_dict = {
                "open": "first", "high": "max",
                "low": "min", "close": "last",
            }
            if "volume" in df_copy.columns:
                agg_dict["volume"] = "sum"
            resampled = df_copy.resample(rule).agg(agg_dict).dropna()
            return resampled.reset_index()
        except Exception as e:  # noqa: BLE001
            log.debug("resample to %s failed: %r", target_tf, e)
            return None