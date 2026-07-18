"""engine.candlestick.sr_confidence
=====================================================================
Day 132 — Support/Resistance confidence scoring.

The book insists: S/R is not a line, it's a ZONE with varying strength.
We score each S/R level on:
  - Touches (how many times price has reacted)
  - Reaction strength (how strongly did price bounce)
  - Volume at the level
  - Age (older levels fade)
  - Liquidity (round-number levels attract more orders)
  - Distance from current price (too far = irrelevant)

Output: SRConfidenceResult with per-level scores + best level summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional  # FIX (SR-1): Optional was used below but never imported -> NameError on import

import numpy as np
import pandas as pd

from utils.logger import get_logger

# BUG FIX: `log` was used at the ATR-fallback debug line below but never
# defined anywhere in this module, so hitting that edge case (zero/NaN ATR)
# raised NameError instead of just logging the fallback.
log = get_logger("engine.candlestick.sr_confidence")


@dataclass
class SRLevel:
    price: float
    side: str                      # "support" / "resistance"
    touches: int
    reaction_strength: float       # average bounce size in ATR
    age_bars: int
    volume_at_level: float
    distance_from_current: float   # in ATR units
    confidence: float              # 0-100

    def to_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "side": self.side,
            "touches": self.touches,
            "reaction_strength": self.reaction_strength,
            "age_bars": self.age_bars,
            "volume_at_level": self.volume_at_level,
            "distance_from_current": self.distance_from_current,
            "confidence": self.confidence,
        }


@dataclass
class SRConfidenceResult:
    levels: list[SRLevel] = field(default_factory=list)
    best_support: Optional[SRLevel] = None
    best_resistance: Optional[SRLevel] = None
    current_price: float = 0.0
    nearest_strong_level: Optional[SRLevel] = None
    summary_score: float = 0.0       # 0-100, how strong is the nearest S/R

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": [l.to_dict() for l in self.levels],
            "best_support": self.best_support.to_dict() if self.best_support else None,
            "best_resistance": self.best_resistance.to_dict() if self.best_resistance else None,
            "current_price": self.current_price,
            "nearest_strong_level": (self.nearest_strong_level.to_dict()
                                      if self.nearest_strong_level else None),
            "summary_score": self.summary_score,
        }


# ----------------------------------------------------------------------
class SupportResistanceConfidence:
    def __init__(self,
                 lookback: int = 200,
                 min_touches: int = 2,
                 cluster_tolerance_atr: float = 0.5,
                 max_age_bars: int = 500) -> None:
        self.lookback = int(lookback)
        self.min_touches = int(min_touches)
        self.cluster_tol = float(cluster_tolerance_atr)
        self.max_age = int(max_age_bars)

    # ----------------------------------------------------------------
    def detect_and_score(self, df: pd.DataFrame,
                          atr_period: int = 14) -> SRConfidenceResult:
        from utils.indicators import atr as atr_indicator
        if len(df) < 50:
            return SRConfidenceResult()
        atr_series = atr_indicator(df, atr_period)
        # FIX (SR-2): removed dead/mistyped walrus-assigned `atrial_per` line that was
        # immediately shadowed by this correct computation (noqa'd dead code left in a
        # risk-scoring module — a maintainability hazard on its own).
        # C11 fix: if ATR is zero or NaN (dead market), use a small fallback
        # based on the price to avoid division-by-zero in _cluster().
        raw_atr = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 0.0
        current_price_tmp = float(df["close"].iloc[-1])
        if raw_atr <= 0:
            raw_atr = max(current_price_tmp * 0.001, 1e-8)  # 0.1% of price as fallback
            log.debug("sr_confidence: ATR was zero/NaN — using fallback %.6f", raw_atr)
        atr_val = raw_atr
        current_price = float(df["close"].iloc[-1])

        # Find swing highs/lows (peaks/troughs)
        window = df.tail(self.lookback).reset_index(drop=True)
        peaks: list[tuple[int, float]] = []
        troughs: list[tuple[int, float]] = []
        for i in range(1, len(window) - 1):
            if (window["high"].iloc[i] > window["high"].iloc[i - 1]
                    and window["high"].iloc[i] > window["high"].iloc[i + 1]):
                peaks.append((i, float(window["high"].iloc[i])))
            if (window["low"].iloc[i] < window["low"].iloc[i - 1]
                    and window["low"].iloc[i] < window["low"].iloc[i + 1]):
                troughs.append((i, float(window["low"].iloc[i])))

        # Cluster peaks into resistance levels
        # FIX (SR-3): volume_at_level was hardcoded to 0.0 despite the module docstring
        # advertising volume as a scoring factor. The data is already on hand here
        # (window["volume"]) — pass it through instead of silently dropping it.
        volume_series = window["volume"] if "volume" in window.columns else None
        resistance_levels = self._cluster(peaks, atr_val, "resistance",
                                            len(window), current_price, volume_series)
        support_levels = self._cluster(troughs, atr_val, "support",
                                        len(window), current_price, volume_series)
        all_levels = resistance_levels + support_levels

        # Filter: at least min_touches
        all_levels = [l for l in all_levels if l.touches >= self.min_touches]
        # Sort by confidence
        all_levels.sort(key=lambda l: l.confidence, reverse=True)

        best_support = next((l for l in all_levels if l.side == "support"), None)
        best_resistance = next((l for l in all_levels if l.side == "resistance"), None)
        # Nearest strong level
        nearest = None
        if all_levels:
            nearest = min(all_levels,
                           key=lambda l: abs(l.distance_from_current))
        summary = nearest.confidence if nearest else 0.0

        return SRConfidenceResult(
            levels=all_levels,
            best_support=best_support,
            best_resistance=best_resistance,
            current_price=current_price,
            nearest_strong_level=nearest,
            summary_score=float(summary),
        )

    # ----------------------------------------------------------------
    def _cluster(self, swings: list[tuple[int, float]],
                  atr_val: float, side: str,
                  total_bars: int, current_price: float,
                  volume_series: Optional[pd.Series] = None) -> list[SRLevel]:
        if not swings:
            return []
        # Cluster swings that are within cluster_tolerance_atr of each other
        tol = atr_val * self.cluster_tol
        clusters: list[list[tuple[int, float]]] = []
        for idx, price in swings:
            placed = False
            for cluster in clusters:
                avg_price = sum(p for _, p in cluster) / len(cluster)
                if abs(price - avg_price) <= tol:
                    cluster.append((idx, price))
                    placed = True
                    break
            if not placed:
                clusters.append([(idx, price)])
        # Build SRLevel per cluster
        levels: list[SRLevel] = []
        for cluster in clusters:
            avg_price = sum(p for _, p in cluster) / len(cluster)
            touches = len(cluster)
            # Reaction strength: distance from avg_price to the swings
            reaction = float(np.std([p for _, p in cluster]) / max(atr_val, 1e-9))
            # Age: bars since last touch
            last_touch_idx = max(idx for idx, _ in cluster)
            age = total_bars - last_touch_idx
            # Distance from current
            distance = abs(avg_price - current_price) / max(atr_val, 1e-9)
            # Confidence scoring
            touch_score = min(1.0, touches / 5.0) * 30       # 0-30
            reaction_score = min(1.0, reaction / 2.0) * 20   # 0-20
            age_score = max(0.0, 1.0 - age / self.max_age) * 20  # 0-20
            distance_score = (1.0 - min(1.0, distance / 5.0)) * 20  # 0-20, closer = better
            # Round number bonus (psychological levels)
            round_bonus = 0
            if avg_price > 0:
                # Check if price is near a round number (e.g. 50000, 100000)
                log_p = np.log10(avg_price)
                if abs(log_p - round(log_p)) < 0.05:
                    round_bonus = 10
            confidence = touch_score + reaction_score + age_score + distance_score + round_bonus
            confidence = float(max(0.0, min(100.0, confidence)))
            # FIX (SR-3): average the actual bar volume at each touch index instead
            # of hardcoding 0.0. Falls back to 0.0 only if no volume column exists.
            if volume_series is not None:
                touch_indices = [idx for idx, _ in cluster if 0 <= idx < len(volume_series)]
                vol_at_level = (float(volume_series.iloc[touch_indices].mean())
                                 if touch_indices else 0.0)
            else:
                vol_at_level = 0.0
            levels.append(SRLevel(
                price=float(avg_price),
                side=side,
                touches=int(touches),
                reaction_strength=float(reaction),
                age_bars=int(age),
                volume_at_level=float(vol_at_level),
                distance_from_current=float(distance),
                confidence=confidence,
            ))
        return levels