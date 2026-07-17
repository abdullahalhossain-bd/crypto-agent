"""
Cross-Asset Confirmation — "Is the macro picture supporting this trade?"
=======================================================================

A crypto BUY is stronger when DXY is weak. A Gold BUY is stronger when USD
is weak. An equity long is stronger when bond yields are falling.

This module takes the candidate trade's symbol and a dict of OHLCV dataframes
for related assets, then checks whether the related assets' recent moves
confirm or contradict the candidate.

Default correlation map (caller can override):
    - BTCUSD ↔ DXY        (inverse — USD weak → BTC up)
    - BTCUSD ↔ GOLD       (positive — both anti-USD)
    - BTCUSD ↔ SPX/NASDAQ (positive — risk-on)
    - ETHUSD ↔ BTCUSD     (positive — same sector)
    - EURUSD ↔ DXY        (inverse)
    - GBPUSD ↔ DXY        (inverse)
    - XAUUSD ↔ DXY        (inverse)

Usage:
    from trading_modules.cross_asset import CrossAssetConfirmation
    checker = CrossAssetConfirmation()
    result = checker.check(
        candidate_symbol="BTCUSD",
        candidate_direction="BUY",
        related_dfs={
            "DXY":   dxy_df,
            "GOLD":  gold_df,
            "NASDAQ": ndx_df,
        },
    )
    if result.confirmed:
        # macro picture supports the trade
    elif result.contradicted:
        # macro picture argues against the trade
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default relationships: (asset_a, asset_b, expected_correlation)
# expected_correlation: "positive" → both should move same direction
#                       "inverse"  → opposite directions
DEFAULT_RELATIONSHIPS: list[tuple[str, str, str]] = [
    ("BTCUSD", "DXY",    "inverse"),
    ("BTCUSD", "GOLD",   "positive"),
    ("BTCUSD", "XAUUSD", "inverse"),     # XAUUSD = gold quoted in USD → inverse to BTC if USD weak
    ("BTCUSD", "NASDAQ", "positive"),
    ("BTCUSD", "SPX",    "positive"),
    ("BTCUSD", "ETHUSD", "positive"),
    ("ETHUSD", "BTCUSD", "positive"),
    ("ETHUSD", "DXY",    "inverse"),
    ("EURUSD", "DXY",    "inverse"),
    ("GBPUSD", "DXY",    "inverse"),
    ("XAUUSD", "DXY",    "inverse"),
    ("GOLD",   "DXY",    "inverse"),
]


@dataclass
class CrossAssetResult:
    confirmed: bool = False          # related assets support the trade
    contradicted: bool = False       # related assets argue against
    neutral: bool = True             # no related assets checked
    score: float = 0.0               # -1..+1 (positive = confirm, negative = contradict)
    confirmations: list[str] = field(default_factory=list)   # ["DXY down → BTCUSD up"]
    contradictions: list[str] = field(default_factory=list)
    missing_assets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "confirmed": self.confirmed,
            "contradicted": self.contradicted,
            "neutral": self.neutral,
            "score": round(self.score, 2),
            "confirmations": self.confirmations,
            "contradictions": self.contradictions,
            "missing_assets": self.missing_assets,
            "notes": self.notes,
        }


class CrossAssetConfirmation:
    """Check whether related assets confirm or contradict a candidate trade.

    Parameters:
        relationships: list of (asset_a, asset_b, "positive"/"inverse")
        trend_lookback: bars to determine each related asset's direction (default 20)
        min_strength_pct: minimum % move over lookback to count as directional (default 0.3)
    """

    def __init__(
        self,
        relationships: Optional[list[tuple[str, str, str]]] = None,
        trend_lookback: int = 20,
        min_strength_pct: float = 0.3,
    ) -> None:
        self.relationships = relationships if relationships is not None else DEFAULT_RELATIONSHIPS
        self.trend_lookback = trend_lookback
        self.min_strength_pct = min_strength_pct

    def check(
        self,
        candidate_symbol: str,
        candidate_direction: str,
        related_dfs: dict[str, pd.DataFrame],
    ) -> CrossAssetResult:
        """Check related assets for confirmation.

        Args:
            candidate_symbol: e.g., "BTCUSD"
            candidate_direction: "BUY" or "SELL"
            related_dfs: dict mapping related asset name → OHLCV df
        """
        direction = candidate_direction.upper()
        if direction not in ("BUY", "SELL"):
            return CrossAssetResult(notes=["invalid direction"])

        confirmations: list[str] = []
        contradictions: list[str] = []
        missing: list[str] = []
        scores: list[float] = []

        for asset_a, asset_b, rel in self.relationships:
            # Find which side of the relationship matches our candidate
            other: Optional[str] = None
            if asset_a == candidate_symbol:
                other = asset_b
            elif asset_b == candidate_symbol:
                other = asset_a
            if other is None:
                continue
            # Need OHLCV for the other asset
            if other not in related_dfs or related_dfs[other] is None:
                missing.append(other)
                continue
            other_df = related_dfs[other]
            if len(other_df) < self.trend_lookback + 1:
                missing.append(other)
                continue

            # Determine direction of related asset
            recent_close = float(other_df["close"].iloc[-1])
            past_close = float(other_df["close"].iloc[-self.trend_lookback])
            if past_close <= 0:
                continue
            pct_change = (recent_close - past_close) / past_close * 100
            if abs(pct_change) < self.min_strength_pct:
                continue  # related asset is flat — no signal

            related_up = pct_change > 0
            # Expected direction of related asset for confirmation
            # If candidate BUY and rel="positive" → related should be UP
            # If candidate BUY and rel="inverse"  → related should be DOWN
            # If candidate SELL and rel="positive" → related should be DOWN
            # If candidate SELL and rel="inverse"  → related should be UP
            if direction == "BUY":
                expected_up = (rel == "positive")
            else:
                expected_up = (rel == "inverse")

            confirms = (related_up == expected_up)
            strength = min(1.0, abs(pct_change) / 3.0)  # cap at 1.0 for 3% move
            if confirms:
                confirmations.append(
                    f"{other} {'up' if related_up else 'down'} {pct_change:+.2f}% "
                    f"→ confirms {candidate_symbol} {direction}"
                )
                scores.append(+strength)
            else:
                contradictions.append(
                    f"{other} {'up' if related_up else 'down'} {pct_change:+.2f}% "
                    f"→ contradicts {candidate_symbol} {direction}"
                )
                scores.append(-strength)

        score = float(np.mean(scores)) if scores else 0.0
        confirmed = score > 0.3 and len(confirmations) > 0
        contradicted = score < -0.3 and len(contradictions) > 0
        neutral = not confirmed and not contradicted

        notes: list[str] = []
        if missing:
            notes.append(f"missing data for: {', '.join(missing)}")
        if confirmed:
            notes.append(f"cross-asset confirmed (score={score:.2f})")
        elif contradicted:
            notes.append(f"cross-asset contradicted (score={score:.2f})")
        else:
            notes.append(f"cross-asset neutral (score={score:.2f})")

        return CrossAssetResult(
            confirmed=confirmed,
            contradicted=contradicted,
            neutral=neutral,
            score=score,
            confirmations=confirmations,
            contradictions=contradictions,
            missing_assets=missing,
            notes=notes,
        )


__all__ = ["CrossAssetConfirmation", "CrossAssetResult", "DEFAULT_RELATIONSHIPS"]
