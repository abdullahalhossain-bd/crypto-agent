"""engine.candlestick.confluence_engine
=====================================================================
Day 129 — Confluence Engine.

The heart of The Candlestick Trading Bible approach. No single
indicator should make a trading decision. Instead, multiple factors
contribute to a weighted "confluence score" in [0, 100]:

  Factors (default weights):
    - Pattern confidence       20%
    - Trend alignment          15%
    - Support/Resistance       15%
    - Moving average           10%
    - Volume                   10%
    - Liquidity (spread proxy)  5%
    - Rejection strength       10%
    - Multi-timeframe          10%
    - ML score                 5%

Final confidence = Σ (factor_score × weight).

Trade decision logic (configurable):
  - confidence >= 75  → HIGH confidence (A+ setup)
  - confidence >= 60  → MEDIUM confidence (A setup)
  - confidence >= 45  → LOW confidence (B setup, require smaller size)
  - confidence < 45   → REJECT

Each factor can VETO the trade (e.g. trend alignment = 0 when trading
against a strong trend). Vetoes are recorded in the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("candlestick.confluence")


@dataclass
class ConfluenceFactor:
    name: str
    score: float                  # 0-100
    weight: float                 # 0-1
    veto: bool = False            # if True, force final score to 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "veto": self.veto,
            "reason": self.reason,
        }


@dataclass
class ConfluenceResult:
    final_confidence: float       # 0-100
    grade: str                    # A+ / A / B / C / REJECT
    factors: list[ConfluenceFactor] = field(default_factory=list)
    vetoed: bool = False
    veto_reason: str = ""
    direction: str = "neutral"    # bullish / bearish / neutral

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_confidence": self.final_confidence,
            "grade": self.grade,
            "factors": [f.to_dict() for f in self.factors],
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "direction": self.direction,
        }


# ----------------------------------------------------------------------
class ConfluenceEngine:
    DEFAULT_WEIGHTS: dict[str, float] = {
        "pattern":        0.20,
        "trend":          0.15,
        "support_resistance": 0.15,
        "moving_average": 0.10,
        "volume":         0.10,
        "liquidity":      0.05,
        "rejection":      0.10,
        "multi_timeframe": 0.10,
        "ml_score":       0.05,
    }

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "A_plus": 75.0,
        "A": 60.0,
        "B": 45.0,
        "C": 30.0,
    }

    def __init__(self,
                 weights: Optional[dict[str, float]] = None,
                 thresholds: Optional[dict[str, float]] = None) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        # Normalise weights to sum to 1
        # H4 fix: warn if custom weights don't sum to 1 — silent normalization
        # can hide configuration mistakes (e.g. a weight of 0.5 that was meant
        # to be 50% but gets normalized to 5% if other weights sum to 10).
        total = sum(self.weights.values())
        if total > 0:
            if weights is not None and abs(total - 1.0) > 0.01:
                log.warning("confluence_engine: custom weights sum to %.4f (not 1.0) — "
                            "normalizing silently. Original weights: %s",
                            total, weights)
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.thresholds = dict(thresholds or self.DEFAULT_THRESHOLDS)

    # ----------------------------------------------------------------
    def evaluate(
        self,
        pattern_score: float = 50.0,
        trend_score: float = 50.0,
        sr_score: float = 50.0,
        ma_score: float = 50.0,
        volume_score: float = 50.0,
        liquidity_score: float = 50.0,
        rejection_score: float = 50.0,
        mtf_score: float = 50.0,
        ml_score: float = 50.0,
        direction: str = "neutral",
        vetoes: Optional[list[str]] = None,
    ) -> ConfluenceResult:
        """Combine all factors into a final confluence score."""
        vetoes = vetoes or []
        factors = [
            ConfluenceFactor("pattern", pattern_score, self.weights["pattern"],
                              veto=("pattern" in vetoes)),
            ConfluenceFactor("trend", trend_score, self.weights["trend"],
                              veto=("trend" in vetoes)),
            ConfluenceFactor("support_resistance", sr_score,
                              self.weights["support_resistance"],
                              veto=("support_resistance" in vetoes)),
            ConfluenceFactor("moving_average", ma_score,
                              self.weights["moving_average"],
                              veto=("moving_average" in vetoes)),
            ConfluenceFactor("volume", volume_score, self.weights["volume"],
                              veto=("volume" in vetoes)),
            ConfluenceFactor("liquidity", liquidity_score,
                              self.weights["liquidity"],
                              veto=("liquidity" in vetoes)),
            ConfluenceFactor("rejection", rejection_score,
                              self.weights["rejection"],
                              veto=("rejection" in vetoes)),
            ConfluenceFactor("multi_timeframe", mtf_score,
                              self.weights["multi_timeframe"],
                              veto=("multi_timeframe" in vetoes)),
            ConfluenceFactor("ml_score", ml_score, self.weights["ml_score"],
                              veto=("ml_score" in vetoes)),
        ]
        # Check vetoes
        vetoed_factors = [f for f in factors if f.veto]
        if vetoed_factors:
            final = 0.0
            grade = "REJECT"
            vetoed = True
            veto_reason = "; ".join(f"{f.name} vetoed" for f in vetoed_factors)
        else:
            final = sum(f.score * f.weight for f in factors)
            grade = self._grade(final)
            vetoed = False
            veto_reason = ""
        return ConfluenceResult(
            final_confidence=float(final),
            grade=grade,
            factors=factors,
            vetoed=vetoed,
            veto_reason=veto_reason,
            direction=direction,
        )

    # ----------------------------------------------------------------
    def _grade(self, score: float) -> str:
        t = self.thresholds
        if score >= t["A_plus"]:
            return "A+"
        if score >= t["A"]:
            return "A"
        if score >= t["B"]:
            return "B"
        if score >= t["C"]:
            return "C"
        return "REJECT"

    # ----------------------------------------------------------------
    def recommend_action(self, result: ConfluenceResult) -> dict[str, Any]:
        """Translate the confluence result into a soft recommendation."""
        if result.vetoed or result.grade == "REJECT":
            return {
                "action": "HOLD",
                "reason": f"vetoed or below C threshold ({result.final_confidence:.1f})",
                "suggested_size_multiplier": 0.0,
            }
        if result.direction == "neutral":
            return {
                "action": "HOLD",
                "reason": "neutral direction",
                "suggested_size_multiplier": 0.0,
            }
        # Size scales with grade
        size_mult = {
            "A+": 1.0,
            "A": 0.75,
            "B": 0.50,
            "C": 0.25,
        }.get(result.grade, 0.0)
        action = "BUY" if result.direction == "bullish" else "SELL"
        return {
            "action": action,
            "reason": f"grade={result.grade} conf={result.final_confidence:.1f}",
            "suggested_size_multiplier": float(size_mult),
        }
