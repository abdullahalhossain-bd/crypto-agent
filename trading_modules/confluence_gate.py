"""
Confluence Gate Module — Multi-Factor Entry Confirmation
==========================================================

Implements the professional confluence checklist:
  1. MTF Trend alignment (H4 + H1 same direction)
  2. Key Zone proximity (Order Block / Demand / Supply / FVG)
  3. Liquidity Sweep detection (stop hunt before reversal)
  4. Candlestick Pattern (final trigger, NOT the primary signal)
  5. Volume confirmation (high volume = strong, low = weak)
  6. RSI confirmation (oversold for BUY, overbought for SELL)
  7. Structure break (BOS for continuation, CHoCH for reversal)
  8. Candle closed (wait for close, no early entry)

The gate enforces ALL checks must pass. If any single check fails,
the signal is WAIT, not EXECUTE.

Inspired by: Professional discretionary trading methodology
Validated by: Orallexa evaluation report (regime_ensemble FAIL proves
              that more indicators ≠ better performance)

Usage:
    from confluence_gate import ConfluenceGate, ConfluenceInput

    gate = ConfluenceGate()
    result = gate.check(ConfluenceInput(
        symbol="BTCUSDT",
        direction="BUY",
        mtf_trend={"H4": "bullish", "H1": "bullish", "M15": "pullback"},
        at_key_zone=True,
        liquidity_sweep=True,
        pattern="Bullish Engulfing",
        pattern_rating=5,  # 1-5 stars
        volume_ratio=2.1,  # 2.1x average volume
        rsi=28.0,
        structure_break="BOS",
        candle_closed=True,
    ))

    if result.signal == "EXECUTE":
        print(f"Entry approved! Confluence score: {result.score:.0%}")
    else:
        print(f"Waiting. Failed checks: {result.failed_checks}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfluenceInput:
    """Input for the confluence gate check."""
    symbol: str
    direction: str  # "BUY" or "SELL"

    # Step 1: MTF Trend
    mtf_trend: dict = field(default_factory=dict)
    # Example: {"H4": "bullish", "H1": "bullish", "M15": "pullback"}

    # Step 2: Key Zone proximity
    at_key_zone: bool = False
    zone_type: str = ""  # "order_block" / "demand" / "supply" / "FVG" / "support" / "resistance"

    # Step 3: Liquidity Sweep
    liquidity_sweep: bool = False
    sweep_direction: str = ""  # "downward" (stop hunt below) or "upward" (stop hunt above)

    # Step 4: Candlestick Pattern (FINAL TRIGGER)
    pattern: str = ""
    pattern_rating: int = 0  # 1-5 stars (5 = strongest)

    # Step 5: Volume
    volume_ratio: float = 1.0  # current volume / average volume (>1.5 = high)
    volume_threshold: float = 1.5  # minimum volume ratio for confirmation

    # Step 6: RSI
    rsi: float = 50.0
    rsi_oversold: float = 35.0  # BUY threshold
    rsi_overbought: float = 65.0  # SELL threshold

    # Step 7: Structure break
    structure_break: str = ""  # "BOS" / "CHoCH" / ""

    # Step 8: Candle timing
    candle_closed: bool = False


@dataclass
class ConfluenceResult:
    """Result of the confluence gate check."""
    signal: str = "WAIT"  # "EXECUTE" or "WAIT"
    score: float = 0.0  # 0.0 to 1.0 (percentage of checks passed)
    checks: dict = field(default_factory=dict)
    failed_checks: list = field(default_factory=list)
    passed_checks: list = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "score": round(self.score, 4),
            "checks": self.checks,
            "failed_checks": self.failed_checks,
            "passed_checks": self.passed_checks,
            "recommendation": self.recommendation,
        }


class ConfluenceGate:
    """
    Multi-factor confluence gate for trade entry confirmation.

    Implements the professional trading principle:
    "Candlestick patterns are the FINAL TRIGGER, not the primary signal."

    The gate enforces ALL checks must pass before EXECUTE signal.
    Any single failure → WAIT.
    """

    def __init__(self, require_all: bool = True, min_score: float = 1.0):
        """
        Args:
            require_all: If True, ALL checks must pass for EXECUTE.
                        If False, score >= min_score is sufficient.
            min_score: Minimum confluence score when require_all=False.
        """
        self.require_all = require_all
        self.min_score = min_score

    def check(self, inp: ConfluenceInput) -> ConfluenceResult:
        """
        Run all confluence checks.

        Returns ConfluenceResult with signal=EXECUTE only if all checks pass.
        """
        result = ConfluenceResult()

        # Run each check
        result.checks = {
            "mtf_trend": self._check_mtf_trend(inp),
            "at_key_zone": self._check_key_zone(inp),
            "liquidity_sweep": self._check_liquidity_sweep(inp),
            "pattern": self._check_pattern(inp),
            "volume": self._check_volume(inp),
            "rsi": self._check_rsi(inp),
            "structure_break": self._check_structure(inp),
            "candle_closed": inp.candle_closed,
        }

        # Classify passed/failed
        for name, passed in result.checks.items():
            if passed:
                result.passed_checks.append(name)
            else:
                result.failed_checks.append(name)

        # Compute score
        total = len(result.checks)
        passed = len(result.passed_checks)
        result.score = passed / total if total > 0 else 0.0

        # Determine signal
        if self.require_all:
            result.signal = "EXECUTE" if len(result.failed_checks) == 0 else "WAIT"
        else:
            result.signal = "EXECUTE" if result.score >= self.min_score else "WAIT"

        # Generate recommendation
        result.recommendation = self._generate_recommendation(inp, result)

        return result

    def _check_mtf_trend(self, inp: ConfluenceInput) -> bool:
        """
        Step 1: Multi-timeframe trend alignment.

        For BUY: H4 and H1 must both be bullish
        For SELL: H4 and H1 must both be bearish
        """
        mtf = inp.mtf_trend
        if not mtf:
            return False

        h4 = str(mtf.get("H4", "")).lower()
        h1 = str(mtf.get("H1", "")).lower()

        if inp.direction == "BUY":
            return h4 in ("bullish", "up") and h1 in ("bullish", "up")
        elif inp.direction == "SELL":
            return h4 in ("bearish", "down") and h1 in ("bearish", "down")
        return False

    def _check_key_zone(self, inp: ConfluenceInput) -> bool:
        """
        Step 2: Price at a key zone.

        BUY: at demand zone / bullish order block / bullish FVG / support
        SELL: at supply zone / bearish order block / bearish FVG / resistance
        """
        if not inp.at_key_zone:
            return False

        zone = inp.zone_type.lower()
        if inp.direction == "BUY":
            return zone in ("demand", "order_block", "fvg", "support", "bullish_ob")
        elif inp.direction == "SELL":
            return zone in ("supply", "order_block", "fvg", "resistance", "bearish_ob")
        return False

    def _check_liquidity_sweep(self, inp: ConfluenceInput) -> bool:
        """
        Step 3: Liquidity sweep (stop hunt) detected.

        BUY: price dipped below previous low (collected sell stops), then reversed up
        SELL: price spiked above previous high (collected buy stops), then reversed down
        """
        if not inp.liquidity_sweep:
            return False

        sweep = inp.sweep_direction.lower()
        if inp.direction == "BUY":
            return sweep in ("downward", "down", "below")
        elif inp.direction == "SELL":
            return sweep in ("upward", "up", "above")
        return False

    def _check_pattern(self, inp: ConfluenceInput) -> bool:
        """
        Step 4: Candlestick pattern detected (FINAL TRIGGER).

        Must have a pattern AND minimum 3-star rating.
        Higher-rated patterns (4-5 stars) are strongly preferred.
        """
        if not inp.pattern:
            return False
        return inp.pattern_rating >= 3

    def _check_volume(self, inp: ConfluenceInput) -> bool:
        """
        Step 5: Volume confirmation.

        Volume ratio must be above threshold (default 1.5x average).
        Low volume patterns are weak and should be ignored.
        """
        return inp.volume_ratio >= inp.volume_threshold

    def _check_rsi(self, inp: ConfluenceInput) -> bool:
        """
        Step 6: RSI confirmation.

        BUY: RSI should be oversold (< 35) — momentum has room to go up
        SELL: RSI should be overbought (> 65) — momentum has room to go down

        Note: In strong trends, RSI may not reach extremes.
        The 35/65 thresholds are more flexible than the classic 30/70.
        """
        if inp.direction == "BUY":
            return inp.rsi <= inp.rsi_oversold
        elif inp.direction == "SELL":
            return inp.rsi >= inp.rsi_overbought
        return False

    def _check_structure(self, inp: ConfluenceInput) -> bool:
        """
        Step 7: Market structure break.

        BOS (Break of Structure) = trend continuation
        CHoCH (Change of Character) = trend reversal

        For BUY in pullback: BOS confirms continuation
        For BUY at reversal: CHoCH confirms reversal
        """
        if not inp.structure_break:
            return False
        return inp.structure_break.upper() in ("BOS", "CHOCH")

    def _generate_recommendation(self, inp: ConfluenceInput, result: ConfluenceResult) -> str:
        """Generate a human-readable recommendation."""
        if result.signal == "EXECUTE":
            return (
                f"✅ EXECUTE {inp.direction} on {inp.symbol}. "
                f"All {len(result.passed_checks)} confluence checks passed. "
                f"Pattern: {inp.pattern} ({'⭐' * inp.pattern_rating}). "
                f"Enter after candle close."
            )

        # WAIT — explain why
        reasons = []
        if "mtf_trend" in result.failed_checks:
            reasons.append("MTF trend not aligned")
        if "at_key_zone" in result.failed_checks:
            reasons.append("Not at a key zone")
        if "liquidity_sweep" in result.failed_checks:
            reasons.append("No liquidity sweep detected")
        if "pattern" in result.failed_checks:
            reasons.append(f"No valid pattern (current: {inp.pattern or 'none'})")
        if "volume" in result.failed_checks:
            reasons.append(f"Volume too low ({inp.volume_ratio:.1f}x vs {inp.volume_threshold}x required)")
        if "rsi" in result.failed_checks:
            reasons.append(f"RSI not in range (current: {inp.rsi:.1f})")
        if "structure_break" in result.failed_checks:
            reasons.append("No BOS/CHoCH confirmation")
        if "candle_closed" in result.failed_checks:
            reasons.append("Candle not yet closed")

        return (
            f"⏸️ WAIT on {inp.symbol}. "
            f"Failed {len(result.failed_checks)}/{len(result.checks)} checks: {', '.join(reasons)}. "
            f"Confluence score: {result.score:.0%}"
        )


class WeightedConfluenceGate(ConfluenceGate):
    """
    Weighted confluence gate — each check has a weight.
    Uses Orallexa-style adaptive weighting.

    Instead of requiring ALL checks to pass, this gate computes a
    weighted score and requires it to exceed a threshold.
    """

    DEFAULT_WEIGHTS = {
        "mtf_trend": 0.20,        # 20% — most important
        "at_key_zone": 0.15,      # 15% — SMC zone
        "liquidity_sweep": 0.15,  # 15% — stop hunt
        "pattern": 0.15,          # 15% — candlestick (final trigger)
        "volume": 0.10,           # 10%
        "rsi": 0.10,              # 10%
        "structure_break": 0.10,  # 10% — BOS/CHoCH
        "candle_closed": 0.05,    # 5% — timing gate
    }

    def __init__(self, weights: Optional[dict] = None, min_score: float = 0.75):
        super().__init__(require_all=False, min_score=min_score)
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def check(self, inp: ConfluenceInput) -> ConfluenceResult:
        """Run weighted confluence check."""
        result = ConfluenceResult()

        # Run each check
        raw_checks = {
            "mtf_trend": self._check_mtf_trend(inp),
            "at_key_zone": self._check_key_zone(inp),
            "liquidity_sweep": self._check_liquidity_sweep(inp),
            "pattern": self._check_pattern(inp),
            "volume": self._check_volume(inp),
            "rsi": self._check_rsi(inp),
            "structure_break": self._check_structure(inp),
            "candle_closed": inp.candle_closed,
        }

        # Compute weighted score
        total_weight = sum(self.weights.values())
        weighted_score = 0.0
        result.checks = {}

        for name, passed in raw_checks.items():
            weight = self.weights.get(name, 0)
            result.checks[name] = passed
            if passed:
                weighted_score += weight
                result.passed_checks.append(name)
            else:
                result.failed_checks.append(name)

        result.score = weighted_score / total_weight if total_weight > 0 else 0.0
        result.signal = "EXECUTE" if result.score >= self.min_score else "WAIT"
        result.recommendation = self._generate_recommendation(inp, result)

        return result
