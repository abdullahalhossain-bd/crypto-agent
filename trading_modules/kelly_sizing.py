"""
Kelly Sizing Module — Position Sizing with Drawdown Protection
================================================================

Implements Kelly Criterion position sizing with safety modifications:
  - Half-Kelly by default (full Kelly is too aggressive under uncertainty)
  - Drawdown-adjusted variant (scales to 0 at max drawdown)
  - Hard cap on any single position (25% of account max)

Philosophy:
  "Full Kelly (1.0) is mathematically optimal ONLY when p_win is known
  exactly; under uncertainty it courts ruin. 0.5f* gives ~75% of the
  growth rate at dramatically lower drawdown risk."

  — Thorp (1997), Ed Miller (2018), Robot Wealth (2026)

Source: Orallexa (review #27) — kelly_sizing.py (220 LOC)
License: MIT

Usage:
    from kelly_sizing import kelly_fraction, kelly_position_size

    # Compute Kelly fraction
    f = kelly_fraction(p_win=0.60, avg_win_pct=0.05, avg_loss_pct=0.03)
    print(f"Full Kelly: {f:.2%}")  # e.g., 0.3333

    # Half-Kelly with drawdown adjustment
    f_adjusted = kelly_position_size(
        p_win=0.60,
        avg_win_pct=0.05,
        avg_loss_pct=0.03,
        account_equity=10000,
        current_drawdown_pct=8.0,
        kelly_fraction=0.5,  # Half-Kelly
    )
    print(f"Position size: ${f_adjusted['position_usd']:.2f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Professional-trader consensus default. Full Kelly (1.0) is
# mathematically optimal ONLY when p_win is known exactly; under
# uncertainty it courts ruin. 0.5f* gives ~75% of the growth rate
# at dramatically lower drawdown risk.
DEFAULT_KELLY_FRACTION = 0.5

# Absolute upper bound on any Kelly recommendation regardless of what
# the math says. A single trade should never risk more than 25% of
# the account in a paper stack that hasn't graduated to real money.
MAX_KELLY_FRACTION_CAP = 0.25

# Drawdown at which position size scales to zero
MAX_DRAWDOWN_FOR_SIZING = 0.15  # 15% drawdown = zero position


@dataclass
class KellyResult:
    """Result of Kelly position sizing calculation."""
    full_kelly: float          # f* = (p*b - q) / b
    adjusted_kelly: float      # After fraction and cap
    position_usd: float        # Dollar position size
    position_pct: float        # Percentage of account
    drawdown_factor: float     # Drawdown adjustment multiplier
    warnings: list[str]        # List of warning messages

    def to_dict(self) -> dict:
        return {
            "full_kelly": round(self.full_kelly, 6),
            "adjusted_kelly": round(self.adjusted_kelly, 6),
            "position_usd": round(self.position_usd, 2),
            "position_pct": round(self.position_pct, 4),
            "drawdown_factor": round(self.drawdown_factor, 4),
            "warnings": self.warnings,
        }


def full_kelly_fraction(
    p_win: float,
    avg_win_pct: float,
    avg_loss_pct: float,
) -> float:
    """
    Return the FULL Kelly fraction f* for a bet.

    f* = (p * b - q) / b, where b = avg_win / avg_loss, q = 1 - p.

    Returns 0 when the edge is negative or nonexistent (never a short
    recommendation — just "don't bet").

    Args:
        p_win: Probability of winning (0 to 1)
        avg_win_pct: Average profit on win as positive fraction (0.05 = 5% up)
        avg_loss_pct: Average loss on loss as positive fraction (0.03 = 3% down)

    Returns:
        Full Kelly fraction f* (0 to large positive, but usually < 1)
    """
    if p_win <= 0 or p_win >= 1:
        return 0.0
    if avg_win_pct <= 0 or avg_loss_pct <= 0:
        return 0.0

    b = avg_win_pct / avg_loss_pct  # Win/loss ratio
    q = 1.0 - p_win

    f = (p_win * b - q) / b

    # Never recommend negative (no short sizing — just "don't trade")
    return max(0.0, f)


def drawdown_factor(current_drawdown_pct: float, max_dd: float = 0.15) -> float:
    """
    Compute drawdown adjustment factor.

    Scales linearly from 1.0 (no drawdown) to 0.0 (at max drawdown).

    At 0% DD → factor = 1.0 (full position)
    At 7.5% DD → factor = 0.5 (half position)
    At 15% DD → factor = 0.0 (no position — stop trading)
    At 20% DD → factor = 0.0 (clamped)
    """
    if current_drawdown_pct <= 0:
        return 1.0
    if current_drawdown_pct >= max_dd:
        return 0.0

    # Linear scale: factor = 1 - (dd / max_dd)
    return 1.0 - (current_drawdown_pct / max_dd)


def kelly_position_size(
    p_win: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    account_equity: float,
    current_drawdown_pct: float = 0.0,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    max_kelly_cap: float = MAX_KELLY_FRACTION_CAP,
    max_drawdown_pct: float = 15.0,
) -> KellyResult:
    """
    Compute Kelly-optimal position size with all safety modifications.

    Pipeline:
    1. Compute full Kelly f* = (p*b - q) / b
    2. Scale by kelly_fraction (default 0.5 = Half-Kelly)
    3. Apply drawdown adjustment (scales to 0 at max_drawdown_pct)
    4. Apply hard cap (max_kelly_cap, default 25%)
    5. Compute dollar position size

    Args:
        p_win: Probability of winning (0 to 1)
        avg_win_pct: Average profit on win (0.05 = 5%)
        avg_loss_pct: Average loss on loss (0.03 = 3%)
        account_equity: Current account equity in USD
        current_drawdown_pct: Current drawdown percentage
        kelly_fraction: Fraction of full Kelly to use (0.5 = Half-Kelly)
        max_kelly_cap: Hard cap on position as fraction of account
        max_drawdown_pct: Drawdown at which position scales to zero

    Returns:
        KellyResult with all calculations and warnings
    """
    warnings = []

    # Step 1: Full Kelly
    f_star = full_kelly_fraction(p_win, avg_win_pct, avg_loss_pct)

    if f_star == 0.0:
        return KellyResult(
            full_kelly=0.0,
            adjusted_kelly=0.0,
            position_usd=0.0,
            position_pct=0.0,
            drawdown_factor=0.0,
            warnings=["No edge detected — do not trade"],
        )

    # Step 2: Scale by Kelly fraction (Half-Kelly default)
    f_adjusted = f_star * kelly_fraction

    # Step 3: Drawdown adjustment
    dd_factor = drawdown_factor(current_drawdown_pct, max_drawdown_pct)
    f_adjusted *= dd_factor

    if dd_factor < 0.3:
        warnings.append(
            f"Severe drawdown ({current_drawdown_pct:.1%}) — position reduced to {dd_factor:.0%}"
        )
    if dd_factor == 0.0:
        return KellyResult(
            full_kelly=f_star,
            adjusted_kelly=0.0,
            position_usd=0.0,
            position_pct=0.0,
            drawdown_factor=0.0,
            warnings=[f"Drawdown {current_drawdown_pct:.1%} >= max {max_drawdown_pct:.0%} — stop trading"],
        )

    # Step 4: Hard cap
    f_adjusted = min(f_adjusted, max_kelly_cap)

    if f_star * kelly_fraction > max_kelly_cap:
        warnings.append(
            f"Kelly recommends {f_star * kelly_fraction:.1%} but capped at {max_kelly_cap:.1%}"
        )

    # Step 5: Compute dollar position
    position_usd = f_adjusted * account_equity

    # Sanity checks
    if f_adjusted > 1.0:
        warnings.append(f"⚠️ Kelly fraction {f_adjusted:.1%} > 100% — check inputs")
    if p_win < 0.50:
        warnings.append(f"⚠️ Win probability {p_win:.1%} < 50% — low confidence trade")

    return KellyResult(
        full_kelly=f_star,
        adjusted_kelly=f_adjusted,
        position_usd=position_usd,
        position_pct=f_adjusted,
        drawdown_factor=dd_factor,
        warnings=warnings,
    )


def r_multiple_position_size(
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_usd: float,
    r_multiples: list[float] = None,
) -> dict:
    """
    Compute R-multiple based position sizing.

    R = risk per trade = |entry - stop| × position_size

    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        risk_per_trade_usd: Dollar amount to risk (e.g., 1% of account)
        r_multiples: Take-profit targets in R multiples (default: [2, 3, 5])

    Returns:
        Dict with position size, R value, and take-profit levels
    """
    if r_multiples is None:
        r_multiples = [2.0, 3.0, 5.0]

    risk_per_unit = abs(entry_price - stop_loss_price)
    if risk_per_unit == 0:
        return {"error": "Entry and stop loss cannot be the same"}

    position_size = risk_per_trade_usd / risk_per_unit
    is_long = entry_price > stop_loss_price

    # Compute take-profit levels
    take_profits = []
    for r in r_multiples:
        if is_long:
            tp = entry_price + (risk_per_unit * r)
        else:
            tp = entry_price - (risk_per_unit * r)
        take_profits.append({
            "r_multiple": r,
            "price": tp,
            "pct_from_entry": abs(tp - entry_price) / entry_price,
        })

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "position_size": position_size,
        "risk_per_unit": risk_per_unit,
        "risk_per_trade_usd": risk_per_trade_usd,
        "is_long": is_long,
        "take_profits": take_profits,
        "breakeven_move_after": r_multiples[0],  # Move SL to BE after first TP
    }
