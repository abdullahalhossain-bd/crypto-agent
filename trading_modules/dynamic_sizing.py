"""
Dynamic Position Sizing — "Bigger bets on A+ setups, tiny bets on B setups"
=============================================================================

A naive bot trades the same size every time. An institutional trader
scales risk with conviction:

    A+ setup (score >= 90) → 1.5x base risk
    A  setup (score >= 80) → 1.0x base risk
    B  setup (score >= 70) → 0.5x base risk
    C  setup (score < 70)  → skip entirely

This module wraps the Kelly position sizing logic and adjusts the
result by a grade-based multiplier. It also enforces hard caps:

    - Max absolute USD risk per trade
    - Max % of equity per trade
    - Min USD risk (avoid dust positions)

Usage:
    from trading_modules.dynamic_sizing import DynamicPositionSizer, SizingInput

    sizer = DynamicPositionSizer(initial_capital=10000)
    result = sizer.size(
        grade="A+",
        score=92,
        win_probability=0.62,
        avg_win_pct=0.03,
        avg_loss_pct=0.02,
        current_equity=10500,
        current_drawdown_pct=2.5,
    )

    if result.skip:
        log.info(f"Skip: {result.reason}")
    else:
        risk_usd = result.risk_usd  # → use this for SL distance / lot sizing
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .kelly_sizing import kelly_position_size, KellyResult

log = logging.getLogger(__name__)


@dataclass
class SizingInput:
    grade: str                      # "A+" / "A" / "B" / "C" / "F"
    score: float                    # 0..100 from the gate
    win_probability: float          # 0..1
    avg_win_pct: float              # e.g., 0.03 = 3%
    avg_loss_pct: float             # e.g., 0.02 = 2%
    current_equity: float
    current_drawdown_pct: float = 0.0


@dataclass
class SizingResult:
    skip: bool
    reason: Optional[str]
    risk_usd: float                 # dollar amount to risk
    kelly_fraction: float           # raw Kelly (before grade scaling)
    grade_multiplier: float         # 0.5 / 1.0 / 1.5 / 0
    position_usd: float             # notional position size
    pct_of_equity: float            # risk / equity

    def to_dict(self) -> dict:
        return {
            "skip": self.skip,
            "reason": self.reason,
            "risk_usd": round(self.risk_usd, 2),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "grade_multiplier": self.grade_multiplier,
            "position_usd": round(self.position_usd, 2),
            "pct_of_equity": round(self.pct_of_equity, 4),
        }


# ----------------------------------------------------------------------
# Grade → multiplier table
# ----------------------------------------------------------------------
GRADE_MULTIPLIERS: dict[str, float] = {
    "A+": 1.5,
    "A":  1.0,
    "B":  0.5,
    "C":  0.0,   # skip
    "F":  0.0,   # skip
}


class DynamicPositionSizer:
    """
    Combine Kelly position sizing with grade-based scaling.

    Parameters:
        initial_capital: for percentage calculations
        max_risk_pct: hard cap — never risk more than this % of equity per trade (default 3%)
        min_risk_usd: skip if computed risk is below this (avoid dust, default $10)
        kelly_fraction: Kelly fraction (0.5 = half-Kelly, default 0.5)
        max_drawdown_pct: halt new trades if drawdown exceeds this (default 15%)
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_risk_pct: float = 0.03,
        min_risk_usd: float = 10.0,
        kelly_fraction: float = 0.5,
        max_drawdown_pct: float = 0.15,
    ) -> None:
        self.initial_capital = initial_capital
        self.max_risk_pct = max_risk_pct
        self.min_risk_usd = min_risk_usd
        self.kelly_fraction = kelly_fraction
        self.max_drawdown_pct = max_drawdown_pct

    def size(self, inp: SizingInput) -> SizingResult:
        """Compute dynamic position size."""
        # 1. Skip if grade is C or F
        multiplier = GRADE_MULTIPLIERS.get(inp.grade, 0.0)
        if multiplier <= 0:
            return SizingResult(
                skip=True, reason=f"Grade {inp.grade} → skip",
                risk_usd=0, kelly_fraction=0, grade_multiplier=0,
                position_usd=0, pct_of_equity=0,
            )

        # 2. Skip if drawdown too deep
        if inp.current_drawdown_pct >= self.max_drawdown_pct:
            return SizingResult(
                skip=True,
                reason=(f"Drawdown {inp.current_drawdown_pct:.1%} >= "
                        f"max {self.max_drawdown_pct:.0%}"),
                risk_usd=0, kelly_fraction=0, grade_multiplier=0,
                position_usd=0, pct_of_equity=0,
            )

        # 3. Skip if equity depleted
        if inp.current_equity <= 0:
            return SizingResult(
                skip=True, reason="Equity depleted",
                risk_usd=0, kelly_fraction=0, grade_multiplier=0,
                position_usd=0, pct_of_equity=0,
            )

        # 4. Compute Kelly fraction
        kelly_res: KellyResult = kelly_position_size(
            p_win=inp.win_probability,
            avg_win_pct=inp.avg_win_pct,
            avg_loss_pct=inp.avg_loss_pct,
            account_equity=inp.current_equity,
            current_drawdown_pct=inp.current_drawdown_pct,
            kelly_fraction=self.kelly_fraction,
        )

        if kelly_res.position_usd <= 0 or kelly_res.kelly_fraction <= 0:
            return SizingResult(
                skip=True,
                reason=(f"No Kelly edge (kelly={kelly_res.kelly_fraction:.4f}, "
                        f"win_prob={inp.win_probability:.2f})"),
                risk_usd=0, kelly_fraction=float(kelly_res.kelly_fraction),
                grade_multiplier=multiplier, position_usd=0, pct_of_equity=0,
            )

        # 5. Apply grade multiplier
        risk_usd = kelly_res.position_usd * multiplier

        # 6. Cap at max_risk_pct of equity
        max_risk_usd = inp.current_equity * self.max_risk_pct
        if risk_usd > max_risk_usd:
            risk_usd = max_risk_usd

        # 7. Skip if below dust threshold
        if risk_usd < self.min_risk_usd:
            return SizingResult(
                skip=True,
                reason=f"Risk ${risk_usd:.2f} below min ${self.min_risk_usd:.2f}",
                risk_usd=0, kelly_fraction=float(kelly_res.kelly_fraction),
                grade_multiplier=multiplier, position_usd=0, pct_of_equity=0,
            )

        # 8. Compute notional position (assuming avg_loss_pct is the stop distance)
        # If avg_loss_pct = 0.02 (2% stop), then position = risk / 0.02
        if inp.avg_loss_pct > 0:
            position_usd = risk_usd / inp.avg_loss_pct
        else:
            position_usd = risk_usd * 50  # fallback 50x leverage assumption

        return SizingResult(
            skip=False, reason=None,
            risk_usd=float(risk_usd),
            kelly_fraction=float(kelly_res.kelly_fraction),
            grade_multiplier=float(multiplier),
            position_usd=float(position_usd),
            pct_of_equity=float(risk_usd / inp.current_equity),
        )
