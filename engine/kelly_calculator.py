"""engine.kelly_calculator
=====================================================================
Fractional Kelly (1/8) position sizing with dynamic caps.

Mathematically optimal position sizing based on win rate and
reward/risk ratio. Superior to fixed percentage because it:
  - Sizes UP when edge is strong (high win rate + good RR)
  - Sizes DOWN when edge is weak (low win rate or consecutive losses)
  - Never risks more than the cap allows

Caps by open positions (anti-concentration):
  0 positions → 5% max
  1 position  → 3% max
  2 positions  → 2% max

Override cap (capital protection):
  win_rate < 40% OR 3 consecutive losses → 1% max

TRENDING_STRONG regime: multiply by 1.2 (up to cap)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

log = get_logger("engine.kelly")


@dataclass
class KellyResult:
    kelly_fraction: float       # raw Kelly f* (can be > 1 or < 0)
    fractional_kelly: float     # 1/8 Kelly (always >= 0)
    position_pct: float         # capped per open-position rules
    position_usdt: float        # in USDT given capital
    risk_usdt: float            # expected risk (position × SL distance)
    reduced: bool = False       # True when override cap applied
    method: str = "kelly"       # "kelly" or "risk_parity_fallback"

    def to_dict(self) -> dict:
        return {
            "kelly_fraction": round(self.kelly_fraction, 4),
            "fractional_kelly": round(self.fractional_kelly, 4),
            "position_pct": round(self.position_pct, 4),
            "position_usdt": round(self.position_usdt, 2),
            "risk_usdt": round(self.risk_usdt, 2),
            "reduced": self.reduced,
            "method": self.method,
        }


# ----------------------------------------------------------------------
class KellyCalculator:
    """Fractional Kelly position sizing with dynamic caps."""

    KELLY_FRACTION = 1 / 8      # Use 1/8 of full Kelly (conservative)

    # Position caps by number of open positions
    CAPS_BY_OPEN = {0: 0.05, 1: 0.03, 2: 0.02}
    OVERRIDE_CAP = 0.01         # 1% when conditions are weak
    TRENDING_STRONG_MULT = 1.2  # Boost in strong trend regime

    def calculate(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        capital_usdt: float,
        sl_distance_pct: float,
        open_positions: int = 0,
        consecutive_losses: int = 0,
        trending_strong: bool = False,
        max_pct_override: Optional[float] = None,
    ) -> KellyResult:
        """Compute Kelly-optimal position size.

        Args:
            win_rate: historical win rate (0-1, exclusive)
            avg_win_pct: average winning trade size as fraction (e.g. 0.05 = +5%)
            avg_loss_pct: average losing trade size as fraction (e.g. 0.03 = -3%)
            capital_usdt: current capital in USDT
            sl_distance_pct: stop-loss distance as fraction (e.g. 0.02 = 2%)
            open_positions: current number of open positions
            consecutive_losses: current consecutive loss streak
            trending_strong: whether the market is in a strong trend
            max_pct_override: optional hard cap (overrides CAPS_BY_OPEN)

        Returns:
            KellyResult with position sizing
        """
        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            log.warning("Kelly: win_rate %.2f out of range, falling back to risk parity", win_rate)
            return self._risk_parity_fallback(capital_usdt, sl_distance_pct, max_pct_override)
        if avg_win_pct <= 0 or avg_loss_pct <= 0:
            log.warning("Kelly: avg_win/avg_loss must be positive, falling back")
            return self._risk_parity_fallback(capital_usdt, sl_distance_pct, max_pct_override)

        # Kelly formula: f* = (b × W - L) / b
        # where b = avg_win / avg_loss, W = win_rate, L = loss_rate
        loss_rate = 1.0 - win_rate
        b = avg_win_pct / avg_loss_pct
        kelly_f = max(0.0, (b * win_rate - loss_rate) / b)
        fractional = kelly_f * self.KELLY_FRACTION

        # Determine cap
        override = win_rate < 0.40 or consecutive_losses >= 3
        reduced = False
        if override:
            cap = self.OVERRIDE_CAP
            reduced = True
        elif max_pct_override is not None:
            cap = float(max_pct_override)
        else:
            cap = self.CAPS_BY_OPEN.get(open_positions, 0.02)

        position_pct = min(fractional, cap)

        # TRENDING_STRONG boost (bounded by cap)
        if trending_strong and not override:
            position_pct = min(position_pct * self.TRENDING_STRONG_MULT, cap)

        # Convert to USDT
        position_usdt = capital_usdt * position_pct
        risk_usdt = position_usdt * sl_distance_pct

        return KellyResult(
            kelly_fraction=round(kelly_f, 4),
            fractional_kelly=round(fractional, 4),
            position_pct=round(position_pct, 4),
            position_usdt=round(position_usdt, 2),
            risk_usdt=round(risk_usdt, 2),
            reduced=reduced,
            method="kelly",
        )

    # ----------------------------------------------------------------
    def _risk_parity_fallback(self, capital: float,
                                sl_distance: float,
                                max_pct: Optional[float]) -> KellyResult:
        """Fallback: risk exactly 1% of capital (risk parity)."""
        risk_pct = 0.01
        cap = max_pct or 0.05
        risk_usdt = capital * risk_pct
        position_usdt = risk_usdt / max(sl_distance, 1e-9) if sl_distance > 0 else 0
        position_pct = min(position_usdt / max(capital, 1e-9), cap)
        return KellyResult(
            kelly_fraction=0.0,
            fractional_kelly=0.0,
            position_pct=round(position_pct, 4),
            position_usdt=round(position_usdt, 2),
            risk_usdt=round(risk_usdt, 2),
            reduced=True,
            method="risk_parity_fallback",
        )
