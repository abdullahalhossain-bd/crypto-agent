"""engine.oco_calculator
=====================================================================
OCO (One-Cancels-Other) exit structure with fee-adjusted net gain.

Structured take-profit ladder:
  TP1 = entry × 1.025  → close 50% qty  (+2.5%)
  TP2 = entry × 1.052  → close 30% qty  (+5.2%)
  TP3 = trailing ATR   → close 20% qty  (open-ended)
  SL  = entry - 1×ATR  → cut entire remaining position

All parameters are configurable. Net expected gain includes fees
(0.1% per leg by default).

Inspired by Centina-Quant's OCOCalculator. Adapted to fit our
position manager and trade quality scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

log = get_logger("engine.oco")


@dataclass
class OCOLevels:
    entry: float
    sl: float               # stop-loss
    tp1: float              # take-profit 1 (close 50%)
    tp2: float              # take-profit 2 (close 30%)
    tp3_trail_atr: float    # ATR distance for trailing stop on remaining 20%
    sl_pct: float           # SL distance as % of entry
    tp1_pct: float          # TP1 gain as % of entry
    tp2_pct: float          # TP2 gain as % of entry
    tp1_rr: float           # reward/risk at TP1
    tp2_rr: float           # reward/risk at TP2
    qty_tp1: float          # 50% closed at TP1
    qty_tp2: float          # 30% closed at TP2
    qty_tp3: float          # 20% trailed to TP3
    net_gain_pct: float     # weighted net expected gain (fees included)
    side: str = "long"      # "long" or "short"

    def to_dict(self) -> dict:
        return {
            "entry": self.entry, "sl": self.sl,
            "tp1": self.tp1, "tp2": self.tp2,
            "tp3_trail_atr": self.tp3_trail_atr,
            "sl_pct": self.sl_pct, "tp1_pct": self.tp1_pct,
            "tp2_pct": self.tp2_pct,
            "tp1_rr": self.tp1_rr, "tp2_rr": self.tp2_rr,
            "qty_tp1": self.qty_tp1, "qty_tp2": self.qty_tp2,
            "qty_tp3": self.qty_tp3,
            "net_gain_pct": self.net_gain_pct,
            "side": self.side,
        }


# ----------------------------------------------------------------------
class OCOCalculator:
    """Structured OCO exit with configurable levels."""

    TP1_PCT = 0.025      # +2.5%
    TP2_PCT = 0.052      # +5.2%
    SL_MULT = 1.0        # ATR multiplier for SL
    TRAIL_MULT = 1.5     # ATR multiplier for trailing stop

    QTY_TP1 = 0.50       # 50% at TP1
    QTY_TP2 = 0.30       # 30% at TP2
    QTY_TP3 = 0.20       # 20% trailing

    FEE = 0.001          # 0.1% per leg

    def __init__(self,
                 tp1_pct: float = TP1_PCT,
                 tp2_pct: float = TP2_PCT,
                 sl_mult: float = SL_MULT,
                 trail_mult: float = TRAIL_MULT,
                 fee: float = FEE) -> None:
        self.tp1_pct = float(tp1_pct)
        self.tp2_pct = float(tp2_pct)
        self.sl_mult = float(sl_mult)
        self.trail_mult = float(trail_mult)
        self.fee = float(fee)

    # ----------------------------------------------------------------
    def calculate(self, entry: float, atr: float,
                    side: str = "long") -> OCOLevels:
        """Compute OCO levels for a trade.

        Args:
            entry: entry price
            atr: current ATR value
            side: "long" or "short"

        Returns:
            OCOLevels with all TP/SL levels + net gain
        """
        if atr <= 0:
            raise ValueError("ATR must be positive")
        if entry <= 0:
            raise ValueError("Entry price must be positive")
        # Minor #11 fix: validate side — an invalid side string would
        # silently default to short (because `direction = 1.0 if ... else -1.0`),
        # producing incorrect SL/TP levels for what should be a long trade.
        side_lower = side.lower().strip()
        if side_lower not in ("long", "short", "buy", "sell"):
            raise ValueError(f"side must be 'long'/'buy' or 'short'/'sell', got {side!r}")
        direction = 1.0 if side_lower in ("long", "buy") else -1.0

        sl = entry - direction * self.sl_mult * atr
        tp1 = entry * (1 + direction * self.tp1_pct)
        tp2 = entry * (1 + direction * self.tp2_pct)

        sl_dist = abs(entry - sl)
        sl_pct = sl_dist / entry * 100

        tp1_dist = abs(tp1 - entry)
        tp2_dist = abs(tp2 - entry)
        tp1_pct_val = tp1_dist / entry * 100
        tp2_pct_val = tp2_dist / entry * 100

        tp1_rr = tp1_dist / sl_dist if sl_dist > 0 else 0
        tp2_rr = tp2_dist / sl_dist if sl_dist > 0 else 0

        # Net expected gain (fees included)
        # H19 fix: TP3 estimate is now ATR-based instead of a fixed 1.3× TP2
        # multiplier. The trailing stop at `trail_mult × ATR` captures roughly
        # (trail_mult - sl_mult) × ATR of directional movement before the
        # trailing stop is hit, which is a more realistic estimate of the
        # runner's expected gain than an arbitrary multiplier on TP2.
        tp3_atr_gain_pct = (self.trail_mult - self.sl_mult) * atr / entry
        tp3_est_pct = max(tp3_atr_gain_pct, self.tp2_pct * 1.1)  # floor at 1.1× TP2
        net_gain = (
            self.QTY_TP1 * (self.tp1_pct - 2 * self.fee) +
            self.QTY_TP2 * (self.tp2_pct - 2 * self.fee) +
            self.QTY_TP3 * (tp3_est_pct - 2 * self.fee)
        )

        return OCOLevels(
            entry=entry, sl=sl, tp1=tp1, tp2=tp2,
            tp3_trail_atr=self.trail_mult * atr,
            sl_pct=round(sl_pct, 4),
            tp1_pct=round(tp1_pct_val, 4),
            tp2_pct=round(tp2_pct_val, 4),
            tp1_rr=round(tp1_rr, 4),
            tp2_rr=round(tp2_rr, 4),
            qty_tp1=self.QTY_TP1,
            qty_tp2=self.QTY_TP2,
            qty_tp3=self.QTY_TP3,
            net_gain_pct=round(net_gain * 100, 4),
            side=side,
        )
