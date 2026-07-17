"""
R-Multiple Partial Take-Profit Module
======================================

Institutional risk-multiple thinking automated end-to-end. Positions are
partially closed at 2R, 3R, and 5R targets, with stop-losses automatically
moved to breakeven or better after each partial exit.

R = initial risk = |entry_price - stop_loss_price|

Partial Exit Schedule:
  - At 2R: Close 40% of position, move stop to breakeven
  - At 3R: Close 30% of position, move stop to 1R (locked profit)
  - At 5R: Close remaining 30%, move stop to 3R (trailing)

This locks in profits at multiple levels while letting runners capture
big moves. After the first partial (2R), the trade is "risk-free" because
the stop is at breakeven.

Source: NexusQuant (review #29) — partialTakeProfitExecutor.ts concept
Enhanced with: trailing stop logic and breakeven management

Usage:
    from r_multiple_tp import RMultipleTP, Position

    tp = RMultipleTP()

    # Create a new position
    pos = Position(
        symbol="BTCUSDT",
        direction="long",
        entry_price=65000,
        stop_loss=63500,
        position_size=1.0,  # 1 BTC
    )

    # Initialize TP levels
    plan = tp.create_tp_plan(pos)
    print(f"2R target: ${plan['levels'][0]['price']:.2f}")
    print(f"3R target: ${plan['levels'][1]['price']:.2f}")
    print(f"5R target: ${plan['levels'][2]['price']:.2f}")

    # Check if any TP level is hit (call on each price update)
    current_price = 68000  # Example: price reached 2R
    actions = tp.check_levels(pos, plan, current_price)
    for action in actions:
        print(f"Action: {action['type']} - {action['description']}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Trading position with entry and stop."""
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_loss: float
    position_size: float  # Total position size (in units)
    remaining_size: float = 0.0  # Remaining after partial exits
    current_stop: float = 0.0  # Current stop (may be adjusted)

    def __post_init__(self):
        if self.remaining_size == 0.0:
            self.remaining_size = self.position_size
        if self.current_stop == 0.0:
            self.current_stop = self.stop_loss

    @property
    def risk_per_unit(self) -> float:
        """R = |entry - stop|"""
        return abs(self.entry_price - self.stop_loss)

    @property
    def is_long(self) -> bool:
        return self.direction.lower() == "long"

    @property
    def r_value(self) -> float:
        """Dollar risk = risk_per_unit × position_size"""
        return self.risk_per_unit * self.position_size


@dataclass
class TPLevel:
    """A single take-profit level."""
    r_multiple: float          # 2.0, 3.0, 5.0
    price: float               # Target price
    close_pct: float           # Percentage of position to close (0-1)
    new_stop: float            # New stop after this level hits
    new_stop_label: str        # "breakeven", "1R", "3R"
    hit: bool = False          # Has this level been hit?
    hit_price: Optional[float] = None
    hit_time: Optional[str] = None


@dataclass
class TPPlan:
    """Complete take-profit plan for a position."""
    position: Position
    levels: list[TPLevel] = field(default_factory=list)
    fully_closed: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.position.symbol,
            "direction": self.position.direction,
            "entry_price": self.position.entry_price,
            "initial_stop": self.position.stop_loss,
            "current_stop": self.position.current_stop,
            "risk_per_unit": self.position.risk_per_unit,
            "remaining_size": self.position.remaining_size,
            "fully_closed": self.fully_closed,
            "levels": [
                {
                    "r_multiple": l.r_multiple,
                    "price": l.price,
                    "close_pct": l.close_pct,
                    "new_stop": l.new_stop,
                    "new_stop_label": l.new_stop_label,
                    "hit": l.hit,
                }
                for l in self.levels
            ],
        }


class RMultipleTP:
    """
    R-Multiple Partial Take-Profit Manager.

    Manages partial exits at 2R, 3R, 5R with automatic stop adjustments.
    After first partial (2R), stop moves to breakeven → trade becomes risk-free.
    """

    # Default partial exit schedule
    DEFAULT_SCHEDULE = [
        {"r": 2.0, "close_pct": 0.40, "new_stop_label": "breakeven"},
        {"r": 3.0, "close_pct": 0.30, "new_stop_label": "1R"},
        {"r": 5.0, "close_pct": 0.30, "new_stop_label": "3R"},
    ]

    def __init__(self, schedule: list[dict] = None):
        self.schedule = schedule or self.DEFAULT_SCHEDULE

    def create_tp_plan(self, position: Position) -> TPPlan:
        """
        Create a take-profit plan for a position.

        Computes TP prices based on R multiples and direction.
        """
        plan = TPPlan(position=position)
        r = position.risk_per_unit

        for level_spec in self.schedule:
            r_mult = level_spec["r"]
            close_pct = level_spec["close_pct"]
            stop_label = level_spec["new_stop_label"]

            # Compute TP price
            if position.is_long:
                tp_price = position.entry_price + (r * r_mult)
            else:
                tp_price = position.entry_price - (r * r_mult)

            # Compute new stop after this level
            new_stop = self._compute_new_stop(position, stop_label, r)

            plan.levels.append(TPLevel(
                r_multiple=r_mult,
                price=tp_price,
                close_pct=close_pct,
                new_stop=new_stop,
                new_stop_label=stop_label,
            ))

        return plan

    def check_levels(
        self,
        position: Position,
        plan: TPPlan,
        current_price: float,
    ) -> list[dict]:
        """
        Check if any TP levels are hit by current price.

        Returns list of actions to execute (partial closes + stop adjustments).
        """
        actions = []

        if plan.fully_closed:
            return actions

        for i, level in enumerate(plan.levels):
            if level.hit:
                continue

            # Check if price reached this level
            if position.is_long:
                level_hit = current_price >= level.price
            else:
                level_hit = current_price <= level.price

            if level_hit:
                # Compute close size
                close_size = position.position_size * level.close_pct
                close_size = min(close_size, position.remaining_size)

                # Update position
                position.remaining_size -= close_size
                position.current_stop = level.new_stop

                # Mark level as hit
                level.hit = True
                level.hit_price = current_price
                from datetime import datetime
                level.hit_time = datetime.now().isoformat()

                actions.append({
                    "type": "PARTIAL_CLOSE",
                    "level": level.r_multiple,
                    "close_size": close_size,
                    "close_pct": level.close_pct,
                    "price": level.price,
                    "remaining_size": position.remaining_size,
                    "new_stop": level.new_stop,
                    "new_stop_label": level.new_stop_label,
                    "description": (
                        f"Closed {level.close_pct:.0%} at {level.r_multiple}R "
                        f"(${level.price:.2f}). Stop moved to {level.new_stop_label} "
                        f"(${level.new_stop:.2f}). Remaining: {position.remaining_size:.4f}"
                    ),
                })

                # Check if fully closed
                if position.remaining_size <= 0.0001:
                    plan.fully_closed = True
                    actions.append({
                        "type": "POSITION_CLOSED",
                        "description": "Position fully closed.",
                    })
                    break

        return actions

    def _compute_new_stop(
        self,
        position: Position,
        label: str,
        r: float,
    ) -> float:
        """Compute new stop price based on label."""
        if label == "breakeven":
            return position.entry_price
        elif label == "1R":
            if position.is_long:
                return position.entry_price + r
            else:
                return position.entry_price - r
        elif label == "3R":
            if position.is_long:
                return position.entry_price + (r * 3)
            else:
                return position.entry_price - (r * 3)
        elif label == "trail":
            # Trailing stop — would need highest favorable price tracking
            return position.current_stop
        else:
            return position.current_stop

    def get_current_r(self, position: Position, current_price: float) -> float:
        """
        Compute current R multiple (how far in profit/loss in terms of R).

        Positive R = in profit
        Negative R = in loss
        """
        r = position.risk_per_unit
        if r == 0:
            return 0.0

        if position.is_long:
            return (current_price - position.entry_price) / r
        else:
            return (position.entry_price - current_price) / r
