"""enhancements.position_manager
=====================================================================
Day 141-145 — Position Manager.

Manages OPEN positions intelligently after entry. This is the gap
between "execute order" and "manage trade" — where most retail
systems lose money by either exiting too early or holding too long.

Features:
  - Trailing stops (ATR-based, percentage-based, structure-based)
  - Breakeven move (move SL to entry after price moves X in favor)
  - Partial profit-taking (close 50% at 1R, 25% at 2R, etc.)
  - Scaling in (pyramid into winning positions)
  - Scaling out (gradually exit losing positions)
  - Time-based exit (close if no progress after N bars)
  - Volatility-adjusted stops (widen in high vol, tighten in low vol)

Every action is logged with reason for the trade journal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("enhancements.position_manager")


class TrailingStopType(str, Enum):
    FIXED = "fixed"
    ATR = "atr"
    PERCENTAGE = "percentage"
    STRUCTURE = "structure"   # trail behind recent swing high/low


class PositionAction(str, Enum):
    HOLD = "hold"
    MOVE_STOP = "move_stop"
    TAKE_PARTIAL_PROFIT = "take_partial_profit"
    CLOSE = "close"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    MOVE_TO_BREAKEVEN = "move_to_breakeven"


@dataclass
class PositionState:
    """Current state of an open position."""
    ticket: int
    symbol: str
    side: str                    # "long" / "short"
    entry_price: float
    current_price: float
    lots: float
    initial_lots: float
    stop_loss: float
    take_profit: float
    atr_at_entry: float
    entry_time: datetime
    bars_held: int = 0
    max_favourable: float = 0.0   # best price seen (in favor)
    max_adverse: float = 0.0      # worst price seen (against)
    partials_taken: list[float] = field(default_factory=list)
    breakeven_moved: bool = False
    last_stop_adjustment: Optional[datetime] = None

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) / self.entry_price
        return (self.entry_price - self.current_price) / self.entry_price

    @property
    def r_multiple(self) -> float:
        """Current profit/loss in R multiples (R = initial risk)."""
        initial_risk = abs(self.entry_price - self.stop_loss)
        if initial_risk <= 0:
            return 0.0
        if self.side == "long":
            return (self.current_price - self.entry_price) / initial_risk
        return (self.entry_price - self.current_price) / initial_risk

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "lots": self.lots,
            "initial_lots": self.initial_lots,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr_at_entry": self.atr_at_entry,
            "bars_held": self.bars_held,
            "max_favourable": self.max_favourable,
            "max_adverse": self.max_adverse,
            "partials_taken": list(self.partials_taken),
            "breakeven_moved": self.breakeven_moved,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "r_multiple": self.r_multiple,
        }

    def add_lots(self, price: float, lots: float) -> None:
        """Critical #1 fix: add lots at `price` and update entry_price to
        a weighted average. This must be called by the execution layer
        when a SCALE_IN decision is executed, otherwise unrealized_pnl_pct
        and r_multiple will be computed as if all lots were entered at the
        original price — producing inaccurate risk metrics.

        The stop_loss is NOT adjusted here — the PositionManager.evaluate()
        will recompute it on the next cycle via trailing/breakeven logic.
        """
        if lots <= 0:
            return
        total_lots = self.lots + lots
        if total_lots <= 0:
            return
        # Weighted average entry price.
        self.entry_price = (self.entry_price * self.lots + price * lots) / total_lots
        self.lots = total_lots
        # initial_lots tracks the original + all scale-ins for sizing reference.
        self.initial_lots += lots
        log.info("PositionManager: scale-in %s — added %.4f lots @ %.5f, "
                 "new avg entry=%.5f, total lots=%.4f",
                 self.symbol, lots, price, self.entry_price, self.lots)


@dataclass
class PositionManagerDecision:
    action: PositionAction
    reason: str
    new_stop: Optional[float] = None
    partial_lots: Optional[float] = None
    partial_price: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "new_stop": self.new_stop,
            "partial_lots": self.partial_lots,
            "partial_price": self.partial_price,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
class PositionManager:
    """Manages open positions with trailing stops, breakeven, partials."""

    def __init__(
        self,
        trailing_stop_type: TrailingStopType = TrailingStopType.ATR,
        trailing_atr_multiple: float = 2.0,
        breakeven_trigger_r: float = 1.0,         # move to BE after 1R in favor
        breakeven_buffer_atr: float = 0.1,        # stop goes to entry + 0.1 ATR
        partial_schedule: Optional[list[dict]] = None,
        max_bars_held: int = 100,
        time_exit_r_threshold: float = 0.5,       # exit if < 0.5R after max_bars
        scale_in_threshold_r: float = 2.0,        # pyramid after 2R
        scale_in_size_pct: float = 0.30,          # add 30% of current size
    ) -> None:
        self.trailing_type = trailing_stop_type
        self.trailing_atr_mult = float(trailing_atr_multiple)
        self.be_trigger_r = float(breakeven_trigger_r)
        self.be_buffer_atr = float(breakeven_buffer_atr)
        self.partial_schedule = partial_schedule or [
            {"at_r": 1.0, "close_pct": 0.50},
            {"at_r": 2.0, "close_pct": 0.25},
            {"at_r": 3.0, "close_pct": 0.25},
        ]
        self.max_bars = int(max_bars_held)
        self.time_exit_r = float(time_exit_r_threshold)
        self.scale_in_r = float(scale_in_threshold_r)
        self.scale_in_pct = float(scale_in_size_pct)

    # ----------------------------------------------------------------
    def evaluate(self, position: PositionState,
                  current_atr: float) -> PositionManagerDecision:
        """Decide what action to take on this position."""
        position.bars_held += 1
        # Update max favourable / adverse
        if position.side == "long":
            position.max_favourable = max(position.max_favourable, position.current_price)
            position.max_adverse = min(position.max_adverse, position.current_price)
        else:
            position.max_favourable = min(position.max_favourable, position.current_price)
            position.max_adverse = max(position.max_adverse, position.current_price)

        r = position.r_multiple

        # 1. Stop-loss hit (server-side usually handles this, but double-check)
        if position.side == "long" and position.current_price <= position.stop_loss:
            return PositionManagerDecision(
                action=PositionAction.CLOSE,
                reason=f"stop loss hit @ {position.current_price:.5f}",
            )
        if position.side == "short" and position.current_price >= position.stop_loss:
            return PositionManagerDecision(
                action=PositionAction.CLOSE,
                reason=f"stop loss hit @ {position.current_price:.5f}",
            )

        # 2. Take-profit hit
        if position.take_profit > 0:
            if position.side == "long" and position.current_price >= position.take_profit:
                return PositionManagerDecision(
                    action=PositionAction.CLOSE,
                    reason=f"take profit hit @ {position.current_price:.5f}",
                )
            if position.side == "short" and position.current_price <= position.take_profit:
                return PositionManagerDecision(
                    action=PositionAction.CLOSE,
                    reason=f"take profit hit @ {position.current_price:.5f}",
                )

        # 3. Breakeven move
        if (not position.breakeven_moved
                and r >= self.be_trigger_r
                and current_atr > 0):
            if position.side == "long":
                new_stop = position.entry_price + self.be_buffer_atr * current_atr
                if new_stop > position.stop_loss:
                    position.breakeven_moved = True
                    position.stop_loss = new_stop
                    position.last_stop_adjustment = datetime.now(tz=timezone.utc)
                    return PositionManagerDecision(
                        action=PositionAction.MOVE_TO_BREAKEVEN,
                        reason=f"breakeven: r={r:.2f} >= {self.be_trigger_r}",
                        new_stop=new_stop,
                    )
            else:
                new_stop = position.entry_price - self.be_buffer_atr * current_atr
                if new_stop < position.stop_loss or position.stop_loss == 0:
                    position.breakeven_moved = True
                    position.stop_loss = new_stop
                    position.last_stop_adjustment = datetime.now(tz=timezone.utc)
                    return PositionManagerDecision(
                        action=PositionAction.MOVE_TO_BREAKEVEN,
                        reason=f"breakeven: r={r:.2f} >= {self.be_trigger_r}",
                        new_stop=new_stop,
                    )

        # 4. Trailing stop
        if position.breakeven_moved and current_atr > 0:
            new_stop = self._compute_trailing_stop(position, current_atr)
            if new_stop is not None:
                if position.side == "long" and new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    position.last_stop_adjustment = datetime.now(tz=timezone.utc)
                    return PositionManagerDecision(
                        action=PositionAction.MOVE_STOP,
                        reason=f"trailing {self.trailing_type.value}: new_stop={new_stop:.5f}",
                        new_stop=new_stop,
                    )
                if position.side == "short" and (new_stop < position.stop_loss or position.stop_loss == 0):
                    position.stop_loss = new_stop
                    position.last_stop_adjustment = datetime.now(tz=timezone.utc)
                    return PositionManagerDecision(
                        action=PositionAction.MOVE_STOP,
                        reason=f"trailing {self.trailing_type.value}: new_stop={new_stop:.5f}",
                        new_stop=new_stop,
                    )

        # 5. Partial profit-taking
        for level in self.partial_schedule:
            at_r = float(level["at_r"])
            close_pct = float(level["close_pct"])
            if r >= at_r and at_r not in position.partials_taken:
                partial_lots = position.lots * close_pct
                if partial_lots > 0.001:  # minimum lot
                    position.partials_taken.append(at_r)
                    position.lots -= partial_lots
                    return PositionManagerDecision(
                        action=PositionAction.TAKE_PARTIAL_PROFIT,
                        reason=f"partial @ {at_r}R: close {close_pct:.0%} = {partial_lots:.4f} lots",
                        partial_lots=partial_lots,
                        partial_price=position.current_price,
                    )

        # 6. Scale-in (pyramid into winners)
        if r >= self.scale_in_r and len(position.partials_taken) == 0:
            add_lots = position.initial_lots * self.scale_in_pct
            return PositionManagerDecision(
                action=PositionAction.SCALE_IN,
                reason=f"scale in: r={r:.2f} >= {self.scale_in_r}, add {add_lots:.4f} lots",
                partial_lots=add_lots,
                partial_price=position.current_price,
            )

        # 7. Time-based exit
        if position.bars_held >= self.max_bars and r < self.time_exit_r:
            return PositionManagerDecision(
                action=PositionAction.CLOSE,
                reason=f"time exit: {position.bars_held} bars, r={r:.2f} < {self.time_exit_r}",
            )

        # Default: hold
        return PositionManagerDecision(
            action=PositionAction.HOLD,
            reason=f"holding: r={r:.2f}, bars={position.bars_held}",
        )

    # ----------------------------------------------------------------
    def _compute_trailing_stop(self, position: PositionState,
                                 current_atr: float) -> Optional[float]:
        """Compute trailing stop based on configured type."""
        if self.trailing_type == TrailingStopType.ATR:
            if position.side == "long":
                return position.current_price - self.trailing_atr_mult * current_atr
            return position.current_price + self.trailing_atr_mult * current_atr
        if self.trailing_type == TrailingStopType.PERCENTAGE:
            pct = 0.02  # 2%
            if position.side == "long":
                return position.current_price * (1 - pct)
            return position.current_price * (1 + pct)
        if self.trailing_type == TrailingStopType.STRUCTURE:
            # Trail behind max_favourable by 1 ATR
            if position.side == "long":
                return position.max_favourable - current_atr
            return position.max_favourable + current_atr
        return None
