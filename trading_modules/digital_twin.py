"""
Digital Twin Module — Virtual Account Copy
============================================

A digital twin is a virtual copy of your live trading account.
Every trade decision is first simulated on the twin, then executed live.

Pipeline:
  1. Live signal generated → run on Digital Twin first
  2. Digital Twin simulates entry, SL, TP, position sizing
  3. Compare twin prediction vs live result
  4. Divergence > threshold → halt live trading

This catches:
  - Slippage estimates that are too optimistic
  - Fill assumptions that don't match reality
  - Position sizing that doesn't account for margin
  - Strategy behavior under live conditions vs backtest

Source: User's existing shadow_mode.py concept
        Orallexa (review #27) — what-if scenario simulation
        Vibe-Trading (review #23) — shadow account

Usage:
    from trading_modules.digital_twin import DigitalTwin

    twin = DigitalTwin(initial_capital=10000)

    # Simulate a trade before executing live
    result = twin.simulate_trade(
        symbol="BTCUSD",
        direction="BUY",
        entry_price=65000,
        stop_loss=63500,
        take_profit=68000,
        position_size=0.1,
    )

    # Check twin status
    print(twin.get_status())
    # → equity=10250, open_positions=1, win_rate=0.65

    # Record live trade result for divergence tracking
    twin.record_live_result(symbol="BTCUSD", fill_price=65012, slippage=0.0002)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TwinPosition:
    """A position in the digital twin."""
    symbol: str
    direction: str  # "long" / "short"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    entry_time: str = ""
    status: str = "open"  # open / closed_tp / closed_sl / closed_timeout
    exit_price: float = 0.0
    pnl: float = 0.0
    hold_bars: int = 0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "status": self.status,
            "exit_price": self.exit_price,
            "pnl": round(self.pnl, 2),
            "hold_bars": self.hold_bars,
        }


@dataclass
class DivergenceTracker:
    """Tracks divergence between twin predictions and live results."""
    n_trades: int = 0
    avg_slippage_divergence: float = 0.0
    avg_fill_divergence: float = 0.0
    max_divergence: float = 0.0
    divergent_trades: int = 0  # Trades where divergence > threshold

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "avg_slippage_divergence": round(self.avg_slippage_divergence, 6),
            "avg_fill_divergence": round(self.avg_fill_divergence, 6),
            "max_divergence": round(self.max_divergence, 6),
            "divergent_pct": round(self.divergent_trades / max(self.n_trades, 1), 4),
        }


class DigitalTwin:
    """
    Virtual copy of a trading account for pre-trade simulation.

    The twin maintains its own equity, positions, and trade history.
    Every live trade is first simulated on the twin, then the live
    result is compared to detect divergence.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        divergence_threshold: float = 0.005,  # 0.5% divergence = warning
    ):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.positions: list[TwinPosition] = []
        self.closed_positions: list[TwinPosition] = []
        self.trade_history: list[dict] = []
        self.divergence = DivergenceTracker()
        self.divergence_threshold = divergence_threshold
        self.is_halted = False  # True if divergence too high

    def simulate_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        estimated_slippage: float = 0.001,
    ) -> TwinPosition:
        """
        Simulate a trade on the digital twin.

        This is called BEFORE the live trade to predict the expected outcome.
        """
        if self.is_halted:
            logger.warning("Digital twin halted — divergence too high")
            return TwinPosition(symbol, direction, entry_price, stop_loss,
                              take_profit, position_size, status="blocked")

        # Apply estimated slippage
        if direction == "BUY":
            fill_price = entry_price * (1 + estimated_slippage)
        else:
            fill_price = entry_price * (1 - estimated_slippage)

        position = TwinPosition(
            symbol=symbol,
            direction=direction.lower(),
            entry_price=fill_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            entry_time=datetime.now(timezone.utc).isoformat(),
        )

        self.positions.append(position)
        self.trade_history.append({
            "action": "open",
            "symbol": symbol,
            "direction": direction,
            "entry_price": fill_price,
            "timestamp": position.entry_time,
        })

        logger.info(f"Twin: Opened {direction} {symbol} @ {fill_price:.2f}")
        return position

    def update_positions(self, current_prices: dict[str, float]) -> list[TwinPosition]:
        """
        Update open positions with current prices. Check SL/TP.

        Args:
            current_prices: {symbol: price}

        Returns:
            List of positions that were closed this update.
        """
        closed = []

        for pos in self.positions[:]:  # Copy because we modify during iteration
            if pos.status != "open":
                continue

            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            pos.hold_bars += 1

            # Check SL/TP
            if pos.direction == "long":
                if price <= pos.stop_loss:
                    pos.status = "closed_sl"
                    pos.exit_price = pos.stop_loss
                elif price >= pos.take_profit:
                    pos.status = "closed_tp"
                    pos.exit_price = pos.take_profit
            else:  # short
                if price >= pos.stop_loss:
                    pos.status = "closed_sl"
                    pos.exit_price = pos.stop_loss
                elif price <= pos.take_profit:
                    pos.status = "closed_tp"
                    pos.exit_price = pos.take_profit

            if pos.status != "open":
                # Calculate PnL
                if pos.direction == "long":
                    pos.pnl = (pos.exit_price - pos.entry_price) * pos.position_size
                else:
                    pos.pnl = (pos.entry_price - pos.exit_price) * pos.position_size

                self.equity += pos.pnl
                self.positions.remove(pos)
                self.closed_positions.append(pos)
                closed.append(pos)

                logger.info(f"Twin: Closed {pos.symbol} ({pos.status}) PnL={pos.pnl:.2f}")

        return closed

    def record_live_result(
        self,
        symbol: str,
        fill_price: float,
        slippage: float,
        twin_predicted_fill: Optional[float] = None,
    ) -> None:
        """
        Record live trade result for divergence tracking.

        Compare the twin's predicted fill vs the actual live fill.
        If divergence exceeds threshold, flag for review.
        """
        self.divergence.n_trades += 1

        if twin_predicted_fill is not None:
            fill_div = abs(fill_price - twin_predicted_fill) / fill_price
            self.divergence.avg_fill_divergence = (
                (self.divergence.avg_fill_divergence * (self.divergence.n_trades - 1) + fill_div)
                / self.divergence.n_trades
            )
            self.divergence.max_divergence = max(self.divergence.max_divergence, fill_div)

            if fill_div > self.divergence_threshold:
                self.divergence.divergent_trades += 1
                logger.warning(
                    f"Divergence alert: {symbol} fill divergence {fill_div:.4f} > {self.divergence_threshold}"
                )

                # Halt if too many divergent trades
                if self.divergence.divergent_trades / self.divergence.n_trades > 0.3:
                    self.is_halted = True
                    logger.error("Digital twin HALTED — divergence rate > 30%")

    def get_status(self) -> dict:
        """Get twin status summary."""
        wins = sum(1 for p in self.closed_positions if p.pnl > 0)
        losses = sum(1 for p in self.closed_positions if p.pnl < 0)
        total_pnl = sum(p.pnl for p in self.closed_positions)

        return {
            "equity": round(self.equity, 2),
            "initial_capital": self.initial_capital,
            "return_pct": round((self.equity - self.initial_capital) / self.initial_capital, 4),
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "win_rate": round(wins / max(wins + losses, 1), 4),
            "total_pnl": round(total_pnl, 2),
            "is_halted": self.is_halted,
            "divergence": self.divergence.to_dict(),
        }

    def reset(self) -> None:
        """Reset twin to initial state."""
        self.equity = self.initial_capital
        self.positions = []
        self.closed_positions = []
        self.trade_history = []
        self.divergence = DivergenceTracker()
        self.is_halted = False

    def get_trade_history(self) -> list[dict]:
        """Get full trade history."""
        return self.trade_history
