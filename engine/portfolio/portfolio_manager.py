"""engine.portfolio.portfolio_manager
=====================================================================
Day 13 — Portfolio Manager.

Responsibilities:
  - Track every open position (across symbols, across strategies)
  - Compute realised + unrealised PnL
  - Aggregate exposure via ExposureModel
  - Take a SignalPool + correlation matrix → produce target allocations
  - Reduce allocation when two strategies want the same correlated symbol

Allocation algorithm (deterministic, capital-aware):
  1. Each actionable strategy gets a "vote" weighted by its
     `regime_affinity[regime]` (1.0 if regime layer is off).
  2. Group votes by symbol. Net vote = Σ signed strengths.
  3. For symbols whose pairwise correlation > threshold, scale the
     smaller-vote symbol's allocation by (1 - |corr|).
  4. Cap total risk by the gross-exposure budget; scale down
     proportionally if exceeded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from engine.portfolio.correlation_matrix import CorrelationMatrix
from engine.portfolio.exposure_model import Exposure, ExposureModel
from engine.signals import Action, Signal
from engine.strategy_runner import SignalPool
from utils.logger import get_logger

log = get_logger("engine.portfolio")


# ----------------------------------------------------------------------
@dataclass
class Position:
    symbol: str
    side: str                 # "long" | "short"
    lots: float
    entry_price: float
    entry_time: datetime
    strategy: str = ""
    stop: float = 0.0
    take: float = 0.0
    atr_at_open: float = 0.0
    ticket: int = 0
    # Updated live
    current_price: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.lots > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "side": self.side, "lots": self.lots,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "strategy": self.strategy, "stop": self.stop, "take": self.take,
            "ticket": self.ticket, "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass
class TargetAllocation:
    """Output of the portfolio manager — what we *want* to hold."""
    symbol: str
    action: Action
    lots: float            # 0 means close
    reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    positions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "n_positions": self.n_positions,
            "positions": list(self.positions),
        }


# ----------------------------------------------------------------------
class PortfolioManager:
    def __init__(
        self,
        initial_equity: float = 10_000.0,
        max_gross_exposure: float = 2.0,
        max_net_exposure: float = 1.0,
        correlation_threshold: float = 0.7,
    ) -> None:
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.realized_pnl = 0.0
        self._positions: dict[int, Position] = {}          # keyed by ticket
        self._positions_by_symbol: dict[str, list[int]] = {}
        self.exposure = ExposureModel(max_gross_exposure, max_net_exposure)
        self.correlation = CorrelationMatrix(lookback=500, min_history=50)
        self.correlation_threshold = float(correlation_threshold)
        self._next_ticket = 1

    # ----------------------------------------------------------------
    # Position lifecycle
    # ----------------------------------------------------------------
    def open_position(self, symbol: str, side: str, lots: float,
                      entry_price: float, strategy: str = "",
                      stop: float = 0.0, take: float = 0.0,
                      atr_at_open: float = 0.0,
                      ticket: Optional[int] = None) -> Position:
        ticket = int(ticket or self._next_ticket)
        self._next_ticket = max(self._next_ticket, ticket + 1)
        pos = Position(
            symbol=symbol, side=side, lots=float(lots),
            entry_price=float(entry_price),
            entry_time=datetime.now(tz=timezone.utc),
            strategy=strategy, stop=float(stop), take=float(take),
            atr_at_open=float(atr_at_open), ticket=ticket,
        )
        self._positions[ticket] = pos
        self._positions_by_symbol.setdefault(symbol, []).append(ticket)
        self._refresh_exposure()
        log.info("OPEN ticket=%d %s %s lots=%.4f @ %.5f strategy=%s",
                 ticket, side, symbol, lots, entry_price, strategy)
        return pos

    def close_position(self, ticket: int, exit_price: float) -> Optional[float]:
        pos = self._positions.get(ticket)
        if pos is None:
            return None
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.lots
        else:
            pnl = (pos.entry_price - exit_price) * pos.lots
        self.realized_pnl += pnl
        self.equity += pnl
        log.info("CLOSE ticket=%d %s lots=%.4f entry=%.5f exit=%.5f pnl=%.2f",
                 ticket, pos.symbol, pos.lots, pos.entry_price, exit_price, pnl)
        # Remove
        del self._positions[ticket]
        if pos.symbol in self._positions_by_symbol:
            self._positions_by_symbol[pos.symbol] = [
                t for t in self._positions_by_symbol[pos.symbol] if t != ticket
            ]
        self._refresh_exposure()
        return pnl

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update current_price + unrealized_pnl on every open position."""
        for pos in self._positions.values():
            px = prices.get(pos.symbol)
            if px is None:
                continue
            pos.current_price = float(px)
            if pos.side == "long":
                pos.unrealized_pnl = (px - pos.entry_price) * pos.lots
            else:
                pos.unrealized_pnl = (pos.entry_price - px) * pos.lots
        self._refresh_exposure()

    # ----------------------------------------------------------------
    # Snapshot
    # ----------------------------------------------------------------
    def snapshot(self) -> PortfolioSnapshot:
        unreal = sum(p.unrealized_pnl for p in self._positions.values())
        return PortfolioSnapshot(
            timestamp=datetime.now(tz=timezone.utc),
            equity=self.equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unreal,
            gross_exposure=self.exposure.gross_exposure,
            net_exposure=self.exposure.net_exposure,
            n_positions=self.exposure.n_open_positions,
            positions=[p.to_dict() for p in self._positions.values()],
        )

    @property
    def positions(self) -> list[Position]:
        return list(self._positions.values())

    def positions_for_symbol(self, symbol: str) -> list[Position]:
        tickets = self._positions_by_symbol.get(symbol, [])
        return [self._positions[t] for t in tickets if t in self._positions]

    # ----------------------------------------------------------------
    # Allocation: pool → target trades
    # ----------------------------------------------------------------
    def allocate(
        self,
        pool: SignalPool,
        regime: str = "trend",
        regime_confidence: float = 1.0,
        risk_unit_per_trade: float = 0.01,
        atrs: Optional[dict[str, float]] = None,
        strategy_affinity: Optional[dict[str, dict[str, float]]] = None,
    ) -> list[TargetAllocation]:
        """Convert a SignalPool into a list of target allocations.

        Steps:
          1. Filter actionable signals.
          2. Apply regime-affinity weight to each signal's strength.
          3. Aggregate by symbol (signed sum).
          4. Penalise symbols correlated with already-targeted symbols.
          5. Check exposure budget; scale down if needed.
        """
        atrs = atrs or {}
        strategy_affinity = strategy_affinity or {}
        targets: list[TargetAllocation] = []

        # 1. Filter actionable
        actionable = pool.actionable
        if not actionable:
            return targets

        # 2-3. Aggregate by symbol (signed vote)
        votes: dict[str, float] = {}
        meta_by_symbol: dict[str, dict[str, Any]] = {}
        for name, sig in actionable.items():
            affinity = 1.0
            if name in strategy_affinity:
                affinity = float(strategy_affinity[name].get(regime, 1.0))
            vote = sig.strength * affinity * regime_confidence
            if sig.action == Action.SELL:
                vote = -vote
            votes[sig.symbol] = votes.get(sig.symbol, 0.0) + vote
            meta_by_symbol.setdefault(sig.symbol, {})
            meta_by_symbol[sig.symbol].setdefault("strategies", []).append(name)

        # 4. Correlation penalty — scale down redundant bets
        targeted_symbols = list(votes.keys())
        scale_factor: dict[str, float] = {s: 1.0 for s in targeted_symbols}
        for i, s_i in enumerate(targeted_symbols):
            for s_j in targeted_symbols[i + 1:]:
                corr = self.correlation.pairwise(s_i, s_j)
                if abs(corr) >= self.correlation_threshold:
                    # Penalise the smaller-vote symbol
                    if abs(votes[s_i]) >= abs(votes[s_j]):
                        scale_factor[s_j] *= (1.0 - abs(corr))
                    else:
                        scale_factor[s_i] *= (1.0 - abs(corr))

        # 5. Build target allocations, capped by exposure budget
        remaining_budget = max(0.0, self.exposure.max_gross - self.exposure.gross_exposure)
        for sym, vote in votes.items():
            scaled_vote = vote * scale_factor[sym]
            if abs(scaled_vote) < 0.05:
                continue
            action = Action.BUY if scaled_vote > 0 else Action.SELL
            risk_units = abs(scaled_vote) * risk_unit_per_trade * 10.0  # heuristic
            if risk_units > remaining_budget:
                risk_units = remaining_budget
            if risk_units <= 0:
                continue
            remaining_budget -= risk_units

            # Convert risk units to lots using ATR if available
            atr_val = atrs.get(sym, 0.0)
            if atr_val > 0 and self.equity > 0:
                lots = (self.equity * risk_units) / (atr_val * 2.0)  # 2 ATR stop
            else:
                lots = 0.01  # placeholder minimum

            targets.append(TargetAllocation(
                symbol=sym, action=action, lots=float(lots),
                reason=f"vote={scaled_vote:.3f} corr_scale={scale_factor[sym]:.2f}",
                meta={
                    "raw_vote": vote,
                    "scaled_vote": scaled_vote,
                    "strategies": meta_by_symbol[sym]["strategies"],
                    "risk_units": risk_units,
                    "atr": atr_val,
                    "regime": regime,
                },
            ))
        return targets

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------
    def _refresh_exposure(self) -> None:
        self.exposure._exposures.clear()
        # Aggregate lots per symbol
        agg: dict[str, Exposure] = {}
        for pos in self._positions.values():
            long_lots = pos.lots if pos.side == "long" else 0.0
            short_lots = pos.lots if pos.side == "short" else 0.0
            net = pos.lots if pos.side == "long" else -pos.lots
            risk = pos.lots * (pos.atr_at_open or 0.0) / max(self.equity, 1.0)
            e = Exposure(
                symbol=pos.symbol,
                net_lots=net,
                long_lots=long_lots,
                short_lots=short_lots,
                notional=pos.lots * pos.entry_price,
                risk_units=risk,
                side=pos.side,
            )
            if pos.symbol in agg:
                agg[pos.symbol] = agg[pos.symbol].merge(e)
            else:
                agg[pos.symbol] = e
        for sym, e in agg.items():
            self.exposure.update_position(
                symbol=sym, net_lots=e.net_lots,
                long_lots=e.long_lots, short_lots=e.short_lots,
                notional=e.notional, risk_units=e.risk_units,
            )
