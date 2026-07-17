"""
Transaction Cost Analysis (TCA)
================================

Measures the true cost of executing trades — beyond just commission.

Components:
    1. Commission           — broker fee
    2. Spread               — half-spread × notional
    3. Slippage             — (executed price - decision price) × qty
    4. Market Impact        — price move caused by your order
    5. Opportunity Cost     — trades not taken due to delay/rejection
    6. Timing Cost          — (execution price - arrival price) × qty

The TCA report shows the gap between paper PnL (no costs) and realized PnL.

Usage:
    from trading_modules.tca import TCAAnalyzer, TradeExecution
    analyzer = TCAAnalyzer()
    tca = analyzer.analyze(TradeExecution(
        symbol="BTCUSD", side="BUY", qty=1.0,
        decision_price=65000, arrival_price=65010,
        execution_price=65025, commission=5.0,
        spread_half=5.0, market_data=df_at_time,
    ))
    print(f"Total cost: ${tca.total_cost:.2f} ({tca.total_cost_bps:.1f} bps)")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeExecution:
    """Details of a single trade execution for TCA."""
    symbol: str
    side: str                     # "BUY" or "SELL"
    qty: float
    decision_price: float         # price when signal was generated
    arrival_price: float          # price when order reached broker
    execution_price: float        # actual fill price
    commission: float = 0.0       # broker commission in $
    spread_half: float = 0.0      # half-spread in price units
    market_data: Optional[pd.DataFrame] = None  # OHLCV around execution
    execution_time_seconds: float = 0.0  # order-to-fill latency


@dataclass
class TCAResult:
    symbol: str
    side: str
    qty: float
    notional: float
    # Cost components (all in $)
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    market_impact_cost: float = 0.0
    timing_cost: float = 0.0
    opportunity_cost: float = 0.0
    total_cost: float = 0.0
    total_cost_bps: float = 0.0   # basis points of notional
    # Analysis
    execution_quality: str = "unknown"  # "good" / "acceptable" / "poor"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side,
            "qty": self.qty, "notional": round(self.notional, 2),
            "commission": round(self.commission, 2),
            "spread_cost": round(self.spread_cost, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "market_impact_cost": round(self.market_impact_cost, 2),
            "timing_cost": round(self.timing_cost, 2),
            "opportunity_cost": round(self.opportunity_cost, 2),
            "total_cost": round(self.total_cost, 2),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "execution_quality": self.execution_quality,
            "notes": self.notes,
        }


class TCAAnalyzer:
    """Transaction Cost Analysis.

    Parameters:
        impact_window_bars: # of bars after execution to measure impact (default 5)
        max_acceptable_bps: total cost above this (bps) = poor execution (default 20)
    """

    def __init__(
        self, impact_window_bars: int = 5, max_acceptable_bps: float = 20,
    ) -> None:
        self.impact_window_bars = impact_window_bars
        self.max_acceptable_bps = max_acceptable_bps

    def analyze(self, exec: TradeExecution) -> TCAResult:
        """Analyze a single trade execution."""
        side = exec.side.upper()
        if side not in ("BUY", "SELL"):
            return TCAResult(
                symbol=exec.symbol, side=exec.side, qty=exec.qty,
                notional=0, notes=["invalid side"],
            )
        qty = float(exec.qty)
        notional = qty * exec.execution_price
        # ── Commission ────────────────────────────────────────────
        commission = float(exec.commission)
        # ── Spread cost ───────────────────────────────────────────
        # Half-spread × qty (you cross the spread to get filled)
        spread_cost = float(exec.spread_half) * qty
        # ── Slippage ──────────────────────────────────────────────
        # Difference between execution price and decision price
        if side == "BUY":
            slippage_per_unit = exec.execution_price - exec.decision_price
        else:
            slippage_per_unit = exec.decision_price - exec.execution_price
        slippage_cost = float(slippage_per_unit) * qty
        # ── Timing cost ───────────────────────────────────────────
        # Difference between arrival price and execution price
        if side == "BUY":
            timing_per_unit = exec.execution_price - exec.arrival_price
        else:
            timing_per_unit = exec.arrival_price - exec.execution_price
        timing_cost = float(timing_per_unit) * qty
        # ── Market impact ─────────────────────────────────────────
        # Price move in the direction of your trade after execution
        # (your order pushed the market against you)
        market_impact_cost = 0.0
        if exec.market_data is not None and len(exec.market_data) > self.impact_window_bars:
            post_prices = exec.market_data["close"].tail(self.impact_window_bars).to_numpy(dtype=float)
            if len(post_prices) > 1:
                # For BUY: if price rose after your buy, that's adverse impact
                # For SELL: if price fell after your sell, that's adverse impact
                price_move = float(post_prices[-1] - exec.execution_price)
                if side == "BUY":
                    adverse_move = max(0, price_move)  # price went up = bad for buyer
                else:
                    adverse_move = max(0, -price_move)  # price went down = bad for seller
                market_impact_cost = adverse_move * qty
        # ── Opportunity cost ──────────────────────────────────────
        # If execution was slow (high latency), you may have missed part of the move
        # Approximate: if latency > 1 second, cost = (latency_seconds × avg_price_change_per_sec × qty)
        opportunity_cost = 0.0
        if exec.execution_time_seconds > 1.0:
            # Conservative estimate: 1 bps per second of latency
            opportunity_cost = notional * 0.0001 * exec.execution_time_seconds

        # ── Totals ────────────────────────────────────────────────
        total_cost = (commission + spread_cost + slippage_cost +
                      market_impact_cost + timing_cost + opportunity_cost)
        total_cost_bps = (total_cost / notional * 10000) if notional > 0 else 0

        # ── Execution quality ─────────────────────────────────────
        if total_cost_bps <= self.max_acceptable_bps * 0.5:
            quality = "good"
        elif total_cost_bps <= self.max_acceptable_bps:
            quality = "acceptable"
        else:
            quality = "poor"

        notes: list[str] = []
        notes.append(f"notional: ${notional:,.2f}")
        notes.append(f"commission: ${commission:.2f} ({commission/notional*10000 if notional else 0:.1f} bps)")
        notes.append(f"spread: ${spread_cost:.2f} ({spread_cost/notional*10000 if notional else 0:.1f} bps)")
        notes.append(f"slippage: ${slippage_cost:.2f} ({slippage_cost/notional*10000 if notional else 0:.1f} bps)")
        if market_impact_cost > 0:
            notes.append(f"market_impact: ${market_impact_cost:.2f} ({market_impact_cost/notional*10000:.1f} bps)")
        if timing_cost > 0:
            notes.append(f"timing: ${timing_cost:.2f}")
        if opportunity_cost > 0:
            notes.append(f"opportunity: ${opportunity_cost:.2f} (latency {exec.execution_time_seconds:.1f}s)")
        notes.append(f"TOTAL: ${total_cost:.2f} ({total_cost_bps:.1f} bps) — {quality}")

        return TCAResult(
            symbol=exec.symbol, side=side, qty=qty, notional=notional,
            commission=commission, spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact_cost,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            total_cost=total_cost, total_cost_bps=float(total_cost_bps),
            execution_quality=quality, notes=notes,
        )


__all__ = ["TCAAnalyzer", "TradeExecution", "TCAResult"]
