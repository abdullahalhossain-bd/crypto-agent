"""execution.execution_engine
=====================================================================
Day 19 — Alpha-aware execution engine.

Wraps the Day-4 ExecutionEngine with:
  - Order slicing (large parent → child orders)
  - Slippage estimation (refuse orders whose predicted slippage
    exceeds the edge)
  - Adaptive execution speed (slow down when vol_ratio is high)
  - Decision trace recording (forwarded to the observability layer)

This module is a *decorator* over the original ExecutionEngine —
the live/paper order-sending mechanics stay in `engine/execution.py`
so we don't duplicate MT5 plumbing.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from execution.order_slicer import OrderSlicer, SlicedOrder
from execution.slippage_model import SlippageModel
from external.env_loader import env
from utils.logger import get_logger, log_trade

log = get_logger("execution.alpha")


# ----------------------------------------------------------------------
@dataclass
class AlphaExecutionResult:
    parent_id: str
    ok: bool
    total_filled_lots: float
    avg_fill_price: float
    slices: list[dict[str, Any]] = field(default_factory=list)
    slippage_estimate: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "ok": self.ok,
            "total_filled_lots": self.total_filled_lots,
            "avg_fill_price": self.avg_fill_price,
            "slices": list(self.slices),
            "slippage_estimate": dict(self.slippage_estimate),
            "reason": self.reason,
        }


# ----------------------------------------------------------------------
class AlphaExecutionEngine:
    """Decorates the original ExecutionEngine with slicing + slippage checks."""

    def __init__(
        self,
        base_engine,  # engine.execution.ExecutionEngine
        slicer: Optional[OrderSlicer] = None,
        slippage: Optional[SlippageModel] = None,
        adaptive_speed: bool = True,
    ) -> None:
        self.base = base_engine
        self.slicer = slicer or OrderSlicer()
        self.slippage = slippage or SlippageModel()
        self.adaptive_speed = bool(adaptive_speed)
        self._parent_counter = 0
        self._parent_counter_lock = threading.Lock()

    # ----------------------------------------------------------------
    def execute(
        self,
        symbol: str,
        action: str,                 # "BUY" or "SELL"
        lots: float,
        entry_price: float,
        atr: float = 0.0,
        adv_lots: float = 0.0,
        volatility_ratio: float = 1.0,
        magic: int = 100000,
    ) -> AlphaExecutionResult:
        """Top-level entry point.

        Returns an `AlphaExecutionResult` with the full slice breakdown
        and slippage estimate, whether or not all slices filled.
        """
        with self._parent_counter_lock:
            self._parent_counter += 1
            parent_id = f"P{self._parent_counter:08d}"
        side = "buy" if action.upper() == "BUY" else "sell"

        # 1. Slippage sanity check
        slip = self.slippage.estimate(
            order_lots=lots, price=entry_price,
            atr=atr, adv_lots=adv_lots, side=side,
        )
        if not slip["ok"]:
            log.warning("SLIPPAGE_REFUSE %s %s lots=%.4f bps=%.2f > max=%.2f",
                        action, symbol, lots, slip["bps"],
                        self.slippage.max_acceptable_bps)
            return AlphaExecutionResult(
                parent_id=parent_id, ok=False,
                total_filled_lots=0.0, avg_fill_price=0.0,
                slippage_estimate=slip,
                reason=f"slippage refusal bps={slip['bps']:.2f}",
            )

        # 2. Slice the parent
        strategy = "adaptive" if self.adaptive_speed else "twap"
        slices = self.slicer.slice(
            parent_lots=lots, side=side, strategy=strategy,
            adv_lots=adv_lots, volatility_ratio=volatility_ratio,
        )
        if not slices:
            return AlphaExecutionResult(
                parent_id=parent_id, ok=False,
                total_filled_lots=0.0, avg_fill_price=0.0,
                slippage_estimate=slip,
                reason="slicer returned 0 slices",
            )

        # 3. Execute each slice via the base engine
        from engine.risk import ApprovedTrade  # local import (avoid cycle)
        from engine.signals import Signal, Action
        slice_records: list[dict[str, Any]] = []
        total_filled = 0.0
        weighted_price = 0.0
        action_enum = Action.BUY if action.upper() == "BUY" else Action.SELL

        # Only paper/simulation runs should collapse inter-slice pacing to a
        # near-zero delay. In live mode, the recommended delay is *the*
        # mechanism that keeps us from moving the book — collapsing it to
        # 0.01s there silently disables the slicer's slippage protection.
        is_paper = env.execution_mode.lower() != "live" or env.simulation_mode
        # Hard ceiling in live mode so a misconfigured slicer can't stall
        # the engine indefinitely (e.g. runaway volatility_ratio multiplier).
        max_live_delay_s = 30.0
        # Major #2 fix: paper mode pacing was capped at 0.01s, which
        # effectively removed all pacing — the slicer's adaptive delay and
        # volatility-based slowdown were both negated. Now paper mode uses
        # 10% of the recommended delay (configurable via env var), which
        # retains relative pacing while still being fast for simulation.
        paper_delay_fraction = float(os.environ.get("PAPER_DELAY_FRACTION", "0.1"))

        for sl in slices:
            # Adaptive pacing: in high vol, wait longer between slices
            delay = sl.recommended_delay_s
            if self.adaptive_speed and volatility_ratio > 1.5:
                delay *= min(3.0, volatility_ratio)
            if delay > 0:
                if is_paper:
                    # Major #2 fix: use a fraction of the delay instead of
                    # capping at 0.01s, so relative pacing is preserved.
                    sleep_for = delay * paper_delay_fraction
                else:
                    sleep_for = min(delay, max_live_delay_s)
                time.sleep(sleep_for)

            # Build a synthetic ApprovedTrade so the base engine stays unchanged
            sig = Signal(
                symbol=symbol, timeframe="M15",
                action=action_enum, strength=0.5,
                price=entry_price, meta={"parent_id": parent_id,
                                          "slice_index": sl.slice_index},
            )
            trade = ApprovedTrade(
                signal=sig, action=action_enum, symbol=symbol,
                lots=sl.lots, entry_price=entry_price,
                stop_loss=0.0, take_profit=0.0,
                risk_amount=0.0, atr_value=atr,
                meta={"parent_id": parent_id},
            )
            try:
                res = self.base.place_order(trade, magic=magic)
                filled = res.ok
                fill_price = res.price
                ticket = res.ticket
                comment = res.comment
                error = ""
            except Exception as e:  # noqa: BLE001
                # A single slice failing (network blip, exchange reject, MT5
                # disconnect, etc.) must not lose the fills already recorded
                # for prior slices in this parent order. Log it, record it
                # as an unfilled slice, and keep going with the remaining
                # slices so the caller gets an accurate partial-fill report.
                log.error(
                    "SLICE_EXCEPTION parent=%s slice=%d symbol=%s lots=%.4f err=%r",
                    parent_id, sl.slice_index, symbol, sl.lots, e,
                )
                filled = False
                fill_price = 0.0
                ticket = None
                comment = f"exception: {e!r}"
                error = str(e)

            slice_records.append({
                "index": sl.slice_index,
                "lots": sl.lots,
                "ok": filled,
                "price": fill_price,
                "ticket": ticket,
                "comment": comment,
                "delay_s": delay,
                "error": error,
            })
            if filled:
                total_filled += sl.lots
                weighted_price += fill_price * sl.lots
            log_trade("slice", parent_id=parent_id,
                      slice=sl.slice_index, symbol=symbol, action=action,
                      lots=sl.lots, price=fill_price, ok=filled)

        avg = (weighted_price / total_filled) if total_filled > 0 else 0.0
        ok = total_filled > 0
        log.info("ALPHA_EXEC parent=%s %s %s filled=%.4f/%.4f avg=%.5f slices=%d",
                 parent_id, action, symbol, total_filled, lots, avg, len(slices))
        return AlphaExecutionResult(
            parent_id=parent_id, ok=ok,
            total_filled_lots=total_filled,
            avg_fill_price=avg,
            slices=slice_records,
            slippage_estimate=slip,
            reason=("partial" if 0 < total_filled < lots
                    else "filled" if ok else "no fills"),
        )