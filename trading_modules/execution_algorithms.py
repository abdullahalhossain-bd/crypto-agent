"""
Execution Algorithms — institutional order execution
====================================================

Pure-Python execution algorithm implementations for splitting large orders
into smaller child orders to minimize market impact:

    1. TWAP    — Time-Weighted Average Price (uniform time-slicing)
    2. VWAP    — Volume-Weighted Average Price (volume-curve-aware)
    3. POV     — Percentage of Volume (participate at x% of market volume)
    4. IS      — Implementation Shortfall (front-load when cost is low)
    5. Sniper  — wait for liquidity, then strike (minimize footprint)

Each algorithm returns a list of (timestamp, quantity) child orders.

Usage:
    from trading_modules.execution_algorithms import (
        twap_schedule, vwap_schedule, pov_schedule, is_schedule, sniper_schedule
    )
    # Split 1000 BTC over 8 hours using TWAP
    schedule = twap_schedule(total_qty=1000, start_time=..., end_time=..., n_slices=48)
    for slice_time, qty in schedule:
        broker.send_order(symbol, qty, time=slice_time)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionSlice:
    timestamp: datetime
    quantity: float
    reason: str = ""


@dataclass
class ExecutionPlan:
    algorithm: str
    slices: list[ExecutionSlice] = field(default_factory=list)
    total_quantity: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "total_quantity": self.total_quantity,
            "n_slices": len(self.slices),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "slices": [
                {"timestamp": s.timestamp.isoformat(), "quantity": round(s.quantity, 6), "reason": s.reason}
                for s in self.slices
            ],
            "notes": self.notes,
        }


# ──────────────────────────────────────────────────────────────────────
# 1. TWAP
# ──────────────────────────────────────────────────────────────────────
def twap_schedule(
    total_qty: float,
    start_time: datetime,
    end_time: datetime,
    n_slices: int = 48,
) -> ExecutionPlan:
    """TWAP — uniformly distribute order over time.

    Args:
        total_qty: total quantity to execute
        start_time: when to start
        end_time: when to finish
        n_slices: number of child orders (default 48 = every 10 min over 8 hours)
    """
    if n_slices < 1:
        n_slices = 1
    duration = (end_time - start_time).total_seconds()
    if duration <= 0:
        return ExecutionPlan(
            algorithm="twap", slices=[ExecutionSlice(start_time, total_qty)],
            total_quantity=total_qty, start_time=start_time, end_time=end_time,
            notes=["invalid duration → single slice"],
        )
    slice_interval = duration / n_slices
    per_slice_qty = total_qty / n_slices
    slices = [
        ExecutionSlice(
            timestamp=start_time + timedelta(seconds=i * slice_interval),
            quantity=per_slice_qty,
            reason=f"twap slice {i + 1}/{n_slices}",
        )
        for i in range(n_slices)
    ]
    return ExecutionPlan(
        algorithm="twap", slices=slices, total_quantity=total_qty,
        start_time=start_time, end_time=end_time,
        notes=[f"{n_slices} uniform slices, {per_slice_qty:.6f} each"],
    )


# ──────────────────────────────────────────────────────────────────────
# 2. VWAP
# ──────────────────────────────────────────────────────────────────────
def vwap_schedule(
    total_qty: float,
    start_time: datetime,
    end_time: datetime,
    volume_curve: np.ndarray,
    n_slices: int = 48,
) -> ExecutionPlan:
    """VWAP — slice according to historical intraday volume distribution.

    Args:
        total_qty: total quantity
        start_time, end_time: execution window
        volume_curve: array of relative volumes per time bucket (e.g., 24 hourly values)
                      — sums don't need to be 1; they'll be normalized
        n_slices: number of child orders
    """
    volume_curve = np.asarray(volume_curve, dtype=float)
    if len(volume_curve) == 0:
        return twap_schedule(total_qty, start_time, end_time, n_slices)
    # Re-sample volume_curve to n_slices
    if len(volume_curve) != n_slices:
        # Simple interpolation
        old_idx = np.linspace(0, len(volume_curve) - 1, len(volume_curve))
        new_idx = np.linspace(0, len(volume_curve) - 1, n_slices)
        volume_curve = np.interp(new_idx, old_idx, volume_curve)
    # Normalize
    total = volume_curve.sum()
    if total <= 0:
        return twap_schedule(total_qty, start_time, end_time, n_slices)
    weights = volume_curve / total
    duration = (end_time - start_time).total_seconds()
    slice_interval = duration / n_slices
    slices = []
    for i in range(n_slices):
        slices.append(ExecutionSlice(
            timestamp=start_time + timedelta(seconds=i * slice_interval),
            quantity=total_qty * weights[i],
            reason=f"vwap slice {i + 1}/{n_slices} (vol share {weights[i]:.1%})",
        ))
    return ExecutionPlan(
        algorithm="vwap", slices=slices, total_quantity=total_qty,
        start_time=start_time, end_time=end_time,
        notes=[f"{n_slices} volume-weighted slices"],
    )


# ──────────────────────────────────────────────────────────────────────
# 3. POV (Percentage of Volume)
# ──────────────────────────────────────────────────────────────────────
def pov_schedule(
    total_qty: float,
    start_time: datetime,
    end_time: datetime,
    participation_rate: float = 0.1,
    expected_market_volume_per_slice: Optional[np.ndarray] = None,
    n_slices: int = 48,
) -> ExecutionPlan:
    """POV — participate at a fixed % of market volume.

    Args:
        total_qty: target quantity (capped — if market can't absorb, you'll execute less)
        participation_rate: fraction of market volume (0.1 = 10%)
        expected_market_volume_per_slice: array of expected market volumes per slice
    """
    if expected_market_volume_per_slice is None:
        # Assume uniform
        expected_market_volume_per_slice = np.ones(n_slices)
    expected_market_volume_per_slice = np.asarray(expected_market_volume_per_slice, dtype=float)
    duration = (end_time - start_time).total_seconds()
    slice_interval = duration / n_slices
    slices = []
    remaining = total_qty
    for i in range(n_slices):
        max_qty = expected_market_volume_per_slice[i] * participation_rate
        qty = min(remaining, max_qty)
        if qty > 0:
            slices.append(ExecutionSlice(
                timestamp=start_time + timedelta(seconds=i * slice_interval),
                quantity=qty,
                reason=f"pov slice {i + 1}/{n_slices} (max {max_qty:.4f} @ {participation_rate:.0%})",
            ))
            remaining -= qty
        if remaining <= 0:
            break
    notes = [f"participation rate {participation_rate:.0%}"]
    if remaining > 0:
        notes.append(f"WARNING: {remaining:.4f} qty unexecuted (market volume too low)")
    return ExecutionPlan(
        algorithm="pov", slices=slices, total_quantity=total_qty - remaining,
        start_time=start_time, end_time=end_time, notes=notes,
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Implementation Shortfall (IS)
# ──────────────────────────────────────────────────────────────────────
def is_schedule(
    total_qty: float,
    start_time: datetime,
    end_time: datetime,
    risk_aversion: float = 0.5,
    n_slices: int = 48,
) -> ExecutionPlan:
    """Implementation Shortfall — front-load when risk aversion is high.

    Higher risk_aversion → front-load execution to reduce timing risk.
    Lower risk_aversion → back-load to wait for better prices.

    The schedule follows a decay curve:
        w(t) ∝ exp(-k * (1 - t))  where k = risk_aversion × 2
    """
    if n_slices < 1:
        n_slices = 1
    t = np.linspace(0, 1, n_slices)
    k = risk_aversion * 2.0
    weights = np.exp(-k * (1 - t))
    weights = weights / weights.sum()
    duration = (end_time - start_time).total_seconds()
    slice_interval = duration / n_slices
    slices = [
        ExecutionSlice(
            timestamp=start_time + timedelta(seconds=i * slice_interval),
            quantity=total_qty * weights[i],
            reason=f"is slice {i + 1}/{n_slices} (weight {weights[i]:.1%})",
        )
        for i in range(n_slices)
    ]
    return ExecutionPlan(
        algorithm="is", slices=slices, total_quantity=total_qty,
        start_time=start_time, end_time=end_time,
        notes=[f"risk_aversion={risk_aversion}, front-loaded" if risk_aversion > 0.5 else "back-loaded"],
    )


# ──────────────────────────────────────────────────────────────────────
# 5. Sniper
# ──────────────────────────────────────────────────────────────────────
def sniper_schedule(
    total_qty: float,
    start_time: datetime,
    end_time: datetime,
    liquidity_threshold: float = 1.5,
    expected_liquidity_curve: Optional[np.ndarray] = None,
    n_slices: int = 48,
) -> ExecutionPlan:
    """Sniper — wait for liquidity spikes, then strike.

    Only executes when liquidity is above `liquidity_threshold` × average.
    Splits the order among the top-N most liquid moments.

    Args:
        liquidity_threshold: minimum liquidity multiple (1.5 = 1.5x average)
        expected_liquidity_curve: array of expected liquidity per slice
    """
    if expected_liquidity_curve is None:
        # Default: assume higher liquidity at open and close
        t = np.linspace(0, 1, n_slices)
        expected_liquidity_curve = 1.0 + 0.5 * np.exp(-((t - 0.1) ** 2) / 0.02) + \
                                    0.8 * np.exp(-((t - 0.9) ** 2) / 0.02)
    expected_liquidity_curve = np.asarray(expected_liquidity_curve, dtype=float)
    avg_liq = float(expected_liquidity_curve.mean())
    if avg_liq <= 0:
        return twap_schedule(total_qty, start_time, end_time, n_slices)
    # Find slices where liquidity exceeds threshold
    high_liq_mask = expected_liquidity_curve >= liquidity_threshold * avg_liq
    if not high_liq_mask.any():
        # Lower threshold to 1.0 if no slices qualify
        high_liq_mask = expected_liquidity_curve >= avg_liq
        if not high_liq_mask.any():
            return twap_schedule(total_qty, start_time, end_time, n_slices)
    # Allocate proportionally to liquidity among qualifying slices
    qualifying_volumes = expected_liquidity_curve[high_liq_mask]
    weights = qualifying_volumes / qualifying_volumes.sum()
    qualifying_indices = np.where(high_liq_mask)[0]
    duration = (end_time - start_time).total_seconds()
    slice_interval = duration / n_slices
    slices = []
    for idx, w in zip(qualifying_indices, weights):
        slices.append(ExecutionSlice(
            timestamp=start_time + timedelta(seconds=idx * slice_interval),
            quantity=total_qty * w,
            reason=f"sniper strike @ slice {idx + 1} (liq {expected_liquidity_curve[idx]:.2f}x avg)",
        ))
    return ExecutionPlan(
        algorithm="sniper", slices=slices, total_quantity=total_qty,
        start_time=start_time, end_time=end_time,
        notes=[f"{len(slices)} strikes (threshold={liquidity_threshold}x avg)"],
    )


__all__ = [
    "ExecutionSlice", "ExecutionPlan",
    "twap_schedule", "vwap_schedule", "pov_schedule",
    "is_schedule", "sniper_schedule",
]
