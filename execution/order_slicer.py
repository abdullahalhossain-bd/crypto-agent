"""execution.order_slicer
=====================================================================
Day 18 — Order slicer.

Breaks a large parent order into smaller child orders spaced over
time, so we don't slam the book with one big market order.

Strategies:
  - "twap" : equal-sized slices at equal time intervals
  - "vpov" : volume-weighted (uses a fake "volume profile" when real
             ADV is unavailable — uniform by default)
  - "adaptive" : starts small (probe), then ramps up if first slice
                 fills cleanly; backs off if slippage is bad

The slicer is intentionally a pure data structure — it doesn't touch
MT5. The execution engine pulls child orders off it as needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SlicedOrder:
    """One child order produced by the slicer."""
    slice_index: int
    lots: float
    side: str
    recommended_delay_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class OrderSlicer:
    def __init__(self, max_slice_lots: float = 0.5,
                 min_slices: int = 1,
                 max_slices: int = 10,
                 default_interval_s: float = 5.0) -> None:
        self.max_slice_lots = float(max_slice_lots)
        self.min_slices = int(min_slices)
        self.max_slices = int(max_slices)
        self.default_interval_s = float(default_interval_s)

    # ----------------------------------------------------------------
    def slice(self, parent_lots: float, side: str,
              strategy: str = "twap",
              adv_lots: float = 0.0,
              volatility_ratio: float = 1.0) -> list[SlicedOrder]:
        """Return the list of child orders for a parent."""
        parent_lots = float(parent_lots)
        if parent_lots <= 0:
            return []
        side = side.lower()

        # Decide slice count: bounded by min/max and by max_slice_lots
        n_by_size = max(1, int(math.ceil(parent_lots / self.max_slice_lots)))
        # In high vol, slice more (slower execution)
        vol_multiplier = max(1.0, volatility_ratio)
        n_by_vol = max(1, int(math.ceil(n_by_size * vol_multiplier)))
        n = int(max(self.min_slices, min(self.max_slices, n_by_vol)))

        if strategy == "twap":
            return self._twap(parent_lots, side, n)
        if strategy == "vpov":
            return self._vpov(parent_lots, side, n, adv_lots)
        if strategy == "adaptive":
            return self._adaptive(parent_lots, side, n, volatility_ratio)
        raise ValueError(f"unknown slicer strategy: {strategy}")

    # ----------------------------------------------------------------
    def _twap(self, lots: float, side: str, n: int) -> list[SlicedOrder]:
        per = lots / n
        out = []
        for i in range(n):
            out.append(SlicedOrder(
                slice_index=i, lots=per, side=side,
                recommended_delay_s=self.default_interval_s,
                metadata={"strategy": "twap"},
            ))
        return out

    def _vpov(self, lots: float, side: str, n: int,
              adv_lots: float) -> list[SlicedOrder]:
        # Without a real intraday volume profile we default to equal slices
        # but include the participation hint in metadata.
        per = lots / n
        participation = (per / adv_lots) if adv_lots > 0 else 0.0
        return [
            SlicedOrder(slice_index=i, lots=per, side=side,
                        recommended_delay_s=self.default_interval_s,
                        metadata={"strategy": "vpov",
                                  "participation": participation})
            for i in range(n)
        ]

    def _adaptive(self, lots: float, side: str, n: int,
                  vol_ratio: float) -> list[SlicedOrder]:
        """First slice is 1/2n, middle slices are 1/n each, last slice
        is the remainder. This probes the book before committing."""
        if n < 2:
            return self._twap(lots, side, n)
        first = lots / (2 * n)
        middle = lots / n
        last = lots - first - middle * (n - 2)
        out = [SlicedOrder(0, first, side,
                            self.default_interval_s * max(1.0, vol_ratio),
                            {"strategy": "adaptive", "phase": "probe"})]
        for i in range(1, n - 1):
            out.append(SlicedOrder(i, middle, side,
                                   self.default_interval_s,
                                   {"strategy": "adaptive", "phase": "main"}))
        out.append(SlicedOrder(n - 1, last, side,
                               self.default_interval_s,
                               {"strategy": "adaptive", "phase": "cleanup"}))
        return out
