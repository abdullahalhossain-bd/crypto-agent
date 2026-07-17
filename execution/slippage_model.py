"""execution.slippage_model
=====================================================================
Day 17 — Slippage estimation model.

Predicts expected slippage (in price units) for a given order size,
current spread, ATR, and volatility regime. Used by the alpha
execution layer to:
  - Decide whether to slice an order
  - Quote a realistic expected fill price for backtest comparison
  - Refuse orders whose predicted slippage would erase the edge

The model is intentionally simple (linear in size + vol term) so it
trains fast and stays interpretable. Replace with a learned model
later if you have enough fill data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SlippageModel:
    """Linear slippage model:
        slip = base_bps * sqrt(participation) + vol_bps * (atr / price) * 100
    """
    base_bps: float = 1.5          # fixed cost per fill
    vol_bps_per_atr: float = 8.0   # volatility multiplier
    spread_bps: float = 0.5        # half-spread penalty
    max_acceptable_bps: float = 25.0  # refuse if predicted > this

    def estimate(
        self,
        order_lots: float,
        price: float,
        atr: float,
        adv_lots: float = 0.0,
        side: str = "buy",
    ) -> dict[str, float]:
        """Return predicted slippage in {bps, price_units, percent}."""
        if price <= 0:
            return {"bps": 0.0, "price_units": 0.0, "percent": 0.0, "ok": True}

        # Participation rate (fraction of ADV); fallback to 0.01 if no ADV
        participation = (order_lots / adv_lots) if adv_lots > 0 else 0.01
        participation = min(1.0, max(0.0, participation))

        # Square-root market impact (Almgren-Chriss lite)
        impact_bps = self.base_bps * (participation ** 0.5)
        vol_bps = self.vol_bps_per_atr * (atr / price) * 100.0
        spread = self.spread_bps
        total_bps = impact_bps + vol_bps + spread
        price_units = total_bps / 10_000.0 * price
        return {
            "bps": float(total_bps),
            "price_units": float(price_units),
            "percent": float(total_bps / 100.0),
            "ok": total_bps <= self.max_acceptable_bps,
            "impact_bps": float(impact_bps),
            "vol_bps": float(vol_bps),
            "spread_bps": float(spread),
        }

    def adjusted_fill_price(self, side: str, price: float,
                            slip_price_units: float) -> float:
        """Apply slippage to a reference price.

        For BUY we pay more; for SELL we receive less.
        """
        if side.lower() in ("buy", "long"):
            return price + slip_price_units
        return price - slip_price_units
