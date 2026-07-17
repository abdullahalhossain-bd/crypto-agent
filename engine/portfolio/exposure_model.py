"""engine.portfolio.exposure_model
=====================================================================
Day 11 — Exposure model.

Tracks net directional exposure per symbol and per side (long/short).
Exposure is measured in *risk units* — i.e. fraction of equity at
risk — rather than raw lot count, so it's comparable across symbols
with different contract sizes or volatilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

log = get_logger("engine.portfolio.exposure")


@dataclass
class Exposure:
    """Per-symbol exposure snapshot."""
    symbol: str
    net_lots: float = 0.0
    long_lots: float = 0.0
    short_lots: float = 0.0
    notional: float = 0.0
    risk_units: float = 0.0  # |lots| * ATR_multiple / equity
    side: str = "flat"       # "long" | "short" | "flat"

    def merge(self, other: "Exposure") -> "Exposure":
        return Exposure(
            symbol=self.symbol,
            net_lots=self.net_lots + other.net_lots,
            long_lots=self.long_lots + other.long_lots,
            short_lots=self.short_lots + other.short_lots,
            notional=self.notional + other.notional,
            risk_units=self.risk_units + other.risk_units,
            side=("long" if self.net_lots + other.net_lots > 0
                  else "short" if self.net_lots + other.net_lots < 0 else "flat"),
        )


class ExposureModel:
    """Maintains the current exposure map across all open positions."""

    def __init__(self, max_gross_exposure: float = 2.0,
                 max_net_exposure: float = 1.0) -> None:
        # In risk units (fraction of equity at risk)
        self.max_gross = float(max_gross_exposure)
        self.max_net = float(max_net_exposure)
        self._exposures: dict[str, Exposure] = {}

    # ----------------------------------------------------------------
    def update_position(self, symbol: str, net_lots: float,
                        long_lots: float = 0.0, short_lots: float = 0.0,
                        notional: float = 0.0, risk_units: float = 0.0) -> None:
        self._exposures[symbol] = Exposure(
            symbol=symbol,
            net_lots=net_lots,
            long_lots=long_lots,
            short_lots=short_lots,
            notional=notional,
            risk_units=risk_units,
            side=("long" if net_lots > 0 else "short" if net_lots < 0 else "flat"),
        )

    def remove_position(self, symbol: str) -> None:
        self._exposures.pop(symbol, None)

    def get(self, symbol: str) -> Optional[Exposure]:
        return self._exposures.get(symbol)

    def all_exposures(self) -> dict[str, Exposure]:
        return dict(self._exposures)

    # ----------------------------------------------------------------
    @property
    def gross_exposure(self) -> float:
        """Sum of |risk_units| across all symbols."""
        return sum(abs(e.risk_units) for e in self._exposures.values())

    @property
    def net_exposure(self) -> float:
        """Signed sum of risk_units."""
        return sum(e.risk_units for e in self._exposures.values())

    @property
    def long_exposure(self) -> float:
        return sum(e.risk_units for e in self._exposures.values() if e.net_lots > 0)

    @property
    def short_exposure(self) -> float:
        return sum(abs(e.risk_units) for e in self._exposures.values() if e.net_lots < 0)

    @property
    def n_open_positions(self) -> int:
        return sum(1 for e in self._exposures.values() if abs(e.net_lots) > 1e-9)

    # ----------------------------------------------------------------
    def would_breach(self, additional_risk: float, side: str = "long") -> dict[str, bool]:
        """Check if adding `additional_risk` would breach limits."""
        new_gross = self.gross_exposure + abs(additional_risk)
        signed = additional_risk if side == "long" else -additional_risk
        new_net = self.net_exposure + signed
        return {
            "gross_breach": new_gross > self.max_gross,
            "net_breach": abs(new_net) > self.max_net,
            "new_gross": new_gross,
            "new_net": new_net,
        }

    def to_dict(self) -> dict[str, dict]:
        return {sym: e.__dict__ for sym, e in self._exposures.items()}
