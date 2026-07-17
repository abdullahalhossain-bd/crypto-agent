"""enhancements.sector_rotation
=====================================================================
Inspired by OpenAlice's sector-rotation module.

Cross-sectional comparison of sector ETFs on multi-period momentum
and volume axes, to detect where capital is rotating.

For crypto (not traditional equities), we adapt the concept:
  - Instead of GICS sector ETFs, we compare major crypto sectors:
    BTC (store of value), ETH (smart contracts), SOL (L1 alt),
    BNB (exchange), XRP (payments), ADA (research), AVAX (L1),
    DOT (parachains), LINK (oracles), MATIC (L2), UNI (DeFi)
  - Benchmark = BTC (the crypto market's "SPY")

Output: ranked table of sectors by:
  - Multi-period returns (1D, 1W, 1M, 3M, 6M)
  - Relative strength vs benchmark
  - Momentum acceleration (1W pace vs 3M pace)
  - Dollar volume share
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("enhancements.sector_rotation")


# ----------------------------------------------------------------------
# Crypto sector universe (analogous to GICS sector ETFs)
# ----------------------------------------------------------------------
CRYPTO_SECTORS: list[dict[str, str]] = [
    {"symbol": "BTCUSD", "sector": "Store of Value", "name": "Bitcoin"},
    {"symbol": "ETHUSD", "sector": "Smart Contracts", "name": "Ethereum"},
    {"symbol": "SOLUSD", "sector": "L1 Alternative", "name": "Solana"},
    {"symbol": "BNBUSD", "sector": "Exchange Token", "name": "BNB"},
    {"symbol": "XRPUSD", "sector": "Payments", "name": "Ripple"},
    {"symbol": "ADAUSD", "sector": "Research L1", "name": "Cardano"},
    {"symbol": "AVAXUSD", "sector": "L1 Alternative", "name": "Avalanche"},
    {"symbol": "DOTUSD", "sector": "Parachains", "name": "Polkadot"},
    {"symbol": "LINKUSD", "sector": "Oracles", "name": "Chainlink"},
    {"symbol": "MATICUSD", "sector": "L2 Scaling", "name": "Polygon"},
    {"symbol": "UNIUSD", "sector": "DeFi", "name": "Uniswap"},
]

BENCHMARK_SYMBOL = "BTCUSD"

PERIOD_DAYS = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "6M": 180}
VOLUME_BASELINE_DAYS = 20


# ----------------------------------------------------------------------
@dataclass
class SectorRotationRow:
    symbol: str
    sector: str
    name: str
    returns: dict[str, Optional[float]] = field(default_factory=dict)
    rel_strength: dict[str, Optional[float]] = field(default_factory=dict)
    momentum_acceleration: Optional[float] = None
    dollar_volume: Optional[float] = None
    dv_share: Optional[float] = None
    rvol: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "name": self.name,
            "returns": {k: v for k, v in self.returns.items()},
            "rel_strength": {k: v for k, v in self.rel_strength.items()},
            "momentum_acceleration": self.momentum_acceleration,
            "dollar_volume": self.dollar_volume,
            "dv_share": self.dv_share,
            "rvol": self.rvol,
        }


# ----------------------------------------------------------------------
class SectorRotationAnalyzer:
    """Compute sector rotation table from OHLCV data."""

    def __init__(self, sectors: Optional[list[dict]] = None,
                 benchmark: str = BENCHMARK_SYMBOL) -> None:
        self.sectors = sectors or CRYPTO_SECTORS
        self.benchmark = benchmark

    # ----------------------------------------------------------------
    def compute(self, ohlcv_by_symbol: dict[str, pd.DataFrame]) -> list[SectorRotationRow]:
        """Compute rotation table.

        Args:
            ohlcv_by_symbol: {symbol: DataFrame with time/open/high/low/close/volume}
        """
        rows: list[SectorRotationRow] = []
        # Benchmark returns
        bench_returns = self._compute_returns(ohlcv_by_symbol.get(self.benchmark))
        # Total dollar volume (for share calculation)
        total_dv = 0.0
        for s in self.sectors:
            df = ohlcv_by_symbol.get(s["symbol"])
            if df is None or df.empty:
                continue
            dv = float((df["close"] * df["volume"]).tail(1).iloc[0]) if "volume" in df.columns else 0.0
            total_dv += dv
        # Per-sector rows
        for s in self.sectors:
            df = ohlcv_by_symbol.get(s["symbol"])
            if df is None or df.empty:
                rows.append(SectorRotationRow(
                    symbol=s["symbol"], sector=s["sector"], name=s["name"],
                ))
                continue
            returns = self._compute_returns(df)
            rel_strength = {}
            for period, ret in returns.items():
                bench = bench_returns.get(period)
                if ret is not None and bench is not None:
                    rel_strength[period] = ret - bench
                else:
                    rel_strength[period] = None
            # Momentum acceleration
            r_1w = returns.get("1W")
            r_3m = returns.get("3M")
            if r_1w is not None and r_3m is not None and r_3m != 0:
                # Per-day pace
                pace_1w = r_1w / PERIOD_DAYS["1W"]
                pace_3m = r_3m / PERIOD_DAYS["3M"]
                accel = pace_1w - pace_3m
            else:
                accel = None
            # Dollar volume + share
            dv = float((df["close"] * df["volume"]).tail(1).iloc[0]) if "volume" in df.columns else None
            dv_share = (dv / total_dv) if (dv is not None and total_dv > 0) else None
            # RVOL
            rvol_val = None
            if "volume" in df.columns and len(df) >= VOLUME_BASELINE_DAYS + 1:
                vol = df["volume"]
                prior_avg = vol.shift(1).rolling(VOLUME_BASELINE_DAYS,
                                                   min_periods=VOLUME_BASELINE_DAYS).mean()
                latest_rvol = float(vol.iloc[-1] / prior_avg.iloc[-1]) if not prior_avg.isna().iloc[-1] and prior_avg.iloc[-1] > 0 else None
                rvol_val = latest_rvol
            rows.append(SectorRotationRow(
                symbol=s["symbol"], sector=s["sector"], name=s["name"],
                returns=returns, rel_strength=rel_strength,
                momentum_acceleration=accel,
                dollar_volume=dv, dv_share=dv_share, rvol=rvol_val,
            ))
        # Sort by 1M relative strength (strongest first)
        rows.sort(key=lambda r: r.rel_strength.get("1M") or -999, reverse=True)
        return rows

    # ----------------------------------------------------------------
    @staticmethod
    def _compute_returns(df: Optional[pd.DataFrame]) -> dict[str, Optional[float]]:
        if df is None or df.empty:
            return {p: None for p in PERIOD_DAYS}
        close = df["close"]
        out: dict[str, Optional[float]] = {}
        for period, days in PERIOD_DAYS.items():
            if len(close) > days:
                prior = close.iloc[-days - 1]
                if prior > 0:
                    out[period] = float((close.iloc[-1] - prior) / prior)
                else:
                    out[period] = None
            else:
                out[period] = None
        return out

    # ----------------------------------------------------------------
    def summary(self, rows: list[SectorRotationRow]) -> dict[str, Any]:
        """Produce a summary interpretation."""
        if not rows:
            return {"status": "no_data"}
        # Leaders (top 3 by 1M rel strength)
        leaders = [r for r in rows if r.rel_strength.get("1M") is not None][:3]
        # Laggards (bottom 3)
        laggards = [r for r in reversed(rows) if r.rel_strength.get("1M") is not None][:3]
        # Accelerating sectors
        accelerating = [r for r in rows if (r.momentum_acceleration or 0) > 0]
        return {
            "leaders": [r.to_dict() for r in leaders],
            "laggards": [r.to_dict() for r in laggards],
            "accelerating_count": len(accelerating),
            "total_sectors": len(rows),
            "benchmark": self.benchmark,
        }
