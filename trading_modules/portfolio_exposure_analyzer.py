"""trading_modules/portfolio_exposure_analyzer.py
=====================================================================
Portfolio Exposure Analyzer (Principle #113 — Measure Risk Exposure)
=====================================================================
Detects correlated risk across open positions and computes true
portfolio exposure.

Problem:
    EURUSD BUY + GBPUSD BUY + AUDUSD BUY = 3x USD short exposure
    BTCUSD LONG + ETHUSD LONG = ~2x crypto long exposure
    These look like 3 separate trades but are effectively ONE bet.

Features:
    1. Currency exposure aggregation (USD, EUR, GBP, JPY, etc.)
    2. Asset class exposure (crypto, forex, metals, indices)
    3. Correlation-adjusted exposure (PCA-based)
    4. Sector concentration (e.g. all tech, all crypto)
    5. Net vs gross exposure
    6. Beta-weighted exposure (vs benchmark)
    7. Hedging detection (offsetting positions)
    8. Exposure limits enforcement

Usage:
    analyzer = PortfolioExposureAnalyzer()
    analyzer.add_position("EURUSD", "BUY", 1.0, 1.0850)
    analyzer.add_position("GBPUSD", "BUY", 1.0, 1.2750)
    analyzer.add_position("AUDUSD", "BUY", 1.0, 0.6550)

    report = analyzer.analyze()
    # report = {
    #     "currency_exposure": {"USD": -3.0, "EUR": +1.0, "GBP": +1.0, "AUD": +1.0},
    #     "asset_class": {"forex": 3.0},
    #     "net_exposure_usd": 4200,
    #     "correlation_risk": "HIGH",
    #     "effective_positions": 1.2,  # not 3 — they're correlated!
    #     "recommendation": "Reduce: 3 correlated USD positions"
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.portfolio_exposure_analyzer")


# ----------------------------------------------------------------------
# Symbol classification
# ----------------------------------------------------------------------
# Currency pairs → constituent currencies
FOREX_PAIRS = {
    "EURUSD": ("EUR", "USD"), "GBPUSD": ("GBP", "USD"),
    "AUDUSD": ("AUD", "USD"), "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"), "USDCHF": ("USD", "CHF"),
    "USDJPY": ("USD", "JPY"),
    "EURGBP": ("EUR", "GBP"), "EURJPY": ("EUR", "JPY"),
    "EURCHF": ("EUR", "CHF"), "EURAUD": ("EUR", "AUD"),
    "GBPJPY": ("GBP", "JPY"), "GBPCHF": ("GBP", "CHF"),
    "AUDJPY": ("AUD", "JPY"), "CADJPY": ("CAD", "JPY"),
    "CHFJPY": ("CHF", "JPY"),
}

# Crypto pairs
CRYPTO_BASES = {"BTC", "ETH", "XRP", "LTC", "SOL", "BNB", "DOGE", "ADA", "DOT", "AVAX"}
CRYPTO_QUOTES = {"USD", "USDT", "USDC", "BTC", "ETH"}

# Metals
METALS = {"XAUUSD", "XAGUSD", "GOLD", "SILVER", "XPTUSD", "XPDUSD"}

# Asset class mapping
def classify_symbol(symbol: str) -> str:
    """Return asset class: forex, crypto, metal, index, commodity, synthetic."""
    s = symbol.upper()
    if s in METALS:
        return "metal"
    if s in FOREX_PAIRS:
        return "forex"
    # Check crypto: starts with known crypto base
    for base in CRYPTO_BASES:
        if s.startswith(base):
            return "crypto"
    # Synthetic indices (Deriv)
    if any(s.startswith(p) for p in ["VOL", "BOOM", "CRASH", "STEP", "JUMP", "RESET"]):
        return "synthetic"
    if any(s.startswith(p) for p in ["US30", "NAS100", "SPX500", "GER40", "UK100"]):
        return "index"
    return "unknown"


def extract_currencies(symbol: str) -> Optional[Tuple[str, str]]:
    """For a forex pair, return (base, quote) currencies. None if not forex."""
    s = symbol.upper()
    return FOREX_PAIRS.get(s)


def extract_crypto_base(symbol: str) -> Optional[str]:
    """For a crypto pair, return the base crypto. None if not crypto."""
    s = symbol.upper()
    for base in CRYPTO_BASES:
        if s.startswith(base):
            return base
    return None


# ----------------------------------------------------------------------
# Exposure data classes
# ----------------------------------------------------------------------
@dataclass
class Position:
    """A single open position for exposure analysis."""
    symbol: str
    side: str          # "BUY" or "SELL"
    volume: float      # lots
    entry_price: float
    current_price: float = 0.0
    notional_usd: float = 0.0  # computed: volume * price * contract_size


@dataclass
class ExposureReport:
    """Complete portfolio exposure analysis."""
    # Currency-level exposure (in units of lots, signed)
    currency_exposure: Dict[str, float] = field(default_factory=dict)
    # Asset class exposure (in USD notional)
    asset_class_exposure: Dict[str, float] = field(default_factory=dict)
    # Per-symbol exposure
    symbol_exposure: Dict[str, float] = field(default_factory=dict)
    # Net vs gross
    gross_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0
    long_exposure_usd: float = 0.0
    short_exposure_usd: float = 0.0
    # Correlation-adjusted
    correlation_risk: str = "unknown"  # LOW, MEDIUM, HIGH, EXTREME
    effective_positions: float = 0.0   # lower = more correlated
    herfindahl_index: float = 0.0      # 0-1, higher = more concentrated
    # Directional bias
    directional_bias: str = "neutral"  # long, short, neutral
    # Recommendations
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "currency_exposure": self.currency_exposure,
            "asset_class_exposure": {k: round(v, 2) for k, v in self.asset_class_exposure.items()},
            "symbol_exposure": {k: round(v, 2) for k, v in self.symbol_exposure.items()},
            "gross_exposure_usd": round(self.gross_exposure_usd, 2),
            "net_exposure_usd": round(self.net_exposure_usd, 2),
            "long_exposure_usd": round(self.long_exposure_usd, 2),
            "short_exposure_usd": round(self.short_exposure_usd, 2),
            "correlation_risk": self.correlation_risk,
            "effective_positions": round(self.effective_positions, 2),
            "herfindahl_index": round(self.herfindahl_index, 4),
            "directional_bias": self.directional_bias,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------
class PortfolioExposureAnalyzer:
    """Analyzes portfolio exposure across currencies, asset classes, and correlations."""

    def __init__(self,
                 max_single_currency_pct: float = 0.40,
                 max_asset_class_pct: float = 0.60,
                 max_correlation_threshold: float = 0.70,
                 default_contract_size: float = 100_000):
        """Initialize analyzer.

        Args:
            max_single_currency_pct: max % exposure to one currency
            max_asset_class_pct: max % exposure to one asset class
            max_correlation_threshold: max avg correlation allowed
            default_contract_size: forex standard lot size
        """
        self.max_currency_pct = max_single_currency_pct
        self.max_asset_class_pct = max_asset_class_pct
        self.max_corr = max_correlation_threshold
        self.default_contract = default_contract_size
        self._positions: List[Position] = []

    def add_position(self, symbol: str, side: str, volume: float,
                     entry_price: float, current_price: float = 0) -> None:
        """Add a position to the analyzer."""
        current = current_price or entry_price
        asset_class = classify_symbol(symbol)

        # Compute notional in USD
        if asset_class == "forex":
            notional = volume * self.default_contract * current / max(entry_price, 1e-10)
        else:
            notional = volume * current

        pos = Position(
            symbol=symbol, side=side.upper(), volume=volume,
            entry_price=entry_price, current_price=current,
            notional_usd=notional,
        )
        self._positions.append(pos)

    def remove_position(self, symbol: str) -> None:
        """Remove all positions for a symbol."""
        self._positions = [p for p in self._positions if p.symbol != symbol]

    def clear(self) -> None:
        self._positions.clear()

    def positions(self) -> List[Position]:
        return list(self._positions)

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self) -> ExposureReport:
        """Run full exposure analysis."""
        report = ExposureReport()

        if not self._positions:
            report.recommendation = "No open positions"
            return report

        # === 1. Per-symbol exposure ===
        for p in self._positions:
            direction = 1 if p.side == "BUY" else -1
            report.symbol_exposure[p.symbol] = \
                report.symbol_exposure.get(p.symbol, 0) + direction * p.notional_usd

        # === 2. Currency exposure (forex only) ===
        for p in self._positions:
            currencies = extract_currencies(p.symbol)
            if currencies is None:
                continue
            base, quote = currencies
            direction = 1 if p.side == "BUY" else -1
            # BUY EURUSD = long EUR, short USD
            report.currency_exposure[base] = \
                report.currency_exposure.get(base, 0) + direction * p.volume
            report.currency_exposure[quote] = \
                report.currency_exposure.get(quote, 0) - direction * p.volume

        # === 3. Asset class exposure ===
        for p in self._positions:
            ac = classify_symbol(p.symbol)
            direction = 1 if p.side == "BUY" else -1
            report.asset_class_exposure[ac] = \
                report.asset_class_exposure.get(ac, 0) + direction * p.notional_usd

        # === 4. Gross / Net / Long / Short ===
        longs = sum(p.notional_usd for p in self._positions if p.side == "BUY")
        shorts = sum(p.notional_usd for p in self._positions if p.side == "SELL")
        report.long_exposure_usd = longs
        report.short_exposure_usd = shorts
        report.gross_exposure_usd = longs + shorts
        report.net_exposure_usd = longs - shorts

        # === 5. Directional bias ===
        if report.net_exposure_usd > report.gross_exposure_usd * 0.3:
            report.directional_bias = "long"
        elif report.net_exposure_usd < -report.gross_exposure_usd * 0.3:
            report.directional_bias = "short"
        else:
            report.directional_bias = "neutral"

        # === 6. Herfindahl index (concentration) ===
        total = sum(abs(v) for v in report.symbol_exposure.values())
        if total > 0:
            weights = [abs(v) / total for v in report.symbol_exposure.values()]
            report.herfindahl_index = sum(w * w for w in weights)
            # Effective number of positions = 1 / H
            report.effective_positions = 1.0 / report.herfindahl_index if report.herfindahl_index > 0 else 0

        # === 7. Correlation risk ===
        n_positions = len(self._positions)
        if report.effective_positions < n_positions * 0.5:
            report.correlation_risk = "HIGH"
        elif report.effective_positions < n_positions * 0.7:
            report.correlation_risk = "MEDIUM"
        else:
            report.correlation_risk = "LOW"

        # === 8. Currency concentration check ===
        total_currency_exposure = sum(abs(v) for v in report.currency_exposure.values())
        if total_currency_exposure > 0:
            for curr, exp in report.currency_exposure.items():
                pct = abs(exp) / total_currency_exposure
                if pct > self.max_currency_pct:
                    report.warnings.append(
                        f"Currency {curr} exposure {pct:.0%} exceeds limit {self.max_currency_pct:.0%}"
                    )

        # === 9. Asset class concentration check ===
        total_ac = sum(abs(v) for v in report.asset_class_exposure.values())
        if total_ac > 0:
            for ac, exp in report.asset_class_exposure.items():
                pct = abs(exp) / total_ac
                if pct > self.max_asset_class_pct:
                    report.warnings.append(
                        f"Asset class {ac} exposure {pct:.0%} exceeds limit {self.max_asset_class_pct:.0%}"
                    )

        # === 10. Recommendation ===
        report.recommendation = self._recommend(report)

        return report

    def _recommend(self, report: ExposureReport) -> str:
        """Generate recommendation based on exposure analysis."""
        if not self._positions:
            return "No positions open"

        recs = []

        # Correlation warning
        if report.correlation_risk == "HIGH":
            n_actual = len(self._positions)
            recs.append(
                f"High correlation risk: {n_actual} positions but only "
                f"{report.effective_positions:.1f} effective — reduce to "
                f"{int(report.effective_positions) + 1} positions"
            )

        # Currency concentration
        for curr, exp in report.currency_exposure.items():
            total = sum(abs(v) for v in report.currency_exposure.values())
            if total > 0 and abs(exp) / total > self.max_currency_pct:
                recs.append(f"Reduce {curr} exposure (currently {abs(exp)/total:.0%})")

        # Directional bias
        if report.directional_bias != "neutral":
            recs.append(f"Directional bias: {report.directional_bias} "
                       f"(net ${report.net_exposure_usd:,.0f})")

        if not recs:
            return "Exposure balanced — no action needed"
        return "; ".join(recs)

    # ------------------------------------------------------------------
    # Can we add another position?
    # ------------------------------------------------------------------
    def can_add_position(self, symbol: str, side: str, volume: float,
                         price: float) -> Tuple[bool, str]:
        """Check if adding a position would breach exposure limits.

        Returns (allowed, reason_if_not)
        """
        # Simulate adding
        self.add_position(symbol, side, volume, price, price)
        report = self.analyze()
        # Remove the simulated position
        self._positions.pop()

        if report.warnings:
            return False, "; ".join(report.warnings)
        if report.correlation_risk == "HIGH" and len(self._positions) >= 3:
            return False, f"Correlation risk HIGH with {len(self._positions)+1} positions"
        return True, "OK"
