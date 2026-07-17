"""trading_modules/portfolio_intelligence_layer.py
=====================================================================
Portfolio Intelligence Layer (Principle #137 — Portfolio-Level Intelligence)
=====================================================================
Provides a unified view of the entire portfolio: positions, risk, exposure,
correlations, and diversification quality.

What It Computes:
    1. CORRELATION MATRIX — rolling correlation between all open positions
    2. CURRENCY EXPOSURE — net exposure per currency (USD, EUR, GBP, etc.)
    3. NET RISK BUDGET — how much risk is currently allocated
    4. DIVERSIFICATION SCORE — 0-1, higher = more diversified
    5. SECTOR CONCENTRATION — Herfindahl index per asset class
    6. HEDGE DETECTION — identify offsetting positions
    7. PORTFOLIO BETA — vs benchmark (e.g., BTC or SPX)
    8. TAIL RISK — estimated portfolio VaR and CVaR

Output:
    PortfolioIntelligence report with all metrics + recommendations

Usage:
    intel = PortfolioIntelligenceLayer()

    # Add positions
    intel.add_position("BTCUSD", "BUY", 0.5, 43250)
    intel.add_position("ETHUSD", "BUY", 3.0, 2580)
    intel.add_position("EURUSD", "SELL", 1.0, 1.085)

    # Add price history for correlation
    intel.set_price_history({
        "BTCUSD": df_btc,
        "ETHUSD": df_eth,
        "EURUSD": df_eur,
    })

    report = intel.analyze()
    # report = {
    #     "diversification_score": 0.65,
    #     "correlation_risk": "MEDIUM",
    #     "net_risk_budget_pct": 4.5,
    #     "portfolio_beta": 0.85,
    #     "var_95": 350.00,
    #     "recommendation": "..."
    # }
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.portfolio_intelligence_layer")


@dataclass
class PortfolioIntelligenceReport:
    """Complete portfolio intelligence report."""
    # Position summary
    total_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    gross_exposure_usd: float = 0.0
    net_exposure_usd: float = 0.0

    # Risk
    net_risk_budget_pct: float = 0.0    # sum of position risks / equity
    portfolio_var_95: float = 0.0       # Value at Risk (95% confidence)
    portfolio_cvar_95: float = 0.0      # Conditional VaR (expected shortfall)
    portfolio_beta: float = 0.0         # vs benchmark

    # Correlation
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    avg_correlation: float = 0.0
    correlation_risk: str = "unknown"   # LOW, MEDIUM, HIGH

    # Diversification
    diversification_score: float = 0.0  # 0-1
    herfindahl_index: float = 0.0       # 0-1, higher = more concentrated
    effective_positions: float = 0.0    # 1/H, lower = more correlated

    # Currency/sector exposure
    currency_exposure: Dict[str, float] = field(default_factory=dict)
    sector_exposure: Dict[str, float] = field(default_factory=dict)

    # Hedging
    hedges_detected: List[Tuple[str, str, float]] = field(default_factory=list)
    hedge_coverage_pct: float = 0.0

    # Recommendations
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_positions": self.total_positions,
            "long_positions": self.long_positions,
            "short_positions": self.short_positions,
            "gross_exposure_usd": round(self.gross_exposure_usd, 2),
            "net_exposure_usd": round(self.net_exposure_usd, 2),
            "net_risk_budget_pct": round(self.net_risk_budget_pct, 2),
            "portfolio_var_95": round(self.portfolio_var_95, 2),
            "portfolio_cvar_95": round(self.portfolio_cvar_95, 2),
            "portfolio_beta": round(self.portfolio_beta, 3),
            "avg_correlation": round(self.avg_correlation, 3),
            "correlation_risk": self.correlation_risk,
            "diversification_score": round(self.diversification_score, 3),
            "herfindahl_index": round(self.herfindahl_index, 4),
            "effective_positions": round(self.effective_positions, 2),
            "currency_exposure": self.currency_exposure,
            "sector_exposure": {k: round(v, 2) for k, v in self.sector_exposure.items()},
            "hedges_detected": [(a, b, round(c, 3)) for a, b, c in self.hedges_detected],
            "hedge_coverage_pct": round(self.hedge_coverage_pct, 2),
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


class PortfolioIntelligenceLayer:
    """Portfolio-level intelligence: correlations, risk, diversification."""

    def __init__(self,
                 equity: float = 10000.0,
                 benchmark: str = "BTCUSD",
                 max_risk_budget_pct: float = 10.0,
                 var_confidence: float = 0.95,
                 var_lookback: int = 100):
        """Initialize intelligence layer.

        Args:
            equity: current account equity
            benchmark: benchmark symbol for beta calculation
            max_risk_budget_pct: max total risk as % of equity
            var_confidence: VaR confidence level (0.95 or 0.99)
            var_lookback: bars to look back for VaR calc
        """
        self.equity = equity
        self.benchmark = benchmark
        self.max_risk_budget = max_risk_budget_pct
        self.var_confidence = var_confidence
        self.var_lookback = var_lookback

        self._lock = threading.RLock()
        self._positions: Dict[str, dict] = {}  # symbol → position details
        self._price_history: Dict[str, pd.DataFrame] = {}
        self._benchmark_returns: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def add_position(self, symbol: str, side: str, volume: float,
                     entry_price: float, sl: float = 0,
                     current_price: float = 0) -> None:
        """Add or update a position."""
        with self._lock:
            self._positions[symbol] = {
                "symbol": symbol, "side": side.upper(),
                "volume": volume, "entry_price": entry_price,
                "current_price": current_price or entry_price,
                "sl": sl,
                "notional_usd": volume * (current_price or entry_price),
                "risk_usd": abs(entry_price - sl) * volume if sl > 0 else 0,
            }

    def remove_position(self, symbol: str) -> None:
        with self._lock:
            self._positions.pop(symbol, None)

    def set_price_history(self, dfs: Dict[str, pd.DataFrame]) -> None:
        """Set price history for correlation + VaR computation."""
        with self._lock:
            self._price_history = dfs
            # Compute benchmark returns
            if self.benchmark in dfs:
                bench = dfs[self.benchmark]
                if not bench.empty and "close" in bench:
                    self._benchmark_returns = bench["close"].pct_change().dropna()

    def update_equity(self, equity: float) -> None:
        self.equity = equity

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self) -> PortfolioIntelligenceReport:
        """Run full portfolio intelligence analysis."""
        report = PortfolioIntelligenceReport()

        with self._lock:
            positions = list(self._positions.values())

        if not positions:
            report.recommendation = "No open positions"
            return report

        # === Position summary ===
        report.total_positions = len(positions)
        report.long_positions = sum(1 for p in positions if p["side"] == "BUY")
        report.short_positions = sum(1 for p in positions if p["side"] == "SELL")
        report.gross_exposure_usd = sum(p["notional_usd"] for p in positions)
        net = sum(p["notional_usd"] if p["side"] == "BUY" else -p["notional_usd"]
                  for p in positions)
        report.net_exposure_usd = net

        # === Risk budget ===
        total_risk = sum(p["risk_usd"] for p in positions)
        report.net_risk_budget_pct = (total_risk / max(self.equity, 1)) * 100

        # === Correlation matrix ===
        report.correlation_matrix, report.avg_correlation = self._compute_correlation()
        if report.avg_correlation < 0.3:
            report.correlation_risk = "LOW"
        elif report.avg_correlation < 0.6:
            report.correlation_risk = "MEDIUM"
        else:
            report.correlation_risk = "HIGH"

        # === Diversification ===
        weights = [p["notional_usd"] / max(report.gross_exposure_usd, 1) for p in positions]
        report.herfindahl_index = sum(w * w for w in weights)
        report.effective_positions = 1.0 / report.herfindahl_index if report.herfindahl_index > 0 else 0
        # Diversification score: combine Herfindahl + correlation
        herf_score = 1.0 - report.herfindahl_index  # 0-1, higher = more diversified
        corr_penalty = report.avg_correlation * 0.5
        report.diversification_score = max(0, min(1, herf_score - corr_penalty))

        # === Currency exposure ===
        report.currency_exposure = self._compute_currency_exposure(positions)

        # === Sector exposure ===
        report.sector_exposure = self._compute_sector_exposure(positions)

        # === Hedge detection ===
        report.hedges_detected = self._detect_hedges(positions, report.correlation_matrix)
        if report.hedges_detected:
            hedged_value = sum(min(
                next((p["notional_usd"] for p in positions if p["symbol"] == a), 0),
                next((p["notional_usd"] for p in positions if p["symbol"] == b), 0)
            ) for a, b, _ in report.hedges_detected)
            report.hedge_coverage_pct = (hedged_value / max(report.gross_exposure_usd, 1)) * 100

        # === VaR + CVaR ===
        report.portfolio_var_95, report.portfolio_cvar_95 = self._compute_var(report)

        # === Portfolio beta ===
        report.portfolio_beta = self._compute_portfolio_beta(positions)

        # === Warnings ===
        if report.net_risk_budget_pct > self.max_risk_budget:
            report.warnings.append(
                f"Risk budget {report.net_risk_budget_pct:.1f}% > limit {self.max_risk_budget}%"
            )
        if report.correlation_risk == "HIGH":
            report.warnings.append(
                f"High correlation ({report.avg_correlation:.2f}) — positions not diversified"
            )
        if report.herfindahl_index > 0.5:
            report.warnings.append(
                f"High concentration (HHI={report.herfindahl_index:.2f})"
            )
        if report.diversification_score < 0.3:
            report.warnings.append(
                f"Poor diversification (score={report.diversification_score:.2f})"
            )

        # === Recommendation ===
        report.recommendation = self._recommend(report)

        return report

    # ------------------------------------------------------------------
    # Correlation computation
    # ------------------------------------------------------------------
    def _compute_correlation(self) -> Tuple[Dict[str, Dict[str, float]], float]:
        """Compute correlation matrix between position returns."""
        symbols = [p["symbol"] for p in self._positions.values()
                  if p["symbol"] in self._price_history]
        if len(symbols) < 2:
            return {}, 0.0

        # Build returns DataFrame
        returns_data = {}
        for sym in symbols:
            df = self._price_history.get(sym)
            if df is not None and not df.empty and "close" in df:
                returns_data[sym] = df["close"].pct_change().dropna().tail(self.var_lookback)

        if len(returns_data) < 2:
            return {}, 0.0

        returns_df = pd.DataFrame(returns_data)
        corr_matrix = returns_df.corr()

        # Convert to dict
        result: Dict[str, Dict[str, float]] = {}
        for a in corr_matrix.columns:
            result[a] = {}
            for b in corr_matrix.columns:
                val = corr_matrix.loc[a, b]
                result[a][b] = float(val) if not pd.isna(val) else 0.0

        # Average off-diagonal correlation
        n = len(corr_matrix)
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            off_diag = corr_matrix.values[mask]
            avg_corr = float(np.nanmean(off_diag)) if len(off_diag) > 0 else 0.0
        else:
            avg_corr = 0.0

        return result, avg_corr

    def _compute_currency_exposure(self, positions: list) -> Dict[str, float]:
        """Compute net exposure per currency."""
        from trading_modules.portfolio_exposure_analyzer import extract_currencies
        exposure: Dict[str, float] = {}
        for p in positions:
            currencies = extract_currencies(p["symbol"])
            if currencies is None:
                continue
            base, quote = currencies
            direction = 1 if p["side"] == "BUY" else -1
            exposure[base] = exposure.get(base, 0) + direction * p["volume"]
            exposure[quote] = exposure.get(quote, 0) - direction * p["volume"]
        return exposure

    def _compute_sector_exposure(self, positions: list) -> Dict[str, float]:
        """Compute exposure per asset class."""
        from trading_modules.portfolio_exposure_analyzer import classify_symbol
        exposure: Dict[str, float] = {}
        for p in positions:
            sector = classify_symbol(p["symbol"])
            direction = 1 if p["side"] == "BUY" else -1
            exposure[sector] = exposure.get(sector, 0) + direction * p["notional_usd"]
        return exposure

    def _detect_hedges(self, positions: list,
                       corr: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
        """Detect hedging relationships.

        A hedge = two positions with opposite directions AND negative correlation.
        """
        hedges = []
        if not corr:
            return hedges
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1:]:
                if p1["side"] == p2["side"]:
                    continue  # same direction = not a hedge
                c = corr.get(p1["symbol"], {}).get(p2["symbol"], 0)
                if c < -0.3:  # negatively correlated
                    hedges.append((p1["symbol"], p2["symbol"], c))
        return hedges

    def _compute_var(self, report: PortfolioIntelligenceReport) -> Tuple[float, float]:
        """Compute portfolio VaR and CVaR.

        Simplified: assumes position returns are normally distributed.
        """
        # Get returns for all positions
        all_returns = []
        for p in self._positions.values():
            df = self._price_history.get(p["symbol"])
            if df is not None and not df.empty and "close" in df:
                rets = df["close"].pct_change().dropna().tail(self.var_lookback)
                # Weight by position notional (relative to equity)
                weight = p["notional_usd"] / max(self.equity, 1)
                direction = 1 if p["side"] == "BUY" else -1
                all_returns.append(rets * weight * direction)

        if not all_returns:
            return 0.0, 0.0

        # Align and sum
        portfolio_returns = pd.concat(all_returns, axis=1).sum(axis=1).dropna()
        if len(portfolio_returns) < 20:
            return 0.0, 0.0

        # VaR: percentile of returns
        pctile = (1 - self.var_confidence) * 100  # 5 for 95%
        var_return = float(np.percentile(portfolio_returns, pctile))
        var_usd = abs(var_return * self.equity)

        # CVaR: mean of returns below VaR
        tail = portfolio_returns[portfolio_returns <= var_return]
        cvar_return = float(tail.mean()) if len(tail) > 0 else var_return
        cvar_usd = abs(cvar_return * self.equity)

        return var_usd, cvar_usd

    def _compute_portfolio_beta(self, positions: list) -> float:
        """Compute portfolio beta vs benchmark."""
        if self._benchmark_returns is None:
            return 0.0

        weighted_betas = []
        total_weight = 0
        for p in positions:
            df = self._price_history.get(p["symbol"])
            if df is None or df.empty:
                continue
            rets = df["close"].pct_change().dropna().tail(self.var_lookback)
            # Align with benchmark
            common_idx = rets.index.intersection(self._benchmark_returns.index)
            if len(common_idx) < 20:
                continue
            asset_rets = rets.loc[common_idx]
            bench_rets = self._benchmark_returns.loc[common_idx]
            cov = float(asset_rets.cov(bench_rets))
            var = float(bench_rets.var())
            if var > 0:
                beta = cov / var
                direction = 1 if p["side"] == "BUY" else -1
                weight = p["notional_usd"]
                weighted_betas.append(beta * direction * weight)
                total_weight += weight

        if total_weight == 0:
            return 0.0
        return sum(weighted_betas) / total_weight

    def _recommend(self, report: PortfolioIntelligenceReport) -> str:
        """Generate recommendation."""
        if not self._positions:
            return "No open positions"

        recs = []
        if report.correlation_risk == "HIGH":
            recs.append(f"Reduce correlation ({report.avg_correlation:.2f}) — close correlated positions")
        if report.herfindahl_index > 0.5:
            recs.append(f"Reduce concentration (HHI={report.herfindahl_index:.2f})")
        if report.net_risk_budget_pct > self.max_risk_budget:
            recs.append(f"Reduce risk ({report.net_risk_budget_pct:.1f}% > {self.max_risk_budget}%)")
        if report.diversification_score < 0.3:
            recs.append(f"Improve diversification (score={report.diversification_score:.2f})")
        if report.portfolio_var_95 > self.equity * 0.05:
            recs.append(f"High VaR (${report.portfolio_var_95:.0f} > 5% equity)")

        if not recs:
            return (f"Portfolio healthy — {report.total_positions} positions, "
                   f"diversification={report.diversification_score:.2f}, "
                   f"risk={report.net_risk_budget_pct:.1f}%")
        return "; ".join(recs)
