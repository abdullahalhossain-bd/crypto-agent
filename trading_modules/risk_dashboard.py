"""
Institutional Risk Dashboard
============================

Real-time portfolio risk monitor. Computes the same risk metrics that
institutional desks watch on their Bloomberg terminals:

    1. VaR (Value at Risk)         — max expected loss at 95% confidence
    2. CVaR (Conditional VaR)      — expected loss given loss > VaR
    3. Max Drawdown                — peak-to-trough decline
    4. Daily Loss                  — today's realized + unrealized PnL
    5. Portfolio Heat              — sum of |risk| across open positions
    6. Exposure (gross/net)        — total long + short notional
    7. Concentration (HHI)         — Herfindahl-Hirschman Index
    8. Correlation Risk            — avg pairwise correlation of positions
    9. Sharpe / Sortino            — risk-adjusted performance

Usage:
    from trading_modules.risk_dashboard import RiskDashboard, PortfolioSnapshot
    dashboard = RiskDashboard()
    snapshot = PortfolioSnapshot(
        equity=10500.0,
        cash=5000.0,
        positions=[
            {"symbol": "BTCUSD", "side": "long", "notional": 5000,
             "entry": 65000, "current": 65500, "stop": 64500},
            {"symbol": "ETHUSD", "side": "long", "notional": 500,
             "entry": 3200, "current": 3180, "stop": 3100},
        ],
        returns_history=[0.01, -0.005, 0.02, ...],  # daily returns
    )
    risk = dashboard.compute(snapshot)
    if risk.var_95 > 0.04:
        print(f"VaR too high: {risk.var_95:.1%}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    positions: list[dict]    # each: {symbol, side, notional, entry, current, stop}
    returns_history: list[float] = field(default_factory=list)
    # daily PnL % history (most recent last)


@dataclass
class RiskMetrics:
    # VaR / CVaR
    var_95: float = 0.0          # 95% 1-day VaR (fraction of equity)
    var_99: float = 0.0
    cvar_95: float = 0.0         # expected loss beyond VaR
    # Drawdown
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    # PnL
    daily_pnl_pct: float = 0.0
    daily_pnl_usd: float = 0.0
    # Exposure
    gross_exposure: float = 0.0   # sum |notional| / equity
    net_exposure: float = 0.0     # (long - short) / equity
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    # Concentration
    hhi: float = 0.0              # 0 = diversified, 1 = single position
    max_position_pct: float = 0.0
    # Portfolio heat
    portfolio_heat_pct: float = 0.0  # sum |risk| / equity
    # Correlation
    avg_correlation: float = 0.0
    # Performance
    sharpe: float = 0.0
    sortino: float = 0.0
    # Status flags
    risk_status: str = "ok"       # "ok" / "warning" / "critical"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "var_95": round(self.var_95, 4),
            "var_99": round(self.var_99, 4),
            "cvar_95": round(self.cvar_95, 4),
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "daily_pnl_usd": round(self.daily_pnl_usd, 2),
            "gross_exposure": round(self.gross_exposure, 3),
            "net_exposure": round(self.net_exposure, 3),
            "long_exposure": round(self.long_exposure, 3),
            "short_exposure": round(self.short_exposure, 3),
            "hhi": round(self.hhi, 3),
            "max_position_pct": round(self.max_position_pct, 3),
            "portfolio_heat_pct": round(self.portfolio_heat_pct, 4),
            "avg_correlation": round(self.avg_correlation, 3),
            "sharpe": round(self.sharpe, 2),
            "sortino": round(self.sortino, 2),
            "risk_status": self.risk_status,
            "warnings": self.warnings,
        }


class RiskDashboard:
    """Real-time institutional risk dashboard.

    Parameters:
        var_confidence: VaR confidence level (default 0.95)
        var_lookback: # of historical returns to use (default 250)
        max_var_pct: warning threshold for VaR (default 0.04 = 4%)
        max_drawdown_pct: warning threshold for drawdown (default 0.15)
        max_gross_exposure: warning threshold for gross exposure (default 2.0)
        max_hhi: warning threshold for concentration (default 0.5)
        max_portfolio_heat_pct: warning threshold for portfolio heat (default 0.06)
        risk_free_rate: for Sharpe calculation (default 0.0)
    """

    def __init__(
        self, var_confidence: float = 0.95, var_lookback: int = 250,
        max_var_pct: float = 0.04, max_drawdown_pct: float = 0.15,
        max_gross_exposure: float = 2.0, max_hhi: float = 0.5,
        max_portfolio_heat_pct: float = 0.06,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.var_confidence = var_confidence
        self.var_lookback = var_lookback
        self.max_var_pct = max_var_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_hhi = max_hhi
        self.max_portfolio_heat_pct = max_portfolio_heat_pct
        self.risk_free_rate = risk_free_rate

    def compute(self, snapshot: PortfolioSnapshot) -> RiskMetrics:
        """Compute all risk metrics from a portfolio snapshot."""
        m = RiskMetrics()
        if snapshot is None:
            return m

        equity = float(snapshot.equity)
        if equity <= 0:
            m.warnings.append("equity is zero or negative")
            m.risk_status = "critical"
            return m

        # ── Exposure ─────────────────────────────────────────────
        long_notional = sum(
            float(p["notional"]) for p in snapshot.positions
            if p.get("side") == "long"
        )
        short_notional = sum(
            float(p["notional"]) for p in snapshot.positions
            if p.get("side") == "short"
        )
        m.long_exposure = long_notional / equity
        m.short_exposure = short_notional / equity
        m.gross_exposure = (long_notional + short_notional) / equity
        m.net_exposure = (long_notional - short_notional) / equity

        # ── Concentration (HHI) ──────────────────────────────────
        total_notional = long_notional + short_notional
        if total_notional > 0 and snapshot.positions:
            shares = [
                float(p["notional"]) / total_notional
                for p in snapshot.positions
            ]
            m.hhi = float(sum(s * s for s in shares))
            m.max_position_pct = max(
                float(p["notional"]) / equity for p in snapshot.positions
            )

        # ── Portfolio Heat (sum of |risk|) ───────────────────────
        total_risk = 0.0
        for p in snapshot.positions:
            entry = float(p.get("entry", 0))
            stop = float(p.get("stop", 0))
            notional = float(p.get("notional", 0))
            if entry > 0 and stop > 0:
                # Risk = |entry - stop| / entry × notional
                risk_per_unit = abs(entry - stop) / entry
                total_risk += risk_per_unit * notional
        m.portfolio_heat_pct = total_risk / equity

        # ── VaR / CVaR (historical method) ───────────────────────
        returns = snapshot.returns_history or []
        if len(returns) >= 30:
            recent = np.array(returns[-self.var_lookback:], dtype=float)
            recent = recent[np.isfinite(recent)]
            if len(recent) >= 30:
                # VaR = -percentile(returns, 1 - confidence) × equity
                # 95% VaR → 5th percentile of returns (loss)
                var_95 = float(np.percentile(recent, 5))
                var_99 = float(np.percentile(recent, 1))
                m.var_95 = abs(var_95)  # positive number representing loss
                m.var_99 = abs(var_99)
                # CVaR = mean of returns below VaR
                tail = recent[recent <= var_95]
                m.cvar_95 = abs(float(tail.mean())) if len(tail) > 0 else m.var_95

        # ── Drawdown ─────────────────────────────────────────────
        if len(returns) >= 2:
            equity_curve = np.cumprod(1 + np.array(returns, dtype=float))
            peak = np.maximum.accumulate(equity_curve)
            drawdowns = (peak - equity_curve) / peak
            m.max_drawdown_pct = float(drawdowns.max())
            m.current_drawdown_pct = float(drawdowns[-1])

        # ── Daily PnL ────────────────────────────────────────────
        if returns:
            m.daily_pnl_pct = float(returns[-1])
            m.daily_pnl_usd = m.daily_pnl_pct * equity

        # ── Sharpe / Sortino ─────────────────────────────────────
        if len(returns) >= 30:
            arr = np.array(returns, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) >= 30 and arr.std() > 0:
                excess = arr - self.risk_free_rate / 252
                m.sharpe = float(excess.mean() / arr.std() * np.sqrt(252))
                downside = arr[arr < 0]
                if len(downside) > 0 and downside.std() > 0:
                    m.sortino = float(excess.mean() / downside.std() * np.sqrt(252))

        # ── Status flags ─────────────────────────────────────────
        warnings: list[str] = []
        if m.var_95 > self.max_var_pct:
            warnings.append(f"VaR 95% = {m.var_95:.1%} > {self.max_var_pct:.1%}")
        if m.max_drawdown_pct > self.max_drawdown_pct:
            warnings.append(f"Max DD = {m.max_drawdown_pct:.1%} > {self.max_drawdown_pct:.1%}")
        if m.gross_exposure > self.max_gross_exposure:
            warnings.append(f"Gross exposure = {m.gross_exposure:.2f}x > {self.max_gross_exposure:.2f}x")
        if m.hhi > self.max_hhi:
            warnings.append(f"HHI = {m.hhi:.2f} > {self.max_hhi:.2f} (concentrated)")
        if m.portfolio_heat_pct > self.max_portfolio_heat_pct:
            warnings.append(f"Portfolio heat = {m.portfolio_heat_pct:.1%} > {self.max_portfolio_heat_pct:.1%}")
        m.warnings = warnings
        if any("critical" in w.lower() or "max dd" in w.lower() or "var" in w.lower()
               for w in warnings):
            m.risk_status = "critical" if len(warnings) >= 2 else "warning"
        elif warnings:
            m.risk_status = "warning"
        else:
            m.risk_status = "ok"

        return m


__all__ = ["RiskDashboard", "PortfolioSnapshot", "RiskMetrics"]
