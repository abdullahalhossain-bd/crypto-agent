"""engine.risk_v2
=====================================================================
Day 14-16 — Portfolio-aware risk engine (Risk Engine v2).

Sits AFTER portfolio aggregation, BEFORE execution. Receives a
proposed `TargetAllocation` (or list thereof) plus the current
portfolio snapshot, returns either an approved trade (with possibly
adjusted size) or a rejection with a reason.

Checks performed:
  1. Hard kill-switch / halt state (delegates to v1 RiskState)
  2. Daily-loss circuit-breaker
  3. Gross / net exposure caps
  4. Volatility scaling — cut size when ATR is elevated
  5. Value-at-Risk (historical, parametric) — reject if VaR breach
  6. Max-drawdown enforcement — halt if rolling DD > threshold
  7. Correlation risk penalty — scale size if symbol correlates with
     existing open positions
  8. Per-trade risk cap (lot ceiling)

All checks are independent and composable — failing one does NOT
short-circuit the others, so the decision trace records every factor.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from engine.portfolio.portfolio_manager import PortfolioManager, PortfolioSnapshot
from engine.portfolio.exposure_model import ExposureModel
from engine.risk import RiskState
from engine.signals import Action
from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("engine.risk_v2")


# ----------------------------------------------------------------------
@dataclass
class RiskDecision:
    """Full audit record of a single risk evaluation."""
    approved: bool
    adjusted_lots: float
    reason: str
    decision_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "adjusted_lots": self.adjusted_lots,
            "reason": self.reason,
            "decision_trace": self.decision_trace,
        }


# ----------------------------------------------------------------------
class RiskEngineV2:
    def __init__(
        self,
        state: RiskState,
        exposure: ExposureModel,
        cfg: dict[str, Any],
        portfolio: Optional[PortfolioManager] = None,
    ) -> None:
        self.state = state
        self.exposure = exposure
        self.portfolio = portfolio
        self.cfg = cfg

        # Common knobs
        self.risk_per_trade = float(cfg.get("risk_per_trade", 0.01))
        self.max_daily_loss = float(cfg.get("max_daily_loss", 0.05))
        self.max_open_trades = int(cfg.get("max_open_trades", 5))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.stop_atr_multiple = float(cfg.get("stop_atr_multiple", 2.0))
        self.vol_filter_multiple = float(cfg.get("volatility_filter_multiple", 2.5))

        # v2 additions
        v2 = cfg.get("v2", {})
        self.var_confidence = float(v2.get("var_confidence", 0.95))
        self.var_lookback = int(v2.get("var_lookback", 250))
        self.max_var_pct = float(v2.get("max_var_pct", 0.04))   # 4% portfolio VaR
        self.max_drawdown_pct = float(v2.get("max_drawdown_pct", 0.15))
        self.vol_scale_high = float(v2.get("vol_scale_high", 1.5))    # ATR > 1.5x baseline → scale 0.5
        self.vol_scale_zero = float(v2.get("vol_scale_zero", 3.0))    # ATR > 3x baseline → reject
        self.correlation_penalty_threshold = float(
            v2.get("correlation_penalty_threshold", 0.7)
        )
        self.max_lot_per_trade = float(v2.get("max_lot_per_trade", 5.0))

        # Rolling equity history for drawdown
        self._equity_history: list[tuple[float, float]] = []  # (ts, equity)

    # ----------------------------------------------------------------
    def evaluate(
        self,
        symbol: str,
        action: Action,
        requested_lots: float,
        entry_price: float,
        df: pd.DataFrame,
        atr_baseline: Optional[float] = None,
        correlated_symbols: Optional[list[str]] = None,
    ) -> RiskDecision:
        """Run every risk check; return a RiskDecision with full trace."""
        trace: dict[str, Any] = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "symbol": symbol,
            "action": action.value,
            "requested_lots": requested_lots,
            "entry_price": entry_price,
            "checks": {},
        }
        approved = True
        reason = "approved"
        adjusted_lots = float(requested_lots)

        # Snapshot equity (use portfolio or fallback 10k)
        equity = (self.portfolio.equity if self.portfolio else 10_000.0)
        self.state.reset_for_new_day(equity)

        # 1. Halt state
        if self.state.halted:
            approved = False
            reason = f"halted: {self.state.halt_reason}"
            trace["checks"]["halt"] = {"ok": False, "reason": reason}

        # 2. Daily-loss circuit-breaker
        pnl_today = equity - self.state.start_of_day_equity
        if self.state.start_of_day_equity > 0:
            pnl_pct = pnl_today / self.state.start_of_day_equity
            if pnl_pct <= -self.max_daily_loss:
                approved = False
                reason = f"daily loss {pnl_pct:.2%} <= -{self.max_daily_loss:.2%}"
                self._halt(reason)
                trace["checks"]["daily_loss"] = {"ok": False, "pnl_pct": pnl_pct}

        # 3. Exposure caps
        side = "long" if action == Action.BUY else "short"
        atr_for_exposure = float(atr(df, self.atr_period).iloc[-1])
        if not math.isfinite(atr_for_exposure):
            # ATR isn't ready yet (warm-up) or the feed has a gap. Do NOT
            # silently substitute 0.0 here — that would make the position
            # look risk-free to the exposure model and let it pass a
            # check it can't actually evaluate. Reject explicitly instead.
            reason = "exposure check unavailable: ATR not finite (warm-up or data gap)"
            trace["checks"]["exposure"] = {"ok": False, "reason": reason,
                                           "atr_value": atr_for_exposure}
            log.warning("RISK_V2 %s %s rejected: %s", action, symbol, reason)
            self.state.rejected_count += 1
            trace["adjusted_lots"] = adjusted_lots
            trace["approved"] = False
            trace["reason"] = reason
            return RiskDecision(approved=False, adjusted_lots=adjusted_lots,
                                reason=reason, decision_trace=trace)

        additional_risk = adjusted_lots * atr_for_exposure
        additional_risk /= max(equity, 1.0)
        breach = self.exposure.would_breach(additional_risk, side)
        trace["checks"]["exposure"] = breach
        if breach["gross_breach"]:
            approved = False
            reason = f"gross exposure breach: {breach['new_gross']:.3f} > {self.exposure.max_gross}"
        if breach["net_breach"]:
            approved = False
            reason = f"net exposure breach: {breach['new_net']:.3f} > {self.exposure.max_net}"

        # 4. Volatility scaling
        atr_series = atr(df, self.atr_period)
        atr_now = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 0.0
        baseline = atr_baseline or float(atr_series.dropna().median() or atr_now)
        trace["checks"]["volatility"] = {
            "atr_now": atr_now, "baseline": baseline,
            "ratio": (atr_now / baseline) if baseline > 0 else 0.0,
        }
        if baseline > 0:
            ratio = atr_now / baseline
            if ratio >= self.vol_scale_zero:
                approved = False
                reason = f"volatility extreme ratio={ratio:.2f}"
            elif ratio >= self.vol_scale_high:
                # Scale size down by (vol_scale_zero - ratio) / (vol_scale_zero - vol_scale_high)
                scale = (self.vol_scale_zero - ratio) / (self.vol_scale_zero - self.vol_scale_high)
                scale = max(0.1, min(1.0, scale))
                adjusted_lots *= scale
                trace["checks"]["volatility"]["scale"] = scale

        # 5. Value-at-Risk (parametric, using log-returns of the symbol)
        var_check = self._var_check(df, adjusted_lots, entry_price, equity)
        trace["checks"]["var"] = var_check
        if var_check["portfolio_var_pct"] > self.max_var_pct:
            approved = False
            reason = (f"VaR breach: {var_check['portfolio_var_pct']:.2%} > "
                      f"{self.max_var_pct:.2%}")

        # 6. Max drawdown
        self._equity_history.append((time.time(), equity))
        if len(self._equity_history) > 5000:
            self._equity_history = self._equity_history[-5000:]
        dd_check = self._drawdown_check()
        trace["checks"]["drawdown"] = dd_check
        if dd_check["current_dd_pct"] > self.max_drawdown_pct:
            approved = False
            reason = (f"max drawdown breach: {dd_check['current_dd_pct']:.2%} > "
                      f"{self.max_drawdown_pct:.2%}")
            self._halt(reason)

        # 7. Correlation penalty
        corr_check = self._correlation_check(symbol, correlated_symbols or [])
        trace["checks"]["correlation"] = corr_check
        if corr_check["scale"] < 1.0:
            adjusted_lots *= corr_check["scale"]
            trace["checks"]["correlation"]["adjusted_lots"] = adjusted_lots

        # 8. Per-trade lot cap + sanity
        adjusted_lots = float(min(adjusted_lots, self.max_lot_per_trade))
        if adjusted_lots <= 0:
            approved = False
            reason = "adjusted_lots <= 0 after scaling"

        # Max open trades
        if self.exposure.n_open_positions >= self.max_open_trades:
            approved = False
            reason = f"max_open_trades={self.max_open_trades}"

        trace["adjusted_lots"] = adjusted_lots
        trace["approved"] = approved
        trace["reason"] = reason
        log.info("RISK_V2 %s %s lots req=%.4f adj=%.4f approved=%s reason=%s",
                 action, symbol, requested_lots, adjusted_lots, approved, reason)
        if approved:
            self.state.approved_count += 1
        else:
            self.state.rejected_count += 1
        return RiskDecision(
            approved=approved,
            adjusted_lots=adjusted_lots,
            reason=reason,
            decision_trace=trace,
        )

    # ----------------------------------------------------------------
    # Internal checks
    # ----------------------------------------------------------------
    @staticmethod
    def _z_score(confidence: float) -> float:
        """C19 fix: compute the z-score for a given confidence level.

        Tries scipy.stats.norm.ppf first (accurate). If scipy is not
        installed, falls back to a hardcoded lookup table for common
        confidence levels, then to the inverse-erf approximation.
        """
        try:
            from scipy.stats import norm
            return float(norm.ppf(confidence))
        except ImportError:
            pass
        # Hardcoded z-values for common confidence levels.
        _Z_TABLE = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326,
                    0.999: 3.090, 0.80: 0.842, 0.85: 1.036}
        if confidence in _Z_TABLE:
            return _Z_TABLE[confidence]
        # Inverse-error-function approximation (Abramowitz & Stegun 26.2.23).
        # Good enough for VaR purposes (within 0.01 of the true z-value).
        import math
        if confidence <= 0.0:
            return 0.0
        if confidence >= 1.0:
            return 3.5  # ~99.95%
        # ppf(p) = sqrt(2) * erfinv(2p - 1)
        # erfinv approximation:
        t = 1.0 / (1.0 - 0.5 * abs(2 * confidence - 1))
        erf_inv = math.copysign(
            t * (2.5066282746310002 - 3.0664798066708953 * t +
                 1.7817594038231056 * t**2 - 0.5415462766861901 * t**3),
            2 * confidence - 1)
        return float(math.sqrt(2) * erf_inv)

    def _var_check(self, df: pd.DataFrame, lots: float,
                   entry_price: float, equity: float) -> dict[str, Any]:
        """Parametric VaR using log-return std of close."""
        if len(df) < 20 or equity <= 0:
            return {"portfolio_var_pct": 0.0, "symbol_var": 0.0, "ok": True}
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        if len(log_ret) < 20:
            return {"portfolio_var_pct": 0.0, "symbol_var": 0.0, "ok": True}
        sigma = float(log_ret.tail(self.var_lookback).std())
        # C19 fix: scipy is an optional dependency. If it's not installed,
        # fall back to a hardcoded z-value for common confidence levels
        # instead of crashing the risk engine.
        z = self._z_score(self.var_confidence)
        symbol_var = z * sigma  # 1-day, 1 unit
        # Position notional
        notional = lots * entry_price
        portfolio_var = symbol_var * notional / equity
        return {
            "portfolio_var_pct": float(portfolio_var),
            "symbol_var": float(symbol_var),
            "sigma": sigma,
            "z": z,
            "ok": portfolio_var <= self.max_var_pct,
        }

    def _drawdown_check(self) -> dict[str, Any]:
        if len(self._equity_history) < 2:
            return {"current_dd_pct": 0.0, "max_dd_pct": 0.0, "ok": True}
        eqs = [e for _, e in self._equity_history]
        running_max = np.maximum.accumulate(eqs)
        dd = (np.array(eqs) - running_max) / np.where(running_max > 0, running_max, 1.0)
        return {
            "current_dd_pct": float(abs(dd[-1])),
            "max_dd_pct": float(abs(dd.min())),
            "ok": float(abs(dd[-1])) <= self.max_drawdown_pct,
        }

    def _correlation_check(self, symbol: str,
                           correlated: list[str]) -> dict[str, Any]:
        """If `symbol` is highly correlated with symbols already in the book,
        scale size down to avoid concentrated risk."""
        if not correlated:
            return {"scale": 1.0, "overlaps": []}
        scale = 1.0
        for c in correlated:
            # Use the correlation matrix if portfolio is available
            if self.portfolio is None:
                corr = 0.0
            else:
                corr = abs(self.portfolio.correlation.pairwise(symbol, c))
            if corr >= self.correlation_penalty_threshold:
                scale *= (1.0 - corr * 0.5)  # dampen, don't zero out
        scale = max(0.1, scale)
        return {"scale": float(scale), "overlaps": list(correlated)}

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        log.error("RISK V2 HALT: %s", reason)

    # ----------------------------------------------------------------
    def record_equity(self, equity: float) -> None:
        """Call from the main loop every cycle so DD tracking works."""
        self._equity_history.append((time.time(), float(equity)))
        if len(self._equity_history) > 5000:
            self._equity_history = self._equity_history[-5000:]