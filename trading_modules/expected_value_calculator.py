"""trading_modules/expected_value_calculator.py
=====================================================================
Expected Value Calculator (Principle #116 — Think in Expected Value)
=====================================================================
Computes the long-term expected value of a trade setup across 1000+
trades, not just a single outcome.

EV = (Win% × Average Win) - (Loss% × Average Loss)

Inputs:
    - Historical win rate for this setup type
    - Average win size (in R multiples)
    - Average loss size (in R multiples, usually 1.0)
    - Sample size (how many historical trades?)
    - Confidence interval (how sure are we of the win rate?)

Outputs:
    - Expected Value per trade (in $ and R)
    - 95% confidence interval
    - Kelly fraction (optimal position size)
    - Risk of ruin (probability of going broke)
    - Sample size adequacy

Usage:
    ev_calc = ExpectedValueCalculator()
    result = ev_calc.calculate(
        win_rate=0.62,
        avg_win_r=1.8,
        avg_loss_r=1.0,
        sample_size=50,
        account_equity=10000,
        risk_per_trade_pct=2.0,
    )
    if result.ev_per_trade_r > 0 and result.kelly_fraction > 0:
        # Positive EV — trade
        size = result.kelly_fraction * 0.25  # quarter Kelly
    else:
        # Negative EV — skip
        pass
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.expected_value_calculator")


@dataclass
class EVResult:
    """Expected value calculation result."""
    # Core EV
    ev_per_trade_r: float = 0.0       # expected R per trade
    ev_per_trade_usd: float = 0.0     # expected $ per trade
    # Win/loss stats
    win_rate: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    # Confidence
    ci_95_lower: float = 0.0          # 95% CI lower bound for EV
    ci_95_upper: float = 0.0
    win_rate_ci_lower: float = 0.0    # 95% CI for win rate
    win_rate_ci_upper: float = 0.0
    # Sizing
    kelly_fraction: float = 0.0       # optimal fraction (0-1)
    kelly_half: float = 0.0           # half Kelly (more conservative)
    risk_of_ruin: float = 0.0         # probability of going broke
    # Sample adequacy
    sample_size: int = 0
    sample_adequate: bool = False
    min_sample_size: int = 30
    # Recommendation
    is_positive_ev: bool = False
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ev_per_trade_r": round(self.ev_per_trade_r, 4),
            "ev_per_trade_usd": round(self.ev_per_trade_usd, 2),
            "win_rate": round(self.win_rate, 4),
            "avg_win_r": round(self.avg_win_r, 2),
            "avg_loss_r": round(self.avg_loss_r, 2),
            "ci_95_lower": round(self.ci_95_lower, 4),
            "ci_95_upper": round(self.ci_95_upper, 4),
            "win_rate_ci_lower": round(self.win_rate_ci_lower, 4),
            "win_rate_ci_upper": round(self.win_rate_ci_upper, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "kelly_half": round(self.kelly_half, 4),
            "risk_of_ruin": round(self.risk_of_ruin, 4),
            "sample_size": self.sample_size,
            "sample_adequate": self.sample_adequate,
            "is_positive_ev": self.is_positive_ev,
            "recommendation": self.recommendation,
        }


class ExpectedValueCalculator:
    """Calculates expected value, Kelly fraction, and risk of ruin."""

    def __init__(self,
                 min_sample_size: int = 30,
                 risk_free_rate: float = 0.0):
        """Initialize calculator.

        Args:
            min_sample_size: minimum historical trades for reliable EV
            risk_free_rate: annual risk-free rate (for Kelly comparison)
        """
        self.min_sample = min_sample_size
        self.risk_free_rate = risk_free_rate

    def calculate(self,
                  win_rate: float,
                  avg_win_r: float = 2.0,
                  avg_loss_r: float = 1.0,
                  sample_size: int = 0,
                  account_equity: float = 10000.0,
                  risk_per_trade_pct: float = 2.0,
                  max_risk_of_ruin: float = 0.01) -> EVResult:
        """Calculate expected value for a trade setup.

        Args:
            win_rate: historical win rate (0-1)
            avg_win_r: average winning trade in R multiples
            avg_loss_r: average losing trade in R multiples (usually 1.0)
            sample_size: how many historical trades is win_rate based on?
            account_equity: current account equity in $
            risk_per_trade_pct: risk per trade as % of equity
            max_risk_of_ruin: acceptable probability of ruin (default 1%)

        Returns:
            EVResult with all metrics
        """
        result = EVResult(
            win_rate=win_rate,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
            sample_size=sample_size,
            min_sample_size=self.min_sample,
        )

        # === 1. Expected Value per trade (in R) ===
        loss_rate = 1.0 - win_rate
        ev_r = (win_rate * avg_win_r) - (loss_rate * avg_loss_r)
        result.ev_per_trade_r = ev_r

        # === 2. EV in USD ===
        risk_usd = account_equity * (risk_per_trade_pct / 100)
        result.ev_per_trade_usd = ev_r * risk_usd

        # === 3. Win rate confidence interval (95%) ===
        # Using normal approximation: CI = p ± 1.96 * sqrt(p*(1-p)/n)
        if sample_size > 0:
            std_error = math.sqrt(win_rate * (1 - win_rate) / sample_size)
            result.win_rate_ci_lower = max(0.0, win_rate - 1.96 * std_error)
            result.win_rate_ci_upper = min(1.0, win_rate + 1.96 * std_error)
            # EV confidence interval (using lower bound win rate)
            ev_lower = (result.win_rate_ci_lower * avg_win_r) - \
                       ((1 - result.win_rate_ci_lower) * avg_loss_r)
            ev_upper = (result.win_rate_ci_upper * avg_win_r) - \
                       ((1 - result.win_rate_ci_upper) * avg_loss_r)
            result.ci_95_lower = ev_lower
            result.ci_95_upper = ev_upper
        else:
            result.win_rate_ci_lower = win_rate
            result.win_rate_ci_upper = win_rate
            result.ci_95_lower = ev_r
            result.ci_95_upper = ev_r

        # === 4. Kelly Criterion ===
        # Kelly = (b*p - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = loss_rate
        b = avg_win_r / max(avg_loss_r, 1e-10)
        kelly = (b * win_rate - loss_rate) / max(b, 1e-10)
        result.kelly_fraction = max(0.0, kelly)
        result.kelly_half = result.kelly_fraction * 0.5  # half Kelly

        # === 5. Risk of Ruin ===
        # Risk of ruin = ((1 - edge) / (1 + edge))^units
        # where edge = EV per trade in R
        if ev_r > 0 and avg_loss_r > 0:
            # Using the classic formula
            edge = ev_r
            units = 1.0 / avg_loss_r  # how many R in our bankroll
            if edge > 0:
                ror = ((1 - edge) / (1 + edge)) ** units
                result.risk_of_ruin = min(1.0, max(0.0, ror))
            else:
                result.risk_of_ruin = 1.0
        else:
            result.risk_of_ruin = 1.0

        # === 6. Sample adequacy ===
        result.sample_adequate = sample_size >= self.min_sample

        # === 7. Positive EV? ===
        result.is_positive_ev = ev_r > 0 and result.kelly_fraction > 0

        # === 8. Recommendation ===
        result.recommendation = self._recommend(result, max_risk_of_ruin)

        return result

    def _recommend(self, result: EVResult, max_ror: float) -> str:
        """Generate human-readable recommendation."""
        if not result.sample_adequate:
            return (f"INSUFFICIENT SAMPLE — only {result.sample_size} trades "
                    f"(need {result.min_sample_size}). EV unreliable.")
        if result.ev_per_trade_r <= 0:
            return (f"NEGATIVE EV — EV={result.ev_per_trade_r:.3f}R per trade. "
                    f"Do NOT trade this setup.")
        if result.risk_of_ruin > max_ror:
            return (f"RISK OF RUIN TOO HIGH — RoR={result.risk_of_ruin:.1%} "
                    f"(max {max_ror:.1%}). Reduce position size or skip.")
        if result.kelly_fraction > 0.25:
            return (f"STRONG EDGE — EV={result.ev_per_trade_r:.3f}R, "
                    f"Kelly={result.kelly_fraction:.1%}. "
                    f"Use half-Kelly ({result.kelly_half:.1%}) for safety.")
        if result.is_positive_ev:
            return (f"POSITIVE EV — EV={result.ev_per_trade_r:.3f}R, "
                    f"Kelly={result.kelly_fraction:.1%}, "
                    f"RoR={result.risk_of_ruin:.2%}. "
                    f"Trade with quarter-Kelly sizing.")
        return "MARGINAL — review setup before trading."

    # ------------------------------------------------------------------
    # Batch EV calculation for multiple setups
    # ------------------------------------------------------------------
    def rank_setups(self, setups: list) -> list:
        """Rank multiple setups by EV.

        Args:
            setups: list of dicts with keys: name, win_rate, avg_win_r,
                    avg_loss_r, sample_size

        Returns:
            List of (name, EVResult) sorted by EV descending
        """
        results = []
        for s in setups:
            r = self.calculate(
                win_rate=s.get("win_rate", 0.5),
                avg_win_r=s.get("avg_win_r", 1.5),
                avg_loss_r=s.get("avg_loss_r", 1.0),
                sample_size=s.get("sample_size", 0),
                account_equity=s.get("equity", 10000),
                risk_per_trade_pct=s.get("risk_pct", 2.0),
            )
            results.append((s.get("name", "?"), r))
        # Sort by EV descending
        results.sort(key=lambda x: x[1].ev_per_trade_r, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Monte Carlo simulation for EV validation
    # ------------------------------------------------------------------
    def monte_carlo(self,
                    win_rate: float,
                    avg_win_r: float,
                    avg_loss_r: float,
                    n_trades: int = 1000,
                    n_simulations: int = 10000,
                    initial_equity: float = 10000,
                    risk_per_trade_pct: float = 2.0) -> Dict[str, Any]:
        """Monte Carlo simulation to validate EV.

        Simulates n_simulations runs of n_trades each, returns
        distribution of final equity.
        """
        risk_usd = initial_equity * (risk_per_trade_pct / 100)
        results = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)

        for sim in range(n_simulations):
            equity = initial_equity
            peak = initial_equity
            max_dd = 0.0
            for _ in range(n_trades):
                if np.random.random() < win_rate:
                    equity += avg_win_r * risk_usd
                else:
                    equity -= avg_loss_r * risk_usd
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / max(peak, 1)
                if dd > max_dd:
                    max_dd = dd
                if equity <= 0:
                    break
            results[sim] = equity
            max_drawdowns[sim] = max_dd

        return {
            "median_final_equity": float(np.median(results)),
            "p05_final_equity": float(np.percentile(results, 5)),
            "p95_final_equity": float(np.percentile(results, 95)),
            "mean_final_equity": float(np.mean(results)),
            "ruin_probability": float(np.mean(results <= 0)),
            "median_max_drawdown_pct": float(np.median(max_drawdowns) * 100),
            "p95_max_drawdown_pct": float(np.percentile(max_drawdowns, 95) * 100),
            "n_simulations": n_simulations,
            "n_trades": n_trades,
        }
