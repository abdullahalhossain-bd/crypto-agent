"""enhancements.monte_carlo
=====================================================================
Day 152-154 — Monte Carlo simulator.

Takes a historical trade PnL series and resamples it 1000s of times
to produce a DISTRIBUTION of possible equity curves. This answers:

  - "What's the worst-case drawdown I should expect?"
  - "What's the probability of going bust?"
  - "What's the 95th percentile equity after N trades?"

Methods:
  - Bootstrap resampling (trade order shuffled with replacement)
  - Path-dependent resampling (preserves autocorrelation)
  - Position-sizing sensitivity (test different risk per trade)

This is critical for setting realistic expectations. A backtest that
shows +50% return with one equity curve might have a 20% probability
of going bust under resampling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("enhancements.monte_carlo")


@dataclass
class MonteCarloResult:
    n_simulations: int
    n_trades_per_sim: int
    # Equity curve statistics
    mean_final_equity: float
    median_final_equity: float
    p5_final_equity: float
    p25_final_equity: float
    p75_final_equity: float
    p95_final_equity: float
    # Drawdown statistics
    mean_max_drawdown_pct: float
    median_max_drawdown_pct: float
    p5_max_drawdown_pct: float          # 5th percentile (lucky)
    p95_max_drawdown_pct: float         # 95th percentile (unlucky)
    # Risk of ruin
    prob_bust: float                     # P(equity < 0)
    prob_drawdown_20pct: float
    prob_drawdown_30pct: float
    prob_drawdown_50pct: float
    # Return statistics
    mean_annual_return_pct: float
    median_annual_return_pct: float
    prob_profitable: float
    # Sample equity curves (for plotting)
    sample_curves: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_simulations": self.n_simulations,
            "n_trades_per_sim": self.n_trades_per_sim,
            "mean_final_equity": self.mean_final_equity,
            "median_final_equity": self.median_final_equity,
            "p5_final_equity": self.p5_final_equity,
            "p25_final_equity": self.p25_final_equity,
            "p75_final_equity": self.p75_final_equity,
            "p95_final_equity": self.p95_final_equity,
            "mean_max_drawdown_pct": self.mean_max_drawdown_pct,
            "median_max_drawdown_pct": self.median_max_drawdown_pct,
            "p5_max_drawdown_pct": self.p5_max_drawdown_pct,
            "p95_max_drawdown_pct": self.p95_max_drawdown_pct,
            "prob_bust": self.prob_bust,
            "prob_drawdown_20pct": self.prob_drawdown_20pct,
            "prob_drawdown_30pct": self.prob_drawdown_30pct,
            "prob_drawdown_50pct": self.prob_drawdown_50pct,
            "mean_annual_return_pct": self.mean_annual_return_pct,
            "median_annual_return_pct": self.median_annual_return_pct,
            "prob_profitable": self.prob_profitable,
            "sample_curves": [list(c) for c in self.sample_curves],
        }


# ----------------------------------------------------------------------
class MonteCarloSimulator:
    def __init__(self,
                 n_simulations: int = 1000,
                 n_trades_per_sim: int = 200,
                 initial_equity: float = 10_000.0,
                 risk_per_trade_pct: float = 0.01,
                 seed: int = 42) -> None:
        self.n_sims = int(n_simulations)
        self.n_trades = int(n_trades_per_sim)
        self.initial_equity = float(initial_equity)
        self.risk_pct = float(risk_per_trade_pct)
        self.rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------
    def run(self, historical_pnls: list[float] | np.ndarray,
              method: str = "bootstrap") -> MonteCarloResult:
        """Run Monte Carlo simulation.

        Args:
            historical_pnls: list of per-trade R-multiples (e.g. +1.0 = 1R win,
                             -1.0 = 1R loss, +2.5 = 2.5R win). These are
                             multiplied by `risk_pct` to get the equity change.
            method: "bootstrap" (shuffle with replacement) or
                    "path" (block-bootstrap to preserve autocorrelation)
        """
        pnls = np.array(historical_pnls, dtype=float)
        if len(pnls) < 10:
            raise ValueError(f"need >= 10 historical trades, got {len(pnls)}")

        final_equities = np.empty(self.n_sims)
        max_drawdowns = np.empty(self.n_sims)
        sample_curves: list[list[float]] = []
        n_bust = 0
        n_dd20 = 0
        n_dd30 = 0
        n_dd50 = 0
        n_profitable = 0

        for i in range(self.n_sims):
            if method == "path":
                sample = self._block_resample(pnls, self.n_trades)
            else:
                sample = self.rng.choice(pnls, size=self.n_trades, replace=True)
            # Simulate equity curve with position sizing
            equity = self.initial_equity
            curve = [equity]
            peak = equity
            max_dd = 0.0
            for pnl in sample:
                # Critical #2 fix: `pnl` represents an R-multiple (e.g. +1.0
                # = one risk unit gained, -1.0 = full risk lost). `risk_pct`
                # is the fraction of equity risked per trade (e.g. 0.01 = 1%).
                # So the equity change per trade is: pnl × risk_pct × equity.
                # The old formula `pnl * risk_pct * 100` was a confusing mix
                # of percentage and multiplier that only worked when pnl was
                # exactly 1% per trade — it over-scaled by 100×.
                #
                # Correct: equity *= (1.0 + pnl * self.risk_pct)
                #   — if pnl = +1.0 (1R win) and risk_pct = 0.01, equity grows by 1%.
                #   — if pnl = -1.0 (1R loss) and risk_pct = 0.01, equity drops by 1%.
                equity *= (1.0 + pnl * self.risk_pct)
                if equity < 0:
                    equity = 0
                    n_bust += 1
                    break
                curve.append(equity)
                peak = max(peak, equity)
                if peak > 0:
                    dd = (peak - equity) / peak
                    max_dd = max(max_dd, dd)
            final_equities[i] = equity
            max_drawdowns[i] = max_dd
            if max_dd >= 0.20:
                n_dd20 += 1
            if max_dd >= 0.30:
                n_dd30 += 1
            if max_dd >= 0.50:
                n_dd50 += 1
            if equity > self.initial_equity:
                n_profitable += 1
            # Keep first 100 curves as samples for plotting
            if i < 100:
                sample_curves.append(curve)

        # Annualised return approximation (assume 252 trading days, ~5 trades/day)
        n_trades_per_year = 252 * 5
        years = self.n_trades / n_trades_per_year if n_trades_per_year > 0 else 1
        final_returns = (final_equities / self.initial_equity) - 1.0
        annual_returns = np.where(years > 0,
                                    (1 + final_returns) ** (1 / years) - 1,
                                    final_returns)

        return MonteCarloResult(
            n_simulations=self.n_sims,
            n_trades_per_sim=self.n_trades,
            mean_final_equity=float(final_equities.mean()),
            median_final_equity=float(np.median(final_equities)),
            p5_final_equity=float(np.percentile(final_equities, 5)),
            p25_final_equity=float(np.percentile(final_equities, 25)),
            p75_final_equity=float(np.percentile(final_equities, 75)),
            p95_final_equity=float(np.percentile(final_equities, 95)),
            mean_max_drawdown_pct=float(max_drawdowns.mean()),
            median_max_drawdown_pct=float(np.median(max_drawdowns)),
            p5_max_drawdown_pct=float(np.percentile(max_drawdowns, 5)),
            p95_max_drawdown_pct=float(np.percentile(max_drawdowns, 95)),
            prob_bust=float(n_bust / self.n_sims),
            prob_drawdown_20pct=float(n_dd20 / self.n_sims),
            prob_drawdown_30pct=float(n_dd30 / self.n_sims),
            prob_drawdown_50pct=float(n_dd50 / self.n_sims),
            mean_annual_return_pct=float(annual_returns.mean()),
            median_annual_return_pct=float(np.median(annual_returns)),
            prob_profitable=float(n_profitable / self.n_sims),
            sample_curves=sample_curves,
        )

    # ----------------------------------------------------------------
    def _block_resample(self, pnls: np.ndarray, n: int,
                          block_size: int = 10) -> np.ndarray:
        """Block bootstrap — preserves short-term autocorrelation."""
        out = []
        while len(out) < n:
            start = self.rng.integers(0, max(1, len(pnls) - block_size))
            block = pnls[start:start + block_size]
            out.extend(block.tolist())
        return np.array(out[:n])

    # ----------------------------------------------------------------
    def risk_assessment(self, result: MonteCarloResult) -> dict[str, Any]:
        """Translate MC results into risk verdict."""
        if result.prob_bust > 0.05:
            risk = "CRITICAL"
            reason = f"probability of bust = {result.prob_bust:.1%} > 5%"
        elif result.p95_max_drawdown_pct > 0.50:
            risk = "HIGH"
            reason = f"95th percentile DD = {result.p95_max_drawdown_pct:.1%} > 50%"
        elif result.p95_max_drawdown_pct > 0.30:
            risk = "MODERATE"
            reason = f"95th percentile DD = {result.p95_max_drawdown_pct:.1%} > 30%"
        elif result.prob_profitable < 0.70:
            risk = "MODERATE"
            reason = f"probability of profit = {result.prob_profitable:.1%} < 70%"
        else:
            risk = "LOW"
            reason = "all risk metrics within acceptable range"
        return {
            "risk_level": risk,
            "reason": reason,
            "prob_bust": result.prob_bust,
            "p95_max_dd": result.p95_max_drawdown_pct,
            "prob_profitable": result.prob_profitable,
        }
