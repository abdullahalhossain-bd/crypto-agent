"""
Monte Carlo Trade Simulation
============================

Before taking a trade, run thousands of simulated outcomes based on
historical win rate, reward:risk ratio, and risk-per-trade to estimate:

    - Probability of profit
    - Expected value (EV)
    - Max drawdown distribution (5th, 50th, 95th percentile)
    - Risk of ruin (probability of blowing the account)
    - Equity curve percentiles

This is the institutional pre-trade risk check that point estimates cannot
provide. A strategy with 60% win rate and 2:1 R:R can still blow up if
risk-per-trade is too aggressive.

Usage:
    from trading_modules.monte_carlo import MonteCarloSimulator, SimulationInput
    sim = MonteCarloSimulator(n_runs=10000, n_trades=100)
    result = sim.run(SimulationInput(
        win_probability=0.55,
        avg_win_pct=0.03,
        avg_loss_pct=0.015,
        risk_per_trade_pct=0.02,
        initial_capital=10000,
    ))
    print(f"Prob of profit: {result.prob_profit:.1%}")
    print(f"Risk of ruin:   {result.risk_of_ruin:.1%}")
    print(f"Median max DD:  {result.median_max_drawdown_pct:.1%}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SimulationInput:
    win_probability: float           # 0..1
    avg_win_pct: float               # e.g., 0.03 = +3% on win
    avg_loss_pct: float              # e.g., 0.015 = -1.5% on loss
    risk_per_trade_pct: float        # fraction of equity risked per trade
    initial_capital: float = 10000.0
    ruin_threshold_pct: float = 0.5  # equity falls below this × initial = ruin


@dataclass
class SimulationResult:
    n_runs: int
    n_trades: int
    prob_profit: float               # P(final equity > initial)
    expected_value: float            # mean final equity
    median_final_equity: float
    p5_final_equity: float           # 5th percentile (worst-case)
    p95_final_equity: float          # 95th percentile (best-case)
    risk_of_ruin: float              # P(equity < ruin_threshold × initial)
    median_max_drawdown_pct: float
    p95_max_drawdown_pct: float
    p5_max_drawdown_pct: float
    equity_curves: list = field(default_factory=list)  # sample of curves

    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "n_trades": self.n_trades,
            "prob_profit": round(self.prob_profit, 4),
            "expected_value": round(self.expected_value, 2),
            "median_final_equity": round(self.median_final_equity, 2),
            "p5_final_equity": round(self.p5_final_equity, 2),
            "p95_final_equity": round(self.p95_final_equity, 2),
            "risk_of_ruin": round(self.risk_of_ruin, 4),
            "median_max_drawdown_pct": round(self.median_max_drawdown_pct, 4),
            "p95_max_drawdown_pct": round(self.p95_max_drawdown_pct, 4),
            "p5_max_drawdown_pct": round(self.p5_max_drawdown_pct, 4),
        }


class MonteCarloSimulator:
    """Monte Carlo trade outcome simulator.

    Parameters:
        n_runs: # of simulated equity curves (default 10000)
        n_trades: # of trades per simulation (default 100)
        seed: RNG seed for reproducibility (default 42)
        keep_curves: # of sample equity curves to keep for plotting (default 50)
    """

    def __init__(
        self, n_runs: int = 10000, n_trades: int = 100,
        seed: int = 42, keep_curves: int = 50,
    ) -> None:
        if n_runs < 100:
            raise ValueError(f"n_runs must be >= 100, got {n_runs}")
        if n_trades < 10:
            raise ValueError(f"n_trades must be >= 10, got {n_trades}")
        self.n_runs = int(n_runs)
        self.n_trades = int(n_trades)
        self.seed = int(seed)
        self.keep_curves = int(keep_curves)

    def run(self, inp: SimulationInput) -> SimulationResult:
        """Run the simulation."""
        rng = np.random.default_rng(self.seed)
        n_runs = self.n_runs
        n_trades = self.n_trades
        p = float(inp.win_probability)
        win_pct = float(inp.avg_win_pct)
        loss_pct = float(inp.avg_loss_pct)
        risk_pct = float(inp.risk_per_trade_pct)
        initial = float(inp.initial_capital)
        ruin_threshold = initial * float(inp.ruin_threshold_pct)

        # Pre-generate all random wins/losses
        wins_mask = rng.random((n_runs, n_trades)) < p

        # Simulate equity curves
        equities = np.full((n_runs, n_trades + 1), initial, dtype=float)
        peak = np.full(n_runs, initial, dtype=float)
        max_dd = np.zeros(n_runs, dtype=float)

        for t in range(n_trades):
            # Per-trade return = ±risk_pct × (win_pct / loss_pct scaling)
            # Win → equity *= (1 + risk_pct × (win_pct / loss_pct))
            # Loss → equity *= (1 - risk_pct)
            # Simpler interpretation: risk_pct fraction is risked; on win we gain
            # risk_pct × (win_pct/loss_pct); on loss we lose risk_pct
            ret = np.where(
                wins_mask[:, t],
                risk_pct * (win_pct / loss_pct) if loss_pct > 0 else risk_pct,
                -risk_pct,
            )
            equities[:, t + 1] = equities[:, t] * (1 + ret)
            # Update peak and drawdown
            peak = np.maximum(peak, equities[:, t + 1])
            dd = (peak - equities[:, t + 1]) / peak
            max_dd = np.maximum(max_dd, dd)

        final_equities = equities[:, -1]
        # Risk of ruin: equity fell below ruin_threshold at any point
        # Check the min equity across all trades
        min_equity = equities.min(axis=1)
        risk_of_ruin = float(np.mean(min_equity < ruin_threshold))
        prob_profit = float(np.mean(final_equities > initial))

        # Sort final equities for percentiles
        sorted_final = np.sort(final_equities)
        sorted_dd = np.sort(max_dd)

        # Keep a sample of equity curves for plotting
        sample_idx = rng.choice(n_runs, size=min(self.keep_curves, n_runs), replace=False)
        sample_curves = equities[sample_idx].tolist()

        return SimulationResult(
            n_runs=n_runs,
            n_trades=n_trades,
            prob_profit=prob_profit,
            expected_value=float(final_equities.mean()),
            median_final_equity=float(np.median(final_equities)),
            p5_final_equity=float(np.percentile(sorted_final, 5)),
            p95_final_equity=float(np.percentile(sorted_final, 95)),
            risk_of_ruin=risk_of_ruin,
            median_max_drawdown_pct=float(np.median(max_dd)),
            p95_max_drawdown_pct=float(np.percentile(sorted_dd, 95)),
            p5_max_drawdown_pct=float(np.percentile(sorted_dd, 5)),
            equity_curves=sample_curves,
        )

    def quick_check(self, inp: SimulationInput) -> dict:
        """One-line summary for quick go/no-go decisions."""
        r = self.run(inp)
        verdict = "GO" if (r.risk_of_ruin < 0.05 and r.prob_profit > 0.6) else "NO-GO"
        return {
            "verdict": verdict,
            "prob_profit": f"{r.prob_profit:.1%}",
            "risk_of_ruin": f"{r.risk_of_ruin:.1%}",
            "median_max_dd": f"{r.median_max_drawdown_pct:.1%}",
            "ev": f"${r.expected_value:,.0f}",
        }


__all__ = ["MonteCarloSimulator", "SimulationInput", "SimulationResult"]
