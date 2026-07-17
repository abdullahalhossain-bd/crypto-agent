"""
Game Theory — strategic decision-making under adversarial conditions
====================================================================

Markets are adversarial — every trade has a counterparty. This module
implements game-theoretic concepts for trading:

    1. Nash Equilibrium          — find stable strategy profiles
    2. Zero-Sum Game Solver      — minimax / maximin for buy vs sell
    3. Adversarial Awareness     — model what other traders will do
    4. Mixed Strategy Nash       — probabilistic strategies
    5. Bayesian Games            — games with incomplete info
    6. Cooperative Game Value    — Shapley value for portfolio attribution

Usage:
    from trading_modules.game_theory import (
        zero_sum_game_solver, mixed_strategy_nash, shapley_value
    )
    # Solve a 2x2 zero-sum payoff matrix
    value, p1_strategy, p2_strategy = zero_sum_game_solver(payoff_matrix)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GameSolution:
    value: float                       # game value (expected payoff for player 1)
    player1_strategy: np.ndarray       # mixed strategy (probabilities)
    player2_strategy: np.ndarray
    nash_equilibria: list[tuple] = field(default_factory=list)
    is_pure_equilibrium: bool = False
    notes: list[str] = field(default_factory=list)


def zero_sum_game_solver(payoff_matrix: np.ndarray) -> GameSolution:
    """Solve a 2-player zero-sum game using linear programming (simplified).

    For player 1 (maximizer): find p that maximizes min_j sum_i p_i * a_ij
    For player 2 (minimizer): find q that minimizes max_i sum_j a_ij * q_j

    Uses a grid search for small matrices; for larger ones, uses iterative LP.

    Args:
        payoff_matrix: (m, n) matrix where a_ij = payoff to player 1
                       when P1 plays i and P2 plays j
    """
    A = np.asarray(payoff_matrix, dtype=float)
    m, n = A.shape
    if m == 0 or n == 0:
        return GameSolution(0, np.array([]), np.array([]), notes=["empty matrix"])

    # Check for pure strategy equilibrium (saddle point)
    row_mins = A.min(axis=1)
    col_maxs = A.max(axis=0)
    max_row_min = float(row_mins.max())
    min_col_max = float(col_maxs.min())
    if abs(max_row_min - min_col_max) < 1e-10:
        # Saddle point exists → pure strategy
        i_star = int(np.argmax(row_mins))
        j_star = int(np.argmin(col_maxs))
        p1 = np.zeros(m); p1[i_star] = 1.0
        p2 = np.zeros(n); p2[j_star] = 1.0
        return GameSolution(
            value=max_row_min, player1_strategy=p1, player2_strategy=p2,
            nash_equilibria=[(i_star, j_star)],
            is_pure_equilibrium=True,
            notes=[f"pure equilibrium at ({i_star}, {j_star})"],
        )

    # Mixed strategy — grid search for 2x2, iterative for larger
    if m == 2 and n == 2:
        return _solve_2x2(A)
    else:
        return _solve_iterative(A)


def _solve_2x2(A: np.ndarray) -> GameSolution:
    """Solve a 2x2 zero-sum game analytically."""
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    # Player 1's mixed strategy: p = (d - c) / (a + d - b - c)
    denom = a + d - b - c
    if abs(denom) < 1e-10:
        # Degenerate — use uniform
        p1 = np.array([0.5, 0.5])
        p2 = np.array([0.5, 0.5])
        value = float(A[0, 0] * 0.5 + A[0, 1] * 0.5)
    else:
        p = (d - c) / denom
        p = max(0.0, min(1.0, p))
        p1 = np.array([p, 1 - p])
        # Player 2: q = (d - b) / (a + d - b - c)
        q = (d - b) / denom
        q = max(0.0, min(1.0, q))
        p2 = np.array([q, 1 - q])
        value = float(a * p * q + b * p * (1 - q) + c * (1 - p) * q + d * (1 - p) * (1 - q))
    return GameSolution(
        value=value, player1_strategy=p1, player2_strategy=p2,
        notes=["2x2 analytical solution"],
    )


def _solve_iterative(A: np.ndarray, n_iter: int = 1000) -> GameSolution:
    """Iterative fictitious play for larger games."""
    m, n = A.shape
    # Start with uniform
    p1 = np.ones(m) / m
    p2 = np.ones(n) / n
    for _ in range(n_iter):
        # Player 1's best response to p2
        expected_p1 = A @ p2
        i_star = int(np.argmax(expected_p1))
        # Player 2's best response to p1
        expected_p2 = p1 @ A
        j_star = int(np.argmin(expected_p2))
        # Update mixed strategies (slowly)
        br1 = np.zeros(m); br1[i_star] = 1.0
        br2 = np.zeros(n); br2[j_star] = 1.0
        p1 = 0.99 * p1 + 0.01 * br1
        p2 = 0.99 * p2 + 0.01 * br2
    value = float(p1 @ A @ p2)
    return GameSolution(
        value=value, player1_strategy=p1, player2_strategy=p2,
        notes=[f"iterative fictitious play ({n_iter} iters)"],
    )


def mixed_strategy_nash(
    payoff_matrix_p1: np.ndarray, payoff_matrix_p2: np.ndarray,
    n_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Find mixed strategy Nash equilibrium for a general-sum 2-player game.

    Uses iterative best response (fictitious play).

    Args:
        payoff_matrix_p1: (m, n) payoff to player 1
        payoff_matrix_p2: (m, n) payoff to player 2
        n_iter: # of iterations

    Returns:
        (p1_strategy, p2_strategy)
    """
    A = np.asarray(payoff_matrix_p1, dtype=float)
    B = np.asarray(payoff_matrix_p2, dtype=float)
    m, n = A.shape
    p1 = np.ones(m) / m
    p2 = np.ones(n) / n
    for _ in range(n_iter):
        # Player 1's best response to p2
        exp_p1 = A @ p2
        i_star = int(np.argmax(exp_p1))
        # Player 2's best response to p1
        exp_p2 = p1 @ B
        j_star = int(np.argmax(exp_p2))
        br1 = np.zeros(m); br1[i_star] = 1.0
        br2 = np.zeros(n); br2[j_star] = 1.0
        p1 = 0.99 * p1 + 0.01 * br1
        p2 = 0.99 * p2 + 0.01 * br2
    return p1, p2


def shapley_value(coalition_values: dict[frozenset, float]) -> dict:
    """Compute Shapley value for cooperative game.

    Useful for attributing portfolio profit to individual strategies/positions.

    Args:
        coalition_values: dict mapping frozenset of player names → coalition value

    Returns:
        dict mapping player name → Shapley value
    """
    # Get all players
    all_players: set = set()
    for coalition in coalition_values:
        all_players.update(coalition)
    players = sorted(all_players)
    n = len(players)
    if n == 0:
        return {}
    shapley: dict[str, float] = {p: 0.0 for p in players}
    # Iterate over all permutations (only feasible for small n)
    from itertools import permutations
    for perm in permutations(players):
        coalition: set = set()
        for i, player in enumerate(perm):
            coalition_with = frozenset(coalition | {player})
            coalition_without = frozenset(coalition)
            v_with = coalition_values.get(coalition_with, 0.0)
            v_without = coalition_values.get(coalition_without, 0.0)
            shapley[player] += (v_with - v_without) / math.factorial(n)
            coalition.add(player)
    return shapley


def adversarial_awareness(
    my_strategy: str, predicted_counter_strategies: dict[str, float],
    payoff_matrix: dict[str, dict[str, float]],
) -> dict:
    """Model what happens when adversaries react to your strategy.

    Args:
        my_strategy: my chosen strategy (e.g., "BUY")
        predicted_counter_strategies: dict of (counter_strategy → probability)
            e.g., {"stop_hunt": 0.3, "fade": 0.4, "follow": 0.3}
        payoff_matrix: dict of dicts
            payoff_matrix[my_strategy][counter] = my expected payoff

    Returns:
        dict with expected_payoff, worst_case, best_case
    """
    if my_strategy not in payoff_matrix:
        return {"expected_payoff": 0, "worst_case": 0, "best_case": 0}
    row = payoff_matrix[my_strategy]
    payoffs = []
    for counter, prob in predicted_counter_strategies.items():
        if counter in row:
            payoffs.append((counter, prob, row[counter]))
    if not payoffs:
        return {"expected_payoff": 0, "worst_case": 0, "best_case": 0}
    expected = sum(p * v for _, p, v in payoffs)
    worst = min(v for _, _, v in payoffs)
    best = max(v for _, _, v in payoffs)
    return {
        "expected_payoff": float(expected),
        "worst_case": float(worst),
        "best_case": float(best),
        "scenarios": [
            {"counter": c, "probability": p, "payoff": v}
            for c, p, v in payoffs
        ],
    }


__all__ = [
    "GameSolution", "zero_sum_game_solver", "mixed_strategy_nash",
    "shapley_value", "adversarial_awareness",
]
