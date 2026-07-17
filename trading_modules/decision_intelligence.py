"""
Decision Intelligence — beyond prediction to optimal decisions
================================================================

Prediction is only half the battle. This module focuses on making
optimal DECISIONS given predictions, uncertainty, and costs.

    1. Expected Utility         — EU = Σ p(s) × U(action, s)
    2. Kelly Fraction            — optimal bet size for repeated bets
    3. Optimal Stopping          — when to enter / exit
    4. Multi-Armed Bandit        — strategy selection under uncertainty
    5. Decision Tree             — explicit if/then expected value calculation
    6. Influence Diagram         — decision + chance + utility nodes

Usage:
    from trading_modules.decision_intelligence import (
        expected_utility, optimal_stop, multi_armed_bandit
    )
    # Expected utility of BUY vs SELL vs WAIT
    eu_buy = expected_utility(
        action="BUY",
        states={"up": 0.5, "down": 0.3, "flat": 0.2},
        utilities={"up": 100, "down": -50, "flat": 0},
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def expected_utility(
    action: str,
    states: dict[str, float],        # state → probability
    utilities: dict[str, float],     # state → utility for this action
) -> float:
    """Compute expected utility of an action.

    EU(action) = Σ_s P(s) × U(action, s)

    Args:
        action: name of the action (for logging)
        states: dict of state name → probability (must sum to ~1)
        utilities: dict of state name → utility for this action
    """
    eu = 0.0
    for state, prob in states.items():
        u = utilities.get(state, 0.0)
        eu += prob * u
    return float(eu)


def optimal_action(
    actions: list[str],
    states: dict[str, float],
    utility_matrix: dict[str, dict[str, float]],
) -> tuple[str, float, dict[str, float]]:
    """Find the action with highest expected utility.

    Args:
        actions: list of action names
        states: dict of state → probability
        utility_matrix: dict of action → (dict of state → utility)

    Returns:
        (best_action, best_eu, all_eus)
    """
    eus: dict[str, float] = {}
    for action in actions:
        eus[action] = expected_utility(action, states, utility_matrix.get(action, {}))
    best = max(eus, key=eus.get)
    return best, eus[best], eus


# ──────────────────────────────────────────────────────────────────────
# Optimal Stopping — when to enter/exit
# ──────────────────────────────────────────────────────────────────────
@dataclass
class OptimalStopResult:
    should_act: bool
    action: str                # "enter" / "exit" / "wait"
    threshold: float
    current_value: float
    expected_value_of_waiting: float
    notes: list[str] = field(default_factory=list)


def optimal_stop(
    current_value: float,
    threshold: float,
    expected_value_of_waiting: float,
    cost_of_waiting: float = 0.0,
) -> OptimalStopResult:
    """Optimal stopping rule — act when current_value ≥ threshold.

    Compare acting now vs waiting one more period.

    Args:
        current_value: value of acting now
        threshold: minimum value to trigger action
        expected_value_of_waiting: E[V(next period)] - cost_of_waiting
        cost_of_waiting: opportunity cost of not acting now

    Returns:
        OptimalStopResult with should_act + action
    """
    net_waiting = expected_value_of_waiting - cost_of_waiting
    if current_value >= threshold and current_value >= net_waiting:
        return OptimalStopResult(
            should_act=True, action="enter",
            threshold=threshold, current_value=current_value,
            expected_value_of_waiting=net_waiting,
            notes=[f"act now: {current_value:.2f} ≥ threshold {threshold:.2f}"],
        )
    elif net_waiting > current_value:
        return OptimalStopResult(
            should_act=False, action="wait",
            threshold=threshold, current_value=current_value,
            expected_value_of_waiting=net_waiting,
            notes=[f"wait: E[waiting]={net_waiting:.2f} > current={current_value:.2f}"],
        )
    else:
        return OptimalStopResult(
            should_act=False, action="wait",
            threshold=threshold, current_value=current_value,
            expected_value_of_waiting=net_waiting,
            notes=[f"wait: current {current_value:.2f} < threshold {threshold:.2f}"],
        )


# ──────────────────────────────────────────────────────────────────────
# Multi-Armed Bandit — strategy selection
# ──────────────────────────────────────────────────────────────────────
@dataclass
class BanditArm:
    name: str
    n_pulls: int = 0
    total_reward: float = 0.0
    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.n_pulls if self.n_pulls > 0 else 0.0


@dataclass
class BanditResult:
    chosen_arm: str
    expected_reward: float
    confidence: float
    all_arms: dict[str, dict]
    notes: list[str] = field(default_factory=list)


class MultiArmedBandit:
    """Thompson Sampling for strategy selection.

    Each "arm" is a trading strategy. The bandit learns which strategy
    works best over time and balances exploration vs exploitation.

    Parameters:
        arms: list of strategy names
        prior_alpha: Beta prior alpha (default 1)
        prior_beta: Beta prior beta (default 1)
    """

    def __init__(
        self, arms: list[str],
        prior_alpha: float = 1.0, prior_beta: float = 1.0,
    ) -> None:
        self.arms = {name: BanditArm(name) for name in arms}
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

    def select(self) -> BanditResult:
        """Select an arm using Thompson Sampling."""
        rng = np.random.default_rng()
        samples: dict[str, float] = {}
        for name, arm in self.arms.items():
            # Beta posterior: alpha = prior + wins, beta = prior + losses
            # wins = total_reward (if reward is binary 0/1)
            # For continuous rewards, use Beta(1 + successes, 1 + failures) approximation
            successes = max(0, int(arm.total_reward))  # assume reward ~ # of wins
            failures = max(0, arm.n_pulls - successes)
            alpha = self.prior_alpha + successes
            beta = self.prior_beta + failures
            samples[name] = float(rng.beta(alpha, beta))
        chosen = max(samples, key=samples.get)
        # Confidence: difference between top 2
        sorted_samples = sorted(samples.values(), reverse=True)
        confidence = (sorted_samples[0] - sorted_samples[1]) if len(sorted_samples) > 1 else 1.0
        all_arms = {
            name: {
                "n_pulls": arm.n_pulls,
                "mean_reward": round(arm.mean_reward, 4),
                "sample": round(samples[name], 4),
            }
            for name, arm in self.arms.items()
        }
        return BanditResult(
            chosen_arm=chosen,
            expected_reward=float(samples[chosen]),
            confidence=float(confidence),
            all_arms=all_arms,
        )

    def update(self, arm_name: str, reward: float) -> None:
        """Update the bandit with the observed reward."""
        if arm_name in self.arms:
            self.arms[arm_name].n_pulls += 1
            self.arms[arm_name].total_reward += float(reward)


# ──────────────────────────────────────────────────────────────────────
# Decision Tree — explicit expected value calculation
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DecisionNode:
    name: str
    node_type: str                # "decision" / "chance" / "terminal"
    children: list = field(default_factory=list)
    probability: float = 1.0      # for chance nodes
    value: float = 0.0            # for terminal nodes


def evaluate_decision_tree(node: DecisionNode) -> float:
    """Recursively evaluate a decision tree, returning expected value."""
    if node.node_type == "terminal":
        return float(node.value)
    elif node.node_type == "chance":
        ev = 0.0
        for child in node.children:
            ev += child.probability * evaluate_decision_tree(child)
        return float(ev)
    elif node.node_type == "decision":
        # Pick the child with highest EV
        if not node.children:
            return 0.0
        evs = [evaluate_decision_tree(c) for c in node.children]
        return float(max(evs))
    return 0.0


# ──────────────────────────────────────────────────────────────────────
# Kelly Criterion (multi-outcome)
# ──────────────────────────────────────────────────────────────────────
def kelly_fraction_multi(
    probabilities: list[float],
    payoffs: list[float],
) -> float:
    """Kelly fraction for a bet with multiple outcomes.

    f* = argmax E[log(1 + f*X)]

    For binary: f* = (p*b - q) / b where b = payoff odds
    For multi-outcome: numerical optimization.

    Args:
        probabilities: list of outcome probabilities
        payoffs: list of outcome payoffs (return per unit bet, e.g., +0.5 = +50%)

    Returns:
        Optimal fraction of bankroll to bet (0 = no bet)
    """
    if len(probabilities) != len(payoffs) or len(probabilities) == 0:
        return 0.0
    # Grid search
    best_f = 0.0
    best_ev = -np.inf
    for f in np.linspace(0, 1, 101):
        ev = 0.0
        for p, x in zip(probabilities, payoffs):
            ev += p * np.log(1 + f * x)
        if ev > best_ev:
            best_ev = ev
            best_f = f
    # Also check negative f (shorting)
    for f in np.linspace(-1, 0, 101):
        ev = 0.0
        for p, x in zip(probabilities, payoffs):
            ev += p * np.log(1 + f * x)
        if ev > best_ev:
            best_ev = ev
            best_f = f
    return float(best_f)


__all__ = [
    "expected_utility", "optimal_action",
    "OptimalStopResult", "optimal_stop",
    "BanditArm", "BanditResult", "MultiArmedBandit",
    "DecisionNode", "evaluate_decision_tree",
    "kelly_fraction_multi",
]
