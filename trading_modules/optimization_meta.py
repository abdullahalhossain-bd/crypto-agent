"""
Metaheuristic Optimization — global search for non-convex problems
====================================================================

Pure-Python implementations of metaheuristic optimizers:

    1. Genetic Algorithm (GA)    — population-based evolutionary search
    2. Particle Swarm Opt (PSO)  — swarm intelligence
    3. Simulated Annealing (SA)  — temperature-based local search
    4. Bayesian Optimization     — Gaussian-process-based global search

Useful for: strategy parameter tuning, portfolio weight optimization
when the objective is non-convex or non-differentiable.

Usage:
    from trading_modules.optimization_meta import (
        genetic_algorithm, particle_swarm, simulated_annealing, bayesian_optimize
    )
    # GA for strategy param tuning
    best_params, best_score = genetic_algorithm(
        objective=lambda params: -backtest(params),
        bounds=[(5, 50), (1, 10)],  # (fast_ema, slow_ema)
        pop_size=30, n_generations=50,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    best_params: np.ndarray
    best_score: float
    n_evaluations: int
    history: list[float] = field(default_factory=list)  # best score per iteration
    converged: bool = False

    def to_dict(self) -> dict:
        return {
            "best_params": [round(float(x), 4) for x in self.best_params],
            "best_score": round(self.best_score, 6),
            "n_evaluations": self.n_evaluations,
            "converged": self.converged,
            "history_length": len(self.history),
        }


# ──────────────────────────────────────────────────────────────────────
# 1. Genetic Algorithm
# ──────────────────────────────────────────────────────────────────────
def genetic_algorithm(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    pop_size: int = 30,
    n_generations: int = 50,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    seed: int = 42,
) -> OptimizationResult:
    """Genetic algorithm for continuous parameter optimization.

    Args:
        objective: function that takes a parameter vector and returns a scalar
                   (higher = better)
        bounds: list of (lower, upper) per dimension
        pop_size: population size
        n_generations: # of generations
        mutation_rate: probability of mutation per gene
        crossover_rate: probability of crossover
    """
    rng = np.random.default_rng(seed)
    n_dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Initialize population
    pop = rng.uniform(lb, ub, size=(pop_size, n_dim))
    fitness = np.array([float(objective(ind)) for ind in pop])
    n_evaluations = pop_size
    history: list[float] = [float(fitness.max())]

    for gen in range(n_generations):
        # Selection (tournament)
        parents = np.zeros_like(pop)
        for i in range(pop_size):
            candidates = rng.choice(pop_size, 3, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            parents[i] = pop[winner]
        # Crossover
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.random() < crossover_rate:
                alpha = rng.random(n_dim)
                children[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                children[i + 1] = (1 - alpha) * parents[i] + alpha * parents[i + 1]
        # Mutation
        for i in range(pop_size):
            if rng.random() < mutation_rate:
                mutation = rng.normal(0, 0.1 * (ub - lb), n_dim)
                children[i] = np.clip(children[i] + mutation, lb, ub)
        # Evaluate
        pop = children
        fitness = np.array([float(objective(ind)) for ind in pop])
        n_evaluations += pop_size
        history.append(float(fitness.max()))
        # Convergence check
        if gen > 10 and abs(history[-1] - history[-10]) < 1e-6:
            return OptimizationResult(
                best_params=pop[np.argmax(fitness)],
                best_score=float(fitness.max()),
                n_evaluations=n_evaluations,
                history=history,
                converged=True,
            )
    best_idx = int(np.argmax(fitness))
    return OptimizationResult(
        best_params=pop[best_idx],
        best_score=float(fitness[best_idx]),
        n_evaluations=n_evaluations,
        history=history,
        converged=False,
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Particle Swarm Optimization
# ──────────────────────────────────────────────────────────────────────
def particle_swarm(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_particles: int = 30,
    n_iterations: int = 50,
    w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
    seed: int = 42,
) -> OptimizationResult:
    """Particle Swarm Optimization.

    Args:
        objective: function (higher = better)
        bounds: list of (lower, upper) per dimension
        n_particles: swarm size
        n_iterations: max iterations
        w: inertia weight
        c1: cognitive (personal best) coefficient
        c2: social (global best) coefficient
    """
    rng = np.random.default_rng(seed)
    n_dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Initialize particles
    positions = rng.uniform(lb, ub, size=(n_particles, n_dim))
    velocities = rng.uniform(-(ub - lb), (ub - lb), size=(n_particles, n_dim))
    personal_best = positions.copy()
    personal_best_fitness = np.array([float(objective(p)) for p in positions])
    n_evaluations = n_particles
    global_best_idx = int(np.argmax(personal_best_fitness))
    global_best = personal_best[global_best_idx].copy()
    global_best_fitness = float(personal_best_fitness[global_best_idx])
    history: list[float] = [global_best_fitness]

    for it in range(n_iterations):
        r1 = rng.random((n_particles, n_dim))
        r2 = rng.random((n_particles, n_dim))
        velocities = (w * velocities +
                      c1 * r1 * (personal_best - positions) +
                      c2 * r2 * (global_best - positions))
        positions = np.clip(positions + velocities, lb, ub)
        # Evaluate
        fitness = np.array([float(objective(p)) for p in positions])
        n_evaluations += n_particles
        # Update personal best
        improved = fitness > personal_best_fitness
        personal_best[improved] = positions[improved]
        personal_best_fitness[improved] = fitness[improved]
        # Update global best
        best_idx = int(np.argmax(personal_best_fitness))
        if personal_best_fitness[best_idx] > global_best_fitness:
            global_best = personal_best[best_idx].copy()
            global_best_fitness = float(personal_best_fitness[best_idx])
        history.append(global_best_fitness)
        # Convergence
        if it > 10 and abs(history[-1] - history[-10]) < 1e-6:
            return OptimizationResult(
                best_params=global_best,
                best_score=global_best_fitness,
                n_evaluations=n_evaluations,
                history=history,
                converged=True,
            )
    return OptimizationResult(
        best_params=global_best,
        best_score=global_best_fitness,
        n_evaluations=n_evaluations,
        history=history,
        converged=False,
    )


# ──────────────────────────────────────────────────────────────────────
# 3. Simulated Annealing
# ──────────────────────────────────────────────────────────────────────
def simulated_annealing(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_iterations: int = 1000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.995,
    seed: int = 42,
) -> OptimizationResult:
    """Simulated Annealing.

    Args:
        objective: function (higher = better)
        bounds: list of (lower, upper) per dimension
        n_iterations: max iterations
        initial_temp: starting temperature
        cooling_rate: temperature decay per iteration
    """
    rng = np.random.default_rng(seed)
    n_dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    current = rng.uniform(lb, ub)
    current_score = float(objective(current))
    n_evaluations = 1
    best = current.copy()
    best_score = current_score
    temp = initial_temp
    history: list[float] = [best_score]

    for it in range(n_iterations):
        # Generate neighbor
        step = rng.normal(0, 0.1 * (ub - lb) * temp, n_dim)
        candidate = np.clip(current + step, lb, ub)
        candidate_score = float(objective(candidate))
        n_evaluations += 1
        # Accept?
        if candidate_score > current_score:
            current = candidate
            current_score = candidate_score
            if candidate_score > best_score:
                best = candidate.copy()
                best_score = candidate_score
        else:
            # Accept worse solution with probability exp(Δ / T)
            delta = candidate_score - current_score
            accept_prob = np.exp(delta / max(temp, 1e-10))
            if rng.random() < accept_prob:
                current = candidate
                current_score = candidate_score
        temp *= cooling_rate
        history.append(best_score)
        if it > 100 and abs(history[-1] - history[-100]) < 1e-6:
            return OptimizationResult(
                best_params=best, best_score=best_score,
                n_evaluations=n_evaluations, history=history, converged=True,
            )
    return OptimizationResult(
        best_params=best, best_score=best_score,
        n_evaluations=n_evaluations, history=history, converged=False,
    )


# ──────────────────────────────────────────────────────────────────────
# 4. Bayesian Optimization (simplified)
# ──────────────────────────────────────────────────────────────────────
def bayesian_optimize(
    objective: Callable[[np.ndarray], float],
    bounds: list[tuple[float, float]],
    n_iterations: int = 30,
    n_initial_samples: int = 5,
    seed: int = 42,
) -> OptimizationResult:
    """Simplified Bayesian Optimization using GP surrogate + Expected Improvement.

    Uses the Gaussian Process from statistics_advanced.py.

    Args:
        objective: function (higher = better)
        bounds: list of (lower, upper) per dimension
        n_iterations: # of BO iterations after initial sampling
        n_initial_samples: # of random initial samples
    """
    # Critical #4 fix: lazy import with fallback — if statistics_advanced
    # is unavailable or creates a circular import, use a simple GP fallback.
    try:
        from .statistics_advanced import gaussian_process_fit
    except (ImportError, Exception) as e:
        log.warning("optimization_meta: statistics_advanced unavailable (%r) — using simple fallback", e)
        gaussian_process_fit = None

    rng = np.random.default_rng(seed)
    n_dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # Initial random samples
    X = rng.uniform(lb, ub, size=(n_initial_samples, n_dim))
    y = np.array([float(objective(x)) for x in X])
    n_evaluations = n_initial_samples
    best_idx = int(np.argmax(y))
    best = X[best_idx].copy()
    best_score = float(y[best_idx])
    history: list[float] = [best_score]

    for it in range(n_iterations):
        # Fit GP surrogate
        # Use a candidate set for acquisition (grid + random)
        n_candidates = 200
        X_candidates = rng.uniform(lb, ub, size=(n_candidates, n_dim))
        # Critical #4 fix: fallback if gaussian_process_fit is unavailable.
        if gaussian_process_fit is not None:
            gp = gaussian_process_fit(X, y, X_candidates, length_scale=0.3, noise=0.01)
        else:
            # Simple fallback: use distance-weighted interpolation
            from collections import namedtuple
            GPResult = namedtuple("GPResult", ["mean", "std"])
            from scipy.spatial.distance import cdist as _cdist
            dists = _cdist(X_candidates, X)
            weights = np.exp(-dists**2 / (2 * 0.3**2))
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
            gp_mean = weights @ y
            gp_std = np.sqrt(np.maximum(0.0, weights @ (y**2) - gp_mean**2))
            gp = GPResult(mean=gp_mean, std=gp_std)
        # Expected Improvement
        best_y = float(y.max())
        improvement = gp.mean - best_y
        ei = improvement * (1 - 0.5 * (1 + np.sign(improvement))) + \
             gp.std * np.exp(-0.5 * (improvement / np.maximum(gp.std, 1e-10)) ** 2) / np.sqrt(2 * np.pi)
        # Select candidate with max EI
        next_idx = int(np.argmax(ei))
        next_x = X_candidates[next_idx]
        # Evaluate true objective
        next_y = float(objective(next_x))
        n_evaluations += 1
        # Update
        X = np.vstack([X, next_x])
        y = np.append(y, next_y)
        if next_y > best_score:
            best = next_x.copy()
            best_score = next_y
        history.append(best_score)
    return OptimizationResult(
        best_params=best, best_score=best_score,
        n_evaluations=n_evaluations, history=history, converged=False,
    )


__all__ = [
    "OptimizationResult",
    "genetic_algorithm", "particle_swarm",
    "simulated_annealing", "bayesian_optimize",
]
