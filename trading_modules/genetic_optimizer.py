"""
Genetic Algorithm Module — Strategy & Parameter Optimization
=============================================================

Uses evolutionary algorithms to optimize:
  - Strategy parameters (SL %, TP %, indicator periods)
  - Strategy selection (which strategies to combine)
  - Position sizing rules
  - Entry/exit conditions

Pipeline:
  1. Initialize population (random parameter sets)
  2. Evaluate fitness (backtest each individual)
  3. Select best performers (tournament selection)
  4. Crossover (combine parent parameters)
  5. Mutate (random changes)
  6. Repeat for N generations

Source: Orallexa (review #27) — strategy_evolver.py concept
        ml4t-3e (review #18) — optimization best practices

Usage:
    from trading_modules.genetic_optimizer import GeneticOptimizer, StrategyGenome

    optimizer = GeneticOptimizer(
        population_size=50,
        n_generations=20,
        mutation_rate=0.1,
    )

    # Define parameter search space
    search_space = {
        "sma_fast": (5, 50),
        "sma_slow": (20, 200),
        "rsi_period": (7, 28),
        "stop_loss_pct": (0.005, 0.05),
        "take_profit_pct": (0.01, 0.10),
    }

    # Run optimization
    best = optimizer.optimize(
        search_space=search_space,
        fitness_fn=my_backtest_function,
        verbose=True,
    )

    print(f"Best params: {best.params}")
    print(f"Best fitness (Sharpe): {best.fitness:.2f}")
"""

from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyGenome:
    """A single strategy parameter set (individual in population)."""
    params: dict[str, float]
    fitness: float = -999.0  # Higher = better
    generation: int = 0

    def copy(self) -> "StrategyGenome":
        return StrategyGenome(params=dict(self.params), fitness=self.fitness, generation=self.generation)


class GeneticOptimizer:
    """
    Genetic Algorithm for strategy parameter optimization.

    Evolves a population of parameter sets over multiple generations,
    selecting for high fitness (typically Sharpe ratio).

    Selection: Tournament (k=3)
    Crossover: Uniform (each parameter randomly from either parent)
    Mutation: Gaussian (add noise to each parameter with probability)
    """

    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 20,
        mutation_rate: float = 0.1,
        mutation_std: float = 0.1,
        elite_pct: float = 0.1,  # Top 10% carried over unchanged
        tournament_size: int = 3,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elite_pct = elite_pct
        self.tournament_size = tournament_size
        self.rng = random.Random(random_state)
        self.np_rng = np.random.RandomState(random_state)

        self.history: list[dict] = []
        self.best_genome: Optional[StrategyGenome] = None

    def optimize(
        self,
        search_space: dict[str, tuple[float, float]],
        fitness_fn: Callable[[dict], float],
        verbose: bool = True,
    ) -> StrategyGenome:
        """
        Run genetic optimization.

        Args:
            search_space: {param_name: (min_value, max_value)}
            fitness_fn: Function that takes params dict → returns fitness (higher=better)
            verbose: Print progress

        Returns:
            Best StrategyGenome found
        """
        # Initialize population
        population = self._init_population(search_space)

        for gen in range(self.n_generations):
            # Evaluate fitness
            for individual in population:
                if individual.fitness == -999.0:
                    individual.fitness = fitness_fn(individual.params)
                individual.generation = gen

            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Track best
            gen_best = population[0]
            if self.best_genome is None or gen_best.fitness > self.best_genome.fitness:
                self.best_genome = gen_best.copy()

            # Track history
            gen_stats = {
                "generation": gen,
                "best_fitness": population[0].fitness,
                "mean_fitness": np.mean([x.fitness for x in population]),
                "worst_fitness": population[-1].fitness,
                "best_params": dict(population[0].params),
            }
            self.history.append(gen_stats)

            if verbose and (gen % 5 == 0 or gen == self.n_generations - 1):
                print(f"  Gen {gen:3d} | Best: {gen_stats['best_fitness']:.4f} | "
                      f"Mean: {gen_stats['mean_fitness']:.4f} | "
                      f"Params: {gen_stats['best_params']}")

            # Create next generation
            next_pop = []

            # Elitism: carry over top performers
            n_elite = max(1, int(self.population_size * self.elite_pct))
            next_pop.extend([x.copy() for x in population[:n_elite]])

            # Fill rest with crossover + mutation
            while len(next_pop) < self.population_size:
                parent1 = self._tournament(population)
                parent2 = self._tournament(population)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, search_space)
                child.fitness = -999.0  # Force re-evaluation
                next_pop.append(child)

            population = next_pop

        # Final evaluation
        for individual in population:
            if individual.fitness == -999.0:
                individual.fitness = fitness_fn(individual.params)
        population.sort(key=lambda x: x.fitness, reverse=True)

        if population[0].fitness > self.best_genome.fitness:
            self.best_genome = population[0].copy()

        return self.best_genome

    def _init_population(self, search_space: dict) -> list[StrategyGenome]:
        """Initialize random population."""
        pop = []
        for _ in range(self.population_size):
            params = {}
            for name, (lo, hi) in search_space.items():
                # Random uniform in range
                if isinstance(lo, int) and isinstance(hi, int):
                    params[name] = self.rng.randint(lo, hi)
                else:
                    params[name] = self.rng.uniform(lo, hi)
            pop.append(StrategyGenome(params=params))
        return pop

    def _tournament(self, population: list[StrategyGenome]) -> StrategyGenome:
        """Tournament selection — pick k random, return best."""
        contestants = self.rng.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda x: x.fitness).copy()

    def _crossover(self, p1: StrategyGenome, p2: StrategyGenome) -> StrategyGenome:
        """Uniform crossover — each parameter randomly from either parent."""
        child_params = {}
        for key in p1.params:
            child_params[key] = self.rng.choice([p1.params[key], p2.params[key]])
        return StrategyGenome(params=child_params)

    def _mutate(self, genome: StrategyGenome, search_space: dict) -> StrategyGenome:
        """Gaussian mutation — add noise to each parameter with probability."""
        for key, (lo, hi) in search_space.items():
            if self.rng.random() < self.mutation_rate:
                # Add Gaussian noise proportional to range
                range_size = hi - lo
                noise = self.np_rng.normal(0, range_size * self.mutation_std)
                new_val = genome.params[key] + noise
                # Clip to search space
                new_val = max(lo, min(hi, new_val))
                if isinstance(lo, int) and isinstance(hi, int):
                    new_val = int(round(new_val))
                genome.params[key] = new_val
        return genome

    def get_history(self) -> list[dict]:
        """Get optimization history for plotting."""
        return self.history

    def get_summary(self) -> dict:
        """Get optimization summary."""
        if not self.history:
            return {"status": "not_run"}
        return {
            "generations": len(self.history),
            "best_fitness": self.best_genome.fitness if self.best_genome else None,
            "best_params": self.best_genome.params if self.best_genome else None,
            "final_gen_best": self.history[-1]["best_fitness"],
            "improvement": self.history[-1]["best_fitness"] - self.history[0]["best_fitness"],
        }
