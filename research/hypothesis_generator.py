"""research.hypothesis_generator
=====================================================================
Day 45-47 — Hypothesis Generator.

Generates CANDIDATE strategies — signal rules, NOT trading logic.
Each hypothesis is a small declarative spec:

    IF <feature> <operator> <threshold> THEN <action>

The hypothesis generator doesn't know about positions, sizing, risk,
or execution. It only emits signal rules that the evaluation pipeline
will test.

Generation strategies:
  - grid       : enumerate combinations of (feature, op, threshold)
  - quantile   : thresholds at quartiles of the feature distribution
  - crossover  : pairs of features whose difference crosses zero
  - random     : Monte Carlo sampling (with seeds for reproducibility)

The generator is intentionally CHEAP — it doesn't run any backtests.
The evaluation pipeline decides which hypotheses survive.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("research.hypothesis")


class Operator(str, Enum):
    GT = ">"
    LT = "<"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


@dataclass
class StrategyHypothesis:
    """A candidate strategy — declarative signal rule."""
    hypothesis_id: str
    name: str
    feature_a: str
    feature_b: Optional[str]      # for crossover hypotheses
    operator: Operator
    threshold: float
    action: str                    # "BUY" | "SELL"
    generation_method: str         # grid | quantile | crossover | random
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "name": self.name,
            "feature_a": self.feature_a,
            "feature_b": self.feature_b,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "action": self.action,
            "generation_method": self.generation_method,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    def evaluate_signal(self, features_df: pd.DataFrame) -> pd.Series:
        """Apply the rule to a feature DataFrame → boolean Series.

        True = signal fires on that bar. Caller decides what to do
        with the signal (typically BUY or SELL depending on `action`).
        """
        if self.feature_a not in features_df.columns:
            return pd.Series(False, index=features_df.index)
        a = features_df[self.feature_a]
        if self.operator == Operator.GT:
            return a > self.threshold
        if self.operator == Operator.LT:
            return a < self.threshold
        if self.feature_b is None or self.feature_b not in features_df.columns:
            return pd.Series(False, index=features_df.index)
        b = features_df[self.feature_b]
        diff = a - b
        if self.operator == Operator.CROSS_ABOVE:
            return (diff > 0) & (diff.shift(1) <= 0)
        if self.operator == Operator.CROSS_BELOW:
            return (diff < 0) & (diff.shift(1) >= 0)
        return pd.Series(False, index=features_df.index)


# ----------------------------------------------------------------------
class HypothesisGenerator:
    """Generates a catalogue of strategy hypotheses."""

    def __init__(self,
                 random_seed: int = 42,
                 max_per_method: int = 100) -> None:
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)
        self.max_per_method = int(max_per_method)
        self._counter = 0

    # ----------------------------------------------------------------
    def generate(self, feature_names: list[str],
                 features_df: Optional[pd.DataFrame] = None,
                 methods: tuple[str, ...] = ("grid", "quantile",
                                              "crossover", "random"),
                 ) -> list[StrategyHypothesis]:
        """Generate hypotheses for the given feature set."""
        out: list[StrategyHypothesis] = []
        if "grid" in methods:
            out.extend(self._grid(feature_names))
        if "quantile" in methods and features_df is not None:
            out.extend(self._quantile(feature_names, features_df))
        if "crossover" in methods:
            out.extend(self._crossover(feature_names))
        if "random" in methods and features_df is not None:
            out.extend(self._random(feature_names, features_df))
        log.info("Generated %d hypotheses (methods=%s)", len(out), methods)
        return out

    # ----------------------------------------------------------------
    def _grid(self, feature_names: list[str]) -> list[StrategyHypothesis]:
        out: list[StrategyHypothesis] = []
        thresholds = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        for feat in feature_names:
            for op, action in [(Operator.GT, "BUY"), (Operator.LT, "SELL")]:
                for thr in thresholds:
                    out.append(self._make(
                        feature_a=feat, feature_b=None,
                        operator=op, threshold=thr,
                        action=action, method="grid",
                        description=f"{feat} {op.value} {thr}",
                    ))
                    if len(out) >= self.max_per_method * 4:
                        return out
        return out

    # ----------------------------------------------------------------
    def _quantile(self, feature_names: list[str],
                  features_df: pd.DataFrame) -> list[StrategyHypothesis]:
        out: list[StrategyHypothesis] = []
        for feat in feature_names:
            if feat not in features_df.columns:
                continue
            series = features_df[feat].dropna()
            if len(series) < 50:
                continue
            qs = [0.1, 0.25, 0.5, 0.75, 0.9]
            thresholds = [float(series.quantile(q)) for q in qs]
            for op, action in [(Operator.GT, "BUY"), (Operator.LT, "SELL")]:
                for q, thr in zip(qs, thresholds):
                    out.append(self._make(
                        feature_a=feat, feature_b=None,
                        operator=op, threshold=thr,
                        action=action, method="quantile",
                        description=f"{feat} {op.value} {thr:.4f} (q={q})",
                        metadata={"quantile": q},
                    ))
                    if len(out) >= self.max_per_method * 4:
                        return out
        return out

    # ----------------------------------------------------------------
    def _crossover(self, feature_names: list[str]) -> list[StrategyHypothesis]:
        out: list[StrategyHypothesis] = []
        # Only pair features from the same family (cheap filter)
        pairs = list(itertools.combinations(feature_names, 2))
        self.rng.shuffle(pairs)
        for a, b in pairs[:self.max_per_method]:
            out.append(self._make(
                feature_a=a, feature_b=b,
                operator=Operator.CROSS_ABOVE, threshold=0.0,
                action="BUY", method="crossover",
                description=f"{a} crosses above {b}",
            ))
            out.append(self._make(
                feature_a=a, feature_b=b,
                operator=Operator.CROSS_BELOW, threshold=0.0,
                action="SELL", method="crossover",
                description=f"{a} crosses below {b}",
            ))
        return out

    # ----------------------------------------------------------------
    def _random(self, feature_names: list[str],
                features_df: pd.DataFrame) -> list[StrategyHypothesis]:
        out: list[StrategyHypothesis] = []
        if not feature_names:
            return out
        for _ in range(self.max_per_method):
            feat = self.rng.choice(feature_names)
            if feat not in features_df.columns:
                continue
            series = features_df[feat].dropna()
            if len(series) < 10:
                continue
            lo, hi = float(series.min()), float(series.max())
            if lo == hi:
                continue
            thr = self.rng.uniform(lo, hi)
            op = self.rng.choice([Operator.GT, Operator.LT])
            action = "BUY" if op == Operator.GT else "SELL"
            out.append(self._make(
                feature_a=feat, feature_b=None,
                operator=op, threshold=thr,
                action=action, method="random",
                description=f"random: {feat} {op.value} {thr:.4f}",
            ))
        return out

    # ----------------------------------------------------------------
    def _make(self, feature_a: str, feature_b: Optional[str],
              operator: Operator, threshold: float, action: str,
              method: str, description: str = "",
              metadata: Optional[dict[str, Any]] = None) -> StrategyHypothesis:
        self._counter += 1
        hid = f"H{self._counter:05d}"
        name = f"{method}_{hid}"
        return StrategyHypothesis(
            hypothesis_id=hid,
            name=name,
            feature_a=feature_a,
            feature_b=feature_b,
            operator=operator,
            threshold=float(threshold),
            action=action,
            generation_method=method,
            description=description,
            metadata=metadata or {},
        )
