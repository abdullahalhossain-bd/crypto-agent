"""enhancements.strategy_ensemble
=====================================================================
Day 160 — Intelligent strategy ensemble.

Different from engine.strategy_runner (which runs strategies in
parallel and produces a consensus vote). The ensemble layer goes
further: it LEARNS which strategy combinations work best in which
regimes, and produces a smarter combination.

Combination methods:
  - VOTE            : simple majority vote (baseline)
  - WEIGHTED        : weight by recent Sharpe
  - REGIME_ROUTING  : use only strategies whose affinity matches current regime
  - BAYESIAN        : Bayesian model averaging (treat each strategy as a hypothesis)
  - STACKING        : train a meta-learner on strategy outputs

For now we implement VOTE, WEIGHTED, and REGIME_ROUTING. STACKING
would require an ML model that takes strategy outputs as features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from engine.signals import Action, Signal
from engine.strategy_runner import SignalPool
from utils.logger import get_logger

log = get_logger("enhancements.ensemble")


class CombinationMethod(str, Enum):
    VOTE = "vote"
    WEIGHTED = "weighted"
    REGIME_ROUTING = "regime_routing"
    BAYESIAN = "bayesian"


@dataclass
class EnsembleResult:
    method: CombinationMethod
    action: Action
    confidence: float                # 0-100
    contributing_strategies: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "action": self.action.value,
            "confidence": self.confidence,
            "contributing_strategies": list(self.contributing_strategies),
            "weights": dict(self.weights),
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
class StrategyEnsemble:
    """Combines signals from multiple strategies intelligently."""

    def __init__(self, method: CombinationMethod = CombinationMethod.REGIME_ROUTING):
        self.method = method
        # Per-strategy rolling Sharpe for WEIGHTED method
        self._strategy_sharpes: dict[str, float] = {}
        # Per-strategy regime affinity (from StrategyMetadata)
        self._regime_affinities: dict[str, dict[str, float]] = {}

    # ----------------------------------------------------------------
    def update_strategy_sharpe(self, strategy_name: str, sharpe: float) -> None:
        """Update rolling Sharpe for a strategy (used by WEIGHTED method)."""
        # Exponential moving average
        old = self._strategy_sharpes.get(strategy_name, 0.0)
        self._strategy_sharpes[strategy_name] = 0.9 * old + 0.1 * float(sharpe)

    def set_regime_affinities(self, affinities: dict[str, dict[str, float]]) -> None:
        self._regime_affinities = dict(affinities)

    # ----------------------------------------------------------------
    def combine(
        self,
        pool: SignalPool,
        current_regime: str = "",
        regime_confidence: float = 1.0,
    ) -> EnsembleResult:
        """Combine the signal pool into a single ensemble decision."""
        if not pool.signals:
            return EnsembleResult(
                method=self.method, action=Action.HOLD,
                confidence=0.0, reason="no signals",
            )
        if self.method == CombinationMethod.VOTE:
            return self._combine_vote(pool)
        if self.method == CombinationMethod.WEIGHTED:
            return self._combine_weighted(pool)
        if self.method == CombinationMethod.REGIME_ROUTING:
            return self._combine_regime_routing(pool, current_regime, regime_confidence)
        if self.method == CombinationMethod.BAYESIAN:
            return self._combine_bayesian(pool)
        return self._combine_vote(pool)  # fallback

    # ----------------------------------------------------------------
    def _combine_vote(self, pool: SignalPool) -> EnsembleResult:
        """Simple majority vote. Each strategy gets equal weight."""
        votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        contributors: list[str] = []
        for name, sig in pool.signals.items():
            votes[sig.action.value] = votes.get(sig.action.value, 0) + 1
            if sig.is_actionable:
                contributors.append(name)
        # Winner
        winner = max(votes, key=votes.get)
        total = sum(votes.values())
        confidence = float(100.0 * votes[winner] / total) if total > 0 else 0.0
        # If tie between BUY and SELL, default to HOLD
        if votes["BUY"] == votes["SELL"] and votes["BUY"] > 0:
            return EnsembleResult(
                method=CombinationMethod.VOTE,
                action=Action.HOLD, confidence=50.0,
                contributing_strategies=contributors,
                reason="BUY/SELL tie — defaulting to HOLD",
                metadata={"votes": votes},
            )
        return EnsembleResult(
            method=CombinationMethod.VOTE,
            action=Action[winner], confidence=confidence,
            contributing_strategies=contributors,
            weights={n: 1.0 for n in pool.signals},
            reason=f"majority vote: {votes}",
            metadata={"votes": votes},
        )

    # ----------------------------------------------------------------
    def _combine_weighted(self, pool: SignalPool) -> EnsembleResult:
        """Weight by rolling Sharpe. Better-performing strategies get more weight."""
        weights: dict[str, float] = {}
        weighted_buy = 0.0
        weighted_sell = 0.0
        total_weight = 0.0
        for name, sig in pool.signals.items():
            sharpe = self._strategy_sharpes.get(name, 0.5)
            # Weight = max(0.1, sharpe) so even negative-Sharpe strategies get a small voice
            w = max(0.1, sharpe)
            weights[name] = w
            total_weight += w
            if sig.action == Action.BUY:
                weighted_buy += w * sig.strength
            elif sig.action == Action.SELL:
                weighted_sell += w * sig.strength
        if total_weight == 0:
            return EnsembleResult(
                method=CombinationMethod.WEIGHTED, action=Action.HOLD,
                confidence=0.0, reason="no weight",
            )
        net = (weighted_buy - weighted_sell) / total_weight
        if abs(net) < 0.15:
            action = Action.HOLD
            confidence = 50.0
        else:
            action = Action.BUY if net > 0 else Action.SELL
            confidence = float(min(100.0, abs(net) * 100.0))
        return EnsembleResult(
            method=CombinationMethod.WEIGHTED,
            action=action, confidence=confidence,
            contributing_strategies=[n for n, s in pool.signals.items() if s.is_actionable],
            weights=weights,
            reason=f"net weighted vote={net:.3f}",
            metadata={"weighted_buy": weighted_buy, "weighted_sell": weighted_sell},
        )

    # ----------------------------------------------------------------
    def _combine_regime_routing(self, pool: SignalPool,
                                  regime: str,
                                  regime_confidence: float) -> EnsembleResult:
        """Only use strategies whose regime affinity matches the current regime."""
        if not regime:
            return self._combine_vote(pool)
        # Filter strategies by affinity
        filtered_signals: dict[str, Signal] = {}
        weights: dict[str, float] = {}
        for name, sig in pool.signals.items():
            affinity = self._regime_affinities.get(name, {}).get(regime, 0.5)
            if affinity < 0.2:
                continue  # skip strategies with very low affinity
            filtered_signals[name] = sig
            weights[name] = affinity * regime_confidence
        if not filtered_signals:
            return EnsembleResult(
                method=CombinationMethod.REGIME_ROUTING,
                action=Action.HOLD, confidence=0.0,
                reason=f"no strategies match regime={regime}",
                metadata={"regime": regime, "regime_confidence": regime_confidence},
            )
        # Weighted vote on filtered set
        weighted_buy = 0.0
        weighted_sell = 0.0
        total_weight = 0.0
        for name, sig in filtered_signals.items():
            w = weights[name]
            total_weight += w
            if sig.action == Action.BUY:
                weighted_buy += w * sig.strength
            elif sig.action == Action.SELL:
                weighted_sell += w * sig.strength
        if total_weight == 0:
            return EnsembleResult(
                method=CombinationMethod.REGIME_ROUTING,
                action=Action.HOLD, confidence=0.0,
                reason="zero total weight",
            )
        net = (weighted_buy - weighted_sell) / total_weight
        if abs(net) < 0.15:
            action = Action.HOLD
            confidence = 50.0
        else:
            action = Action.BUY if net > 0 else Action.SELL
            confidence = float(min(100.0, abs(net) * 100.0))
        return EnsembleResult(
            method=CombinationMethod.REGIME_ROUTING,
            action=action, confidence=confidence,
            contributing_strategies=list(filtered_signals.keys()),
            weights=weights,
            reason=f"regime={regime} conf={regime_confidence:.2f} net={net:.3f}",
            metadata={"regime": regime, "regime_confidence": regime_confidence,
                      "n_filtered": len(filtered_signals),
                      "n_total": len(pool.signals)},
        )

    # ----------------------------------------------------------------
    def _combine_bayesian(self, pool: SignalPool) -> EnsembleResult:
        """Bayesian model averaging — each strategy is a 'hypothesis'.

        Posterior P(action | signals) ∝ Σ P(action | strategy_i) * P(strategy_i)
        where P(strategy_i) is the prior (rolling Sharpe).
        """
        # For simplicity, treat each strategy's signal as a noisy estimate
        # of the "true" action. Weighted average of indicators.
        weights: dict[str, float] = {}
        buy_prob = 0.0
        sell_prob = 0.0
        total = 0.0
        for name, sig in pool.signals.items():
            sharpe = self._strategy_sharpes.get(name, 0.5)
            prior = max(0.1, sharpe)
            weights[name] = prior
            total += prior
            # Each strategy votes with its strength
            if sig.action == Action.BUY:
                buy_prob += prior * sig.strength
                sell_prob += prior * (1 - sig.strength) * 0.5
            elif sig.action == Action.SELL:
                sell_prob += prior * sig.strength
                buy_prob += prior * (1 - sig.strength) * 0.5
            else:
                # HOLD splits evenly
                buy_prob += prior * 0.25
                sell_prob += prior * 0.25
        if total == 0:
            return EnsembleResult(
                method=CombinationMethod.BAYESIAN, action=Action.HOLD,
                confidence=0.0, reason="no weight",
            )
        buy_prob /= total
        sell_prob /= total
        if buy_prob > sell_prob and buy_prob > 0.4:
            action = Action.BUY
            confidence = float(buy_prob * 100.0)
        elif sell_prob > buy_prob and sell_prob > 0.4:
            action = Action.SELL
            confidence = float(sell_prob * 100.0)
        else:
            action = Action.HOLD
            confidence = 50.0
        return EnsembleResult(
            method=CombinationMethod.BAYESIAN,
            action=action, confidence=confidence,
            contributing_strategies=[n for n, s in pool.signals.items() if s.is_actionable],
            weights=weights,
            reason=f"P(BUY)={buy_prob:.2f} P(SELL)={sell_prob:.2f}",
            metadata={"buy_prob": buy_prob, "sell_prob": sell_prob},
        )
