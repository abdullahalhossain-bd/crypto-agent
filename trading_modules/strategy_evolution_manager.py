"""trading_modules/strategy_evolution_manager.py
=====================================================================
Strategy Evolution Manager (Principle #170, #171)
=====================================================================
Manages the continuous evolution of trading strategies:
    COLLECT → EVALUATE → RETRAIN → VALIDATE → DEPLOY → (repeat)

Different from strategy_lifecycle_manager.py (which tracks stages),
this module actually orchestrates the retraining pipeline:
    - Collects new training data
    - Triggers retraining when edge decays
    - Validates new models on out-of-sample data
    - Deploys new versions with shadow testing
    - Rolls back if new version underperforms

Evolution Pipeline:
    1. DATA COLLECTION — accumulate new trade outcomes + features
    2. DECAY DETECTION — monitor for edge decline
    3. RETRAIN TRIGGER — when decay > threshold, start retraining
    4. VALIDATION — test new model on held-out data
    5. SHADOW TEST — run new model in parallel (no real trades)
    6. PROMOTION — if shadow outperforms, promote to production
    7. ROLLBACK — if promoted model underperforms, revert

Usage:
    mgr = StrategyEvolutionManager()

    # Collect data
    mgr.collect_trade("momentum", features={...}, outcome="win", pnl=42)

    # Check if retrain needed
    if mgr.should_retrain("momentum"):
        new_version = mgr.trigger_retrain("momentum")
        # ... retrain model ...
        mgr.validate("momentum", new_version, validation_data)

    # Promote or rollback
    if mgr.shadow_performance_good("momentum", new_version):
        mgr.promote("momentum", new_version)
    else:
        mgr.rollback("momentum")
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.strategy_evolution_manager")


class EvolutionStage(str, Enum):
    COLLECTING = "collecting"      # gathering new data
    DECAY_DETECTED = "decay_detected"  # edge declining
    RETRAINING = "retraining"      # model being retrained
    VALIDATING = "validating"      # testing new model
    SHADOW_TESTING = "shadow_testing"  # running in parallel
    PROMOTED = "promoted"          # new version deployed
    ROLLED_BACK = "rolled_back"    # reverted to old version
    STABLE = "stable"              # no evolution needed


@dataclass
class StrategyVersion:
    """A version of a strategy."""
    version_id: str = ""
    created_at: str = ""
    parent_version: str = ""       # which version it replaced
    # Performance metrics
    validation_ev_r: float = 0.0
    validation_win_rate: float = 0.0
    validation_sharpe: float = 0.0
    shadow_ev_r: float = 0.0
    shadow_trades: int = 0
    # Status
    stage: EvolutionStage = EvolutionStage.COLLECTING
    is_active: bool = False
    # Training data
    training_samples: int = 0
    # Notes
    notes: str = ""


@dataclass
class EvolutionReport:
    """Evolution status report for a strategy."""
    strategy: str
    current_version: str = ""
    current_stage: EvolutionStage = EvolutionStage.STABLE
    versions_total: int = 0
    active_version: Optional[StrategyVersion] = None
    shadow_version: Optional[StrategyVersion] = None
    # Decay
    current_ev_r: float = 0.0
    historical_ev_r: float = 0.0
    decay_pct: float = 0.0
    retrain_recommended: bool = False
    # Data
    samples_collected: int = 0
    samples_needed: int = 100
    # History
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "current_version": self.current_version,
            "current_stage": self.current_stage.value,
            "versions_total": self.versions_total,
            "current_ev_r": round(self.current_ev_r, 3),
            "historical_ev_r": round(self.historical_ev_r, 3),
            "decay_pct": round(self.decay_pct, 3),
            "retrain_recommended": self.retrain_recommended,
            "samples_collected": self.samples_collected,
            "samples_needed": self.samples_needed,
            "recommendations": self.recommendations,
        }


class StrategyEvolutionManager:
    """Manages strategy evolution pipeline."""

    def __init__(self,
                 min_samples_for_retrain: int = 100,
                 decay_threshold_pct: float = 0.25,
                 shadow_min_trades: int = 20,
                 shadow_outperform_threshold: float = 0.1,
                 rollback_underperform_threshold: float = -0.2):
        """Initialize evolution manager.

        Args:
            min_samples_for_retrain: minimum new samples before retraining
            decay_threshold_pct: edge decline > this → retrain
            shadow_min_trades: trades needed in shadow before promotion
            shadow_outperform_threshold: shadow must outperform by this R
            rollback_underperform_threshold: rollback if underperform by this R
        """
        self.min_samples = min_samples_for_retrain
        self.decay_threshold = decay_threshold_pct
        self.shadow_min = shadow_min_trades
        self.shadow_outperform = shadow_outperform_threshold
        self.rollback_threshold = rollback_underperform_threshold

        self._lock = threading.RLock()
        # strategy → list of versions
        self._versions: Dict[str, List[StrategyVersion]] = defaultdict(list)
        # strategy → collected training data
        self._training_data: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=1000))
        # strategy → historical performance
        self._historical_ev: Dict[str, float] = {}
        # strategy → current EV
        self._current_ev: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_trade(self, strategy: str,
                      features: Dict[str, Any],
                      outcome: str, pnl: float, r_multiple: float) -> None:
        """Collect a trade outcome for future retraining."""
        with self._lock:
            self._training_data[strategy].append({
                "timestamp": time.time(),
                "features": features,
                "outcome": outcome,
                "pnl": pnl,
                "r_multiple": r_multiple,
                "win": outcome == "win",
            })
            # Update current EV
            trades = list(self._training_data[strategy])[-50:]
            if trades:
                self._current_ev[strategy] = float(np.mean([t["r_multiple"] for t in trades]))

    # ------------------------------------------------------------------
    # Decay detection
    # ------------------------------------------------------------------
    def should_retrain(self, strategy: str) -> bool:
        """Check if a strategy needs retraining."""
        with self._lock:
            current = self._current_ev.get(strategy, 0.0)
            historical = self._historical_ev.get(strategy, current)
            samples = len(self._training_data.get(strategy, []))

        if historical == 0:
            return False

        decay = (historical - current) / max(abs(historical), 0.01)
        return decay > self.decay_threshold and samples >= self.min_samples

    def get_decay(self, strategy: str) -> float:
        """Get edge decay percentage for a strategy."""
        with self._lock:
            current = self._current_ev.get(strategy, 0.0)
            historical = self._historical_ev.get(strategy, current)
        if historical == 0:
            return 0.0
        return (historical - current) / max(abs(historical), 0.01)

    # ------------------------------------------------------------------
    # Retrain pipeline
    # ------------------------------------------------------------------
    def trigger_retrain(self, strategy: str) -> str:
        """Trigger a retraining cycle. Returns new version ID."""
        with self._lock:
            current_versions = self._versions.get(strategy, [])
            parent = current_versions[-1].version_id if current_versions else ""
            version_id = f"{strategy}_v{len(current_versions) + 1}_{int(time.time())}"

            new_version = StrategyVersion(
                version_id=version_id,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                parent_version=parent,
                stage=EvolutionStage.RETRAINING,
                training_samples=len(self._training_data.get(strategy, [])),
            )
            self._versions[strategy].append(new_version)

        log.info("evolution: triggered retrain for %s → %s", strategy, version_id)
        return version_id

    def validate(self, strategy: str, version_id: str,
                 validation_ev_r: float, validation_win_rate: float,
                 validation_sharpe: float) -> bool:
        """Validate a retrained model on out-of-sample data.

        Returns True if validation passed.
        """
        with self._lock:
            versions = self._versions.get(strategy, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                return False

            version.validation_ev_r = validation_ev_r
            version.validation_win_rate = validation_win_rate
            version.validation_sharpe = validation_sharpe
            version.stage = EvolutionStage.VALIDATING

        # Validation passes if EV is positive
        passed = validation_ev_r > 0
        if passed:
            log.info("evolution: %s %s validated (EV=%.2fR, WR=%.0f%%)",
                     strategy, version_id, validation_ev_r, validation_win_rate * 100)
        else:
            log.warning("evolution: %s %s validation FAILED (EV=%.2fR)",
                       strategy, version_id, validation_ev_r)
        return passed

    def start_shadow_test(self, strategy: str, version_id: str) -> None:
        """Start shadow testing a new version (parallel to production)."""
        with self._lock:
            versions = self._versions.get(strategy, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is not None:
                version.stage = EvolutionStage.SHADOW_TESTING
                version.shadow_trades = 0
                version.shadow_ev_r = 0.0

        log.info("evolution: %s %s shadow testing started", strategy, version_id)

    def record_shadow_trade(self, strategy: str, version_id: str,
                            r_multiple: float) -> None:
        """Record a shadow trade result."""
        with self._lock:
            versions = self._versions.get(strategy, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                return
            version.shadow_trades += 1
            # Rolling average
            n = version.shadow_trades
            version.shadow_ev_r = (version.shadow_ev_r * (n - 1) + r_multiple) / n

    def shadow_performance_good(self, strategy: str,
                                version_id: str) -> bool:
        """Check if shadow version is outperforming production.

        Returns True if safe to promote.
        """
        with self._lock:
            versions = self._versions.get(strategy, [])
            new_version = next((v for v in versions if v.version_id == version_id), None)
            if new_version is None or new_version.shadow_trades < self.shadow_min:
                return False
            current_ev = self._current_ev.get(strategy, 0.0)
        # Shadow must outperform by threshold
        return new_version.shadow_ev_r > current_ev + self.shadow_outperform

    def promote(self, strategy: str, version_id: str) -> None:
        """Promote a shadow version to production."""
        with self._lock:
            versions = self._versions.get(strategy, [])
            # Deactivate old active version
            for v in versions:
                if v.is_active:
                    v.is_active = False
            # Activate new version
            new_version = next((v for v in versions if v.version_id == version_id), None)
            if new_version is not None:
                new_version.is_active = True
                new_version.stage = EvolutionStage.PROMOTED
                # Update historical EV
                self._historical_ev[strategy] = new_version.shadow_ev_r
                self._current_ev[strategy] = new_version.shadow_ev_r
                # Clear training data (start fresh)
                self._training_data[strategy].clear()

        log.info("evolution: %s %s PROMOTED to production", strategy, version_id)

    def rollback(self, strategy: str) -> Optional[str]:
        """Rollback to the previous version.

        Returns the version ID rolled back to, or None.
        """
        with self._lock:
            versions = self._versions.get(strategy, [])
            if len(versions) < 2:
                return None
            # Deactivate current
            current = next((v for v in versions if v.is_active), None)
            if current:
                current.is_active = False
                current.stage = EvolutionStage.ROLLED_BACK
            # Activate previous
            previous = versions[-2]
            previous.is_active = True
            previous.stage = EvolutionStage.PROMOTED

        log.warning("evolution: %s rolled back to %s", strategy, previous.version_id)
        return previous.version_id

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def report(self, strategy: str) -> EvolutionReport:
        """Get evolution report for a strategy."""
        with self._lock:
            versions = self._versions.get(strategy, [])
            active = next((v for v in versions if v.is_active), None)
            shadow = next((v for v in versions if v.stage == EvolutionStage.SHADOW_TESTING), None)
            current_ev = self._current_ev.get(strategy, 0.0)
            historical_ev = self._historical_ev.get(strategy, current_ev)
            samples = len(self._training_data.get(strategy, []))

        decay = 0.0
        if historical_ev != 0:
            decay = (historical_ev - current_ev) / max(abs(historical_ev), 0.01)

        report = EvolutionReport(
            strategy=strategy,
            current_version=active.version_id if active else "none",
            current_stage=active.stage if active else EvolutionStage.STABLE,
            versions_total=len(versions),
            active_version=active,
            shadow_version=shadow,
            current_ev_r=current_ev,
            historical_ev_r=historical_ev,
            decay_pct=decay,
            retrain_recommended=self.should_retrain(strategy),
            samples_collected=samples,
            samples_needed=self.min_samples,
            version_history=[
                {"version": v.version_id, "stage": v.stage.value,
                 "active": v.is_active, "ev_r": v.validation_ev_r}
                for v in versions[-5:]  # last 5 versions
            ],
        )

        # Recommendations
        if report.retrain_recommended:
            report.recommendations.append(
                f"RETRAIN recommended — decay {decay:.0%}, {samples} samples collected"
            )
        if shadow and shadow.shadow_trades >= self.shadow_min:
            if self.shadow_performance_good(strategy, shadow.version_id):
                report.recommendations.append(
                    f"PROMOTE {shadow.version_id} — shadow outperforming by "
                    f"{shadow.shadow_ev_r - current_ev:.2f}R"
                )
            else:
                report.recommendations.append(
                    f"Shadow {shadow.version_id} not yet outperforming "
                    f"({shadow.shadow_ev_r:.2f}R vs {current_ev:.2f}R)"
                )
        if not report.recommendations:
            report.recommendations.append("Stable — no evolution actions needed")

        return report

    def all_reports(self) -> List[EvolutionReport]:
        """Get reports for all strategies."""
        with self._lock:
            strategies = list(self._versions.keys())
        return [self.report(s) for s in strategies]

    def summary(self) -> Dict[str, Any]:
        """Get summary of all evolution activity."""
        with self._lock:
            total_versions = sum(len(v) for v in self._versions.values())
            active_strategies = sum(1 for versions in self._versions.values()
                                   if any(v.is_active for v in versions))
            shadow_testing = sum(1 for versions in self._versions.values()
                                if any(v.stage == EvolutionStage.SHADOW_TESTING
                                      for v in versions))
        return {
            "total_strategies": len(self._versions),
            "active_strategies": active_strategies,
            "shadow_testing": shadow_testing,
            "total_versions": total_versions,
            "total_samples_collected": sum(len(d) for d in self._training_data.values()),
        }
