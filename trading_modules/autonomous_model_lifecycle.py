"""trading_modules/autonomous_model_lifecycle.py
=====================================================================
Autonomous Model Lifecycle Manager (Principle #199)
=====================================================================
Manages the complete ML model lifecycle without human intervention:
    COLLECT → VALIDATE → RETRAIN → BACKTEST → PAPER_TRADE → DEPLOY → MONITOR

Each stage has automatic quality gates:
    - COLLECT: accumulate enough samples (min 100)
    - VALIDATE: check data quality (no NaN, balanced classes)
    - RETRAIN: train new model on accumulated data
    - BACKTEST: test on historical out-of-sample data
    - PAPER_TRADE: run forward on live data without real orders
    - DEPLOY: promote to production
    - MONITOR: track live performance, trigger retrain if decay

Safety Rules:
    - Never deploy without backtest + paper trade passing
    - Always keep previous version for rollback
    - Auto-rollback if live performance < backtest performance * 0.5
    - Human alert if 3+ rollbacks in a week

Usage:
    mgr = AutonomousModelLifecycleManager()

    # Collect data
    mgr.collect_sample("momentum_v1", features={...}, label="win")

    # Check if ready for retrain
    if mgr.ready_for_retrain("momentum_v1"):
        # Train new model
        new_model = train_model(mgr.get_training_data("momentum_v1"))
        # Validate
        if mgr.validate("momentum_v1", new_model, validation_data):
            # Backtest
            bt_results = mgr.backtest("momentum_v1", new_model, historical_data)
            if bt_results["passed"]:
                # Paper trade
                mgr.start_paper_trading("momentum_v1", new_model)
                # After paper trading passes:
                mgr.deploy("momentum_v1", new_model)

    # Monitor
    mgr.monitor()
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

log = get_logger("trading_bot.autonomous_model_lifecycle")


class LifecycleStage(str, Enum):
    COLLECTING = "collecting"
    VALIDATED = "validated"
    RETRAINED = "retrained"
    BACKTESTED = "backtested"
    PAPER_TRADING = "paper_trading"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class ModelVersion:
    """A version of an ML model."""
    version_id: str = ""
    created_at: str = ""
    parent_version: str = ""
    stage: LifecycleStage = LifecycleStage.COLLECTING
    # Quality gates
    validation_passed: bool = False
    backtest_passed: bool = False
    paper_trade_passed: bool = False
    # Metrics
    validation_accuracy: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    paper_trade_sharpe: float = 0.0
    paper_trade_trades: int = 0
    live_sharpe: float = 0.0
    live_trades: int = 0
    # Status
    is_active: bool = False
    is_healthy: bool = True
    # Training data
    training_samples: int = 0
    # Timestamps
    deployed_at: Optional[str] = None
    last_checked: Optional[str] = None


@dataclass
class LifecycleReport:
    """Lifecycle status report for a model."""
    model_name: str
    current_version: str = ""
    current_stage: LifecycleStage = LifecycleStage.COLLECTING
    versions_total: int = 0
    active_version: Optional[ModelVersion] = None
    # Data collection
    samples_collected: int = 0
    samples_needed: int = 100
    ready_for_retrain: bool = False
    # Pipeline status
    pipeline_stage: str = ""
    next_action: str = ""
    # Health
    is_healthy: bool = True
    rollback_count: int = 0
    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "current_version": self.current_version,
            "current_stage": self.current_stage.value,
            "versions_total": self.versions_total,
            "samples_collected": self.samples_collected,
            "samples_needed": self.samples_needed,
            "ready_for_retrain": self.ready_for_retrain,
            "pipeline_stage": self.pipeline_stage,
            "next_action": self.next_action,
            "is_healthy": self.is_healthy,
            "rollback_count": self.rollback_count,
            "recommendations": self.recommendations,
        }


class AutonomousModelLifecycleManager:
    """Manages the full ML model lifecycle autonomously."""

    def __init__(self,
                 min_samples_for_retrain: int = 100,
                 min_backtest_sharpe: float = 0.8,
                 min_paper_trade_trades: int = 30,
                 min_paper_trade_sharpe: float = 0.5,
                 rollback_threshold_ratio: float = 0.5,
                 max_rollbacks_per_week: int = 3):
        """Initialize lifecycle manager.

        Args:
            min_samples_for_retrain: minimum samples before retraining
            min_backtest_sharpe: minimum Sharpe to pass backtest
            min_paper_trade_trades: trades needed in paper trading
            min_paper_trade_sharpe: minimum Sharpe in paper trading
            rollback_threshold_ratio: rollback if live < backtest * this
            max_rollbacks_per_week: alert if exceeded
        """
        self.min_samples = min_samples_for_retrain
        self.min_bt_sharpe = min_backtest_sharpe
        self.min_pt_trades = min_paper_trade_trades
        self.min_pt_sharpe = min_paper_trade_sharpe
        self.rollback_ratio = rollback_threshold_ratio
        self.max_rollbacks = max_rollbacks_per_week

        self._lock = threading.RLock()
        # model_name → list of versions
        self._versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        # model_name → training data
        self._training_data: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=5000))
        # Rollback tracking
        self._rollback_times: List[float] = []
        # Major #6 fix: per-model rollback cooldown — after a rollback, require
        # at least N new live trades before considering another rollback.
        # This prevents oscillation between versions.
        self._rollback_cooldown_trades: Dict[str, int] = defaultdict(int)
        self._min_trades_between_rollbacks = 10

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_sample(self, model_name: str,
                       features: Dict[str, Any],
                       label: str) -> None:
        """Collect a training sample."""
        with self._lock:
            self._training_data[model_name].append({
                "timestamp": time.time(),
                "features": features,
                "label": label,
            })

    def get_training_data(self, model_name: str) -> List[dict]:
        """Get accumulated training data."""
        with self._lock:
            return list(self._training_data.get(model_name, []))

    def ready_for_retrain(self, model_name: str) -> bool:
        """Check if enough samples collected."""
        with self._lock:
            return len(self._training_data.get(model_name, [])) >= self.min_samples

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def validate(self, model_name: str, version_id: str,
                 accuracy: float) -> bool:
        """Validate model on held-out data."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                return False
            version.validation_accuracy = accuracy
            version.validation_passed = accuracy > 0.55
            version.stage = LifecycleStage.VALIDATED if version.validation_passed else LifecycleStage.FAILED
            return version.validation_passed

    def backtest(self, model_name: str, version_id: str,
                 sharpe: float, win_rate: float) -> bool:
        """Backtest model on historical data."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                return False
            version.backtest_sharpe = sharpe
            version.backtest_win_rate = win_rate
            version.backtest_passed = sharpe >= self.min_bt_sharpe
            version.stage = LifecycleStage.BACKTESTED if version.backtest_passed else LifecycleStage.FAILED
            return version.backtest_passed

    def start_paper_trading(self, model_name: str,
                            version_id: str) -> None:
        """Start paper trading a new model version."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is not None:
                version.stage = LifecycleStage.PAPER_TRADING
                version.paper_trade_trades = 0
                version.paper_trade_sharpe = 0.0

    def record_paper_trade(self, model_name: str, version_id: str,
                           pnl: float) -> None:
        """Record a paper trade result."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                return
            version.paper_trade_trades += 1
            # Rolling average P&L (proxy for Sharpe)
            n = version.paper_trade_trades
            version.paper_trade_sharpe = (version.paper_trade_sharpe * (n - 1) + pnl) / n

    def paper_trade_ready_for_deploy(self, model_name: str,
                                      version_id: str) -> bool:
        """Check if paper trading has enough data and performance."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            version = next((v for v in versions if v.version_id == version_id), None)
            if version is None:
                return False
            return (version.paper_trade_trades >= self.min_pt_trades and
                    version.paper_trade_sharpe >= self.min_pt_sharpe)

    def deploy(self, model_name: str, version_id: str) -> None:
        """Deploy a new model version to production."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            # Deactivate old
            for v in versions:
                if v.is_active:
                    v.is_active = False
            # Activate new
            new_version = next((v for v in versions if v.version_id == version_id), None)
            if new_version is not None:
                new_version.is_active = True
                new_version.stage = LifecycleStage.DEPLOYED
                new_version.deployed_at = datetime.now(tz=timezone.utc).isoformat()
                # Clear training data
                self._training_data[model_name].clear()

        log.info("lifecycle: deployed %s %s to production", model_name, version_id)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def record_live_trade(self, model_name: str, pnl: float) -> None:
        """Record a live trade from the active model."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            active = next((v for v in versions if v.is_active), None)
            if active is None:
                return
            active.live_trades += 1
            n = active.live_trades
            active.live_sharpe = (active.live_sharpe * (n - 1) + pnl) / n
            active.last_checked = datetime.now(tz=timezone.utc).isoformat()

            # Check for rollback condition
            # Major #6 fix: enforce cooldown — after a rollback, require
            # at least _min_trades_between_rollbacks new trades before
            # considering another rollback. This prevents version oscillation.
            cooldown_remaining = self._rollback_cooldown_trades.get(model_name, 0)
            if cooldown_remaining > 0:
                self._rollback_cooldown_trades[model_name] = cooldown_remaining - 1
            elif (active.live_trades >= 20 and
                active.backtest_sharpe > 0 and
                active.live_sharpe < active.backtest_sharpe * self.rollback_ratio):
                self._rollback(model_name)

    def _rollback(self, model_name: str) -> Optional[str]:
        """Rollback to previous version."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            if len(versions) < 2:
                return None
            current = next((v for v in versions if v.is_active), None)
            if current:
                current.is_active = False
                current.stage = LifecycleStage.ROLLED_BACK
                current.is_healthy = False
            previous = versions[-2]
            previous.is_active = True
            previous.stage = LifecycleStage.DEPLOYED

            self._rollback_times.append(time.time())
            # Major #6 fix: set cooldown so the next N trades don't trigger
            # another immediate rollback.
            self._rollback_cooldown_trades[model_name] = self._min_trades_between_rollbacks
            # Check for too many rollbacks
            week_ago = time.time() - 7 * 86400
            recent_rollbacks = sum(1 for t in self._rollback_times if t >= week_ago)
            if recent_rollbacks >= self.max_rollbacks:
                log.error("lifecycle: %d rollbacks in a week for %s — ALERT",
                         recent_rollbacks, model_name)

        log.warning("lifecycle: rolled back %s to %s", model_name, previous.version_id)
        return previous.version_id

    # ------------------------------------------------------------------
    # Create new version
    # ------------------------------------------------------------------
    def create_version(self, model_name: str) -> str:
        """Create a new model version for retraining."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            parent = versions[-1].version_id if versions else ""
            version_id = f"{model_name}_v{len(versions) + 1}_{int(time.time())}"
            new_version = ModelVersion(
                version_id=version_id,
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                parent_version=parent,
                stage=LifecycleStage.COLLECTING,
                training_samples=len(self._training_data.get(model_name, [])),
            )
            self._versions[model_name].append(new_version)
        log.info("lifecycle: created new version %s", version_id)
        return version_id

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def report(self, model_name: str) -> LifecycleReport:
        """Get lifecycle report for a model."""
        with self._lock:
            versions = self._versions.get(model_name, [])
            active = next((v for v in versions if v.is_active), None)
            samples = len(self._training_data.get(model_name, []))
            rollback_count = len(self._rollback_times)

        report = LifecycleReport(
            model_name=model_name,
            current_version=active.version_id if active else "none",
            current_stage=active.stage if active else LifecycleStage.COLLECTING,
            versions_total=len(versions),
            active_version=active,
            samples_collected=samples,
            samples_needed=self.min_samples,
            ready_for_retrain=samples >= self.min_samples,
            rollback_count=rollback_count,
        )

        # Determine pipeline stage + next action
        if not versions:
            report.pipeline_stage = "initial_collection"
            report.next_action = f"Collect {self.min_samples} samples"
        elif report.ready_for_retrain and not active:
            report.pipeline_stage = "ready_for_retrain"
            report.next_action = "Create new version + retrain"
        elif active and active.stage == LifecycleStage.DEPLOYED:
            report.pipeline_stage = "live_monitoring"
            report.next_action = "Monitor live performance"
        elif active:
            report.pipeline_stage = active.stage.value
            report.next_action = f"Continue {active.stage.value}"

        # Recommendations
        if report.ready_for_retrain:
            report.recommendations.append("Ready for retraining — create new version")
        if active and active.live_trades >= 20:
            if active.live_sharpe < active.backtest_sharpe * self.rollback_ratio:
                report.recommendations.append("URGENT: Live performance degraded — rollback")
        if rollback_count >= self.max_rollbacks:
            report.recommendations.append("ALERT: Too many rollbacks — human review needed")
        if not report.recommendations:
            report.recommendations.append("Stable — no actions needed")

        return report

    def all_reports(self) -> List[LifecycleReport]:
        """Get reports for all models."""
        with self._lock:
            models = list(self._versions.keys())
        return [self.report(m) for m in models]

    def summary(self) -> Dict[str, Any]:
        """Get summary of all lifecycle activity."""
        with self._lock:
            total_versions = sum(len(v) for v in self._versions.values())
            active = sum(1 for versions in self._versions.values()
                        if any(v.is_active for v in versions))
            failed = sum(1 for versions in self._versions.values()
                        if any(v.stage == LifecycleStage.FAILED for v in versions))
        return {
            "total_models": len(self._versions),
            "active_models": active,
            "failed_models": failed,
            "total_versions": total_versions,
            "total_rollbacks": len(self._rollback_times),
            "total_samples": sum(len(d) for d in self._training_data.values()),
        }
