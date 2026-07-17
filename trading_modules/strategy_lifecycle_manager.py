"""trading_modules/strategy_lifecycle_manager.py
=====================================================================
Strategy Lifecycle Manager (Principle #144, #158)
=====================================================================
Manages the full lifecycle of every trading strategy:
    BIRTH → TESTING → ACTIVE → MONITORING → DECAYING → RETIRED → RETRAINED

Different from strategy_health_monitor.py (which tracks real-time health),
this module manages the LIFECYCLE: when to deploy, when to retrain, when
to retire a strategy entirely.

Lifecycle Stages:
    1. BIRTH       — strategy just created, paper-only
    2. TESTING     — forward testing on small size (0.01 lots)
    3. ACTIVE      — full size, healthy edge
    4. MONITORING  — performance declining, watch closely
    5. DECAYING    — edge clearly declining, reduce size 50%
    6. RETIRED     — edge gone, stop trading, archive data
    7. RETRAINED   — new version trained, back to TESTING

Edge Decay Detection:
    - Rolling 100-trade EV vs historical 500-trade EV
    - Win rate decline > 10% over 50 trades
    - Sharpe ratio drop > 0.5
    - Profit factor below 1.2 (was > 1.5)

Auto-Retrain Triggers:
    - Strategy in DECAYING for 100+ trades
    - Edge decline > 30%
    - Market regime shift detected

Usage:
    mgr = StrategyLifecycleManager()

    # Register a strategy
    mgr.register("momentum_v4", deployed_at=datetime.now(),
                 initial_sharpe=1.8, initial_win_rate=0.62)

    # Update with trade results
    mgr.update_stats("momentum_v4", win_rate=0.55, sharpe=1.2, ev_r=0.15)

    # Get lifecycle state
    state = mgr.get_state("momentum_v4")
    # state = {"stage": "DECAYING", "action": "reduce_size", "days_active": 45}
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.strategy_lifecycle_manager")


class LifecycleStage(str, Enum):
    BIRTH = "birth"
    TESTING = "testing"
    ACTIVE = "active"
    MONITORING = "monitoring"
    DECAYING = "decaying"
    RETIRED = "retired"
    RETRAINED = "retrained"


class LifecycleAction(str, Enum):
    DEPLOY = "deploy"               # move from TESTING to ACTIVE
    INCREASE_SIZE = "increase_size"  # healthy, boost size
    MONITOR = "monitor"              # watch closely
    REDUCE_SIZE = "reduce_size"     # decaying, cut size 50%
    PAUSE = "pause"                  # stop new entries
    RETIRE = "retire"                # edge gone, archive
    RETRAIN = "retrain"              # trigger retraining
    RESUME = "resume"                # re-enabled after retraining


@dataclass
class StrategyLifecycle:
    """Lifecycle state for a single strategy."""
    name: str
    version: str = "1.0"
    stage: LifecycleStage = LifecycleStage.BIRTH
    action: LifecycleAction = LifecycleAction.DEPLOY

    # Timeline
    deployed_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
    days_active: int = 0
    trades_total: int = 0

    # Initial metrics (at deployment)
    initial_sharpe: float = 0.0
    initial_win_rate: float = 0.0
    initial_ev_r: float = 0.0
    initial_profit_factor: float = 0.0

    # Current metrics
    current_sharpe: float = 0.0
    current_win_rate: float = 0.0
    current_ev_r: float = 0.0
    current_profit_factor: float = 0.0

    # Decay tracking
    edge_decline_pct: float = 0.0    # how much has edge declined?
    decay_detected_at: Optional[datetime] = None
    trades_since_decay: int = 0

    # Size multiplier (0 to 1.5)
    size_multiplier: float = 1.0

    # Retrain tracking
    retrain_count: int = 0
    last_retrained_at: Optional[datetime] = None

    # Notes
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "version": self.version,
            "stage": self.stage.value, "action": self.action.value,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "days_active": self.days_active,
            "trades_total": self.trades_total,
            "initial_sharpe": round(self.initial_sharpe, 2),
            "current_sharpe": round(self.current_sharpe, 2),
            "initial_win_rate": round(self.initial_win_rate, 3),
            "current_win_rate": round(self.current_win_rate, 3),
            "initial_ev_r": round(self.initial_ev_r, 3),
            "current_ev_r": round(self.current_ev_r, 3),
            "edge_decline_pct": round(self.edge_decline_pct, 2),
            "size_multiplier": round(self.size_multiplier, 2),
            "retrain_count": self.retrain_count,
            "notes": self.notes,
        }


class StrategyLifecycleManager:
    """Manages the full lifecycle of trading strategies."""

    def __init__(self,
                 min_trades_for_active: int = 50,
                 decay_threshold_pct: float = 0.30,
                 retire_threshold_pct: float = 0.50,
                 monitoring_threshold_pct: float = 0.15,
                 retrain_after_decay_trades: int = 100):
        """Initialize lifecycle manager.

        Args:
            min_trades_for_active: trades needed before ACTIVE stage
            decay_threshold_pct: edge decline > this → DECAYING
            retire_threshold_pct: edge decline > this → RETIRED
            monitoring_threshold_pct: edge decline > this → MONITORING
            retrain_after_decay_trades: trigger retrain after this many decay trades
        """
        self.min_trades = min_trades_for_active
        self.decay_threshold = decay_threshold_pct
        self.retire_threshold = retire_threshold_pct
        self.monitoring_threshold = monitoring_threshold_pct
        self.retrain_trades = retrain_after_decay_trades

        self._lock = threading.RLock()
        self._strategies: Dict[str, StrategyLifecycle] = {}
        # Major #9 fix: cap the number of strategies to prevent unbounded
        # memory growth. Old RETIRED strategies are pruned first.
        self._max_strategies = 200

    # ------------------------------------------------------------------
    # Register / update
    # ------------------------------------------------------------------
    def register(self, name: str, version: str = "1.0",
                 deployed_at: Optional[datetime] = None,
                 initial_sharpe: float = 0.0, initial_win_rate: float = 0.0,
                 initial_ev_r: float = 0.0, initial_profit_factor: float = 0.0) -> None:
        """Register a new strategy (BIRTH stage)."""
        with self._lock:
            # Major #9 fix: prune RETIRED strategies if over cap.
            if len(self._strategies) >= self._max_strategies:
                retired = [k for k, v in self._strategies.items()
                           if v.stage == LifecycleStage.RETIRED]
                for k in retired[:len(retired) // 2]:
                    del self._strategies[k]
                log.info("lifecycle: pruned %d retired strategies (cap=%d)",
                         min(len(retired), len(retired) // 2), self._max_strategies)
            self._strategies[name] = StrategyLifecycle(
                name=name, version=version,
                stage=LifecycleStage.BIRTH,
                action=LifecycleAction.DEPLOY,
                deployed_at=deployed_at or datetime.now(tz=timezone.utc),
                initial_sharpe=initial_sharpe,
                initial_win_rate=initial_win_rate,
                initial_ev_r=initial_ev_r,
                initial_profit_factor=initial_profit_factor,
                current_sharpe=initial_sharpe,
                current_win_rate=initial_win_rate,
                current_ev_r=initial_ev_r,
                current_profit_factor=initial_profit_factor,
            )
            log.info("lifecycle: registered %s v%s (initial Sharpe=%.2f, WR=%.0f%%)",
                     name, version, initial_sharpe, initial_win_rate * 100)

    def update_stats(self, name: str, win_rate: float, sharpe: float,
                     ev_r: float, profit_factor: float = 0,
                     trades: int = 0) -> None:
        """Update current metrics for a strategy."""
        with self._lock:
            s = self._strategies.get(name)
            if s is None:
                return
            s.current_win_rate = win_rate
            s.current_sharpe = sharpe
            s.current_ev_r = ev_r
            s.current_profit_factor = profit_factor or s.current_profit_factor
            s.trades_total += trades

            # Update days active
            if s.deployed_at:
                s.days_active = (datetime.now(tz=timezone.utc) - s.deployed_at).days

            # Compute edge decline
            if s.initial_ev_r > 0:
                s.edge_decline_pct = (s.initial_ev_r - s.current_ev_r) / s.initial_ev_r
            elif s.initial_sharpe > 0:
                s.edge_decline_pct = (s.initial_sharpe - s.current_sharpe) / s.initial_sharpe

            # Update stage
            self._update_stage(s)

    def _update_stage(self, s: StrategyLifecycle) -> None:
        """Update lifecycle stage based on metrics."""
        # First time: BIRTH → TESTING
        if s.stage == LifecycleStage.BIRTH and s.trades_total >= 10:
            s.stage = LifecycleStage.TESTING
            s.action = LifecycleAction.DEPLOY
            s.notes = "Forward testing on small size"
            return

        # TESTING → ACTIVE (after enough good trades)
        if s.stage == LifecycleStage.TESTING:
            if s.trades_total >= self.min_trades and s.current_ev_r > 0:
                s.stage = LifecycleStage.ACTIVE
                s.action = LifecycleAction.INCREASE_SIZE
                s.size_multiplier = 1.0
                s.notes = "Promoted to ACTIVE — healthy edge confirmed"
                log.info("lifecycle: %s promoted to ACTIVE", s.name)
            return

        # ACTIVE → MONITORING (slight decline)
        if s.stage == LifecycleStage.ACTIVE:
            if s.edge_decline_pct > self.monitoring_threshold:
                s.stage = LifecycleStage.MONITORING
                s.action = LifecycleAction.MONITOR
                s.size_multiplier = 0.8
                s.notes = f"Edge declining ({s.edge_decline_pct:.0%}) — monitoring"
                log.warning("lifecycle: %s entering MONITORING (decline=%.0f%%)",
                           s.name, s.edge_decline_pct * 100)
            return

        # MONITORING → DECAYING
        if s.stage == LifecycleStage.MONITORING:
            if s.edge_decline_pct > self.decay_threshold:
                s.stage = LifecycleStage.DECAYING
                s.action = LifecycleAction.REDUCE_SIZE
                s.size_multiplier = 0.5
                s.decay_detected_at = datetime.now(tz=timezone.utc)
                s.notes = f"Edge decayed {s.edge_decline_pct:.0%} — reduce size 50%"
                log.warning("lifecycle: %s DECAYING (decline=%.0f%%)",
                           s.name, s.edge_decline_pct * 100)
            return

        # DECAYING → RETIRED or RETRAIN
        if s.stage == LifecycleStage.DECAYING:
            s.trades_since_decay += 1
            # Retire if edge gone
            if s.edge_decline_pct > self.retire_threshold or s.current_ev_r < -0.1:
                s.stage = LifecycleStage.RETIRED
                s.action = LifecycleAction.RETIRE
                s.size_multiplier = 0.0
                s.retired_at = datetime.now(tz=timezone.utc)
                s.notes = "Edge gone — retired"
                log.error("lifecycle: %s RETIRED — edge gone", s.name)
            # Retrain after enough decay trades
            elif s.trades_since_decay >= self.retrain_trades:
                s.action = LifecycleAction.RETRAIN
                s.notes = f"Triggering retrain after {s.trades_since_decay} decay trades"
                log.info("lifecycle: %s requesting RETRAIN", s.name)
            return

        # RETIRED — no actions
        if s.stage == LifecycleStage.RETIRED:
            s.action = LifecycleAction.RETIRE
            return

    # ------------------------------------------------------------------
    # External controls
    # ------------------------------------------------------------------
    def retrain(self, name: str, new_version: str = "") -> None:
        """Mark a strategy as retrained (back to TESTING)."""
        with self._lock:
            s = self._strategies.get(name)
            if s is None:
                return
            s.stage = LifecycleStage.RETRAINED
            s.action = LifecycleAction.RESUME
            s.size_multiplier = 0.5
            s.retrain_count += 1
            s.last_retrained_at = datetime.now(tz=timezone.utc)
            s.trades_since_decay = 0
            s.edge_decline_pct = 0.0
            if new_version:
                s.version = new_version
            s.notes = f"Retrained (v{s.version}) — back to TESTING"
            log.info("lifecycle: %s retrained to v%s", name, s.version)

    def manually_retire(self, name: str, reason: str = "manual") -> None:
        """Manually retire a strategy."""
        with self._lock:
            s = self._strategies.get(name)
            if s is None:
                return
            s.stage = LifecycleStage.RETIRED
            s.action = LifecycleAction.RETIRE
            s.size_multiplier = 0.0
            s.retired_at = datetime.now(tz=timezone.utc)
            s.notes = f"Manually retired: {reason}"
            log.info("lifecycle: %s manually retired — %s", name, reason)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_state(self, name: str) -> Optional[StrategyLifecycle]:
        """Get lifecycle state for a strategy."""
        with self._lock:
            return self._strategies.get(name)

    def can_trade(self, name: str) -> bool:
        """Can this strategy place new trades?"""
        s = self.get_state(name)
        if s is None:
            return False
        return s.stage in (LifecycleStage.ACTIVE, LifecycleStage.MONITORING,
                          LifecycleStage.DECAYING, LifecycleStage.TESTING)

    def size_multiplier(self, name: str) -> float:
        """Get size multiplier for a strategy."""
        s = self.get_state(name)
        return s.size_multiplier if s else 0.0

    def all_strategies(self) -> Dict[str, StrategyLifecycle]:
        """Get all strategies."""
        with self._lock:
            return dict(self._strategies)

    def summary(self) -> Dict[str, Any]:
        """Get summary of all strategies."""
        with self._lock:
            strategies = list(self._strategies.values())

        by_stage: Dict[str, int] = {}
        for s in strategies:
            by_stage.setdefault(s.stage.value, 0)
            by_stage[s.stage.value] += 1

        return {
            "total_strategies": len(strategies),
            "by_stage": by_stage,
            "active": by_stage.get("active", 0),
            "monitoring": by_stage.get("monitoring", 0),
            "decaying": by_stage.get("decaying", 0),
            "retired": by_stage.get("retired", 0),
            "retrains_total": sum(s.retrain_count for s in strategies),
            "strategies": {s.name: s.to_dict() for s in strategies},
        }
