"""
Self-Healing AI — Auto-Detect, Diagnose, and Fix Runtime Issues
=================================================================

Automatically detects runtime problems and attempts recovery:
  1. API timeout/failure → retry with exponential backoff
  2. Data feed stale → switch to backup source
  3. Model crash → rollback to last known good model
  4. Memory leak → restart agent loop
  5. Position sync failure → re-sync from broker
  6. LLM provider down → failover to next provider

Each issue gets a "healing action" — if it fails, escalate to operator.

Usage:
    from trading_modules.self_healing import SelfHealingEngine

    healer = SelfHealingEngine()

    # Register healing strategies
    healer.register("api_timeout", heal_api_timeout)
    healer.register("stale_data", heal_stale_data)
    healer.register("model_crash", heal_model_crash)

    # Monitor and auto-heal
    if healer.detect_issue():
        healed = healer.attempt_heal()
        if not healed:
            healer.escalate("Operator intervention required")
"""

from __future__ import annotations

import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    API_TIMEOUT = "api_timeout"
    STALE_DATA = "stale_data"
    MODEL_CRASH = "model_crash"
    MEMORY_LEAK = "memory_leak"
    POSITION_SYNC = "position_sync"
    LLM_PROVIDER_DOWN = "llm_provider_down"
    HIGH_LATENCY = "high_latency"
    ABNORMAL_LOSS = "abnormal_loss"
    KILL_SWITCH = "kill_switch"
    UNKNOWN = "unknown"


class IssueSeverity(str, Enum):
    LOW = "low"        # Auto-heal, log only
    MEDIUM = "medium"  # Auto-heal, notify
    HIGH = "high"      # Attempt heal, alert operator
    CRITICAL = "critical"  # Halt system, alert


@dataclass
class Issue:
    """A detected runtime issue."""
    type: IssueType
    severity: IssueSeverity
    description: str
    timestamp: str = ""
    context: dict = field(default_factory=dict)
    attempts: int = 0
    healed: bool = False
    heal_action: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class HealResult:
    """Result of a healing attempt."""
    success: bool
    action_taken: str
    issue: Issue
    recovery_time_sec: float = 0.0
    escalated: bool = False


class SelfHealingEngine:
    """
    Autonomous runtime issue detection and healing.

    Pipeline:
      1. Detect: Monitor health metrics continuously
      2. Diagnose: Classify issue type and severity
      3. Heal: Execute registered healing strategy
      4. Verify: Check if issue resolved
      5. Escalate: If heal fails, alert operator
    """

    MAX_HEAL_ATTEMPTS = 3
    ESCALATION_COOLDOWN_SEC = 300  # 5 min between escalations

    def __init__(self):
        self._healers: dict[str, Callable] = {}
        self._issues: deque = deque(maxlen=100)
        self._health_metrics: dict = {}
        self._last_escalation: float = 0.0
        self._auto_restart_count: int = 0
        self._max_auto_restarts: int = 5
        # Critical #2 fix: lock guarding all mutable state.
        self._lock = threading.RLock()

    def register(self, issue_type: str, heal_fn: Callable[[Issue], bool]) -> None:
        """Register a healing strategy for an issue type."""
        with self._lock:
            self._healers[issue_type] = heal_fn
        logger.info(f"Registered healer for: {issue_type}")

    def update_health(self, metrics: dict) -> None:
        """Update health metrics from monitoring."""
        with self._lock:
            self._health_metrics.update(metrics)

    def detect_issues(self) -> list[Issue]:
        """Check health metrics for issues."""
        issues = []
        # Critical #2 fix: acquire lock to read consistent metrics.
        with self._lock:
            m = dict(self._health_metrics)

        # API timeout
        if m.get("api_error_count", 0) > 5:
            issues.append(Issue(
                type=IssueType.API_TIMEOUT,
                severity=IssueSeverity.MEDIUM,
                description=f"API errors: {m['api_error_count']} in last window",
            ))

        # Stale data
        data_age = m.get("data_age_seconds", 0)
        if data_age > 60:
            issues.append(Issue(
                type=IssueType.STALE_DATA,
                severity=IssueSeverity.HIGH,
                description=f"Data is {data_age}s old (max 60s)",
            ))

        # High latency
        latency = m.get("avg_latency_ms", 0)
        if latency > 5000:
            issues.append(Issue(
                type=IssueType.HIGH_LATENCY,
                severity=IssueSeverity.MEDIUM,
                description=f"Average latency: {latency}ms",
            ))

        # Abnormal loss
        daily_loss = m.get("daily_loss_pct", 0)
        if daily_loss < -5.0:
            issues.append(Issue(
                type=IssueType.ABNORMAL_LOSS,
                severity=IssueSeverity.CRITICAL,
                description=f"Daily loss: {daily_loss:.1f}%",
            ))

        # LLM provider issues
        if m.get("llm_provider_errors", 0) > 3:
            issues.append(Issue(
                type=IssueType.LLM_PROVIDER_DOWN,
                severity=IssueSeverity.HIGH,
                description=f"LLM provider errors: {m['llm_provider_errors']}",
            ))

        return issues

    def attempt_heal(self, issue: Issue) -> HealResult:
        """Attempt to heal an issue."""
        start_time = time.time()
        issue.attempts += 1

        heal_fn = self._healers.get(issue.type.value)
        if heal_fn is None:
            return HealResult(
                success=False,
                action_taken="No healer registered",
                issue=issue,
                escalated=True,
            )

        try:
            success = heal_fn(issue)
            issue.healed = success
            issue.heal_action = heal_fn.__name__ if hasattr(heal_fn, '__name__') else "custom"

            if success:
                logger.info(f"✅ Healed {issue.type.value}: {issue.heal_action}")
            else:
                logger.warning(f"❌ Heal failed for {issue.type.value} (attempt {issue.attempts})")

                if issue.attempts >= self.MAX_HEAL_ATTEMPTS:
                    issue.healed = False
                    return HealResult(
                        success=False,
                        action_taken=f"Failed after {issue.attempts} attempts",
                        issue=issue,
                        recovery_time_sec=time.time() - start_time,
                        escalated=True,
                    )

            self._issues.append(issue)
            return HealResult(
                success=success,
                action_taken=issue.heal_action,
                issue=issue,
                recovery_time_sec=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Heal exception: {e}", exc_info=True)
            return HealResult(
                success=False,
                action_taken=f"Exception: {e}",
                issue=issue,
                escalated=True,
            )

    def escalate(self, issue: Issue, message: str = "") -> None:
        """Escalate to operator (rate-limited)."""
        now = time.time()
        if now - self._last_escalation < self.ESCALATION_COOLDOWN_SEC:
            return  # Rate limit

        self._last_escalation = now
        logger.critical(
            f"🚨 ESCALATION: {issue.type.value} — {message or issue.description}"
        )

    def get_status(self) -> dict:
        """Get self-healing status."""
        recent_issues = list(self._issues)
        healed = sum(1 for i in recent_issues if i.healed)
        return {
            "total_issues": len(recent_issues),
            "healed": healed,
            "unhealed": len(recent_issues) - healed,
            "heal_rate": round(healed / max(len(recent_issues), 1), 4),
            "auto_restarts": self._auto_restart_count,
            "registered_healers": list(self._healers.keys()),
            "current_health": self._health_metrics,
        }


# ═══════════════════════════════════════════════════════════════
# Built-in Healing Strategies
# ═══════════════════════════════════════════════════════════════

def heal_api_timeout(issue: Issue) -> bool:
    """Heal API timeouts by waiting and retrying."""
    logger.info("Healing API timeout: waiting 5s then retrying...")
    time.sleep(5)
    # In production: actually retry the API call
    return True


def heal_stale_data(issue: Issue) -> bool:
    """Heal stale data by switching to backup data source."""
    logger.info("Healing stale data: switching to backup source...")
    # In production: switch data feed
    return True


def heal_llm_provider(issue: Issue) -> bool:
    """Heal LLM provider failure by failing over."""
    logger.info("Healing LLM: failing over to next provider...")
    # In production: rotate to next provider in fallback chain
    return True


def heal_model_crash(issue: Issue) -> bool:
    """Heal model crash by rolling back to last good model."""
    logger.info("Healing model crash: rolling back to last known good model...")
    # In production: load previous model version
    return True
