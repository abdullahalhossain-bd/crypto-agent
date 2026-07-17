"""monitoring.alert_system
=====================================================================
Day 75-77 — Smart Alert System.

Rules:
  - NO ALERT SPAM. Each alert type has a min-interval between firings.
  - Severity tiers: INFO / WARN / CRITICAL
  - Routing: console + log file + (optional) webhook
  - Deduplication: identical alerts within the cooldown window are
    suppressed
  - Escalation: if a WARN condition persists for N cycles, it escalates
    to CRITICAL

Operators can mute specific alert types temporarily without disabling
the underlying monitor.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("monitoring.alerts")


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    alert_id: str
    category: str          # system | trading | risk | alpha
    severity: AlertSeverity
    title: str
    message: str
    ts: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "category": self.category,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "ts": self.ts,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
class AlertSystem:
    def __init__(
        self,
        log_path: str = "data/alerts.jsonl",
        default_cooldown_s: float = 300.0,
        critical_cooldown_s: float = 60.0,
        escalation_cycles: int = 5,
        webhook_url: Optional[str] = None,
    ) -> None:
        self.log_path = log_path
        self.default_cooldown_s = float(default_cooldown_s)
        self.critical_cooldown_s = float(critical_cooldown_s)
        self.escalation_cycles = int(escalation_cycles)
        self.webhook_url = webhook_url
        self._lock = threading.Lock()
        # Minor #7 fix: dedicated lock for file writes — the main _lock is
        # released before _emit() is called, so concurrent fire() calls can
        # interleave file writes and corrupt the JSONL log.
        self._file_lock = threading.Lock()
        self._last_fired: dict[str, float] = {}      # alert_id -> ts
        self._persistence: dict[str, int] = {}         # alert_id -> consecutive cycles
        self._muted: set[str] = set()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # ----------------------------------------------------------------
    def fire(self, alert: Alert) -> bool:
        """Fire an alert. Returns True if it was actually emitted (not
        suppressed by cooldown or mute)."""
        with self._lock:
            if alert.alert_id in self._muted:
                return False
            now = time.time()
            cooldown = (self.critical_cooldown_s
                        if alert.severity == AlertSeverity.CRITICAL
                        else self.default_cooldown_s)
            last = self._last_fired.get(alert.alert_id, 0.0)
            if now - last < cooldown:
                # Suppressed by cooldown — but track persistence for escalation
                self._persistence[alert.alert_id] = (
                    self._persistence.get(alert.alert_id, 0) + 1
                )
                # Escalate if persisted — but only emit once per escalation,
                # then suppress until cooldown expires.
                if (alert.severity == AlertSeverity.WARN
                        and self._persistence[alert.alert_id] >= self.escalation_cycles):
                    # Escalate to CRITICAL but apply CRITICAL cooldown
                    alert.severity = AlertSeverity.CRITICAL
                    critical_last = self._last_fired.get(alert.alert_id + "_critical", 0.0)
                    if now - critical_last < self.critical_cooldown_s:
                        return False  # already escalated recently — suppress
                    self._last_fired[alert.alert_id + "_critical"] = now
                    # Fall through to emit
                else:
                    return False
            else:
                self._persistence[alert.alert_id] = 1
            self._last_fired[alert.alert_id] = now
            alert.ts = now

        # Emit
        self._emit(alert)
        return True

    # ----------------------------------------------------------------
    def _emit(self, alert: Alert) -> None:
        # Console
        if alert.severity == AlertSeverity.CRITICAL:
            log.error("ALERT [%s] %s: %s", alert.severity.value, alert.title, alert.message)
        elif alert.severity == AlertSeverity.WARN:
            log.warning("ALERT [%s] %s: %s", alert.severity.value, alert.title, alert.message)
        else:
            log.info("ALERT [%s] %s: %s", alert.severity.value, alert.title, alert.message)
        # File
        try:
            # Minor #7 fix: acquire _file_lock to prevent concurrent writes
            # from interleaving and corrupting the JSONL log.
            with self._file_lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(alert.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("alert log write failed: %r", e)
        # Webhook (best-effort)
        if self.webhook_url:
            try:
                import urllib.request
                req = urllib.request.Request(
                    self.webhook_url,
                    data=json.dumps(alert.to_dict()).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=2.0)
            except Exception as e:  # noqa: BLE001
                log.debug("webhook post failed: %r", e)

    # ----------------------------------------------------------------
    def mute(self, alert_id: str) -> None:
        self._muted.add(alert_id)
        log.info("Alert %s muted", alert_id)

    def unmute(self, alert_id: str) -> None:
        self._muted.discard(alert_id)
        log.info("Alert %s unmuted", alert_id)

    def clear_persistence(self, alert_id: str) -> None:
        self._persistence.pop(alert_id, None)

    # ----------------------------------------------------------------
    def evaluate_health_alerts(
        self,
        system_health: Optional[dict[str, Any]] = None,
        trading_health: Optional[dict[str, Any]] = None,
        risk_health: Optional[dict[str, Any]] = None,
        alpha_health: Optional[dict[str, Any]] = None,
    ) -> list[Alert]:
        """Convert health-dict issues into structured alerts."""
        alerts: list[Alert] = []
        for source, health, category in [
            ("system", system_health, "system"),
            ("trading", trading_health, "trading"),
            ("risk", risk_health, "risk"),
            ("alpha", alpha_health, "alpha"),
        ]:
            if not health:
                continue
            for issue in health.get("issues", []):
                sev = (AlertSeverity.CRITICAL
                       if health.get("status") == "critical"
                       else AlertSeverity.WARN)
                alerts.append(Alert(
                    alert_id=f"{source}:{issue[:50]}",
                    category=category,
                    severity=sev,
                    title=f"{source} health issue",
                    message=issue,
                    metadata={"source_status": health.get("status")},
                ))
        # Fire them all (cooldown applies)
        fired = []
        for a in alerts:
            if self.fire(a):
                fired.append(a)
        return fired
