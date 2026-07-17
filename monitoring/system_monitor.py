"""monitoring.system_monitor
=====================================================================
Day 71 — System-layer monitor.

Tracks the health of the *infrastructure*, not the trading:
  - Cycle latency (p50, p95, p99)
  - Uptime (cycles succeeded vs. crashed)
  - MT5 connection stability
  - Memory + file handle counts
  - Watchdog state (kill switch, error budget)
"""
from __future__ import annotations

import os
try:
    import resource
except ImportError:
    resource = None  # Windows doesn't have resource module
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from utils.logger import get_logger

log = get_logger("monitoring.system")


@dataclass
class SystemHealth:
    status: str                  # "ok" | "degraded" | "critical"
    uptime_pct: float
    cycle_p50_ms: float
    cycle_p95_ms: float
    cycle_p99_ms: float
    mt5_connected: bool
    kill_switch_armed: bool
    error_budget_remaining: int
    memory_rss_mb: float
    open_file_descriptors: int
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "uptime_pct": self.uptime_pct,
            "cycle_p50_ms": self.cycle_p50_ms,
            "cycle_p95_ms": self.cycle_p95_ms,
            "cycle_p99_ms": self.cycle_p99_ms,
            "mt5_connected": self.mt5_connected,
            "kill_switch_armed": self.kill_switch_armed,
            "error_budget_remaining": self.error_budget_remaining,
            "memory_rss_mb": self.memory_rss_mb,
            "open_file_descriptors": self.open_file_descriptors,
            "issues": list(self.issues),
        }


# ----------------------------------------------------------------------
class SystemMonitor:
    def __init__(self, latency_window: int = 500,
                 uptime_window: int = 1000,
                 p95_threshold_ms: float = 2000.0,
                 p99_threshold_ms: float = 5000.0) -> None:
        self.latency_window = latency_window
        self.uptime_window = uptime_window
        self.p95_threshold_ms = float(p95_threshold_ms)
        self.p99_threshold_ms = float(p99_threshold_ms)
        self._latencies: deque[float] = deque(maxlen=latency_window)
        self._cycle_ok: deque[bool] = deque(maxlen=uptime_window)
        self._mt5_connected: bool = False
        self._kill_switch_armed: bool = False
        self._error_budget_remaining: int = 10

    # ----------------------------------------------------------------
    def record_cycle(self, latency_s: float, ok: bool) -> None:
        self._latencies.append(latency_s * 1000.0)  # ms
        self._cycle_ok.append(ok)

    def set_mt5_status(self, connected: bool) -> None:
        self._mt5_connected = bool(connected)

    def set_kill_switch(self, armed: bool) -> None:
        self._kill_switch_armed = bool(armed)

    def set_error_budget(self, remaining: int) -> None:
        self._error_budget_remaining = int(remaining)

    # ----------------------------------------------------------------
    def health(self) -> SystemHealth:
        import numpy as np
        issues: list[str] = []
        if not self._latencies:
            return SystemHealth(
                status="ok", uptime_pct=100.0, cycle_p50_ms=0.0,
                cycle_p95_ms=0.0, cycle_p99_ms=0.0,
                mt5_connected=self._mt5_connected,
                kill_switch_armed=self._kill_switch_armed,
                error_budget_remaining=self._error_budget_remaining,
                memory_rss_mb=self._mem_rss_mb(),
                open_file_descriptors=self._fd_count(),
                issues=["no cycle data yet"],
            )
        arr = np.array(self._latencies)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        uptime = (sum(self._cycle_ok) / len(self._cycle_ok) * 100.0
                  if self._cycle_ok else 100.0)
        status = "ok"
        if self._kill_switch_armed:
            status = "critical"
            issues.append("kill switch armed")
        if uptime < 95:
            status = "degraded" if status != "critical" else status
            issues.append(f"uptime {uptime:.1f}% < 95%")
        if p95 > self.p95_threshold_ms:
            status = "degraded" if status != "critical" else status
            issues.append(f"p95 latency {p95:.0f}ms > {self.p95_threshold_ms:.0f}ms")
        if p99 > self.p99_threshold_ms:
            status = "critical"
            issues.append(f"p99 latency {p99:.0f}ms > {self.p99_threshold_ms:.0f}ms")
        if self._error_budget_remaining <= 2:
            status = "critical"
            issues.append(f"error budget low ({self._error_budget_remaining})")
        return SystemHealth(
            status=status, uptime_pct=float(uptime),
            cycle_p50_ms=p50, cycle_p95_ms=p95, cycle_p99_ms=p99,
            mt5_connected=self._mt5_connected,
            kill_switch_armed=self._kill_switch_armed,
            error_budget_remaining=self._error_budget_remaining,
            memory_rss_mb=self._mem_rss_mb(),
            open_file_descriptors=self._fd_count(),
            issues=issues,
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _mem_rss_mb() -> float:
        try:
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _fd_count() -> int:
        try:
            return len(os.listdir("/proc/self/fd"))
        except Exception:  # noqa: BLE001
            return 0
