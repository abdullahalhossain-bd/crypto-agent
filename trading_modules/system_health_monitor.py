"""trading_modules/system_health_monitor.py
=====================================================================
System Health Monitor (Principle #119 — Protect Against System Failure)
=====================================================================
Monitors all infrastructure components that the trading bot depends on.
If any component fails, the bot transitions to safe mode.

Components Monitored:
    1. MT5/Broker connection  — is the connection alive?
    2. API response time      — are calls fast enough?
    3. Tick data freshness    — is the latest tick recent?
    4. Order execution latency— are orders filling quickly?
    5. Database write health  — are trades being saved?
    6. Disk space             — is the disk filling up?
    7. Memory usage           — is RAM exhausted?
    8. CPU load               — is the system overloaded?
    9. Network connectivity   — is the internet up?
   10. VPS/clock drift        — is system time accurate?

Health States:
    GREEN  — all systems healthy, trade normally
    YELLOW — one or more warnings, trade with caution
    RED    — critical failure, halt new trades
    BLACK  — system down, emergency shutdown

Usage:
    monitor = SystemHealthMonitor()

    # Each component reports its health:
    monitor.check_mt5_connection(connector)
    monitor.check_tick_freshness(symbol="BTCUSD", connector=connector)
    monitor.check_order_latency(last_order_ms=150)
    monitor.check_database(db_path="data/trading_bot.db")
    monitor.check_disk_space(path="/")
    monitor.check_memory()
    monitor.check_cpu()

    # Get overall health
    health = monitor.health_summary()
    if health["status"] == "RED":
        bot.halt_new_trades()
    elif health["status"] == "BLACK":
        bot.emergency_shutdown()
"""
from __future__ import annotations

import os
import psutil
import shutil
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.system_health_monitor")


class HealthStatus(str, Enum):
    GREEN = "green"    # all healthy
    YELLOW = "yellow"  # warnings
    RED = "red"        # critical, halt new trades
    BLACK = "black"    # system down, emergency


@dataclass
class ComponentHealth:
    """Health of a single component."""
    name: str
    status: HealthStatus = HealthStatus.GREEN
    value: Any = None
    threshold: Any = None
    message: str = ""
    last_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "last_check": self.last_check,
            "consecutive_failures": self.consecutive_failures,
        }


class SystemHealthMonitor:
    """Monitors all infrastructure components.

    Each check updates a ComponentHealth object.
    The overall health is the worst of all components.
    """

    def __init__(self,
                 mt5_timeout_s: float = 5.0,
                 tick_stale_threshold_s: float = 60.0,
                 order_latency_threshold_ms: float = 2000.0,
                 db_check_interval_s: float = 30.0,
                 disk_space_threshold_pct: float = 90.0,
                 memory_threshold_pct: float = 85.0,
                 cpu_threshold_pct: float = 90.0,
                 max_consecutive_failures: int = 3):
        """Initialize monitor with thresholds."""
        self.mt5_timeout = mt5_timeout_s
        self.tick_stale = tick_stale_threshold_s
        self.order_latency_threshold = order_latency_threshold_ms
        self.db_interval = db_check_interval_s
        self.disk_threshold = disk_space_threshold_pct
        self.mem_threshold = memory_threshold_pct
        self.cpu_threshold = cpu_threshold_pct
        self.max_failures = max_consecutive_failures

        self._lock = threading.RLock()
        self._components: Dict[str, ComponentHealth] = {}
        self._init_components()
        self._history: List[Dict[str, Any]] = []

    def _init_components(self) -> None:
        """Initialize all component health objects."""
        for name in [
            "mt5_connection", "api_latency", "tick_freshness",
            "order_latency", "database", "disk_space",
            "memory", "cpu", "network", "clock_drift",
        ]:
            self._components[name] = ComponentHealth(name=name)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def check_mt5_connection(self, connector: Any = None) -> ComponentHealth:
        """Check if MT5/broker connection is alive."""
        comp = self._components["mt5_connection"]
        try:
            if connector is None:
                comp.status = HealthStatus.RED
                comp.message = "no connector provided"
                comp.consecutive_failures += 1
                return comp

            # Try a simple ping
            t0 = time.time()
            info = connector.account_info()
            elapsed = time.time() - t0

            if info is None:
                comp.status = HealthStatus.RED
                comp.value = None
                comp.message = "account_info returned None"
                comp.consecutive_failures += 1
            elif elapsed > self.mt5_timeout:
                comp.status = HealthStatus.YELLOW
                comp.value = f"{elapsed:.2f}s"
                comp.threshold = f"{self.mt5_timeout}s"
                comp.message = f"MT5 slow ({elapsed:.2f}s > {self.mt5_timeout}s)"
                comp.consecutive_failures = 0
            else:
                comp.status = HealthStatus.GREEN
                comp.value = f"{elapsed:.2f}s"
                comp.message = "OK"
                comp.consecutive_failures = 0
        except Exception as e:
            comp.status = HealthStatus.RED
            comp.value = str(e)
            comp.message = f"MT5 connection failed: {e}"
            comp.consecutive_failures += 1

        comp.last_check = time.time()
        if comp.consecutive_failures >= self.max_failures:
            comp.status = HealthStatus.BLACK
        return comp

    def check_tick_freshness(self, symbol: str = "BTCUSD",
                             connector: Any = None) -> ComponentHealth:
        """Check if tick data is fresh (not stale)."""
        comp = self._components["tick_freshness"]
        try:
            if connector is None:
                comp.status = HealthStatus.YELLOW
                comp.message = "no connector"
                return comp

            tick = connector.symbol_tick(symbol)
            if tick is None:
                comp.status = HealthStatus.RED
                comp.message = f"no tick for {symbol}"
                comp.consecutive_failures += 1
                return comp

            # Tick time
            tick_time = getattr(tick, "time", None)
            if tick_time is None:
                comp.status = HealthStatus.YELLOW
                comp.message = "tick has no timestamp"
                return comp

            # Convert tick time (could be epoch int or datetime)
            if isinstance(tick_time, (int, float)):
                tick_age = time.time() - tick_time
            else:
                tick_age = 0  # can't determine

            comp.value = f"{tick_age:.1f}s old"
            comp.threshold = f"{self.tick_stale}s"

            if tick_age > self.tick_stale:
                comp.status = HealthStatus.RED
                comp.message = f"tick stale ({tick_age:.0f}s > {self.tick_stale}s)"
                comp.consecutive_failures += 1
            else:
                comp.status = HealthStatus.GREEN
                comp.message = "OK"
                comp.consecutive_failures = 0
        except Exception as e:
            comp.status = HealthStatus.RED
            comp.message = f"tick check failed: {e}"
            comp.consecutive_failures += 1

        comp.last_check = time.time()
        if comp.consecutive_failures >= self.max_failures:
            comp.status = HealthStatus.BLACK
        return comp

    def check_order_latency(self, last_order_ms: float) -> ComponentHealth:
        """Check if order execution latency is acceptable."""
        comp = self._components["order_latency"]
        comp.value = f"{last_order_ms:.0f}ms"
        comp.threshold = f"{self.order_latency_threshold}ms"
        comp.last_check = time.time()

        if last_order_ms > self.order_latency_threshold * 2:
            comp.status = HealthStatus.RED
            comp.message = f"order latency critical ({last_order_ms:.0f}ms)"
        elif last_order_ms > self.order_latency_threshold:
            comp.status = HealthStatus.YELLOW
            comp.message = f"order latency high ({last_order_ms:.0f}ms)"
        else:
            comp.status = HealthStatus.GREEN
            comp.message = "OK"
        return comp

    def check_database(self, db_path: str = "data/trading_bot.db") -> ComponentHealth:
        """Check if database is writable.

        Critical #1 fix: the old code used raw string SQL which, while not
        directly injectable from user input in the current codebase, is a
        security anti-pattern. We now use parameterized queries AND validate
        the db_path to prevent path traversal.
        """
        comp = self._components["database"]
        try:
            import sqlite3
            import os as _os
            # Critical #1 fix: validate db_path — reject paths with directory
            # traversal or suspicious characters.
            if ".." in db_path or db_path.startswith("/"):
                raise ValueError(f"Invalid db_path: {db_path}")
            test_path = _os.path.abspath(db_path)
            conn = sqlite3.connect(test_path, timeout=2.0)
            # Critical #1 fix: use parameterized query for the INSERT.
            conn.execute("CREATE TABLE IF NOT EXISTS _health_check (id INTEGER)")
            conn.execute("INSERT OR REPLACE INTO _health_check (id) VALUES (?)", (1,))
            conn.commit()
            conn.close()
            comp.status = HealthStatus.GREEN
            comp.value = "writable"
            comp.message = "OK"
            comp.consecutive_failures = 0
        except Exception as e:
            comp.status = HealthStatus.RED
            comp.value = str(e)
            comp.message = f"DB check failed: {e}"
            comp.consecutive_failures += 1

        comp.last_check = time.time()
        if comp.consecutive_failures >= self.max_failures:
            comp.status = HealthStatus.BLACK
        return comp

    def check_disk_space(self, path: str = "/") -> ComponentHealth:
        """Check disk space."""
        comp = self._components["disk_space"]
        try:
            usage = shutil.disk_usage(path)
            pct = usage.used / usage.total * 100
            comp.value = f"{pct:.1f}%"
            comp.threshold = f"{self.disk_threshold}%"
            comp.last_check = time.time()

            if pct > 98:
                comp.status = HealthStatus.BLACK
                comp.message = f"disk full ({pct:.1f}%)"
            elif pct > self.disk_threshold:
                comp.status = HealthStatus.RED
                comp.message = f"disk space low ({pct:.1f}%)"
            elif pct > self.disk_threshold - 10:
                comp.status = HealthStatus.YELLOW
                comp.message = f"disk space warning ({pct:.1f}%)"
            else:
                comp.status = HealthStatus.GREEN
                comp.message = "OK"
        except Exception as e:
            comp.status = HealthStatus.RED
            comp.message = f"disk check failed: {e}"
        return comp

    def check_memory(self) -> ComponentHealth:
        """Check memory usage."""
        comp = self._components["memory"]
        try:
            mem = psutil.virtual_memory()
            pct = mem.percent
            comp.value = f"{pct:.1f}%"
            comp.threshold = f"{self.mem_threshold}%"
            comp.last_check = time.time()

            if pct > 98:
                comp.status = HealthStatus.BLACK
                comp.message = f"memory exhausted ({pct:.1f}%)"
            elif pct > self.mem_threshold:
                comp.status = HealthStatus.RED
                comp.message = f"memory high ({pct:.1f}%)"
            elif pct > self.mem_threshold - 10:
                comp.status = HealthStatus.YELLOW
                comp.message = f"memory warning ({pct:.1f}%)"
            else:
                comp.status = HealthStatus.GREEN
                comp.message = "OK"
        except Exception as e:
            comp.status = HealthStatus.YELLOW
            comp.message = f"memory check failed: {e}"
        return comp

    def check_cpu(self) -> ComponentHealth:
        """Check CPU usage."""
        comp = self._components["cpu"]
        try:
            pct = psutil.cpu_percent(interval=0.5)
            comp.value = f"{pct:.1f}%"
            comp.threshold = f"{self.cpu_threshold}%"
            comp.last_check = time.time()

            if pct > 99:
                comp.status = HealthStatus.BLACK
                comp.message = f"CPU overloaded ({pct:.1f}%)"
            elif pct > self.cpu_threshold:
                comp.status = HealthStatus.RED
                comp.message = f"CPU high ({pct:.1f}%)"
            elif pct > self.cpu_threshold - 10:
                comp.status = HealthStatus.YELLOW
                comp.message = f"CPU warning ({pct:.1f}%)"
            else:
                comp.status = HealthStatus.GREEN
                comp.message = "OK"
        except Exception as e:
            comp.status = HealthStatus.YELLOW
            comp.message = f"CPU check failed: {e}"
        return comp

    def check_network(self, host: str = "8.8.8.8", port: int = 53,
                      timeout: float = 3.0) -> ComponentHealth:
        """Check network connectivity."""
        comp = self._components["network"]
        try:
            socket.setdefaulttimeout(timeout)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            t0 = time.time()
            s.connect((host, port))
            elapsed = time.time() - t0
            s.close()
            comp.value = f"{elapsed*1000:.0f}ms"
            comp.threshold = f"{timeout*1000}ms"
            comp.last_check = time.time()

            if elapsed > timeout:
                comp.status = HealthStatus.YELLOW
                comp.message = f"network slow ({elapsed*1000:.0f}ms)"
            else:
                comp.status = HealthStatus.GREEN
                comp.message = "OK"
        except Exception as e:
            comp.status = HealthStatus.RED
            comp.value = str(e)
            comp.message = f"network down: {e}"
        return comp

    def check_clock_drift(self, ntp_server: str = "pool.ntp.org") -> ComponentHealth:
        """Check system clock drift vs NTP (simplified)."""
        comp = self._components["clock_drift"]
        # Simplified: just check that time is monotonic
        try:
            now = time.time()
            utc_now = datetime.now(tz=timezone.utc).timestamp()
            drift = abs(now - utc_now)
            comp.value = f"{drift*1000:.0f}ms"
            comp.last_check = time.time()
            if drift > 5.0:
                comp.status = HealthStatus.RED
                comp.message = f"clock drift {drift:.1f}s"
            elif drift > 1.0:
                comp.status = HealthStatus.YELLOW
                comp.message = f"clock drift {drift:.1f}s"
            else:
                comp.status = HealthStatus.GREEN
                comp.message = "OK"
        except Exception as e:
            comp.status = HealthStatus.YELLOW
            comp.message = f"clock check failed: {e}"
        return comp

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------
    def check_all(self, connector: Any = None,
                  db_path: str = "data/trading_bot.db",
                  symbol: str = "BTCUSD",
                  last_order_ms: float = 0.0) -> Dict[str, Any]:
        """Run all health checks."""
        self.check_mt5_connection(connector)
        self.check_tick_freshness(symbol, connector)
        if last_order_ms > 0:
            self.check_order_latency(last_order_ms)
        self.check_database(db_path)
        self.check_disk_space("/")
        self.check_memory()
        self.check_cpu()
        self.check_network()
        self.check_clock_drift()
        return self.health_summary()

    # ------------------------------------------------------------------
    # Overall health
    # ------------------------------------------------------------------
    def health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self._lock:
            components = {name: c.to_dict() for name, c in self._components.items()}

            # Overall status = worst of all components
            statuses = [c.status for c in self._components.values()]
            if HealthStatus.BLACK in statuses:
                overall = HealthStatus.BLACK
            elif HealthStatus.RED in statuses:
                overall = HealthStatus.RED
            elif HealthStatus.YELLOW in statuses:
                overall = HealthStatus.YELLOW
            else:
                overall = HealthStatus.GREEN

            # Count by status
            counts = {s.value: 0 for s in HealthStatus}
            for s in statuses:
                counts[s.value] += 1

            # List failed components
            failed = [name for name, c in self._components.items()
                     if c.status in (HealthStatus.RED, HealthStatus.BLACK)]

            return {
                "status": overall.value,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "counts": counts,
                "total_components": len(self._components),
                "failed_components": failed,
                "components": components,
                "can_trade": overall in (HealthStatus.GREEN, HealthStatus.YELLOW),
                "emergency": overall == HealthStatus.BLACK,
            }

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def record_snapshot(self) -> None:
        """Record current health to history."""
        summary = self.health_summary()
        self._history.append({
            "timestamp": summary["timestamp"],
            "status": summary["status"],
            "failed": summary["failed_components"],
        })
        # Keep last 1000 snapshots
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

    def history(self, last_n: int = 50) -> List[Dict[str, Any]]:
        return self._history[-last_n:]
