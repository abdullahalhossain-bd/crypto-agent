"""utils/indicators/diagnostics.py
=====================================================================
Indicator Diagnostics (Improvement #20)
=====================================================================
Tracks performance and quality metrics for indicator computation.

Metrics:
    - execution_time_ms (per indicator)
    - indicator_count (total registered)
    - failed_count (computations that raised)
    - warning_count (data quality warnings)
    - cache_hits / cache_misses
    - memory_usage (estimated)
    - data_quality_score (0-1)

Usage:
    from utils.indicators.diagnostics import Diagnostics

    diag = Diagnostics()
    with diag.track("rsi_14"):
        result = rsi(close, 14)
    # diag.stats["rsi_14"] = {"time_ms": 0.42, "calls": 1, ...}
"""
from __future__ import annotations

import threading
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd


@dataclass
class IndicatorMetric:
    """Per-indicator metrics."""
    name: str = ""
    calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    failures: int = 0
    warnings: int = 0
    last_value: Any = None
    last_call_at: float = 0.0

    def record_call(self, time_ms: float, value: Any = None,
                    failed: bool = False, warning: bool = False) -> None:
        self.calls += 1
        self.total_time_ms += time_ms
        self.avg_time_ms = self.total_time_ms / self.calls
        if time_ms > self.max_time_ms:
            self.max_time_ms = time_ms
        if failed:
            self.failures += 1
        if warning:
            self.warnings += 1
        self.last_value = value
        self.last_call_at = time.time()


class Diagnostics:
    """Central diagnostics tracker for the indicator engine."""

    def __init__(self, enable_memory_tracking: bool = False):
        self._lock = threading.RLock()
        self._metrics: Dict[str, IndicatorMetric] = {}
        self._enable_memory = enable_memory_tracking
        if enable_memory_tracking:
            tracemalloc.start()
        self._data_quality_warnings: List[str] = []

    @contextmanager
    def track(self, indicator_name: str) -> Iterator[IndicatorMetric]:
        """Context manager to track an indicator computation.

        Example:
            with diag.track("rsi_14"):
                result = rsi(close, 14)
        """
        with self._lock:
            if indicator_name not in self._metrics:
                self._metrics[indicator_name] = IndicatorMetric(name=indicator_name)
            metric = self._metrics[indicator_name]

        t0 = time.time()
        failed = False
        try:
            yield metric
        except Exception:
            failed = True
            raise
        finally:
            elapsed_ms = (time.time() - t0) * 1000
            with self._lock:
                metric.record_call(elapsed_ms, failed=failed)

    def record_warning(self, msg: str) -> None:
        with self._lock:
            self._data_quality_warnings.append(msg)

    def stats(self) -> Dict[str, Any]:
        """Aggregate stats for all indicators."""
        with self._lock:
            total_calls = sum(m.calls for m in self._metrics.values())
            total_time = sum(m.total_time_ms for m in self._metrics.values())
            total_failures = sum(m.failures for m in self._metrics.values())
            total_warnings = len(self._data_quality_warnings)
            memory = None
            if self._enable_memory:
                current, peak = tracemalloc.get_traced_memory()
                memory = {"current_mb": current / 1e6, "peak_mb": peak / 1e6}

            per_indicator = {}
            for name, m in self._metrics.items():
                per_indicator[name] = {
                    "calls": m.calls,
                    "total_ms": round(m.total_time_ms, 3),
                    "avg_ms": round(m.avg_time_ms, 3),
                    "max_ms": round(m.max_time_ms, 3),
                    "failures": m.failures,
                    "warnings": m.warnings,
                }

            return {
                "indicator_count": len(self._metrics),
                "total_calls": total_calls,
                "total_time_ms": round(total_time, 3),
                "failed_count": total_failures,
                "warning_count": total_warnings,
                "data_quality_score": max(0.0, 1.0 - total_warnings * 0.05 - total_failures * 0.1),
                "memory_usage": memory,
                "per_indicator": per_indicator,
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._data_quality_warnings.clear()

    def slowest_indicators(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the N slowest indicators by avg time."""
        with self._lock:
            sorted_metrics = sorted(
                self._metrics.values(),
                key=lambda m: m.avg_time_ms,
                reverse=True
            )[:top_n]
            return [
                {"name": m.name, "avg_ms": round(m.avg_time_ms, 3),
                 "calls": m.calls, "max_ms": round(m.max_time_ms, 3)}
                for m in sorted_metrics
            ]


# Global diagnostics instance
_GLOBAL_DIAGNOSTICS: Optional[Diagnostics] = None


def get_diagnostics() -> Diagnostics:
    global _GLOBAL_DIAGNOSTICS
    if _GLOBAL_DIAGNOSTICS is None:
        _GLOBAL_DIAGNOSTICS = Diagnostics(enable_memory_tracking=False)
    return _GLOBAL_DIAGNOSTICS
