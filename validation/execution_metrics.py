"""validation.execution_metrics
=====================================================================
Day 96-100 — Real execution metrics collector.

Tracks the ACTUAL behaviour of orders sent to the broker, so we can
quantify the gap between backtest assumptions and live reality.

Metrics collected per order:
  - Requested fill price vs actual fill price (slippage)
  - Order submission timestamp vs fill timestamp (latency)
  - Partial fill rate (was the order filled in one chunk or many?)
  - Rejection rate + reasons
  - Commission actually paid

Aggregates produced:
  - Slippage distribution (mean, p50, p95, p99, max)
  - Latency distribution (mean, p50, p95, p99, max)
  - Fill success rate
  - Rejection reason breakdown
  - Latency-vs-slippage correlation (does slow execution cost more?)

These metrics feed the readiness gate: if p95 slippage > 2x backtest
assumption, the system is NOT ready for live capital.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("validation.exec_metrics")


@dataclass
class FillRecord:
    """One order's execution record."""
    order_id: str
    ts_submit: str
    ts_fill: Optional[str] = None
    symbol: str = ""
    side: str = ""                # BUY / SELL
    requested_lots: float = 0.0
    filled_lots: float = 0.0
    requested_price: float = 0.0
    fill_price: float = 0.0
    slippage_bps: Optional[float] = None
    latency_ms: Optional[float] = None
    status: str = "submitted"     # submitted / filled / rejected / partial
    rejection_reason: str = ""
    commission_paid: float = 0.0
    n_partial_fills: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
@dataclass
class ExecutionDistribution:
    """Statistical distribution of an execution metric."""
    n: int
    mean: float
    median: float
    p95: float
    p99: float
    max: float
    min: float
    std: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


# ----------------------------------------------------------------------
class ExecutionMetricsCollector:
    """Collects + analyses real fill data from the execution engine."""

    def __init__(self, storage_path: str = "data/execution_metrics.jsonl",
                 max_records: int = 10_000) -> None:
        self.storage_path = storage_path
        self.max_records = int(max_records)
        self._records: deque[FillRecord] = deque(maxlen=max_records)
        self._pending: dict[str, FillRecord] = {}
        self._load()

    # ----------------------------------------------------------------
    def record_submission(
        self,
        order_id: str,
        symbol: str,
        side: str,
        requested_lots: float,
        requested_price: float,
    ) -> FillRecord:
        rec = FillRecord(
            order_id=order_id,
            ts_submit=datetime.now(tz=timezone.utc).isoformat(),
            symbol=symbol, side=side,
            requested_lots=float(requested_lots),
            requested_price=float(requested_price),
        )
        self._pending[order_id] = rec
        self._persist(rec)
        return rec

    # ----------------------------------------------------------------
    def record_fill(
        self,
        order_id: str,
        filled_lots: float,
        fill_price: float,
        commission: float = 0.0,
        n_partial_fills: int = 1,
        status: str = "filled",
        rejection_reason: str = "",
    ) -> Optional[FillRecord]:
        rec = self._pending.get(order_id)
        if rec is None:
            log.warning("fill record for unknown order %s", order_id)
            return None
        rec.ts_fill = datetime.now(tz=timezone.utc).isoformat()
        rec.filled_lots = float(filled_lots)
        rec.fill_price = float(fill_price)
        rec.commission_paid = float(commission)
        rec.n_partial_fills = int(n_partial_fills)
        rec.status = status
        rec.rejection_reason = rejection_reason
        # Compute slippage + latency
        if rec.requested_price > 0 and fill_price > 0:
            raw_slip = (fill_price - rec.requested_price) / rec.requested_price
            # For SELL orders, slippage is positive when fill < requested
            if rec.side.upper() == "SELL":
                raw_slip = -raw_slip
            rec.slippage_bps = float(raw_slip * 10_000.0)
        try:
            t1 = datetime.fromisoformat(rec.ts_submit)
            t2 = datetime.fromisoformat(rec.ts_fill)
            rec.latency_ms = float((t2 - t1).total_seconds() * 1000.0)
        except Exception:  # noqa: BLE001
            pass
        self._records.append(rec)
        self._persist(rec)
        del self._pending[order_id]
        log.debug("fill recorded: %s status=%s slip=%.1fbps lat=%.0fms",
                  order_id, status, rec.slippage_bps or 0, rec.latency_ms or 0)
        return rec

    # ----------------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------------
    def analyse(self) -> dict[str, Any]:
        if not self._records:
            return {"n_records": 0, "status": "no_data"}
        records = list(self._records)
        slippages = [r.slippage_bps for r in records if r.slippage_bps is not None]
        latencies = [r.latency_ms for r in records if r.latency_ms is not None]
        n_filled = sum(1 for r in records if r.status == "filled")
        n_partial = sum(1 for r in records if r.status == "partial")
        n_rejected = sum(1 for r in records if r.status == "rejected")
        n_total = len(records)
        fill_rate = n_filled / n_total if n_total else 0.0
        rejection_reasons: dict[str, int] = defaultdict(int)
        for r in records:
            if r.status == "rejected" and r.rejection_reason:
                rejection_reasons[r.rejection_reason] += 1

        # Latency-slippage correlation
        corr = 0.0
        pairs = [(r.latency_ms, r.slippage_bps) for r in records
                 if r.latency_ms is not None and r.slippage_bps is not None]
        if len(pairs) >= 10:
            arr = np.array(pairs)
            try:
                corr = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            except Exception:  # noqa: BLE001
                corr = 0.0

        return {
            "n_records": n_total,
            "n_pending": len(self._pending),
            "fill_rate": fill_rate,
            "partial_fill_rate": n_partial / n_total if n_total else 0.0,
            "rejection_rate": n_rejected / n_total if n_total else 0.0,
            "rejection_reasons": dict(rejection_reasons),
            "slippage_distribution": self._distribution(slippages),
            "latency_distribution_ms": self._distribution(latencies),
            "latency_slippage_correlation": corr,
            "avg_commission_paid": float(np.mean([r.commission_paid for r in records])),
            "n_partial_fills_avg": float(np.mean([r.n_partial_fills for r in records])),
        }

    # ----------------------------------------------------------------
    @staticmethod
    def _distribution(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "mean": 0, "median": 0, "p95": 0, "p99": 0, "max": 0, "min": 0, "std": 0}
        arr = np.array(values)
        return ExecutionDistribution(
            n=int(len(arr)),
            mean=float(arr.mean()),
            median=float(np.median(arr)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max=float(arr.max()),
            min=float(arr.min()),
            std=float(arr.std()) if len(arr) > 1 else 0.0,
        ).to_dict()

    # ----------------------------------------------------------------
    def readiness_check(self,
                         max_acceptable_p95_slippage_bps: float = 15.0,
                         max_acceptable_p95_latency_ms: float = 2000.0,
                         min_fill_rate: float = 0.95,
                         min_samples: int = 50) -> dict[str, Any]:
        """Institutional readiness check for execution layer."""
        analysis = self.analyse()
        if analysis.get("n_records", 0) < min_samples:
            return {
                "ready": False,
                "reason": f"insufficient samples ({analysis.get('n_records', 0)} < {min_samples})",
                "analysis": analysis,
            }
        slip = analysis.get("slippage_distribution", {})
        lat = analysis.get("latency_distribution_ms", {})
        checks = {
            "p95_slippage_ok": slip.get("p95", 0) <= max_acceptable_p95_slippage_bps,
            "p95_latency_ok": lat.get("p95", 0) <= max_acceptable_p95_latency_ms,
            "fill_rate_ok": analysis.get("fill_rate", 0) >= min_fill_rate,
        }
        all_pass = all(checks.values())
        return {
            "ready": bool(all_pass),
            "checks": checks,
            "thresholds": {
                "max_p95_slippage_bps": max_acceptable_p95_slippage_bps,
                "max_p95_latency_ms": max_acceptable_p95_latency_ms,
                "min_fill_rate": min_fill_rate,
                "min_samples": min_samples,
            },
            "actuals": {
                "p95_slippage_bps": slip.get("p95", 0),
                "p95_latency_ms": lat.get("p95", 0),
                "fill_rate": analysis.get("fill_rate", 0),
                "n_records": analysis.get("n_records", 0),
            },
            "analysis": analysis,
        }

    # ----------------------------------------------------------------
    def _persist(self, rec: FillRecord) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("exec metrics persist failed: %r", e)

    def _load(self) -> None:
        if not os.path.isfile(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        # Critical #3 fix: filter unknown fields before
                        # passing to FillRecord constructor. The old code
                        # used FillRecord(**d) which raises TypeError if
                        # the JSON contains extra fields from a future version.
                        valid_fields = {f.name for f in FillRecord.__dataclass_fields__.values()}
                        filtered = {k: v for k, v in d.items() if k in valid_fields}
                        rec = FillRecord(**filtered)
                        if rec.status == "submitted":
                            self._pending[rec.order_id] = rec
                        else:
                            self._records.append(rec)
                    except Exception:  # noqa: BLE001
                        continue
            log.info("exec metrics loaded: %d records, %d pending",
                     len(self._records), len(self._pending))
        except Exception as e:  # noqa: BLE001
            log.warning("exec metrics load failed: %r", e)
