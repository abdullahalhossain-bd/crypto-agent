"""observability.decision_trace
=====================================================================
Day 28 — Decision trace recorder.

Every trade decision (whether approved or rejected) is recorded as a
`DecisionTrace` containing the full chain of evidence:

    signal → ML score → regime → portfolio check → risk checks →
    execution slices → final outcome

This is the single most important artefact for institutional audit:
when a trade goes wrong, you must be able to reconstruct exactly why
the bot took it.

Traces are persisted as JSON-lines to `data/decision_traces.jsonl`
so they can be ingested into a data warehouse or grepped locally.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("observability.trace")


@dataclass
class DecisionTrace:
    """One trade decision's full audit trail."""
    trace_id: str
    ts: str
    symbol: str
    timeframe: str
    action: str
    # Stage outputs (each is a small dict)
    signal: dict[str, Any] = field(default_factory=dict)
    ml_score: Optional[dict[str, Any]] = None
    regime: Optional[dict[str, Any]] = None
    portfolio: Optional[dict[str, Any]] = None
    risk_decision: Optional[dict[str, Any]] = None
    execution: Optional[dict[str, Any]] = None
    outcome: Optional[dict[str, Any]] = None
    final_status: str = "pending"  # approved | rejected | executed | failed
    final_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ----------------------------------------------------------------------
class DecisionTraceRecorder:
    """Persists every DecisionTrace as a JSON-lines file."""

    def __init__(self, path: str = "data/decision_traces.jsonl") -> None:
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ----------------------------------------------------------------
    def record(self, trace: DecisionTrace) -> None:
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(trace.to_json() + "\n")
        log.debug("trace recorded %s status=%s", trace.trace_id, trace.final_status)

    # ----------------------------------------------------------------
    def query(self, symbol: Optional[str] = None,
              status: Optional[str] = None,
              limit: int = 100) -> list[dict[str, Any]]:
        """Cheap grep-style query — for ops debugging."""
        if not os.path.isfile(self.path):
            return []
        out: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if symbol and d.get("symbol") != symbol:
                    continue
                if status and d.get("final_status") != status:
                    continue
                out.append(d)
                if len(out) >= limit:
                    break
        return out

    # ----------------------------------------------------------------
    @staticmethod
    def make_trace(symbol: str, timeframe: str, action: str) -> DecisionTrace:
        import uuid
        return DecisionTrace(
            trace_id=uuid.uuid4().hex[:16],
            ts=datetime.now(tz=timezone.utc).isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            action=action,
        )
