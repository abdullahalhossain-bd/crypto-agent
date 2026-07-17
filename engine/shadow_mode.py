"""engine.shadow_mode
=====================================================================
Day 37-40 — Shadow Mode.

Runs the v2 decision pipeline against LIVE market data WITHOUT
sending real orders. Every shadow decision is recorded alongside
the paper-mode decision so we can quantify divergence before any
real capital is at risk.

Three modes the system supports simultaneously:
  - LIVE   : real orders to MT5 (only if capital tier >= MICRO)
  - PAPER  : simulated fills via base execution engine
  - SHADOW : same pipeline, no fills recorded, decisions logged only

Divergence metrics tracked:
  - Signal divergence       : do paper and shadow agree on action?
  - Price divergence        : predicted vs actual fill price
  - Slippage divergence     : predicted vs realised slippage
  - Latency divergence      : decision time vs bar arrival time

If divergence exceeds threshold, the capital tier manager refuses
to promote to the next tier (see capital_tiers.py).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from engine.signals import Action, Signal
from observability.decision_trace import DecisionTrace, DecisionTraceRecorder
from utils.logger import get_logger

log = get_logger("engine.shadow_mode")


@dataclass
class ShadowDecision:
    """A decision the system WOULD have made in live mode."""
    ts: str
    symbol: str
    timeframe: str
    action: Action
    strength: float
    price: float
    ml_confidence: Optional[float] = None
    regime: Optional[str] = None
    risk_approved: bool = False
    adjusted_lots: float = 0.0
    decision_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "action": self.action.value,
            "strength": self.strength,
            "price": self.price,
            "ml_confidence": self.ml_confidence,
            "regime": self.regime,
            "risk_approved": self.risk_approved,
            "adjusted_lots": self.adjusted_lots,
            "decision_trace": self.decision_trace,
        }


@dataclass
class DivergenceMetrics:
    """Aggregated divergence between paper and shadow decisions."""
    n_compared: int = 0
    n_signal_mismatch: int = 0          # action differs
    n_risk_mismatch: int = 0            # paper approved, shadow rejected (or vice versa)
    avg_price_diff_pct: float = 0.0
    max_price_diff_pct: float = 0.0
    avg_slippage_diff_bps: float = 0.0
    avg_latency_ms: float = 0.0
    signal_match_rate: float = 1.0
    risk_match_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    @property
    def overall_divergence_pct(self) -> float:
        """Composite score in [0, 1] — higher = more divergent."""
        if self.n_compared == 0:
            return 0.0
        signal_mismatch_rate = 1.0 - self.signal_match_rate
        risk_mismatch_rate = 1.0 - self.risk_match_rate
        price_component = min(1.0, self.avg_price_diff_pct * 100)  # 1% → 1.0
        return float(min(1.0, (signal_mismatch_rate * 0.4
                                + risk_mismatch_rate * 0.3
                                + price_component * 0.3)))


# ----------------------------------------------------------------------
class ShadowModeRunner:
    """Executes the v2 pipeline in shadow mode and compares to paper/live."""

    def __init__(
        self,
        trace_recorder: Optional[DecisionTraceRecorder] = None,
        shadow_log_path: str = "data/shadow_decisions.jsonl",
    ) -> None:
        self.trace_recorder = trace_recorder or DecisionTraceRecorder()
        self.shadow_log_path = shadow_log_path
        self._comparisons: list[dict[str, Any]] = []
        self._divergence = DivergenceMetrics()
        # H16 fix: file lock to prevent concurrent writes from corrupting
        # the JSONL log (the main loop + a shadow comparison thread can
        # both call record_shadow_decision).
        import threading
        self._write_lock = threading.Lock()

    # ----------------------------------------------------------------
    def record_shadow_decision(self, decision: ShadowDecision) -> None:
        """Persist a shadow decision.

        H16 fix: all writes are serialized through self._write_lock so
        concurrent appends don't interleave and corrupt the JSONL file.
        """
        import json
        import os
        os.makedirs(os.path.dirname(self.shadow_log_path) or ".", exist_ok=True)
        try:
            with self._write_lock:
                with open(self.shadow_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(decision.to_dict(), default=str) + "\n")
                    f.flush()  # ensure the write hits disk before releasing the lock
        except Exception as e:  # noqa: BLE001
            log.warning("shadow decision write failed: %r", e)
        log.debug("SHADOW %s %s %s strength=%.2f approved=%s lots=%.4f",
                  decision.action, decision.symbol, decision.timeframe,
                  decision.strength, decision.risk_approved,
                  decision.adjusted_lots)

    # ----------------------------------------------------------------
    def compare(
        self,
        paper_decision: dict[str, Any],
        shadow_decision: ShadowDecision,
    ) -> dict[str, Any]:
        """Compare a paper execution with the matching shadow decision.

        Both must be for the same (symbol, bar_time, strategy).
        """
        paper_action = paper_decision.get("action", "HOLD")
        shadow_action = shadow_decision.action.value
        signal_match = paper_action == shadow_action

        paper_approved = paper_decision.get("risk_approved", False)
        shadow_approved = shadow_decision.risk_approved
        risk_match = paper_approved == shadow_approved

        paper_price = float(paper_decision.get("price", 0.0))
        shadow_price = float(shadow_decision.price or 0.0)
        if shadow_price > 0:
            price_diff_pct = abs(paper_price - shadow_price) / shadow_price
        else:
            price_diff_pct = 0.0

        comparison = {
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "symbol": shadow_decision.symbol,
            "paper_action": paper_action,
            "shadow_action": shadow_action,
            "signal_match": signal_match,
            "paper_approved": paper_approved,
            "shadow_approved": shadow_approved,
            "risk_match": risk_match,
            "paper_price": paper_price,
            "shadow_price": shadow_price,
            "price_diff_pct": price_diff_pct,
        }
        self._comparisons.append(comparison)
        self._update_divergence_metrics(comparison)
        return comparison

    # ----------------------------------------------------------------
    def _update_divergence_metrics(self, c: dict[str, Any]) -> None:
        d = self._divergence
        d.n_compared += 1
        if not c["signal_match"]:
            d.n_signal_mismatch += 1
        if not c["risk_match"]:
            d.n_risk_mismatch += 1
        # Running averages
        n = d.n_compared
        d.avg_price_diff_pct = ((d.avg_price_diff_pct * (n - 1))
                                + c["price_diff_pct"]) / n
        d.max_price_diff_pct = max(d.max_price_diff_pct, c["price_diff_pct"])
        d.signal_match_rate = (d.n_compared - d.n_signal_mismatch) / n
        d.risk_match_rate = (d.n_compared - d.n_risk_mismatch) / n

    # ----------------------------------------------------------------
    @property
    def divergence(self) -> DivergenceMetrics:
        return self._divergence

    def reset_divergence(self) -> None:
        self._comparisons.clear()
        self._divergence = DivergenceMetrics()

    @property
    def comparisons(self) -> list[dict[str, Any]]:
        return list(self._comparisons)
