"""engine.guardrails
=====================================================================
Day 34-36 — Live Trading Guardrails.

Hard, non-negotiable constraints enforced BEFORE any order is sent
to the live broker. Different from risk_v2 (which is per-trade);
guardrails are SYSTEM-WIDE circuit breakers that operate on the
aggregate portfolio state.

Each guardrail has:
  - check()    → returns GuardrailResult(passed, reason, severity)
  - reset()    → clears latched state (operator action required)

Severity levels:
  - INFO    → log and continue
  - WARN    → block new trades, allow position management
  - HALT    → block all trading + arm kill switch
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("engine.guardrails")


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    HALT = "HALT"


@dataclass
class GuardrailResult:
    name: str
    passed: bool
    severity: Severity
    reason: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None
    latched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity.value,
            "reason": self.reason,
            "value": self.value,
            "threshold": self.threshold,
            "latched": self.latched,
        }


# ----------------------------------------------------------------------
class Guardrail:
    """Base class. Subclasses implement `check`."""
    name: str = "abstract"

    def __init__(self) -> None:
        self._latched = False
        self._latched_reason = ""

    def check(self, ctx: dict[str, Any]) -> GuardrailResult:  # noqa: ARG002
        raise NotImplementedError

    def reset(self) -> None:
        self._latched = False
        self._latched_reason = ""
        log.info("Guardrail %s reset", self.name)

    @property
    def latched(self) -> bool:
        return self._latched


# ----------------------------------------------------------------------
class MaxDailyLossGuardrail(Guardrail):
    name = "max_daily_loss"

    def __init__(self, threshold_pct: float = 0.05) -> None:
        super().__init__()
        self.threshold_pct = float(threshold_pct)

    def check(self, ctx: dict[str, Any]) -> GuardrailResult:
        equity = float(ctx.get("equity", 0.0))
        start_of_day = float(ctx.get("start_of_day_equity", equity))
        if start_of_day <= 0:
            return GuardrailResult(self.name, True, Severity.INFO,
                                    "no start-of-day equity")
        pnl_pct = (equity - start_of_day) / start_of_day
        if pnl_pct <= -self.threshold_pct:
            self._latched = True
            self._latched_reason = f"daily loss {pnl_pct:.2%}"
            return GuardrailResult(
                self.name, False, Severity.HALT,
                f"daily loss {pnl_pct:.2%} <= -{self.threshold_pct:.2%}",
                value=abs(pnl_pct), threshold=self.threshold_pct, latched=True,
            )
        return GuardrailResult(self.name, True, Severity.INFO,
                                value=abs(pnl_pct), threshold=self.threshold_pct)


class MaxDrawdownGuardrail(Guardrail):
    name = "max_drawdown"

    def __init__(self, threshold_pct: float = 0.15) -> None:
        super().__init__()
        self.threshold_pct = float(threshold_pct)

    def check(self, ctx: dict[str, Any]) -> GuardrailResult:
        dd = float(ctx.get("current_drawdown_pct", 0.0))
        if dd > self.threshold_pct:
            self._latched = True
            self._latched_reason = f"drawdown {dd:.2%}"
            return GuardrailResult(
                self.name, False, Severity.HALT,
                f"drawdown {dd:.2%} > {self.threshold_pct:.2%}",
                value=dd, threshold=self.threshold_pct, latched=True,
            )
        return GuardrailResult(self.name, True, Severity.INFO,
                                value=dd, threshold=self.threshold_pct)


class MaxExposureGuardrail(Guardrail):
    name = "max_exposure"

    def __init__(self, max_gross: float = 2.0, max_net: float = 1.0,
                 max_per_symbol: float = 0.5) -> None:
        super().__init__()
        self.max_gross = float(max_gross)
        self.max_net = float(max_net)
        self.max_per_symbol = float(max_per_symbol)

    def check(self, ctx: dict[str, Any]) -> GuardrailResult:
        gross = float(ctx.get("gross_exposure", 0.0))
        net = abs(float(ctx.get("net_exposure", 0.0)))
        per_symbol = ctx.get("per_symbol_exposure", {})
        worst_symbol = ""
        worst_value = 0.0
        for sym, val in per_symbol.items():
            v = abs(float(val))
            if v > worst_value:
                worst_value = v
                worst_symbol = sym
        if gross > self.max_gross:
            return GuardrailResult(self.name, False, Severity.WARN,
                                    f"gross {gross:.3f} > {self.max_gross}",
                                    value=gross, threshold=self.max_gross)
        if net > self.max_net:
            return GuardrailResult(self.name, False, Severity.WARN,
                                    f"net {net:.3f} > {self.max_net}",
                                    value=net, threshold=self.max_net)
        if worst_value > self.max_per_symbol:
            return GuardrailResult(
                self.name, False, Severity.WARN,
                f"symbol {worst_symbol} exposure {worst_value:.3f} > {self.max_per_symbol}",
                value=worst_value, threshold=self.max_per_symbol,
            )
        return GuardrailResult(self.name, True, Severity.INFO)


class MaxCorrelatedExposureGuardrail(Guardrail):
    """Block when sum of |exposure| * |correlation| > threshold."""
    name = "max_correlated_exposure"

    def __init__(self, threshold: float = 1.5,
                 correlation_threshold: float = 0.6) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.correlation_threshold = float(correlation_threshold)

    def check(self, ctx: dict[str, Any]) -> GuardrailResult:
        per_symbol = ctx.get("per_symbol_exposure", {})
        corr_matrix = ctx.get("correlation_matrix", {})
        # corr_matrix might be a pandas DataFrame — use .empty for truthiness
        corr_empty = (corr_matrix is None
                      or (hasattr(corr_matrix, "empty") and corr_matrix.empty)
                      or (isinstance(corr_matrix, dict) and not corr_matrix))
        if not per_symbol or corr_empty:
            return GuardrailResult(self.name, True, Severity.INFO,
                                    "no correlation data")
        # Compute pairwise correlation-weighted exposure
        symbols = list(per_symbol.keys())
        correlated_exposure = 0.0
        for i, s_i in enumerate(symbols):
            for s_j in symbols[i + 1:]:
                # corr_matrix may be a pandas DataFrame or a dict-of-dicts
                try:
                    if hasattr(corr_matrix, "loc"):
                        if s_i in corr_matrix.index and s_j in corr_matrix.columns:
                            corr = abs(float(corr_matrix.loc[s_i, s_j]))
                        else:
                            corr = 0.0
                    else:
                        corr = abs(float(corr_matrix.get(s_i, {}).get(s_j, 0.0)))
                except Exception:  # noqa: BLE001
                    corr = 0.0
                if corr >= self.correlation_threshold:
                    correlated_exposure += (abs(per_symbol[s_i])
                                            + abs(per_symbol[s_j])) * corr
        if correlated_exposure > self.threshold:
            return GuardrailResult(
                self.name, False, Severity.WARN,
                f"correlated exposure {correlated_exposure:.3f} > {self.threshold}",
                value=correlated_exposure, threshold=self.threshold,
            )
        return GuardrailResult(self.name, True, Severity.INFO,
                                value=correlated_exposure,
                                threshold=self.threshold)


class AnomalyKillSwitchGuardrail(Guardrail):
    """Triggered by external anomaly signals (e.g. data feed gaps,
    latency spikes, unexpected broker errors)."""
    name = "anomaly_kill_switch"

    def __init__(self, max_anomaly_score: float = 0.8) -> None:
        super().__init__()
        self.max_anomaly_score = float(max_anomaly_score)

    def check(self, ctx: dict[str, Any]) -> GuardrailResult:
        score = float(ctx.get("anomaly_score", 0.0))
        if score >= self.max_anomaly_score:
            self._latched = True
            return GuardrailResult(
                self.name, False, Severity.HALT,
                f"anomaly score {score:.2f} >= {self.max_anomaly_score}",
                value=score, threshold=self.max_anomaly_score, latched=True,
            )
        return GuardrailResult(self.name, True, Severity.INFO,
                                value=score, threshold=self.max_anomaly_score)


# ----------------------------------------------------------------------
class GuardrailEngine:
    """Runs every guardrail against the current portfolio context."""

    def __init__(self, guardrails: Optional[list[Guardrail]] = None) -> None:
        self.guardrails = guardrails or self._default_guardrails()
        self._last_results: list[GuardrailResult] = []

    @staticmethod
    def _default_guardrails() -> list[Guardrail]:
        return [
            MaxDailyLossGuardrail(threshold_pct=0.05),
            MaxDrawdownGuardrail(threshold_pct=0.15),
            MaxExposureGuardrail(max_gross=2.0, max_net=1.0,
                                 max_per_symbol=0.5),
            MaxCorrelatedExposureGuardrail(threshold=1.5),
            AnomalyKillSwitchGuardrail(max_anomaly_score=0.8),
        ]

    # ----------------------------------------------------------------
    def evaluate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Run all guardrails. Returns aggregated decision + per-rail results."""
        results = [g.check(ctx) for g in self.guardrails]
        self._last_results = results
        worst = Severity.INFO
        for r in results:
            if not r.passed:
                if r.severity == Severity.HALT:
                    worst = Severity.HALT
                    break
                if r.severity == Severity.WARN and worst != Severity.HALT:
                    worst = Severity.WARN
        # Decide trading permission
        if worst == Severity.HALT:
            permission = "halt"
        elif worst == Severity.WARN:
            permission = "block_new"
        else:
            permission = "allow"
        return {
            "permission": permission,
            "worst_severity": worst.value,
            "results": [r.to_dict() for r in results],
            "any_latched": any(r.latched for r in results),
            "ts": time.time(),
        }

    def reset_all(self) -> None:
        for g in self.guardrails:
            g.reset()

    @property
    def last_results(self) -> list[GuardrailResult]:
        return list(self._last_results)
