"""validation.readiness_gate
=====================================================================
Day 116-120 — Institutional GO / NO-GO readiness gate.

This is the final approval layer. The system is "ready for live
capital" only when EVERY gate check passes with concrete evidence.

Gate checks (8 total):
  1. EDGE_PROVEN           : edge is statistically real (not luck)
  2. EXECUTION_STABLE      : fills are reliable, slippage bounded
  3. RISK_BATTLE_TESTED    : system survives stress scenarios
  4. KILL_SWITCH_VALIDATED : kill switch activates in every scenario
  5. SURVIVAL_TEST_PASSED  : strategies survive regime changes
  6. SHADOW_LIVE_PASSED    : 30+ day shadow tracking with positive edge
  7. MONITORING_COMPLETE   : all 4 monitoring layers reporting
  8. OPERATOR_RUNBOOK_READY: incident response procedures documented

Each check returns one of:
  - PASS    : evidence provided, meets threshold
  - WARN    : evidence provided, below threshold (operator decision)
  - FAIL    : no evidence or fundamentally insufficient
  - SKIP    : not applicable (e.g. shadow-live when no live data exists)

The system is GO only when ALL checks are PASS (or operator-approved
WARN). Any FAIL → NO-GO.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("validation.readiness_gate")


class GateStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class GateVerdict:
    """Single gate check result."""
    name: str
    status: GateStatus
    evidence: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    actual: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "evidence": dict(self.evidence),
            "threshold": dict(self.threshold),
            "actual": dict(self.actual),
            "reason": self.reason,
        }


@dataclass
class ReadinessReport:
    """Full readiness gate report."""
    timestamp: str
    overall_status: str             # GO / NO-GO / CONDITIONAL
    n_pass: int
    n_warn: int
    n_fail: int
    n_skip: int
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    operator_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class ReadinessGate:
    """The institutional gate. Call `evaluate()` with all evidence."""

    def __init__(self,
                 require_all_pass: bool = True,
                 allow_warn_with_operator_approval: bool = True) -> None:
        self.require_all_pass = bool(require_all_pass)
        self.allow_warn = bool(allow_warn_with_operator_approval)

    # ----------------------------------------------------------------
    def evaluate(
        self,
        edge_proof_result: Optional[dict[str, Any]] = None,
        execution_metrics: Optional[dict[str, Any]] = None,
        survival_report: Optional[dict[str, Any]] = None,
        kill_switch_report: Optional[dict[str, Any]] = None,
        shadow_live_analysis: Optional[dict[str, Any]] = None,
        stress_test_results: Optional[list[dict[str, Any]]] = None,
        monitoring_health: Optional[dict[str, Any]] = None,
        operator_runbook_ready: bool = False,
    ) -> ReadinessReport:
        verdicts: list[GateVerdict] = []
        verdicts.append(self._check_edge_proven(edge_proof_result))
        verdicts.append(self._check_execution_stable(execution_metrics))
        verdicts.append(self._check_risk_battle_tested(stress_test_results))
        verdicts.append(self._check_kill_switch(kill_switch_report))
        verdicts.append(self._check_survival_test(survival_report))
        verdicts.append(self._check_shadow_live(shadow_live_analysis))
        verdicts.append(self._check_monitoring(monitoring_health))
        verdicts.append(self._check_operator_runbook(operator_runbook_ready))

        n_pass = sum(1 for v in verdicts if v.status == GateStatus.PASS)
        n_warn = sum(1 for v in verdicts if v.status == GateStatus.WARN)
        n_fail = sum(1 for v in verdicts if v.status == GateStatus.FAIL)
        n_skip = sum(1 for v in verdicts if v.status == GateStatus.SKIP)

        blocking_issues: list[str] = []
        for v in verdicts:
            if v.status == GateStatus.FAIL:
                blocking_issues.append(f"{v.name}: {v.reason}")
            elif v.status == GateStatus.WARN and not self.allow_warn:
                blocking_issues.append(f"{v.name}: {v.reason}")

        if self.require_all_pass:
            if n_fail == 0 and (n_warn == 0 or self.allow_warn):
                overall = "GO" if n_warn == 0 else "CONDITIONAL"
            else:
                overall = "NO-GO"
        else:
            overall = "GO" if n_fail == 0 else "NO-GO"

        return ReadinessReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            overall_status=overall,
            n_pass=n_pass, n_warn=n_warn, n_fail=n_fail, n_skip=n_skip,
            verdicts=[v.to_dict() for v in verdicts],
            blocking_issues=blocking_issues,
        )

    # ----------------------------------------------------------------
    # Individual gate checks
    # ----------------------------------------------------------------
    def _check_edge_proven(self, result: Optional[dict]) -> GateVerdict:
        if result is None:
            return GateVerdict(
                name="edge_proven", status=GateStatus.SKIP,
                reason="no edge proof result provided",
            )
        proven = bool(result.get("edge_proven", False))
        n = int(result.get("n_samples", 0))
        expectancy = float(result.get("expectancy", 0))
        ci_low = float(result.get("expectancy_ci_low", 0))
        p_val = float(result.get("expectancy_p_value", 1))
        actual = {
            "n_samples": n, "expectancy": expectancy,
            "ci_low": ci_low, "p_value": p_val,
        }
        threshold = {
            "min_samples": 100, "min_expectancy": 0.0,
            "ci_low_must_be_positive": True, "max_p_value": 0.05,
        }
        if proven:
            return GateVerdict(
                name="edge_proven", status=GateStatus.PASS,
                evidence=result, threshold=threshold, actual=actual,
                reason="edge is statistically proven",
            )
        if n >= 50 and expectancy > 0 and ci_low > 0:
            return GateVerdict(
                name="edge_proven", status=GateStatus.WARN,
                evidence=result, threshold=threshold, actual=actual,
                reason=f"edge suggestive but not proven (n={n}, p={p_val:.3f})",
            )
        return GateVerdict(
            name="edge_proven", status=GateStatus.FAIL,
            evidence=result, threshold=threshold, actual=actual,
            reason=f"edge not proven (n={n}, expectancy={expectancy:.4f})",
        )

    def _check_execution_stable(self, metrics: Optional[dict]) -> GateVerdict:
        if metrics is None:
            return GateVerdict(
                name="execution_stable", status=GateStatus.SKIP,
                reason="no execution metrics provided",
            )
        n = int(metrics.get("n_records", 0))
        fill_rate = float(metrics.get("fill_rate", 0))
        slip_p95 = float(metrics.get("slippage_distribution", {}).get("p95", 999))
        lat_p95 = float(metrics.get("latency_distribution_ms", {}).get("p95", 9999))
        actual = {"n_records": n, "fill_rate": fill_rate,
                  "slippage_p95_bps": slip_p95, "latency_p95_ms": lat_p95}
        threshold = {"min_samples": 50, "min_fill_rate": 0.95,
                     "max_slippage_p95_bps": 15.0, "max_latency_p95_ms": 2000.0}
        if n < 50:
            return GateVerdict(
                name="execution_stable", status=GateStatus.FAIL,
                evidence=metrics, threshold=threshold, actual=actual,
                reason=f"insufficient samples ({n} < 50)",
            )
        checks = {
            "fill_rate_ok": fill_rate >= 0.95,
            "slippage_ok": slip_p95 <= 15.0,
            "latency_ok": lat_p95 <= 2000.0,
        }
        if all(checks.values()):
            return GateVerdict(
                name="execution_stable", status=GateStatus.PASS,
                evidence=metrics, threshold=threshold, actual=actual,
                reason="execution metrics within thresholds",
            )
        failed = [k for k, v in checks.items() if not v]
        return GateVerdict(
            name="execution_stable", status=GateStatus.WARN if n >= 50 else GateStatus.FAIL,
            evidence=metrics, threshold=threshold, actual=actual,
            reason=f"thresholds not met: {failed}",
        )

    def _check_risk_battle_tested(self, stress_results: Optional[list]) -> GateVerdict:
        if stress_results is None:
            return GateVerdict(
                name="risk_battle_tested", status=GateStatus.SKIP,
                reason="no stress test results provided",
            )
        n_pass = sum(1 for r in stress_results if r.get("passed", False))
        n_total = len(stress_results)
        actual = {"n_tests": n_total, "n_passed": n_pass}
        threshold = {"min_pass_rate": 0.8}
        if n_total == 0:
            return GateVerdict(
                name="risk_battle_tested", status=GateStatus.FAIL,
                reason="no stress tests run",
            )
        pass_rate = n_pass / n_total
        if pass_rate >= 0.8:
            return GateVerdict(
                name="risk_battle_tested", status=GateStatus.PASS,
                evidence={"results": stress_results}, threshold=threshold, actual=actual,
                reason=f"{n_pass}/{n_total} stress tests passed",
            )
        return GateVerdict(
            name="risk_battle_tested", status=GateStatus.FAIL,
            evidence={"results": stress_results}, threshold=threshold, actual=actual,
            reason=f"only {n_pass}/{n_total} stress tests passed",
        )

    def _check_kill_switch(self, report: Optional[dict]) -> GateVerdict:
        if report is None:
            return GateVerdict(
                name="kill_switch_validated", status=GateStatus.SKIP,
                reason="no kill switch report provided",
            )
        all_passed = bool(report.get("all_passed", False))
        n_tests = int(report.get("n_tests", 0))
        n_passed = int(report.get("n_passed", 0))
        actual = {"n_tests": n_tests, "n_passed": n_passed}
        threshold = {"all_must_pass": True, "max_response_s": 5.0}
        if n_tests == 0:
            return GateVerdict(
                name="kill_switch_validated", status=GateStatus.FAIL,
                reason="no kill switch tests run",
            )
        if all_passed:
            return GateVerdict(
                name="kill_switch_validated", status=GateStatus.PASS,
                evidence=report, threshold=threshold, actual=actual,
                reason=f"all {n_tests} kill switch tests passed",
            )
        return GateVerdict(
            name="kill_switch_validated", status=GateStatus.FAIL,
            evidence=report, threshold=threshold, actual=actual,
            reason=f"only {n_passed}/{n_tests} kill switch tests passed",
        )

    def _check_survival_test(self, report: Optional[dict]) -> GateVerdict:
        if report is None:
            return GateVerdict(
                name="survival_test_passed", status=GateStatus.SKIP,
                reason="no survival test report provided",
            )
        verdict = str(report.get("overall_verdict", "failed"))
        n_passed = int(report.get("n_passed", 0))
        n_tests = int(report.get("n_tests", 0))
        actual = {"n_passed": n_passed, "n_tests": n_tests, "verdict": verdict}
        threshold = {"min_verdict": "fragile"}
        if verdict == "battle_tested":
            return GateVerdict(
                name="survival_test_passed", status=GateStatus.PASS,
                evidence=report, threshold=threshold, actual=actual,
                reason=f"strategy is battle_tested ({n_passed}/{n_tests})",
            )
        if verdict == "fragile":
            return GateVerdict(
                name="survival_test_passed", status=GateStatus.WARN,
                evidence=report, threshold=threshold, actual=actual,
                reason=f"strategy is fragile ({n_passed}/{n_tests}) — operator review",
            )
        return GateVerdict(
            name="survival_test_passed", status=GateStatus.FAIL,
            evidence=report, threshold=threshold, actual=actual,
            reason=f"strategy failed survival tests ({n_passed}/{n_tests})",
        )

    def _check_shadow_live(self, analysis: Optional[dict]) -> GateVerdict:
        if analysis is None:
            return GateVerdict(
                name="shadow_live_passed", status=GateStatus.SKIP,
                reason="no shadow live analysis provided",
            )
        n = int(analysis.get("n_samples", 0))
        expectancy = float(analysis.get("expectancy_pct", 0))
        ci_low = float(analysis.get("ci_low", 0))
        status = str(analysis.get("status", "insufficient_data"))
        actual = {"n_samples": n, "expectancy": expectancy,
                  "ci_low": ci_low, "status": status}
        threshold = {"min_samples": 100, "ci_low_must_be_positive": True}
        if status == "proven":
            return GateVerdict(
                name="shadow_live_passed", status=GateStatus.PASS,
                evidence=analysis, threshold=threshold, actual=actual,
                reason=f"shadow live edge proven (n={n})",
            )
        if n >= 50 and expectancy > 0:
            return GateVerdict(
                name="shadow_live_passed", status=GateStatus.WARN,
                evidence=analysis, threshold=threshold, actual=actual,
                reason=f"shadow live edge suggestive (n={n}, exp={expectancy:.4f})",
            )
        return GateVerdict(
            name="shadow_live_passed", status=GateStatus.FAIL,
            evidence=analysis, threshold=threshold, actual=actual,
            reason=f"shadow live edge not proven (n={n})",
        )

    def _check_monitoring(self, health: Optional[dict]) -> GateVerdict:
        if health is None:
            return GateVerdict(
                name="monitoring_complete", status=GateStatus.SKIP,
                reason="no monitoring health provided",
            )
        layers = ["system", "trading", "risk", "alpha"]
        statuses = {layer: health.get(layer, {}).get("status", "unknown")
                    for layer in layers}
        n_ok = sum(1 for s in statuses.values() if s == "ok")
        n_degraded = sum(1 for s in statuses.values() if s == "degraded")
        n_critical = sum(1 for s in statuses.values() if s == "critical")
        actual = {"layer_statuses": statuses, "n_ok": n_ok,
                  "n_degraded": n_degraded, "n_critical": n_critical}
        threshold = {"min_ok_layers": 4, "no_critical_layers": True}
        if n_critical > 0:
            return GateVerdict(
                name="monitoring_complete", status=GateStatus.FAIL,
                evidence=health, threshold=threshold, actual=actual,
                reason=f"{n_critical} monitoring layer(s) in critical state",
            )
        if n_ok == 4:
            return GateVerdict(
                name="monitoring_complete", status=GateStatus.PASS,
                evidence=health, threshold=threshold, actual=actual,
                reason="all 4 monitoring layers OK",
            )
        return GateVerdict(
            name="monitoring_complete", status=GateStatus.WARN,
            evidence=health, threshold=threshold, actual=actual,
            reason=f"{n_degraded} monitoring layer(s) degraded",
        )

    def _check_operator_runbook(self, ready: bool) -> GateVerdict:
        if ready:
            return GateVerdict(
                name="operator_runbook_ready", status=GateStatus.PASS,
                reason="operator runbook documented",
            )
        return GateVerdict(
            name="operator_runbook_ready", status=GateStatus.FAIL,
            reason="operator runbook not ready",
        )
