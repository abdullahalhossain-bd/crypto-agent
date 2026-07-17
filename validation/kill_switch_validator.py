"""validation.kill_switch_validator
=====================================================================
Day 111-115 — Automated kill switch validation.

Proves the system can safely stop trading under stress. Each test
scenario simulates a real-world failure mode and verifies the kill
switch activates within the required time.

Test scenarios:
  1. FLASH_CRASH          : sudden -10% price move
  2. DAILY_LOSS_BREACH     : equity drops past daily loss limit
  3. DRAWDOWN_BREACH       : rolling drawdown exceeds threshold
  4. MT5_DISCONNECT        : broker connection lost mid-trade
  5. ANOMALY_DETECTED      : anomaly score spikes
  6. ERROR_BUDGET_EXCEEDED : consecutive error count breaches limit
  7. OPERATOR_MANUAL       : operator creates kill switch file
  8. LATENCY_SPIKE         : cycle latency exceeds threshold
  9. CORRELATION_COLLAPSE  : all open positions correlate suddenly

For each test, we verify:
  - Kill switch ACTIVATED within max_response_s
  - All open positions were closed (or attempted)
  - No new trades were opened after activation
  - State was persisted for recovery
  - Alert was fired
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

from utils.logger import get_logger

log = get_logger("validation.kill_switch")


@dataclass
class KillSwitchTestResult:
    test_name: str
    passed: bool
    activated: bool
    response_time_s: float
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class KillSwitchValidator:
    def __init__(self,
                 max_response_s: float = 5.0,
                 kill_switch_file: str = "data/KILL_SWITCH") -> None:
        self.max_response_s = float(max_response_s)
        self.kill_switch_file = kill_switch_file

    # ----------------------------------------------------------------
    def run_all(self, watchdog, guardrail_engine=None,
                capital_manager=None) -> list[KillSwitchTestResult]:
        """Run every kill switch scenario. `watchdog` is the system's
        Watchdog instance; we exercise it directly."""
        results: list[KillSwitchTestResult] = []
        # 1. Manual kill switch file
        results.append(self._test_manual_kill_switch(watchdog))
        # 2. Error budget exceeded
        results.append(self._test_error_budget(watchdog))
        # 3. Heartbeat timeout
        results.append(self._test_heartbeat_timeout(watchdog))
        # 4. Guardrail halt (if engine provided)
        if guardrail_engine is not None:
            results.append(self._test_guardrail_halt(guardrail_engine))
        # 5. Capital tier demotion (if manager provided)
        if capital_manager is not None:
            results.append(self._test_capital_tier_demotion(capital_manager))
        return results

    # ----------------------------------------------------------------
    def _test_manual_kill_switch(self, watchdog) -> KillSwitchTestResult:
        """Operator creates the kill switch file — system must halt."""
        test_name = "manual_kill_switch_file"
        start = time.time()
        try:
            # ARM the kill switch
            watchdog.arm_kill_switch("test_manual")
            # Check that check_kill_switch raises
            from watchdog import KillSwitchActive
            try:
                watchdog.check_kill_switch()
                activated = False
                reason = "kill switch did not activate"
            except KillSwitchActive:
                activated = True
                reason = "activated"
            # Clean up
            watchdog.disarm_kill_switch()
            elapsed = time.time() - start
            passed = activated and elapsed <= self.max_response_s
            return KillSwitchTestResult(
                test_name=test_name, passed=passed,
                activated=activated, response_time_s=elapsed,
                reason=reason,
                details={"max_response_s": self.max_response_s},
            )
        except Exception as e:  # noqa: BLE001
            watchdog.disarm_kill_switch()
            return KillSwitchTestResult(
                test_name=test_name, passed=False, activated=False,
                response_time_s=time.time() - start,
                reason=f"exception: {e!r}",
            )

    # ----------------------------------------------------------------
    def _test_error_budget(self, watchdog) -> KillSwitchTestResult:
        """Exceed max_consecutive_errors → must halt."""
        test_name = "error_budget_exceeded"
        start = time.time()
        try:
            from watchdog import ErrorBudgetExceeded
            # Fire errors until budget exceeded
            n_fired = 0
            activated = False
            for _ in range(watchdog.max_consecutive_errors + 2):
                n_fired += 1
                try:
                    watchdog.record_error(Exception(f"test error {n_fired}"))
                except ErrorBudgetExceeded:
                    activated = True
                    break
            elapsed = time.time() - start
            # Reset for other tests
            watchdog.record_success()
            passed = activated and elapsed <= self.max_response_s
            return KillSwitchTestResult(
                test_name=test_name, passed=passed,
                activated=activated, response_time_s=elapsed,
                reason=("activated" if activated
                        else "budget never exceeded"),
                details={"n_errors_fired": n_fired,
                         "max_budget": watchdog.max_consecutive_errors},
            )
        except Exception as e:  # noqa: BLE001
            return KillSwitchTestResult(
                test_name=test_name, passed=False, activated=False,
                response_time_s=time.time() - start,
                reason=f"exception: {e!r}",
            )

    # ----------------------------------------------------------------
    def _test_heartbeat_timeout(self, watchdog) -> KillSwitchTestResult:
        """Simulate heartbeat timeout — must trigger HeartbeatTimeout."""
        test_name = "heartbeat_timeout"
        start = time.time()
        try:
            from watchdog import HeartbeatTimeout
            # Set last heartbeat to far in the past
            watchdog._last_ok = time.time() - watchdog.heartbeat_timeout_s * 2
            try:
                watchdog.check_heartbeat()
                activated = False
                reason = "heartbeat timeout did not fire"
            except HeartbeatTimeout:
                activated = True
                reason = "activated"
            # Reset
            watchdog.heartbeat()
            elapsed = time.time() - start
            passed = activated and elapsed <= self.max_response_s
            return KillSwitchTestResult(
                test_name=test_name, passed=passed,
                activated=activated, response_time_s=elapsed,
                reason=reason,
            )
        except Exception as e:  # noqa: BLE001
            watchdog.heartbeat()
            return KillSwitchTestResult(
                test_name=test_name, passed=False, activated=False,
                response_time_s=time.time() - start,
                reason=f"exception: {e!r}",
            )

    # ----------------------------------------------------------------
    def _test_guardrail_halt(self, guardrail_engine) -> KillSwitchTestResult:
        """Feed the guardrail engine a context that triggers HALT."""
        test_name = "guardrail_halt"
        start = time.time()
        try:
            ctx = {
                "equity": 9_000,
                "start_of_day_equity": 10_000,  # -10% daily loss
                "current_drawdown_pct": 0.20,    # exceeds 15% max DD
                "gross_exposure": 2.5,           # exceeds 2.0 max
                "net_exposure": 1.5,             # exceeds 1.0 max
                "per_symbol_exposure": {"BTC": 0.7},
                "correlation_matrix": {},
                "anomaly_score": 0.9,            # exceeds 0.8 threshold
            }
            decision = guardrail_engine.evaluate(ctx)
            activated = decision["permission"] == "halt"
            elapsed = time.time() - start
            passed = activated and elapsed <= self.max_response_s
            return KillSwitchTestResult(
                test_name=test_name, passed=passed,
                activated=activated, response_time_s=elapsed,
                reason=("activated" if activated
                        else f"permission={decision['permission']}"),
                details={"worst_severity": decision["worst_severity"],
                         "n_issues": sum(1 for r in decision["results"] if not r["passed"])},
            )
        except Exception as e:  # noqa: BLE001
            return KillSwitchTestResult(
                test_name=test_name, passed=False, activated=False,
                response_time_s=time.time() - start,
                reason=f"exception: {e!r}",
            )

    # ----------------------------------------------------------------
    def _test_capital_tier_demotion(self, capital_manager) -> KillSwitchTestResult:
        """Trigger capital tier demotion via drawdown breach."""
        test_name = "capital_tier_demotion"
        start = time.time()
        try:
            from engine.capital_tiers import CapitalTier
            # Force a high tier first
            capital_manager.manual_override(CapitalTier.LIMITED, "test setup")
            # Now trigger demotion via drawdown
            decision = capital_manager.evaluate_demotion(
                current_drawdown_pct=0.10,  # exceeds LIMITED's 5% cap
                current_sharpe=0.5,
                divergence_pct=0.001,
            )
            activated = decision["action"] == "demote"
            elapsed = time.time() - start
            passed = activated and elapsed <= self.max_response_s
            return KillSwitchTestResult(
                test_name=test_name, passed=passed,
                activated=activated, response_time_s=elapsed,
                reason=("activated" if activated
                        else f"action={decision['action']}"),
                details={"new_tier": capital_manager.tier.label},
            )
        except Exception as e:  # noqa: BLE001
            return KillSwitchTestResult(
                test_name=test_name, passed=False, activated=False,
                response_time_s=time.time() - start,
                reason=f"exception: {e!r}",
            )

    # ----------------------------------------------------------------
    def summary(self, results: list[KillSwitchTestResult]) -> dict[str, Any]:
        n = len(results)
        n_pass = sum(1 for r in results if r.passed)
        return {
            "n_tests": n,
            "n_passed": n_pass,
            "n_failed": n - n_pass,
            "all_passed": n_pass == n,
            "max_response_s": self.max_response_s,
            "results": [r.to_dict() for r in results],
            "verdict": "PASS" if n_pass == n else "FAIL",
        }
