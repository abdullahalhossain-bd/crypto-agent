"""validation package — Phase 6 proof layers (Day 91+).

Closes the gap between "architecturally complete" and "real-money ready"
by adding the proof layers institutions require before go-live:

  - shadow_live_engine  : long-duration shadow tracking with outcome matching
  - execution_metrics   : real fill / slippage / latency distribution collector
  - edge_proof          : statistical edge validation (bootstrap CIs, t-tests)
  - survival_test       : strategy robustness through historical regime changes
  - kill_switch_validator : automated kill switch scenario tests
  - readiness_gate      : institutional GO / NO-GO gate with explicit metrics
  - operator_runbook    : incident response procedures
"""
from validation.shadow_live_engine import ShadowLiveEngine, ShadowOutcome  # noqa: F401
from validation.execution_metrics import (  # noqa: F401
    ExecutionMetricsCollector, ExecutionDistribution,
)
from validation.edge_proof import EdgeProofEngine, EdgeProofResult  # noqa: F401
from validation.survival_test import SurvivalTestRunner, SurvivalReport  # noqa: F401
from validation.kill_switch_validator import (  # noqa: F401
    KillSwitchValidator, KillSwitchTestResult,
)
from validation.readiness_gate import ReadinessGate, GateVerdict  # noqa: F401
from validation.operator_runbook import OperatorRunbook  # noqa: F401

__all__ = [
    "ShadowLiveEngine", "ShadowOutcome",
    "ExecutionMetricsCollector", "ExecutionDistribution",
    "EdgeProofEngine", "EdgeProofResult",
    "SurvivalTestRunner", "SurvivalReport",
    "KillSwitchValidator", "KillSwitchTestResult",
    "ReadinessGate", "GateVerdict",
    "OperatorRunbook",
]
