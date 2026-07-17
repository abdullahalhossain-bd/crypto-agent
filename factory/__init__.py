"""factory package — strategy discovery CI/CD (Day 56-70)."""
from factory.strategy_ci import StrategyCI, CIPipelineResult  # noqa: F401
from factory.strategy_versioning import (  # noqa: F401
    StrategyVersionStore, StrategyVersion,
)
from factory.decay_detector import DecayDetector, DecayReport  # noqa: F401
from factory.auto_retirement import AutoRetirement, RetirementDecision  # noqa: F401

__all__ = [
    "StrategyCI", "CIPipelineResult",
    "StrategyVersionStore", "StrategyVersion",
    "DecayDetector", "DecayReport",
    "AutoRetirement", "RetirementDecision",
]
