"""engine.capital_tiers
=====================================================================
Day 31-33 — Capital Tier System.

Institutional principle: NEVER deploy full capital on day one.
Capital is released in tiers as the system proves itself.

  Tier 0  PAPER      0% real capital — pure simulation
  Tier 1  MICRO      0.1–1% of bankroll — prove live execution works
  Tier 2  LIMITED    1–5% of bankroll — prove risk controls hold
  Tier 3  FULL       5–20% of bankroll — calibrated allocation

Promotion rules (every tier has explicit, automated criteria):
  - Min cycles at current tier
  - Min realised Sharpe
  - Max drawdown observed
  - Live-vs-paper divergence below threshold
  - No kill-switch events

Demotion rules:
  - Drawdown exceeds tier-specific limit
  - Sharpe collapses
  - Divergence exceeds threshold
  - Operator manual override

The tier is persisted in `data/capital_tier.json` so a restart doesn't
silently re-promote the bot.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("engine.capital_tiers")


class CapitalTier(IntEnum):
    PAPER = 0
    MICRO = 1
    LIMITED = 2
    FULL = 3

    @property
    def label(self) -> str:
        return {0: "PAPER", 1: "MICRO", 2: "LIMITED", 3: "FULL"}[int(self)]


# Default capital fraction per tier
TIER_FRACTIONS: dict[CapitalTier, float] = {
    CapitalTier.PAPER:  0.0,
    CapitalTier.MICRO:  0.005,    # 0.5%
    CapitalTier.LIMITED: 0.025,   # 2.5%
    CapitalTier.FULL:    0.10,    # 10%
}

# Default tier limits (per-tier hard caps)
TIER_LIMITS: dict[CapitalTier, dict[str, float]] = {
    CapitalTier.PAPER:   {"max_daily_loss_pct": 1.0, "max_dd_pct": 1.0,
                          "max_position_lots": 0.0},
    CapitalTier.MICRO:   {"max_daily_loss_pct": 0.005, "max_dd_pct": 0.02,
                          "max_position_lots": 0.05},
    CapitalTier.LIMITED: {"max_daily_loss_pct": 0.02, "max_dd_pct": 0.05,
                          "max_position_lots": 0.5},
    CapitalTier.FULL:    {"max_daily_loss_pct": 0.05, "max_dd_pct": 0.12,
                          "max_position_lots": 5.0},
}


@dataclass
class TierPromotionCriteria:
    """Automated criteria to promote from one tier to the next."""
    min_cycles_at_tier: int = 500          # ~ 40 minutes @ 5s cycles
    min_sharpe: float = 0.5
    max_drawdown_pct: float = 0.03
    max_live_paper_divergence_pct: float = 0.01
    max_kill_switch_events: int = 0
    min_win_rate: float = 0.40


# ----------------------------------------------------------------------
@dataclass
class CapitalTierState:
    """Persistable state of the tier system."""
    current_tier: int = 0
    tier_entered_at: float = 0.0
    cycles_at_tier: int = 0
    promotion_history: list[dict[str, Any]] = field(default_factory=list)
    demotion_history: list[dict[str, Any]] = field(default_factory=list)
    last_evaluation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CapitalTierState":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ----------------------------------------------------------------------
class CapitalTierManager:
    """Manages tier transitions with explicit, auditable criteria."""

    def __init__(
        self,
        state: Optional[CapitalTierState] = None,
        state_path: str = "data/capital_tier.json",
        fractions: Optional[dict[CapitalTier, float]] = None,
        limits: Optional[dict[CapitalTier, dict[str, float]]] = None,
        criteria: Optional[TierPromotionCriteria] = None,
    ) -> None:
        self.state = state or CapitalTierState(
            current_tier=int(CapitalTier.PAPER),
            tier_entered_at=time.time(),
        )
        self.state_path = state_path
        self.fractions = fractions or dict(TIER_FRACTIONS)
        # Store tier limits privately to avoid clashing with the public
        # `get_limits()` accessor.
        self._tier_limits = limits or {k: dict(v) for k, v in TIER_LIMITS.items()}
        self.criteria = criteria or TierPromotionCriteria()
        self._load()

    # ----------------------------------------------------------------
    @property
    def tier(self) -> CapitalTier:
        return CapitalTier(self.state.current_tier)

    @property
    def capital_fraction(self) -> float:
        return self.fractions[self.tier]

    def get_limits(self) -> dict[str, float]:
        return self._tier_limits[CapitalTier(self.state.current_tier)]

    def get_fraction(self) -> float:
        return self.fractions[CapitalTier(self.state.current_tier)]

    # ----------------------------------------------------------------
    def record_cycle(self) -> None:
        self.state.cycles_at_tier += 1

    # ----------------------------------------------------------------
    def evaluate_promotion(
        self,
        sharpe: float,
        max_drawdown_pct: float,
        live_paper_divergence_pct: float,
        kill_switch_events: int,
        win_rate: float,
    ) -> dict[str, Any]:
        """Check if we should promote. Returns the decision audit."""
        current = self.tier
        next_tier = CapitalTier(min(int(current) + 1, int(CapitalTier.FULL)))
        if next_tier == current:
            return {"action": "no_promotion", "reason": "already at top tier"}

        c = self.criteria
        checks = {
            "min_cycles":       self.state.cycles_at_tier >= c.min_cycles_at_tier,
            "min_sharpe":       sharpe >= c.min_sharpe,
            "max_drawdown":     max_drawdown_pct <= c.max_drawdown_pct,
            "max_divergence":   live_paper_divergence_pct <= c.max_live_paper_divergence_pct,
            "no_kill_switch":   kill_switch_events <= c.max_kill_switch_events,
            "min_win_rate":     win_rate >= c.min_win_rate,
        }
        all_pass = all(checks.values())
        decision = {
            "ts": time.time(),
            "from_tier": current.label,
            "to_tier": next_tier.label,
            "checks": checks,
            "metrics": {
                "sharpe": sharpe,
                "max_drawdown_pct": max_drawdown_pct,
                "live_paper_divergence_pct": live_paper_divergence_pct,
                "kill_switch_events": kill_switch_events,
                "win_rate": win_rate,
                "cycles_at_tier": self.state.cycles_at_tier,
            },
            "action": "promote" if all_pass else "hold",
        }
        if all_pass:
            self._promote(next_tier, decision)
        self.state.last_evaluation = time.time()
        self._save()
        return decision

    def evaluate_demotion(
        self,
        current_drawdown_pct: float,
        current_sharpe: float,
        divergence_pct: float,
        reason: str = "",
    ) -> dict[str, Any]:
        """Auto-demote if any hard limit is breached."""
        limits = self.get_limits()
        breach = (
            current_drawdown_pct > limits["max_dd_pct"]
            or divergence_pct > 0.03  # 3% divergence = paper vs live disagree
            or (self.tier != CapitalTier.PAPER and current_sharpe < -0.5)
        )
        decision = {
            "ts": time.time(),
            "from_tier": self.tier.label,
            "reason": reason or "auto-eval",
            "current_drawdown_pct": current_drawdown_pct,
            "current_sharpe": current_sharpe,
            "divergence_pct": divergence_pct,
            "limits": limits,
            "action": "demote" if breach else "hold",
        }
        if breach:
            new_tier = CapitalTier(max(0, int(self.tier) - 1))
            self._demote(new_tier, decision)
        self._save()
        return decision

    def manual_override(self, new_tier: CapitalTier,
                        reason: str = "manual") -> None:
        """Operator-initiated tier change (bypass criteria)."""
        decision = {
            "ts": time.time(),
            "from_tier": self.tier.label,
            "to_tier": new_tier.label,
            "reason": reason,
            "action": "manual",
        }
        if new_tier > self.tier:
            self._promote(new_tier, decision)
        else:
            self._demote(new_tier, decision)
        self._save()

    # ----------------------------------------------------------------
    def _promote(self, new_tier: CapitalTier, decision: dict[str, Any]) -> None:
        log.info("CAPITAL PROMOTION %s -> %s", self.tier.label, new_tier.label)
        self.state.promotion_history.append(decision)
        self.state.current_tier = int(new_tier)
        self.state.tier_entered_at = time.time()
        self.state.cycles_at_tier = 0

    def _demote(self, new_tier: CapitalTier, decision: dict[str, Any]) -> None:
        log.warning("CAPITAL DEMOTION %s -> %s reason=%s",
                    self.tier.label, new_tier.label, decision.get("reason"))
        self.state.demotion_history.append(decision)
        self.state.current_tier = int(new_tier)
        self.state.tier_entered_at = time.time()
        self.state.cycles_at_tier = 0

    # ----------------------------------------------------------------
    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        tmp = self.state_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception as e:  # noqa: BLE001
            log.error("capital tier state save failed: %r", e)

    def _load(self) -> None:
        if not os.path.isfile(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                self.state = CapitalTierState.from_dict(json.load(f))
            log.info("Capital tier loaded: %s (cycles at tier=%d)",
                     self.tier.label, self.state.cycles_at_tier)
        except Exception as e:  # noqa: BLE001
            log.warning("capital tier load failed: %r — starting fresh", e)
