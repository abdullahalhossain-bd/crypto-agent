"""
Kill Conditions Module — 4-Gate Risk Protection
=================================================

Implements a 4-gate kill switch system that halts trading when risk
thresholds are breached. Each gate is evaluated in order; the first
failing gate produces the verdict.

Gates (evaluated in order):
  1. MAX_CUMULATIVE_LOSS ($500 default) — hard dollar loss cap
     Hit → WAIT 7 days
  2. MIN_SHARPE_ROLLING_14D (0.0 default) — 14-day rolling Sharpe
     must be > 0. Below → WAIT 3 days
  3. MAX_DRAWDOWN_PCT (15% default) — peak-to-trough intraday
     Hit → WAIT 7 days
  4. MIN_BRIER_ROLLING_30D (0.20 default) — calibration metric
     required to ENTER real money. Above → GATED (not "wait")

Real-money gate:
  is_ready_for_real_money(state) is the stricter check specifically
  for the paper → real transition. Requires ALL of: min-30d-Brier
  under threshold, positive rolling Sharpe, no active kill, and
  >= 30 days of paper trade history.

Source: Orallexa (review #27)
Design principle: "False-negative (system stays in WAIT slightly too
long) is preferred over false-positive (system trades through danger)."

Usage:
    from kill_conditions import KillConditions, PortfolioState

    kc = KillConditions()
    state = PortfolioState(
        cumulative_loss_usd=450,
        rolling_sharpe_14d=1.2,
        current_drawdown_pct=8.0,
        rolling_brier_30d=0.15,
        paper_trade_days=45,
    )

    decision = kc.check(state)
    if not decision.can_trade:
        print(f"KILL: {decision.trigger_reason}")
        print(f"Cooldown until: {decision.cooldown_until_utc}")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Any


# ═══════════════════════════════════════════════════════════════
# Threshold Constants
# ═══════════════════════════════════════════════════════════════

# Gate 1: Hard dollar cap on cumulative loss
MAX_CUMULATIVE_LOSS_USD = 500.0
MAX_LOSS_COOLDOWN_DAYS = 7

# Gate 2: 14-day rolling Sharpe must be positive
MIN_SHARPE_ROLLING_14D = 0.0
MIN_SHARPE_COOLDOWN_DAYS = 3

# Gate 3: Maximum drawdown percentage
MAX_DRAWDOWN_PCT = 0.15
MAX_DRAWDOWN_COOLDOWN_DAYS = 7

# Gate 4: Brier score for real-money gate (lower = better calibration)
MIN_BRIER_ROLLING_30D = 0.20  # Max acceptable Brier

# Real-money gate requirements
MIN_PAPER_TRADE_DAYS = 30


@dataclass
class PortfolioState:
    """Current portfolio state for kill condition evaluation."""
    cumulative_loss_usd: float = 0.0
    rolling_sharpe_14d: float = 0.0
    current_drawdown_pct: float = 0.0
    rolling_brier_30d: float = 0.25  # Default above threshold (not ready)
    paper_trade_days: int = 0
    trade_count_14d: int = 0  # Closed trades in last 14 days (for cold-start bypass)
    # Active cooldown tracking
    active_cooldown_until: Optional[str] = None  # ISO 8601 UTC
    active_cooldown_reason: Optional[str] = None


@dataclass
class KillDecision:
    """Result of kill condition check."""
    can_trade: bool
    state: str  # "OK" | "WAIT" | "GATED"
    trigger_reason: Optional[str] = None
    cooldown_until_utc: Optional[str] = None  # ISO 8601 UTC
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "can_trade": self.can_trade,
            "state": self.state,
            "trigger_reason": self.trigger_reason,
            "cooldown_until_utc": self.cooldown_until_utc,
            "checks": self.checks,
        }


class KillConditions:
    """
    4-gate kill switch system.

    Philosophy: "We prefer false-negative (system stays in WAIT slightly
    too long) over false-positive (system trades through danger)."
    """

    # Minimum closed trades in 14d before Gate 2 (Sharpe) is enforced.
    # Below this threshold the gate is SKIPPED (cold-start / insufficient data).
    MIN_TRADES_FOR_SHARPE_GATE: int = 10

    def __init__(
        self,
        max_cumulative_loss: float = MAX_CUMULATIVE_LOSS_USD,
        max_loss_cooldown_days: int = MAX_LOSS_COOLDOWN_DAYS,
        min_sharpe: float = MIN_SHARPE_ROLLING_14D,
        min_sharpe_cooldown_days: int = MIN_SHARPE_COOLDOWN_DAYS,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        max_drawdown_cooldown_days: int = MAX_DRAWDOWN_COOLDOWN_DAYS,
        min_brier: float = MIN_BRIER_ROLLING_30D,
        min_trades_for_sharpe_gate: int = 10,
        enforce_brier_gate: bool = False,
    ):
        self.max_cumulative_loss = max_cumulative_loss
        self.max_loss_cooldown_days = max_loss_cooldown_days
        self.min_sharpe = min_sharpe
        self.min_sharpe_cooldown_days = min_sharpe_cooldown_days
        self.max_drawdown_pct = max_drawdown_pct
        self.max_drawdown_cooldown_days = max_drawdown_cooldown_days
        self.min_brier = min_brier
        self.min_trades_for_sharpe_gate = min_trades_for_sharpe_gate
        self.enforce_brier_gate = enforce_brier_gate

    def check(self, state: PortfolioState) -> KillDecision:
        """
        Evaluate all 4 gates in order. First failure produces the verdict.

        Returns KillDecision with can_trade=False if any gate fails.
        """
        now = datetime.now(timezone.utc)
        checks = {}

        # Check if we're in an active cooldown
        if state.active_cooldown_until:
            try:
                cooldown_end = datetime.fromisoformat(
                    state.active_cooldown_until.replace("Z", "+00:00")
                )
                if now < cooldown_end:
                    return KillDecision(
                        can_trade=False,
                        state="WAIT",
                        trigger_reason=f"Active cooldown: {state.active_cooldown_reason}",
                        cooldown_until_utc=state.active_cooldown_until,
                        checks={"active_cooldown": False},
                    )
            except (ValueError, TypeError):
                pass  # Invalid cooldown timestamp, continue checks

        # Gate 1: Cumulative loss
        loss_ok = state.cumulative_loss_usd < self.max_cumulative_loss
        checks["cumulative_loss"] = loss_ok
        if not loss_ok:
            cooldown = now + timedelta(days=self.max_loss_cooldown_days)
            return KillDecision(
                can_trade=False,
                state="WAIT",
                trigger_reason=f"Cumulative loss ${state.cumulative_loss_usd:.2f} >= ${self.max_cumulative_loss:.2f}",
                cooldown_until_utc=cooldown.isoformat(),
                checks=checks,
            )

        # Gate 2: Rolling Sharpe (with cold-start bypass)
        # If fewer than min_trades_for_sharpe_gate closed trades exist in the
        # last 14 days, we skip this gate entirely — the Sharpe ratio is
        # statistically meaningless with too few observations, and enforcing
        # it creates a chicken-and-egg lock (need trades to prove Sharpe,
        # but need Sharpe > 0 to trade).
        if state.trade_count_14d < self.min_trades_for_sharpe_gate:
            checks["rolling_sharpe_14d"] = True  # skipped — insufficient data
        else:
            sharpe_ok = state.rolling_sharpe_14d > self.min_sharpe
            checks["rolling_sharpe_14d"] = sharpe_ok
            if not sharpe_ok:
                cooldown = now + timedelta(days=self.min_sharpe_cooldown_days)
                return KillDecision(
                    can_trade=False,
                    state="WAIT",
                    trigger_reason=f"14d rolling Sharpe {state.rolling_sharpe_14d:.2f} <= {self.min_sharpe}",
                    cooldown_until_utc=cooldown.isoformat(),
                    checks=checks,
                )

        # Gate 3: Drawdown
        dd_ok = state.current_drawdown_pct < self.max_drawdown_pct
        checks["max_drawdown"] = dd_ok
        if not dd_ok:
            cooldown = now + timedelta(days=self.max_drawdown_cooldown_days)
            return KillDecision(
                can_trade=False,
                state="WAIT",
                trigger_reason=f"Drawdown {state.current_drawdown_pct:.1%} >= {self.max_drawdown_pct:.0%}",
                cooldown_until_utc=cooldown.isoformat(),
                checks=checks,
            )

        # Gate 4: Brier score (GATED, not WAIT — requires operator override)
        # This gate is ONLY enforced in real-money mode (enforce_brier_gate=True).
        # In demo/paper mode it is always skipped — Brier calibration requires
        # 30+ days of trade outcomes and is meaningless for a fresh bot.
        # Use is_ready_for_real_money() for the strict paper→live transition check.
        if self.enforce_brier_gate:
            brier_ok = state.rolling_brier_30d <= self.min_brier
            checks["brier_30d"] = brier_ok
            if not brier_ok:
                return KillDecision(
                    can_trade=False,
                    state="GATED",
                    trigger_reason=f"30d Brier {state.rolling_brier_30d:.3f} > {self.min_brier} (requires operator override)",
                    checks=checks,
                )
        else:
            checks["brier_30d"] = True  # skipped — not enforcing in demo/paper

        # All gates passed
        return KillDecision(
            can_trade=True,
            state="OK",
            checks=checks,
        )

    def is_ready_for_real_money(self, state: PortfolioState) -> bool:
        """
        Stricter check for paper → real money transition.

        Requires ALL of:
        - 30-day Brier under threshold
        - Positive rolling Sharpe
        - No active kill (all gates pass)
        - >= 30 days of paper trade history
        """
        decision = self.check(state)
        return (
            decision.can_trade
            and state.rolling_brier_30d <= self.min_brier
            and state.rolling_sharpe_14d > 0
            and state.paper_trade_days >= MIN_PAPER_TRADE_DAYS
        )

    def save_state(self, state: PortfolioState, path: Path | str) -> None:
        """Persist portfolio state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(state), f, indent=2, default=str)

    def load_state(self, path: Path | str) -> PortfolioState:
        """Load portfolio state from JSON."""
        path = Path(path)
        if not path.exists():
            return PortfolioState()
        with open(path) as f:
            data = json.load(f)
        return PortfolioState(**data)
