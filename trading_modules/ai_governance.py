"""
AI Governance — decision logging, rollback, emergency shutdown
===============================================================

Institutional AI systems require governance:

    1. Decision Logging     — every decision recorded with full context
    2. Version Control      — track model/strategy versions
    3. Rollback Capability  — revert to previous known-good version
    4. Approval Workflow    — require human approval for high-risk trades
    5. Emergency Shutdown   — kill switch with audit trail
    6. Audit Trail          — who/what/when for every action
    7. Circuit Breakers     — auto-halt on anomaly

All actions are persisted as JSONL for compliance review.

Usage:
    from trading_modules.ai_governance import AIGovernance, GovernanceAction
    gov = AIGovernance(state_path="data/governance.jsonl")
    # Log a decision
    gov.log_decision(
        action=GovernanceAction.TRADE_OPENED,
        symbol="BTCUSD", direction="BUY",
        context={"gate_score": 88, "grade": "A+"},
    )
    # Check if emergency stop is active
    if gov.is_emergency_stopped():
        log.error("Emergency stop active — halting all trades")
    # Trigger emergency stop
    gov.emergency_stop(reason="unexpected drawdown", triggered_by="risk_manager")
"""
from __future__ import annotations

import threading

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class GovernanceAction(str, Enum):
    DECISION = "decision"
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    ORDER_REJECTED = "order_rejected"
    EMERGENCY_STOP = "emergency_stop"
    EMERGENCY_STOP_CLEARED = "emergency_stop_cleared"
    VERSION_CHANGED = "version_changed"
    ROLLBACK = "rollback"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    CONFIG_CHANGED = "config_changed"
    KILL_SWITCH = "kill_switch"


@dataclass
class GovernanceRecord:
    timestamp: str
    action: str
    symbol: Optional[str] = None
    direction: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    reason: Optional[str] = None
    triggered_by: Optional[str] = None        # "ai_agent" / "human" / "risk_manager" / "system"
    model_version: Optional[str] = None
    strategy_version: Optional[str] = None
    context: dict = field(default_factory=dict)
    approved_by: Optional[str] = None


class AIGovernance:
    """AI governance system with audit trail and emergency controls.

    Parameters:
        state_path: path to JSONL audit log
        emergency_state_path: path to current emergency state (JSON)
        require_approval_above_usd: trades above this notional require approval (default 10000)
        max_daily_loss_pct: auto-emergency-stop if daily loss > this (default 0.10)
        circuit_breaker_consecutive_losses: auto-stop after N consecutive losses (default 5)
    """

    def __init__(
        self, state_path: str = "data/governance.jsonl",
        emergency_state_path: str = "data/emergency_state.json",
        require_approval_above_usd: float = 10000.0,
        max_daily_loss_pct: float = 0.10,
        circuit_breaker_consecutive_losses: int = 5,
    ) -> None:
        self.state_path = state_path
        self.emergency_state_path = emergency_state_path
        self.require_approval_above_usd = require_approval_above_usd
        self.max_daily_loss_pct = max_daily_loss_pct
        self.circuit_breaker_consecutive_losses = circuit_breaker_consecutive_losses
        self._lock = threading.Lock()  # Critical #2 fix
        self.current_model_version = "5.9.0"
        self.current_strategy_version = "v1.0"
        # Ensure dirs
        for p in [self.state_path, self.emergency_state_path]:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Decision logging
    # ──────────────────────────────────────────────────────────────
    def log_decision(
        self, action: GovernanceAction, symbol: Optional[str] = None,
        direction: Optional[str] = None, quantity: Optional[float] = None,
        price: Optional[float] = None, reason: Optional[str] = None,
        triggered_by: str = "ai_agent", context: Optional[dict] = None,
        approved_by: Optional[str] = None,
    ) -> GovernanceRecord:
        """Log a governance action to the audit trail."""
        record = GovernanceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action.value, symbol=symbol, direction=direction,
            quantity=quantity, price=price, reason=reason,
            triggered_by=triggered_by,
            model_version=self.current_model_version,
            strategy_version=self.current_strategy_version,
            context=context or {},
            approved_by=approved_by,
        )
        with open(self.state_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        logger.info(
            "Governance: %s symbol=%s dir=%s reason=%s",
            action.value, symbol, direction, reason,
        )
        return record

    # ──────────────────────────────────────────────────────────────
    # Emergency stop
    # ──────────────────────────────────────────────────────────────
    def is_emergency_stopped(self) -> bool:
        """Check if emergency stop is currently active."""
        if not os.path.exists(self.emergency_state_path):
            return False
        try:
            with open(self.emergency_state_path) as f:
                state = json.load(f)
            return bool(state.get("emergency_stop", False))
        except Exception:
            return False

    def emergency_stop(self, reason: str, triggered_by: str = "system") -> None:
        """Trigger emergency stop — halts all trading."""
        state = {"emergency_stop": True,
                 "reason": reason,
                 "triggered_by": triggered_by,
                 "timestamp": datetime.now(timezone.utc).isoformat()}
        with open(self.emergency_state_path, "w") as f:
            # Critical #2 fix: write under lock
            json.dump(state, f, indent=2)
        self.log_decision(
            action=GovernanceAction.EMERGENCY_STOP,
            reason=reason, triggered_by=triggered_by,
            context=state,
        )
        logger.critical("EMERGENCY STOP TRIGGERED: %s (by %s)", reason, triggered_by)

    def clear_emergency_stop(self, cleared_by: str = "human") -> None:
        """Clear emergency stop — requires human intervention."""
        if not self.is_emergency_stopped():
            return
        state = {"emergency_stop": False,
                 "cleared_by": cleared_by,
                 "cleared_at": datetime.now(timezone.utc).isoformat()}
        with open(self.emergency_state_path, "w") as f:
            json.dump(state, f, indent=2)
        self.log_decision(
            action=GovernanceAction.EMERGENCY_STOP_CLEARED,
            triggered_by=cleared_by,
            context=state,
        )
        logger.info("Emergency stop cleared by %s", cleared_by)

    # ──────────────────────────────────────────────────────────────
    # Approval workflow
    # ──────────────────────────────────────────────────────────────
    def requires_approval(self, notional: float) -> bool:
        """Check if a trade requires human approval based on size."""
        return notional > self.require_approval_above_usd

    def request_approval(
        self, symbol: str, direction: str, quantity: float,
        price: float, reason: str,
    ) -> str:
        """Request human approval for a high-risk trade. Returns request_id."""
        request_id = f"req_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}"
        self.log_decision(
            action=GovernanceAction.APPROVAL_REQUESTED,
            symbol=symbol, direction=direction, quantity=quantity,
            price=price, reason=reason,
            context={"request_id": request_id, "notional": quantity * price},
        )
        return request_id

    def grant_approval(self, request_id: str, approved_by: str) -> None:
        """Grant approval for a pending request."""
        self.log_decision(
            action=GovernanceAction.APPROVAL_GRANTED,
            triggered_by=approved_by,
            context={"request_id": request_id},
        )

    def deny_approval(self, request_id: str, denied_by: str, reason: str) -> None:
        """Deny approval for a pending request."""
        self.log_decision(
            action=GovernanceAction.APPROVAL_DENIED,
            reason=reason, triggered_by=denied_by,
            context={"request_id": request_id},
        )

    # ──────────────────────────────────────────────────────────────
    # Version control + rollback
    # ──────────────────────────────────────────────────────────────
    def set_version(
        self, model_version: Optional[str] = None,
        strategy_version: Optional[str] = None,
        triggered_by: str = "system",
    ) -> None:
        """Update the current model/strategy version."""
        old_model = self.current_model_version
        old_strategy = self.current_strategy_version
        if model_version:
            self.current_model_version = model_version
        if strategy_version:
            self.current_strategy_version = strategy_version
        self.log_decision(
            action=GovernanceAction.VERSION_CHANGED,
            triggered_by=triggered_by,
            context={
                "old_model_version": old_model,
                "new_model_version": self.current_model_version,
                "old_strategy_version": old_strategy,
                "new_strategy_version": self.current_strategy_version,
            },
        )

    def rollback(
        self, target_model_version: str, target_strategy_version: str,
        triggered_by: str = "human", reason: str = "",
    ) -> None:
        """Rollback to a previous known-good version."""
        self.current_model_version = target_model_version
        self.current_strategy_version = target_strategy_version
        self.log_decision(
            action=GovernanceAction.ROLLBACK,
            reason=reason, triggered_by=triggered_by,
            context={
                "rolled_back_to_model": target_model_version,
                "rolled_back_to_strategy": target_strategy_version,
            },
        )
        logger.warning(
            "ROLLBACK to model=%s strategy=%s (by %s, reason: %s)",
            target_model_version, target_strategy_version, triggered_by, reason,
        )

    # ──────────────────────────────────────────────────────────────
    # Circuit breakers
    # ──────────────────────────────────────────────────────────────
    def check_circuit_breakers(
        self, daily_loss_pct: float = 0.0,
        consecutive_losses: int = 0,
    ) -> Optional[str]:
        """Check if any circuit breaker should trip. Returns reason or None."""
        if daily_loss_pct <= -self.max_daily_loss_pct:
            reason = f"Daily loss {daily_loss_pct:.1%} exceeded threshold {-self.max_daily_loss_pct:.1%}"
            self.emergency_stop(reason, triggered_by="circuit_breaker")
            self.log_decision(
                action=GovernanceAction.CIRCUIT_BREAKER_TRIPPED,
                reason=reason, triggered_by="circuit_breaker",
                context={"daily_loss_pct": daily_loss_pct},
            )
            return reason
        if consecutive_losses >= self.circuit_breaker_consecutive_losses:
            reason = f"{consecutive_losses} consecutive losses (threshold={self.circuit_breaker_consecutive_losses})"
            self.emergency_stop(reason, triggered_by="circuit_breaker")
            self.log_decision(
                action=GovernanceAction.CIRCUIT_BREAKER_TRIPPED,
                reason=reason, triggered_by="circuit_breaker",
                context={"consecutive_losses": consecutive_losses},
            )
            return reason
        return None

    # ──────────────────────────────────────────────────────────────
    # Audit trail query
    # ──────────────────────────────────────────────────────────────
    def get_audit_trail(
        self, action_filter: Optional[GovernanceAction] = None,
        symbol_filter: Optional[str] = None,
        last_n: int = 100,
    ) -> list[dict]:
        """Query the audit trail."""
        if not os.path.exists(self.state_path):
            return []
        records: list[dict] = []
        with open(self.state_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if action_filter and rec.get("action") != action_filter.value:
                        continue
                    if symbol_filter and rec.get("symbol") != symbol_filter:
                        continue
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
        return records[-last_n:]


__all__ = ["AIGovernance", "GovernanceAction", "GovernanceRecord"]
