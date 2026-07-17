"""engine.recovery
=====================================================================
Day 27 — Crash recovery.

On startup, the bot tries to reconcile its persisted state with what
the broker reports. Three scenarios:

  1. State file missing → fresh start (no positions to recover)
  2. State file present, broker has matching positions → resume
  3. State file present, broker has DIFFERENT positions → reconcile
     (log every discrepancy, keep broker's view as source of truth
     but flag orphaned state entries for operator review)

The recovery module is intentionally conservative: it NEVER sends
orders. It only updates in-memory state. The operator must approve
any corrective action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("engine.recovery")


@dataclass
class RecoveryReport:
    """Summary of what recovery did (or didn't) do."""
    state_file_found: bool = False
    state_file_valid: bool = False
    broker_positions_count: int = 0
    state_positions_count: int = 0
    matched: list[int] = field(default_factory=list)
    orphaned_in_state: list[int] = field(default_factory=list)
    orphaned_in_broker: list[int] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_file_found": self.state_file_found,
            "state_file_valid": self.state_file_valid,
            "broker_positions_count": self.broker_positions_count,
            "state_positions_count": self.state_positions_count,
            "matched": list(self.matched),
            "orphaned_in_state": list(self.orphaned_in_state),
            "orphaned_in_broker": list(self.orphaned_in_broker),
            "actions_taken": list(self.actions_taken),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }


# ----------------------------------------------------------------------
class RecoveryManager:
    def __init__(self, portfolio, connector=None) -> None:
        self.portfolio = portfolio
        self.connector = connector

    # ----------------------------------------------------------------
    def recover(self, persisted_state: dict[str, Any]) -> RecoveryReport:
        """Reconcile persisted state with broker (if available).

        `persisted_state` is the dict loaded from `data/state.json`.
        """
        report = RecoveryReport()
        report.state_file_found = bool(persisted_state)

        # Broker positions (live source of truth if available)
        broker_positions: list[Any] = []
        if self.connector is not None:
            try:
                broker_positions = list(self.connector.positions() or [])
            except Exception as e:  # noqa: BLE001
                log.warning("recovery: cannot query broker positions: %r", e)
        report.broker_positions_count = len(broker_positions)

        # Persisted positions
        persisted_positions = persisted_state.get("positions", [])
        report.state_positions_count = len(persisted_positions)

        if not persisted_positions:
            report.actions_taken.append("no persisted positions — fresh start")
            return report

        # Reconcile by ticket
        broker_tickets = {int(getattr(p, "ticket", 0)) for p in broker_positions}
        state_tickets = {int(p.get("ticket", 0)) for p in persisted_positions if p.get("ticket")}
        report.matched = list(broker_tickets & state_tickets)
        report.orphaned_in_state = list(state_tickets - broker_tickets)
        report.orphaned_in_broker = list(broker_tickets - state_tickets)

        # Reload matched positions into the portfolio manager.
        # When there is no broker connector (paper mode, Linux dev, etc.),
        # we trust the persisted state as ground truth and restore
        # every persisted position.
        restore_set = set(report.matched) if broker_positions else state_tickets
        for p in persisted_positions:
            ticket = int(p.get("ticket", 0))
            if ticket in restore_set:
                try:
                    self.portfolio.open_position(
                        symbol=p["symbol"],
                        side=p.get("side", "long"),
                        lots=float(p.get("lots", 0.0)),
                        entry_price=float(p.get("entry_price", 0.0)),
                        strategy=p.get("strategy", ""),
                        stop=float(p.get("stop", 0.0)),
                        take=float(p.get("take", 0.0)),
                        atr_at_open=float(p.get("atr_at_open", 0.0)),
                        ticket=ticket,
                    )
                    report.actions_taken.append(f"restored ticket={ticket}")
                except Exception as e:  # noqa: BLE001
                    log.warning("recovery: failed to restore ticket %s: %r", ticket, e)

        if report.orphaned_in_state and broker_positions:
            # Only warn about orphans when we actually have a broker to
            # compare against — in paper mode every state position is
            # treated as authoritative and "orphan" is misleading.
            report.actions_taken.append(
                f"WARNING: {len(report.orphaned_in_state)} positions in state "
                f"not found at broker — manual review needed"
            )
            log.error("Recovery orphans in state: %s", report.orphaned_in_state)
        if report.orphaned_in_broker:
            report.actions_taken.append(
                f"WARNING: {len(report.orphaned_in_broker)} positions at broker "
                f"not in state — likely opened externally"
            )
            log.warning("Recovery orphans at broker: %s", report.orphaned_in_broker)

        return report
