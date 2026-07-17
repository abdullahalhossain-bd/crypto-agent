"""enhancements.trading_as_git
=====================================================================
Inspired by OpenAlice's TradingGit.

Trading-as-Git: instead of an AI agent firing orders directly, every
account operation is staged, committed, reviewed, and pushed through
an approval gate — exactly like git.

Flow:
    1. stage(operation)   — add an operation to the staging area
    2. commit(message)    — prepare the staged operations with a hash
    3. review()           — operator (or guard pipeline) reviews
    4. push()             — execute all committed operations
    5. reject(reason)     — reject the commit instead of pushing

Operations:
    - place_order(symbol, side, lots, price, sl, tp)
    - modify_order(ticket, new_sl, new_tp)
    - close_position(ticket)
    - cancel_order(ticket)

Every commit is hashed (SHA-256) and logged immutably. The guard
pipeline runs BEFORE push — it can reject the commit if risk limits
are breached.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from utils.logger import get_logger

log = get_logger("enhancements.trading_as_git")


class OperationAction(str, Enum):
    PLACE_ORDER = "place_order"
    MODIFY_ORDER = "modify_order"
    CLOSE_POSITION = "close_position"
    CANCEL_ORDER = "cancel_order"


class OperationStatus(str, Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    PUSHED = "pushed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class Operation:
    action: OperationAction
    symbol: str = ""
    side: str = ""             # buy / sell
    lots: float = 0.0
    price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    ticket: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "side": self.side,
            "lots": self.lots,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "ticket": self.ticket,
            "metadata": dict(self.metadata),
        }


@dataclass
class CommitEntry:
    hash: str
    message: str
    operations: list[Operation]
    timestamp: str
    parent_hash: Optional[str] = None
    status: OperationStatus = OperationStatus.COMMITTED
    push_results: list[dict[str, Any]] = field(default_factory=list)
    reject_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "message": self.message,
            "operations": [op.to_dict() for op in self.operations],
            "timestamp": self.timestamp,
            "parent_hash": self.parent_hash,
            "status": self.status.value,
            "push_results": list(self.push_results),
            "reject_reason": self.reject_reason,
        }


# ----------------------------------------------------------------------
class TradingGit:
    """Trading-as-Git operation manager."""

    def __init__(self,
                 executor: Optional[Callable[[Operation], dict[str, Any]]] = None,
                 guard_pipeline: Optional[Callable[[Operation], Optional[str]]] = None,
                 log_path: str = "data/trading_commits.jsonl") -> None:
        self.executor = executor
        self.guard_pipeline = guard_pipeline
        self.log_path = log_path
        self._staging: list[Operation] = []
        self._pending_message: Optional[str] = None
        self._pending_hash: Optional[str] = None
        self._commits: list[CommitEntry] = []
        self._head: Optional[str] = None
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # ----------------------------------------------------------------
    # git add
    # ----------------------------------------------------------------
    def stage(self, operation: Operation) -> dict[str, Any]:
        """Add an operation to the staging area."""
        self._staging.append(operation)
        log.info("STAGED %s %s %s lots=%.4f", operation.action.value,
                 operation.side, operation.symbol, operation.lots)
        return {
            "staged": True,
            "index": len(self._staging) - 1,
            "operation": operation.to_dict(),
        }

    def stage_place_order(self, symbol: str, side: str, lots: float,
                           price: float = 0.0, stop_loss: float = 0.0,
                           take_profit: float = 0.0, **meta: Any) -> dict[str, Any]:
        return self.stage(Operation(
            action=OperationAction.PLACE_ORDER,
            symbol=symbol, side=side, lots=lots, price=price,
            stop_loss=stop_loss, take_profit=take_profit, metadata=meta,
        ))

    def stage_close_position(self, ticket: int, **meta: Any) -> dict[str, Any]:
        return self.stage(Operation(
            action=OperationAction.CLOSE_POSITION, ticket=ticket, metadata=meta,
        ))

    # ----------------------------------------------------------------
    # git commit
    # ----------------------------------------------------------------
    def commit(self, message: str) -> dict[str, Any]:
        """Prepare the staged operations into a commit with a hash."""
        if not self._staging:
            raise ValueError("nothing to commit: staging area is empty")
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        commit_hash = self._generate_hash(message, self._staging, timestamp, self._head)
        self._pending_message = message
        self._pending_hash = commit_hash
        log.info("COMMIT prepared hash=%s ops=%d msg='%s'",
                 commit_hash, len(self._staging), message)
        return {
            "prepared": True,
            "hash": commit_hash,
            "message": message,
            "operation_count": len(self._staging),
        }

    # ----------------------------------------------------------------
    # git push
    # ----------------------------------------------------------------
    def push(self) -> dict[str, Any]:
        """Execute all committed operations through the guard pipeline + executor."""
        if not self._staging:
            raise ValueError("nothing to push: staging area is empty")
        if not self._pending_message or not self._pending_hash:
            raise ValueError("nothing to push: commit first")

        operations = list(self._staging)
        message = self._pending_message
        commit_hash = self._pending_hash

        # Run guard pipeline on each operation
        results: list[dict[str, Any]] = []
        all_ok = True
        for op in operations:
            # Guard check
            if self.guard_pipeline is not None:
                rejection = self.guard_pipeline(op)
                if rejection:
                    results.append({
                        "action": op.action.value,
                        "success": False,
                        "status": "rejected",
                        "error": f"[guard] {rejection}",
                    })
                    all_ok = False
                    continue
            # Execute
            if self.executor is not None:
                try:
                    raw = self.executor(op)
                    results.append({
                        "action": op.action.value,
                        "success": True,
                        "status": "pushed",
                        "result": raw,
                    })
                except Exception as e:  # noqa: BLE001
                    results.append({
                        "action": op.action.value,
                        "success": False,
                        "status": "failed",
                        "error": str(e),
                    })
                    all_ok = False
            else:
                # No executor — paper mode
                results.append({
                    "action": op.action.value,
                    "success": True,
                    "status": "paper",
                    "result": {"simulated": True},
                })

        # Record commit
        commit = CommitEntry(
            hash=commit_hash, message=message,
            operations=operations,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            parent_hash=self._head,
            status=OperationStatus.PUSHED if all_ok else OperationStatus.FAILED,
            push_results=results,
        )
        self._commits.append(commit)
        self._head = commit_hash
        self._persist(commit)
        # Clear staging
        self._staging.clear()
        self._pending_message = None
        self._pending_hash = None
        log.info("PUSH hash=%s ops=%d success=%s", commit_hash,
                 len(operations), all_ok)
        return {
            "pushed": all_ok,
            "hash": commit_hash,
            "message": message,
            "results": results,
        }

    # ----------------------------------------------------------------
    # git reject
    # ----------------------------------------------------------------
    def reject(self, reason: str) -> dict[str, Any]:
        """Reject the pending commit instead of pushing."""
        if not self._pending_hash:
            raise ValueError("nothing to reject: no pending commit")
        commit = CommitEntry(
            hash=self._pending_hash,
            message=self._pending_message or "",
            operations=list(self._staging),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            parent_hash=self._head,
            status=OperationStatus.REJECTED,
            reject_reason=reason,
        )
        self._commits.append(commit)
        self._persist(commit)
        self._staging.clear()
        self._pending_message = None
        self._pending_hash = None
        log.warning("REJECT hash=%s reason='%s'", commit.hash, reason)
        return {"rejected": True, "hash": commit.hash, "reason": reason}

    # ----------------------------------------------------------------
    # git status / log
    # ----------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "staged_count": len(self._staging),
            "pending_commit": self._pending_hash is not None,
            "head": self._head,
            "total_commits": len(self._commits),
        }

    def log(self, n: int = 20) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self._commits[-n:]]

    def head(self) -> Optional[CommitEntry]:
        return self._commits[-1] if self._commits else None

    # ----------------------------------------------------------------
    @staticmethod
    def _generate_hash(message: str, operations: list[Operation],
                         timestamp: str, parent: Optional[str]) -> str:
        content = {
            "message": message,
            "operations": [op.to_dict() for op in operations],
            "timestamp": timestamp,
            "parent_hash": parent,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:8]

    def _persist(self, commit: CommitEntry) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(commit.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("commit persist failed: %r", e)
