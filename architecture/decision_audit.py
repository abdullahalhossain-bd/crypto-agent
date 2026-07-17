"""architecture/decision_audit.py
=====================================================================
AI Decision Audit Trail (Improvement #10)
=====================================================================
Every trade decision is recorded end-to-end so we can replay it later
and answer "WHY did the bot take this trade?".

Audit record contains:
    1. Inputs at decision time:
       - Bar OHLCV
       - All feature values (FeatureVector snapshot)
       - Account equity, open positions, drawdown
       - Recent trade history
    2. Pipeline trace:
       - Strategy output (signal action, strength, factor breakdown)
       - Each risk gate verdict (pass/fail + reason)
       - Wisdom gate verdict (which principles passed/failed)
       - Final approved trade details (lots, SL, TP, R:R)
    3. Outcome (filled later when position closes):
       - Entry price, exit price, PnL, hold time
       - Was the decision correct? (profitable or not)

Storage:
    - In-memory ring buffer (last 1000 decisions) for fast query
    - SQLite table `decision_audit` for permanent record
    - Each decision has a UUID correlation_id that ties inputs → pipeline → outcome

Usage:
    audit = DecisionAuditor(db_path="data/trading_bot.db")
    audit_id = audit.start_decision(symbol="BTCUSD", cycle=42)
    audit.add_strategy_output(audit_id, signal=signal)
    audit.add_risk_verdicts(audit_id, verdicts)
    audit.add_wisdom_verdict(audit_id, verdict)
    audit.finalize_decision(audit_id, approved_trade)
    # Later, when trade closes:
    audit.record_outcome(audit_id, exit_price=43100, pnl=42.50)

    # Query:
    records = audit.query(symbol="BTCUSD", profitable_only=True)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.architecture.decision_audit")


@dataclass
class DecisionRecord:
    """Complete audit record for a single trade decision."""
    audit_id: str = ""
    correlation_id: str = ""
    timestamp: str = ""
    symbol: str = ""
    cycle: int = 0
    # Inputs
    bar_close: float = 0.0
    feature_vector: Dict[str, Any] = field(default_factory=dict)
    account_equity: float = 0.0
    open_positions: int = 0
    current_drawdown_pct: float = 0.0
    # Pipeline
    strategy_action: str = ""
    strategy_strength: float = 0.0
    strategy_meta: Dict[str, Any] = field(default_factory=dict)
    risk_verdicts: List[Dict[str, Any]] = field(default_factory=list)
    wisdom_verdict: Dict[str, Any] = field(default_factory=dict)
    # Output
    approved: bool = False
    final_lots: float = 0.0
    final_sl: float = 0.0
    final_tp: float = 0.0
    entry_price: float = 0.0
    ticket: int = 0
    # Outcome (filled later)
    exit_price: float = 0.0
    pnl: float = 0.0
    hold_time_s: float = 0.0
    outcome: str = "open"  # open, win, loss, breakeven
    closed_at: str = ""


class DecisionAuditor:
    """Records every trade decision for audit and post-hoc analysis."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS decision_audit (
        audit_id TEXT PRIMARY KEY,
        correlation_id TEXT,
        timestamp TEXT,
        symbol TEXT,
        cycle INTEGER,
        bar_close REAL,
        feature_vector TEXT,          -- JSON
        account_equity REAL,
        open_positions INTEGER,
        current_drawdown_pct REAL,
        strategy_action TEXT,
        strategy_strength REAL,
        strategy_meta TEXT,           -- JSON
        risk_verdicts TEXT,           -- JSON
        wisdom_verdict TEXT,          -- JSON
        approved INTEGER,
        final_lots REAL,
        final_sl REAL,
        final_tp REAL,
        entry_price REAL,
        ticket INTEGER,
        exit_price REAL,
        pnl REAL,
        hold_time_s REAL,
        outcome TEXT,
        closed_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_audit_symbol ON decision_audit(symbol);
    CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON decision_audit(timestamp);
    CREATE INDEX IF NOT EXISTS idx_audit_outcome ON decision_audit(outcome);
    """

    def __init__(self,
                 db_path: str = "data/trading_bot.db",
                 ring_buffer_size: int = 1000):
        self._lock = threading.RLock()
        self._ring: Deque[DecisionRecord] = deque(maxlen=ring_buffer_size)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        # Co-Founder Audit Fix: leak-safe connection. The previous pattern
        # `conn = sqlite3.connect(...); ...; conn.close()` would leak the
        # connection on ANY exception (IntegrityError, DatabaseError, etc.)
        # between connect and close. Use try/finally to guarantee closure.
        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.executescript(self.SCHEMA)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("audit: DB init failed: %r", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Decision lifecycle
    # ------------------------------------------------------------------
    def start_decision(self,
                       symbol: str,
                       cycle: int,
                       feature_vector: Optional[Dict[str, Any]] = None,
                       account_equity: float = 0.0,
                       open_positions: int = 0,
                       current_drawdown_pct: float = 0.0,
                       bar_close: float = 0.0) -> str:
        """Start recording a new decision. Returns audit_id."""
        record = DecisionRecord(
            audit_id=str(uuid.uuid4()),
            correlation_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            symbol=symbol,
            cycle=cycle,
            bar_close=bar_close,
            feature_vector=feature_vector or {},
            account_equity=account_equity,
            open_positions=open_positions,
            current_drawdown_pct=current_drawdown_pct,
        )
        with self._lock:
            self._ring.append(record)
        return record.audit_id

    def add_strategy_output(self, audit_id: str,
                            signal: Any) -> None:
        record = self._find(audit_id)
        if record is None:
            return
        record.strategy_action = getattr(signal, "action", None).value \
            if hasattr(getattr(signal, "action", None), "value") else str(getattr(signal, "action", ""))
        record.strategy_strength = float(getattr(signal, "strength", 0.0))
        record.strategy_meta = dict(getattr(signal, "meta", {}))

    def add_risk_verdicts(self, audit_id: str,
                          verdicts: List[Any]) -> None:
        record = self._find(audit_id)
        if record is None:
            return
        record.risk_verdicts = [
            {
                "gate": v.gate_name,
                "passed": v.passed,
                "reason": v.reason,
                "modified_lots": v.modified_lots,
                "modified_sl": v.modified_sl,
                "modified_tp": v.modified_tp,
            }
            for v in verdicts
        ]

    def add_wisdom_verdict(self, audit_id: str, verdict: Any) -> None:
        record = self._find(audit_id)
        if record is None:
            return
        record.wisdom_verdict = {
            "approved": getattr(verdict, "approved", False),
            "position_multiplier": getattr(verdict, "position_multiplier", 0.0),
            "checks_passed": getattr(verdict, "checks_passed", 0),
            "failed_principles": getattr(verdict, "failed_principles", []),
        }

    def finalize_decision(self, audit_id: str,
                          approved: bool,
                          lots: float = 0.0,
                          sl: float = 0.0,
                          tp: float = 0.0,
                          entry_price: float = 0.0,
                          ticket: int = 0) -> None:
        record = self._find(audit_id)
        if record is None:
            return
        record.approved = approved
        record.final_lots = lots
        record.final_sl = sl
        record.final_tp = tp
        record.entry_price = entry_price
        record.ticket = ticket
        self._persist(record)

    def record_outcome(self, audit_id: str,
                       exit_price: float,
                       pnl: float,
                       hold_time_s: float,
                       outcome: str = "open") -> None:
        record = self._find(audit_id)
        if record is None:
            return
        record.exit_price = exit_price
        record.pnl = pnl
        record.hold_time_s = hold_time_s
        if outcome == "open":
            record.outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        else:
            record.outcome = outcome
        record.closed_at = datetime.now(tz=timezone.utc).isoformat()
        self._persist(record)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def _find(self, audit_id: str) -> Optional[DecisionRecord]:
        with self._lock:
            for r in self._ring:
                if r.audit_id == audit_id:
                    return r
        return None

    def query(self,
              symbol: Optional[str] = None,
              approved_only: bool = False,
              profitable_only: bool = False,
              last_n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for r in reversed(self._ring):
                if symbol and r.symbol != symbol:
                    continue
                if approved_only and not r.approved:
                    continue
                if profitable_only and r.pnl <= 0:
                    continue
                out.append(asdict(r))
                if len(out) >= last_n:
                    break
            return out

    def summary_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._ring)
            approved = sum(1 for r in self._ring if r.approved)
            closed = sum(1 for r in self._ring if r.outcome != "open")
            wins = sum(1 for r in self._ring if r.outcome == "win")
            losses = sum(1 for r in self._ring if r.outcome == "loss")
            total_pnl = sum(r.pnl for r in self._ring if r.outcome != "open")
        return {
            "total_decisions": total,
            "approved": approved,
            "approval_rate": approved / max(total, 1),
            "closed": closed,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(closed, 1),
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / max(closed, 1),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist(self, record: DecisionRecord) -> None:
        # C17 fix: don't silently swallow DB write errors. Retry once
        # after a short delay; if it still fails, log at ERROR level
        # (not WARNING) so monitoring catches it. The audit trail is a
        # compliance requirement — a missing record is a real problem.
        #
        # Co-Founder Audit Fix: leak-safe connection. The previous code
        # called conn.close() only on the success path — any exception
        # between connect() and close() leaked the connection. With 100
        # symbols per cycle and 5s poll, that's ~720 leaks/hour.
        max_retries = 2
        for attempt in range(max_retries):
            conn = None
            try:
                conn = sqlite3.connect(self._db_path, timeout=5.0)
                conn.execute("""
                    INSERT OR REPLACE INTO decision_audit
                    (audit_id, correlation_id, timestamp, symbol, cycle,
                     bar_close, feature_vector, account_equity, open_positions,
                     current_drawdown_pct, strategy_action, strategy_strength,
                     strategy_meta, risk_verdicts, wisdom_verdict, approved,
                     final_lots, final_sl, final_tp, entry_price, ticket,
                     exit_price, pnl, hold_time_s, outcome, closed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.audit_id, record.correlation_id, record.timestamp,
                    record.symbol, record.cycle, record.bar_close,
                    json.dumps(record.feature_vector, default=str),
                    record.account_equity, record.open_positions,
                    record.current_drawdown_pct,
                    record.strategy_action, record.strategy_strength,
                    json.dumps(record.strategy_meta, default=str),
                    json.dumps(record.risk_verdicts, default=str),
                    json.dumps(record.wisdom_verdict, default=str),
                    int(record.approved),
                    record.final_lots, record.final_sl, record.final_tp,
                    record.entry_price, record.ticket,
                    record.exit_price, record.pnl, record.hold_time_s,
                    record.outcome, record.closed_at,
                ))
                conn.commit()
                return  # success
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    import time as _t
                    _t.sleep(0.1 * (attempt + 1))
                    continue
                log.error("audit: persist FAILED (compliance risk) — audit_id=%s: %r",
                          record.audit_id, e)
                return
            except Exception as e:  # noqa: BLE001
                log.error("audit: persist FAILED (compliance risk) — audit_id=%s: %r",
                          record.audit_id, e)
                return
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
