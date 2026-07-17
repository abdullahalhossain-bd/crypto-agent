"""
Database Manager — SQLite trade persistence for institutional trading
=====================================================================

Stores ALL trading activity in a local SQLite database:
    - trades table: every opened + closed trade with full context
    - signals table: every signal generated (BUY/SELL/HOLD)
    - equity_history table: equity snapshot per cycle
    - positions table: current open positions (synced with MT5)

Usage:
    from database import Database
    db = Database("data/trading_bot.db")
    db.save_trade_open(symbol="BTCUSD", action="BUY", lots=0.01, ...)
    db.save_trade_close(ticket=12345, exit_price=65500, pnl=50.0, reason="TP")
    db.save_signal(symbol="BTCUSD", action="BUY", strength=0.85, ...)
    db.save_equity(cycle=42, equity=10050.25, open_count=3)
    stats = db.get_stats()
"""
from __future__ import annotations

import sqlite3
import json
import os
import shutil
import time
import contextlib
from datetime import datetime, timezone
from typing import Optional, Any
from dataclasses import dataclass

from utils.logger import get_logger

log = get_logger("database")

# Audit-fix H2: retry config for "database is locked" errors.
_DB_BUSY_RETRY_ATTEMPTS = 5
_DB_BUSY_BASE_DELAY_S = 0.05


class Database:
    """SQLite database for trade persistence."""

    def __init__(self, path: str = "data/trading_bot.db") -> None:
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._init_schema()
        self._run_migrations()
        log.info("Database initialized: %s", path)

    @contextlib.contextmanager
    def _conn(self):
        """P0-2 fix: context manager that GUARANTEES connection close.

        BUGFIX (external audit): previously _conn() returned a raw
        sqlite3.Connection. Using `with self._conn() as conn:` relied on
        sqlite3's __exit__ which only commits/rolls back — it does NOT
        close the connection, causing file descriptor leaks under load.

        Now _conn() is a generator-based context manager (via
        contextlib.contextmanager) that closes the connection in finally.
        This replaces the separate _conn_ctx() method — all existing
        `with self._conn() as conn:` calls now get proper cleanup.
        """
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")  # 10s
        try:
            yield conn
        finally:
            conn.close()

    def _conn_ctx(self):
        """Deprecated — use _conn() directly. Kept for backward compat."""
        return self._conn()

    def _exec_with_retry(self, conn, sql: str, params: tuple = ()):
        """Audit-fix H2: retry on sqlite3.OperationalError 'database is locked'.

        Exponential backoff: 50ms, 100ms, 200ms, 400ms, 800ms.
        """
        last_err = None
        for attempt in range(_DB_BUSY_RETRY_ATTEMPTS):
            try:
                return conn.execute(sql, params)
            except sqlite3.OperationalError as e:
                last_err = e
                msg = str(e).lower()
                if "locked" in msg or "busy" in msg:
                    delay = _DB_BUSY_BASE_DELAY_S * (2 ** attempt)
                    log.warning("DB: locked/busy (attempt %d/%d) — retrying in %.3fs: %r",
                                attempt + 1, _DB_BUSY_RETRY_ATTEMPTS, delay, e)
                    time.sleep(delay)
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def _init_schema(self) -> None:
        with self._conn() as conn:
            # ── Trades table ──────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,          -- BUY / SELL
                    lots REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL DEFAULT 0,
                    pnl_pct REAL DEFAULT 0,
                    status TEXT DEFAULT 'open',    -- open / closed
                    open_time TEXT NOT NULL,
                    close_time TEXT,
                    close_reason TEXT,             -- SL / TP / manual / kill
                    magic INTEGER DEFAULT 100000,
                    -- Context at entry time
                    gate_score REAL,
                    grade TEXT,
                    strategy_type TEXT,
                    regime TEXT,
                    rsi REAL,
                    ema_fast REAL,
                    ema_slow REAL,
                    atr REAL,
                    confidence_pct REAL,
                    -- Metadata
                    mode TEXT DEFAULT 'demo',      -- demo / live
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # ── Signals table ─────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,           -- BUY / SELL / HOLD
                    strength REAL,
                    price REAL,
                    strategy_type TEXT,
                    rsi REAL,
                    ema_fast REAL,
                    ema_slow REAL,
                    momentum REAL,
                    bull_factors INTEGER,
                    bear_factors INTEGER,
                    cycle INTEGER
                )
            """)

            # ── Equity history ────────────────────────────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cycle INTEGER,
                    equity REAL NOT NULL,
                    cash REAL,
                    open_positions INTEGER,
                    pnl_pct REAL,
                    approved_count INTEGER,
                    rejected_count INTEGER
                )
            """)

            # ── Positions snapshot (synced with MT5) ──────────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE,
                    symbol TEXT NOT NULL,
                    action TEXT,
                    lots REAL,
                    entry_price REAL,
                    current_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    profit REAL DEFAULT 0,
                    magic INTEGER,
                    open_time TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # ── Settings table (for auto-upgrade tracking) ────────────
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # ── Decisions table (P0-8 FIX, Phase 3) ───────────────────
            # Merged from architecture/decision_audit.py's private table.
            # Every trade decision — whether it resulted in an order or was
            # rejected — is recorded here so the "silently doing nothing"
            # bug (master_orchestrator.py:499-509 pattern) is structurally
            # impossible: every cycle produces a visible decision record.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    audit_id TEXT PRIMARY KEY,
                    correlation_id TEXT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    cycle INTEGER,
                    bar_close REAL,
                    feature_vector TEXT,
                    account_equity REAL,
                    open_positions INTEGER,
                    current_drawdown_pct REAL,
                    strategy_action TEXT,
                    strategy_strength REAL,
                    strategy_meta TEXT,
                    risk_verdicts TEXT,
                    wisdom_verdict TEXT,
                    approved INTEGER DEFAULT 0,
                    final_lots REAL DEFAULT 0,
                    final_sl REAL DEFAULT 0,
                    final_tp REAL DEFAULT 0,
                    entry_price REAL DEFAULT 0,
                    ticket INTEGER DEFAULT 0,
                    reject_reason TEXT,
                    exit_price REAL DEFAULT 0,
                    pnl REAL DEFAULT 0,
                    hold_time_s REAL DEFAULT 0,
                    outcome TEXT DEFAULT 'open',
                    closed_at TEXT
                )
            """)

            # ── Schema version (Phase 13 req #74) ─────────────────────
            # PRAGMA user_version is the SQLite-blessed way to track schema
            # revisions. Bump on every additive migration; read on startup
            # to decide whether to run ALTER TABLE migrations.
            conn.execute("PRAGMA user_version = 1")

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_equity_cycle ON equity_history(cycle)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_approved ON decisions(approved)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp)")

            conn.commit()

    # ================================================================
    # MIGRATIONS (Phase 13 req #74)
    # ================================================================
    SCHEMA_VERSION = 2  # bumped in Phase 6 for slippage column

    def _column_exists(self, conn, table: str, column: str) -> bool:
        """Audit-fix M13: check if a column exists before ALTER TABLE.

        Avoids the 'duplicate column name' error when the migration was
        already applied (e.g. fresh schema includes the column already).
        """
        try:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return any(r[1] == column for r in rows)
        except sqlite3.Error:
            return False

    def _run_migrations(self) -> None:
        """Run additive schema migrations based on PRAGMA user_version.

        Phase 13: SQLite's PRAGMA user_version tracks schema revisions.
        Each migration is a forward-only ALTER TABLE or CREATE TABLE IF
        NOT EXISTS. No down-migrations — always forward.

        Audit-fix C13: forward-only migrations are intentional; rollback
        would risk data loss for an institutional trading DB. To revert,
        restore from the daily backup instead.

        To add a migration:
          1. Bump SCHEMA_VERSION
          2. Add an `if current < N:` block that runs the ALTER
          3. Test with a DB from the previous version
        """
        with self._conn() as conn:
            row = conn.execute("PRAGMA user_version").fetchone()
            current = row[0] if row else 0

            if current < 1:
                conn.execute("PRAGMA user_version = 1")
                log.info("DB: migration to version 1 complete")

            if current < 2:
                # Phase 6: add slippage_bps column to trades table
                # for recording actual vs expected fill price.
                # Audit-fix M13: check column existence before ALTER.
                if not self._column_exists(conn, "trades", "slippage_bps"):
                    try:
                        conn.execute(
                            "ALTER TABLE trades ADD COLUMN slippage_bps REAL DEFAULT 0")
                        log.info("DB: migration to version 2 complete (slippage_bps column)")
                    except sqlite3.Error as e:
                        log.warning("DB: v2 migration issue: %r", e)
                else:
                    log.info("DB: v2 migration skipped — slippage_bps already present")
                conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

            conn.commit()

    # ================================================================
    # TRADES
    # ================================================================
    def save_trade_open(
        self, ticket: int, symbol: str, action: str, lots: float,
        entry_price: float, stop_loss: float, take_profit: float,
        magic: int = 100000, mode: str = "demo",
        gate_score: float = 0, grade: str = "", strategy_type: str = "",
        regime: str = "", rsi: float = 0, ema_fast: float = 0,
        ema_slow: float = 0, atr: float = 0, confidence_pct: float = 0,
        slippage_bps: float = 0.0,
    ) -> int:
        """Record a new trade opening. Returns the database row ID.

        Phase 6: slippage_bps parameter added — the actual fill slippage
        in basis points (|fill_price - signal_price| / signal_price * 10000).
        """
        now = datetime.now(timezone.utc).isoformat()
        # Audit-fix H3 + L6: validate mode and use INSERT with conflict handling
        # so an existing ticket is updated (not silently overwritten) and mode
        # is constrained to the documented enum.
        if mode not in ("demo", "live", "paper"):
            raise ValueError(f"mode must be 'demo'|'live'|'paper', got {mode!r}")
        with self._conn() as conn:
            # If ticket already exists and is open, this is a duplicate send —
            # log and skip instead of overwriting the original entry.
            existing = self._exec_with_retry(
                conn,
                "SELECT id, status, entry_price FROM trades WHERE ticket=?",
                (ticket,)).fetchone()
            if existing and existing["status"] == "open":
                log.warning("DB: ticket=%d already open (id=%d) — skipping duplicate INSERT",
                            ticket, existing["id"])
                return existing["id"]
            cursor = self._exec_with_retry(conn, """
                INSERT OR REPLACE INTO trades
                    (ticket, symbol, action, lots, entry_price, stop_loss,
                     take_profit, status, open_time, magic, mode, gate_score,
                     grade, strategy_type, regime, rsi, ema_fast, ema_slow,
                     atr, confidence_pct, slippage_bps)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (ticket, symbol, action, lots, entry_price, stop_loss,
                  take_profit, now, magic, mode, gate_score, grade,
                  strategy_type, regime, rsi, ema_fast, ema_slow, atr,
                  confidence_pct, slippage_bps))
            conn.commit()
            trade_id = cursor.lastrowid
            log.info("DB: trade opened id=%d ticket=%d %s %s %.4f @ %.2f (slip=%.1fbps)",
                     trade_id, ticket, action, symbol, lots, entry_price, slippage_bps)
            return trade_id

    def save_trade_close(
        self, ticket: int, exit_price: float, pnl: float,
        reason: str = "manual",
    ) -> bool:
        """Record a trade closing. Returns True if updated."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            # Get entry price for pnl_pct
            row = conn.execute(
                "SELECT entry_price, lots FROM trades WHERE ticket=?", (ticket,)
            ).fetchone()
            if row is None:
                log.warning("DB: trade close — ticket %d not found", ticket)
                return False
            entry = float(row["entry_price"])
            lots = float(row["lots"])
            # P0-16 fix: PnL% must account for trade direction (action).
            # For BUY: pnl% = (exit - entry) / entry
            # For SELL: pnl% = (entry - exit) / entry
            action_row = conn.execute(
                "SELECT action FROM trades WHERE ticket=?", (ticket,)
            ).fetchone()
            action = action_row["action"] if action_row else "BUY"
            if action == "SELL":
                pnl_pct = (entry - exit_price) / entry * 100 if entry > 0 else 0
            else:
                pnl_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0
            conn.execute("""
                UPDATE trades SET
                    exit_price=?, pnl=?, pnl_pct=?, status='closed',
                    close_time=?, close_reason=?
                WHERE ticket=?
            """, (exit_price, pnl, pnl_pct, now, reason, ticket))
            conn.commit()
            log.info("DB: trade closed ticket=%d exit=%.2f pnl=%.2f reason=%s",
                     ticket, exit_price, pnl, reason)
            return True

    def get_open_trades(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='open' ORDER BY open_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_closed_trades(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='closed' ORDER BY close_time DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_trades(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================
    # SIGNALS
    # ================================================================
    def save_signal(
        self, symbol: str, action: str, strength: float, price: float,
        strategy_type: str = "", rsi: float = 0, ema_fast: float = 0,
        ema_slow: float = 0, momentum: float = 0,
        bull_factors: int = 0, bear_factors: int = 0, cycle: int = 0,
    ) -> int:
        # Audit-fix L12: validate strength range and action enum.
        if action not in ("BUY", "SELL", "HOLD"):
            raise ValueError(f"action must be BUY|SELL|HOLD, got {action!r}")
        if not (-1.0 <= float(strength) <= 1.0):
            log.warning("DB: signal strength out of [-1,1] range: %r — clamping", strength)
            strength = max(-1.0, min(1.0, float(strength)))
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = self._exec_with_retry(conn, """
                INSERT INTO signals
                    (timestamp, symbol, action, strength, price, strategy_type,
                     rsi, ema_fast, ema_slow, momentum, bull_factors, bear_factors, cycle)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, symbol, action, strength, price, strategy_type,
                  rsi, ema_fast, ema_slow, momentum, bull_factors, bear_factors, cycle))
            conn.commit()
            return cursor.lastrowid

    def get_recent_signals(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================
    # EQUITY HISTORY
    # ================================================================
    def save_equity(
        self, cycle: int, equity: float, cash: float = 0,
        open_positions: int = 0, pnl_pct: float = 0,
        approved_count: int = 0, rejected_count: int = 0,
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cursor = conn.execute("""
                INSERT INTO equity_history
                    (timestamp, cycle, equity, cash, open_positions, pnl_pct,
                     approved_count, rejected_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, cycle, equity, cash, open_positions, pnl_pct,
                  approved_count, rejected_count))
            conn.commit()
            return cursor.lastrowid

    def get_equity_history(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM equity_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ================================================================
    # POSITIONS SYNC
    # ================================================================
    def sync_positions(self, positions: list[dict]) -> None:
        """Sync current MT5 positions into database.

        Audit-fix C20 + M9: use a single transaction with deferred DELETE
        so that if any INSERT fails, the entire sync is rolled back and the
        previous positions remain intact. The previous 'DELETE then INSERT'
        pattern could lose all open-position rows if the INSERT failed midway.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                # P0-3 fix: SQLite does NOT support `CREATE TABLE ... LIKE ...`
                # (that's MySQL syntax). Use `CREATE TABLE ... AS SELECT ... LIMIT 0`
                # to clone the schema, then DELETE all rows.
                conn.execute("DROP TABLE IF EXISTS positions_staging")
                conn.execute("CREATE TABLE positions_staging AS SELECT * FROM positions LIMIT 0")
                conn.execute("DELETE FROM positions_staging")
                for p in positions:
                    self._exec_with_retry(conn, """
                        INSERT INTO positions_staging
                            (ticket, symbol, action, lots, entry_price, current_price,
                             stop_loss, take_profit, profit, magic, open_time, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        p.get("ticket", 0), p.get("symbol", ""),
                        p.get("action", ""), p.get("lots", 0),
                        p.get("entry_price", 0), p.get("current_price", 0),
                        p.get("stop_loss", 0), p.get("take_profit", 0),
                        p.get("profit", 0), p.get("magic", 0),
                        p.get("open_time", now), now,
                    ))
                # Atomic swap: rename old, rename staging, drop old.
                conn.execute("DROP TABLE IF EXISTS positions_old")
                conn.execute("ALTER TABLE positions RENAME TO positions_old")
                conn.execute("ALTER TABLE positions_staging RENAME TO positions")
                conn.execute("DROP TABLE positions_old")
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                log.error("DB: sync_positions rolled back — positions preserved: %r", e)
                raise

    # ================================================================
    # STATISTICS
    # ================================================================
    def get_stats(self) -> dict:
        """Get comprehensive trading statistics."""
        with self._conn() as conn:
            # Trade stats
            total = conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()["c"]
            wins = conn.execute(
                "SELECT COUNT(*) as c FROM trades WHERE status='closed' AND pnl > 0"
            ).fetchone()["c"]
            losses = conn.execute(
                "SELECT COUNT(*) as c FROM trades WHERE status='closed' AND pnl < 0"
            ).fetchone()["c"]
            open_count = conn.execute(
                "SELECT COUNT(*) as c FROM trades WHERE status='open'"
            ).fetchone()["c"]
            total_pnl = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as s FROM trades WHERE status='closed'"
            ).fetchone()["s"]
            avg_win = conn.execute(
                "SELECT COALESCE(AVG(pnl), 0) as a FROM trades WHERE status='closed' AND pnl > 0"
            ).fetchone()["a"]
            avg_loss = conn.execute(
                "SELECT COALESCE(AVG(pnl), 0) as a FROM trades WHERE status='closed' AND pnl < 0"
            ).fetchone()["a"]
            best_trade = conn.execute(
                "SELECT COALESCE(MAX(pnl), 0) as m FROM trades WHERE status='closed'"
            ).fetchone()["m"]
            worst_trade = conn.execute(
                "SELECT COALESCE(MIN(pnl), 0) as m FROM trades WHERE status='closed'"
            ).fetchone()["m"]

            # Signal stats
            total_signals = conn.execute("SELECT COUNT(*) as c FROM signals").fetchone()["c"]
            buy_signals = conn.execute(
                "SELECT COUNT(*) as c FROM signals WHERE action='BUY'"
            ).fetchone()["c"]
            sell_signals = conn.execute(
                "SELECT COUNT(*) as c FROM signals WHERE action='SELL'"
            ).fetchone()["c"]
            hold_signals = conn.execute(
                "SELECT COUNT(*) as c FROM signals WHERE action='HOLD'"
            ).fetchone()["c"]

            # Latest equity
            latest_eq = conn.execute(
                "SELECT * FROM equity_history ORDER BY id DESC LIMIT 1"
            ).fetchone()

            # By symbol
            by_symbol = conn.execute("""
                SELECT symbol,
                       COUNT(*) as total,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                       COALESCE(SUM(pnl), 0) as total_pnl
                FROM trades WHERE status='closed'
                GROUP BY symbol
            """).fetchall()

            decided = wins + losses
            win_rate = (wins / decided * 100) if decided > 0 else 0
            profit_factor = (
                (avg_win * wins) / abs(avg_loss * losses)
                if avg_loss != 0 and losses > 0 else float('inf') if wins > 0 else 0
            )

            return {
                "total_trades": total,
                "open_trades": open_count,
                "closed_trades": decided,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "best_trade": round(best_trade, 2),
                "worst_trade": round(worst_trade, 2),
                "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999.0,
                "total_signals": total_signals,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "hold_signals": hold_signals,
                "latest_equity": dict(latest_eq) if latest_eq else None,
                "by_symbol": [dict(r) for r in by_symbol],
            }

    # ================================================================
    # SETTINGS
    # ================================================================
    def set_setting(self, key: str, value: str) -> None:
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
            """, (key, value))
            conn.commit()

    def get_setting(self, key: str, default: str = "") -> str:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
            return row["value"] if row else default

    # ================================================================
    # DECISIONS  (P0-8 FIX, Phase 3 — merged from decision_audit.py)
    # ================================================================
    def save_decision(self, audit_id: str, correlation_id: str,
                      symbol: str, cycle: int, bar_close: float,
                      feature_vector: dict, account_equity: float,
                      open_positions: int, current_drawdown_pct: float,
                      strategy_action: str, strategy_strength: float,
                      strategy_meta: dict) -> None:
        """Insert a new decision record (approved=unknown at this stage)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO decisions
                    (audit_id, correlation_id, timestamp, symbol, cycle, bar_close,
                     feature_vector, account_equity, open_positions,
                     current_drawdown_pct, strategy_action, strategy_strength,
                     strategy_meta, approved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                audit_id, correlation_id, now, symbol, cycle, bar_close,
                json.dumps(feature_vector, default=str), account_equity,
                open_positions, current_drawdown_pct, strategy_action,
                strategy_strength, json.dumps(strategy_meta, default=str),
            ))
            conn.commit()

    def update_decision_risk(self, audit_id: str, risk_verdicts: list) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET risk_verdicts=? WHERE audit_id=?",
                (json.dumps(risk_verdicts, default=str), audit_id),
            )
            conn.commit()

    def update_decision_wisdom(self, audit_id: str, wisdom_verdict: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET wisdom_verdict=? WHERE audit_id=?",
                (json.dumps(wisdom_verdict, default=str), audit_id),
            )
            conn.commit()

    def finalize_decision(self, audit_id: str, approved: bool,
                          final_lots: float = 0.0, final_sl: float = 0.0,
                          final_tp: float = 0.0, entry_price: float = 0.0,
                          ticket: int = 0, reject_reason: str = "") -> None:
        """P0-1 FIX (Phase 3): Every decision ends with a finalize call —
        approved=True with a ticket, or approved=False with a reason.
        The 'silently doing nothing' bug is now structurally impossible
        because finalize is the only exit path from _process_symbol."""
        with self._conn() as conn:
            conn.execute("""
                UPDATE decisions SET
                    approved=?, final_lots=?, final_sl=?, final_tp=?,
                    entry_price=?, ticket=?, reject_reason=?
                WHERE audit_id=?
            """, (
                1 if approved else 0,
                final_lots, final_sl, final_tp, entry_price,
                ticket, reject_reason, audit_id,
            ))
            conn.commit()

    def record_decision_outcome(self, ticket: int, exit_price: float,
                                 pnl: float, hold_time_s: float,
                                 outcome: str) -> None:
        """Closed-trade outcome for a decision (filled when position closes)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE decisions SET
                    exit_price=?, pnl=?, hold_time_s=?, outcome=?, closed_at=?
                WHERE ticket=?
            """, (exit_price, pnl, hold_time_s, outcome, now, ticket))
            conn.commit()

    def get_decisions(self, limit: int = 50, approved_only: bool = False,
                      rejected_only: bool = False) -> list[dict]:
        # Audit-fix H18: add `id` as secondary sort key so ordering is
        # deterministic even when two decisions share a timestamp.
        with self._conn() as conn:
            q = "SELECT * FROM decisions"
            if approved_only:
                q += " WHERE approved=1"
            elif rejected_only:
                q += " WHERE approved=0"
            q += " ORDER BY timestamp DESC, audit_id DESC LIMIT ?"
            rows = self._exec_with_retry(conn, q, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_consecutive_losses(self) -> int:
        """P0-4 FIX (Phase 3): Read consecutive losing trades from DB.
        Used as the canonical source for ConsecutiveLossGate — survives
        process restarts (portfolio_manager_v2's in-memory history does not).

        Audit-fix C11: fall back to the `trades` table if the `decisions`
        table has no rows with outcomes yet (e.g. paper mode where decisions
        are not always recorded). This prevents the gate from incorrectly
        returning 0 and allowing overtrading after a string of real losses.

        BUGFIX (external audit): wrap both queries in try/except — if the
        `closed_at` column doesn't exist in an older DB schema, the query
        would crash with OperationalError. Now falls back gracefully.
        """
        with self._conn() as conn:
            rows = []
            try:
                rows = self._exec_with_retry(
                    conn,
                    "SELECT pnl FROM decisions WHERE outcome IN ('win','loss') "
                    "ORDER BY closed_at DESC LIMIT 50").fetchall()
            except sqlite3.OperationalError:
                # Column 'closed_at' may not exist in older schema — try
                # without ORDER BY (just get the most recent 50 by rowid).
                try:
                    rows = self._exec_with_retry(
                        conn,
                        "SELECT pnl FROM decisions WHERE outcome IN ('win','loss') "
                        "ORDER BY rowid DESC LIMIT 50").fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                # Fallback: use closed trades table (C11 fix).
                try:
                    rows = self._exec_with_retry(
                        conn,
                        "SELECT pnl FROM trades WHERE status='closed' AND pnl IS NOT NULL "
                        "ORDER BY close_time DESC LIMIT 50").fetchall()
                except sqlite3.OperationalError:
                    try:
                        rows = self._exec_with_retry(
                            conn,
                            "SELECT pnl FROM trades WHERE pnl IS NOT NULL "
                            "ORDER BY rowid DESC LIMIT 50").fetchall()
                    except sqlite3.OperationalError:
                        rows = []
            count = 0
            for r in rows:
                try:
                    val = float(r["pnl"]) if "pnl" in r.keys() else float(r[0])
                except (TypeError, ValueError, IndexError):
                    continue
                if val < 0:
                    count += 1
                else:
                    break
            return count

    def get_realized_pnl_today(self) -> float:
        """P0-3 FIX (Phase 3): Sum of pnl for trades closed since UTC midnight."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as s FROM decisions "
                "WHERE outcome IN ('win','loss','breakeven') "
                "AND closed_at >= date('now')"
            ).fetchone()
            return float(row["s"]) if row else 0.0

    # ================================================================
    # MAINTENANCE
    # ================================================================
    def cleanup_old_signals(self, keep_days: int = 30) -> int:
        """Delete signals older than keep_days.

        Audit-fix M2: keep_days is now configurable from `config.database.signal_retention_days`
        so backtest pipelines can keep more history. Default remains 30.
        """
        with self._conn() as conn:
            cursor = self._exec_with_retry(
                conn,
                "DELETE FROM signals WHERE timestamp < datetime('now', ?)",
                (f"-{keep_days} days",))
            conn.commit()
            return cursor.rowcount

    def reset_all(self, confirm: bool = False) -> None:
        """Delete all data — use with caution!

        Audit-fix L17: require explicit `confirm=True` to prevent accidental
        wipe (e.g. a stray CLI flag or test runner). Logs the caller's intent.
        """
        if not confirm:
            raise RuntimeError(
                "reset_all() now requires confirm=True — refusing to wipe DB "
                "without explicit acknowledgement (audit L17)")
        with self._conn() as conn:
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM signals")
            conn.execute("DELETE FROM equity_history")
            conn.execute("DELETE FROM positions")
            conn.commit()
            log.warning("DB: ALL DATA RESET (confirmed by caller)")

    # ================================================================
    # Phase 7: DB corruption auto-repair (backs up before deleting)
    # ================================================================
    def health_check(self) -> bool:
        """Verify the database is readable and not corrupted.

        Returns True if healthy, False if corrupted.
        Phase 7 req #43: corrupted DB must be backed up before repair,
        never silently discarded.

        Audit-fix C1 / H6: actually verify the `PRAGMA integrity_check` result.
        The previous implementation only checked that the call didn't raise,
        which returns True even when integrity_check returns error rows like
        'database disk image is malformed'. Now we require the first row's
        value to be 'ok'.
        """
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1 FROM trades LIMIT 1")
                conn.execute("SELECT 1 FROM decisions LIMIT 1")
                conn.execute("SELECT 1 FROM signals LIMIT 1")
                # C1 fix: integrity_check returns 'ok' on healthy DBs and
                # one or more error-description rows on corrupt DBs.
                rows = conn.execute("PRAGMA integrity_check").fetchall()
                if not rows:
                    log.error("DB: integrity_check returned no rows — treating as corrupt")
                    return False
                first = str(rows[0][0]) if rows[0] else ""
                if first.lower() != "ok":
                    log.error("DB: integrity_check FAILED: %s", first)
                    return False
            return True
        except Exception as e:
            log.error("DB: health check FAILED — corruption detected: %r", e)
            return False

    def repair_with_backup(self) -> bool:
        """Back up the corrupted DB file, then recreate it fresh.

        Phase 7 req #43: NEVER silently discard trading history.
        The corrupted file is renamed to <path>.corrupt_<timestamp> so
        the operator can inspect it later. A fresh empty DB is created.

        Audit-fix C2 / H12: VERIFY the backup before deleting the original.
        The previous implementation deleted the original DB even if the
        backup failed (e.g. disk full), guaranteeing data loss. Now the
        backup is verified to be a non-zero file with at least the same
        byte size as the original before any deletion is permitted.

        Returns True if repair succeeded.
        """
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.path}.corrupt_{ts}"
        backup_ok = False
        try:
            shutil.copy2(self.path, backup_path)
            # Verify backup: must exist and be at least as large as the
            # source file (allows for the rare case where copy2 partially
            # wrote — in which case size will not match).
            src_size = os.path.getsize(self.path)
            bak_size = os.path.getsize(backup_path) if os.path.exists(backup_path) else 0
            if bak_size == 0:
                log.error("DB: backup is EMPTY — aborting repair to preserve original")
                return False
            if bak_size < src_size:
                log.error("DB: backup (%d bytes) smaller than source (%d bytes) — aborting",
                          bak_size, src_size)
                return False
            log.warning("DB: backed up corrupted DB to %s (%d bytes verified)",
                        backup_path, bak_size)
            backup_ok = True
        except Exception as e:
            log.error("DB: could not back up corrupted DB: %r", e)

        if not backup_ok:
            # C2 fix: do NOT delete the original if backup failed.
            log.error("DB: repair ABORTED — original DB preserved at %s for manual recovery",
                      self.path)
            return False

        try:
            os.remove(self.path)
            # Also remove WAL and SHM files if they exist
            for suffix in ("-wal", "-shm"):
                wal = self.path + suffix
                if os.path.exists(wal):
                    os.remove(wal)
            self._init_schema()
            log.info("DB: recreated fresh schema after corruption repair")
            return True
        except Exception as e:
            log.error("DB: repair failed: %r", e)
            return False


__all__ = ["Database"]
