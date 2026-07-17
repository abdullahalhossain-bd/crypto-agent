"""agents.memory_log
=====================================================================
Append-only JSONL decision log with reflection.

Inspired by TradingAgents' TradingMemoryLog.

Two-phase design:
  Phase A (immediate): store decision right after PM decides
  Phase B (deferred):  once the trade's outcome is known, reflect

The log is re-injected into future agent prompts as "past_context" —
this is how the system LEARNS from past decisions without doing
dangerous live weight updates.

FIXES (Batch 2 audit):
  - C7/X1: added `resolve_trade_outcome()`, a thin public wrapper the
    caller (main trading loop) is expected to invoke once a trade's
    P&L is known. `agent_graph.py` also exposes this on the graph
    object so callers don't need to reach into internals.
  - C8/X5: added a cross-process file lock (best-effort `fcntl` on
    POSIX, plus an in-process `threading.Lock`) around every read
    and write so concurrent graphs can't corrupt the log.
  - H12/M13: storage format switched from a hand-rolled markdown
    separator scheme to JSON Lines (one JSON object per line). This
    removes the fragile "|"-split tag parsing and the risk of a
    decision's own text colliding with the separator string.
  - L6: `max_entries` is now actually enforced (oldest entries are
    trimmed on every write).
  - L13: writes are flushed and fsync'd immediately.

NOTE ON MIGRATION: log files written by the previous markdown-based
version of this module are NOT automatically converted. Lines that
don't parse as JSON are skipped with a warning rather than crashing,
so old logs won't break the app, but their entries won't be picked
up as "past context" until new JSONL entries accumulate.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agents.schemas import parse_rating
from external.llm_provider import LLMProvider, LLMMessage
from utils.logger import get_logger

log = get_logger("agents.memory_log")

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX platforms
    _HAS_FCNTL = False


@dataclass
class MemoryEntry:
    ticker: str
    trade_date: str
    rating: str
    decision: str
    reflection: str = ""
    raw_return: Optional[float] = None
    alpha_return: Optional[float] = None
    status: str = "pending"  # pending / resolved
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "rating": self.rating,
            "decision": self.decision,
            "reflection": self.reflection,
            "raw_return": self.raw_return,
            "alpha_return": self.alpha_return,
            "status": self.status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(
            ticker=d.get("ticker", ""),
            trade_date=d.get("trade_date", ""),
            rating=d.get("rating", "Hold"),
            decision=d.get("decision", ""),
            reflection=d.get("reflection", ""),
            raw_return=d.get("raw_return"),
            alpha_return=d.get("alpha_return"),
            status=d.get("status", "pending"),
            timestamp=d.get("timestamp", ""),
        )


class _FileLock:
    """Best-effort cross-process + in-process lock guarding the log file.

    Major #4 fix: on Windows, `fcntl` is unavailable. We now try the
    `portalocker` library (if installed) for cross-platform file locking.
    If neither `fcntl` nor `portalocker` is available, we fall back to
    a per-process `threading.Lock` — this is safe for single-process
    usage (the common case) but NOT for multi-process on Windows.

    Install `portalocker` (`pip install portalocker`) for true
    cross-platform multi-process safety.
    """

    _proc_locks: dict[str, threading.Lock] = {}
    _proc_locks_guard = threading.Lock()

    # Try to import portalocker for cross-platform locking.
    try:
        import portalocker as _portalocker
        _HAS_PORTALOCKER = True
    except ImportError:
        _HAS_PORTALOCKER = False

    def __init__(self, path: Path) -> None:
        self._lock_path = str(path) + ".lock"
        with _FileLock._proc_locks_guard:
            self._thread_lock = _FileLock._proc_locks.setdefault(
                str(path), threading.Lock(),
            )
        self._fh = None

    def __enter__(self) -> "_FileLock":
        self._thread_lock.acquire()
        self._fh = open(self._lock_path, "a")
        if _HAS_FCNTL:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        elif _FileLock._HAS_PORTALOCKER:
            _FileLock._portalocker.lock(self._fh, _FileLock._portalocker.LOCK_EX)
        # else: no cross-process lock available — in-process only.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if _HAS_FCNTL and self._fh is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            elif _FileLock._HAS_PORTALOCKER and self._fh is not None:
                _FileLock._portalocker.unlock(self._fh)
        finally:
            if self._fh is not None:
                self._fh.close()
            self._thread_lock.release()


# ----------------------------------------------------------------------
class TradingMemoryLog:
    """Append-only JSONL decision log."""

    def __init__(self, path: str = "data/agent_memory_log.jsonl",
                 max_entries: Optional[int] = 100,
                 llm: Optional[LLMProvider] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.llm = llm or LLMProvider()
        self._lock = _FileLock(self.path)

    # ----------------------------------------------------------------
    # Phase A: store decision
    # ----------------------------------------------------------------
    def store_decision(self, ticker: str, trade_date: str,
                         final_decision: str) -> None:
        """Append a pending entry after the PM decides."""
        rating = parse_rating(final_decision)
        entry = MemoryEntry(
            ticker=ticker, trade_date=trade_date, rating=rating,
            decision=final_decision, status="pending",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )
        with self._lock:
            # Check for an existing pending entry for the same
            # ticker/date to avoid duplicates (M2).
            entries = self._load_entries_unlocked()
            if any(e.ticker == ticker and e.trade_date == trade_date
                   and e.status == "pending" for e in entries):
                log.info("Pending entry already exists for %s %s; skipping duplicate",
                          ticker, trade_date)
                return
            entries.append(entry)
            entries = self._trim(entries)
            self._write_all_unlocked(entries)
        log.info("Stored decision: %s %s %s", trade_date, ticker, rating)

    # ----------------------------------------------------------------
    # Phase B: reflect once outcome is known
    # ----------------------------------------------------------------
    def reflect_and_resolve(self, ticker: str, trade_date: str,
                              raw_return: float,
                              alpha_return: float = 0.0,
                              benchmark: str = "BTC") -> str:
        """Generate a reflection and mark the entry resolved."""
        with self._lock:
            entries = self._load_entries_unlocked()
            target = None
            for e in entries:
                if e.ticker == ticker and e.trade_date == trade_date and e.status == "pending":
                    target = e
                    break
            if target is None:
                log.warning("No pending entry for %s %s", ticker, trade_date)
                return ""
            reflection = self._generate_reflection(
                target.decision, raw_return, alpha_return, benchmark,
            )
            target.reflection = reflection
            target.raw_return = raw_return
            target.alpha_return = alpha_return
            target.status = "resolved"
            target.timestamp = datetime.now(tz=timezone.utc).isoformat()
            self._write_all_unlocked(entries)
        log.info("Resolved %s %s: return=%.2f%% alpha=%.2f%%",
                 ticker, trade_date, raw_return * 100, alpha_return * 100)
        return reflection

    # Convenience alias — this is the entry point the main trading
    # loop should call once a trade's outcome is known (fixes C7/X1:
    # previously nothing in the codebase ever called this).
    resolve_trade_outcome = reflect_and_resolve

    # ----------------------------------------------------------------
    def _generate_reflection(self, decision: str, raw_return: float,
                               alpha_return: float,
                               benchmark: str) -> str:
        """LLM generates a 2-4 sentence reflection on the decision."""
        system_prompt = (
            "You are a trading analyst reviewing your own past decision now that "
            "the outcome is known. Write exactly 2-4 sentences of plain prose "
            "(no bullets, no headers). Cover: was the call correct (cite alpha), "
            "which part of the thesis held or failed, and one concrete lesson."
        )
        user_prompt = (
            f"Decision:\n{decision[:1000]}\n\n"
            f"Raw return: {raw_return:.2%}\n"
            f"Alpha vs {benchmark}: {alpha_return:.2%}\n\n"
            "Write your reflection (2-4 sentences)."
        )
        try:
            resp = self.llm.chat(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                max_tokens=200,
                temperature=0.3,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Reflection LLM call raised: %r", e)
            return f"[Reflection unavailable: {e}]"
        return resp.text if resp.success else f"[Reflection unavailable: {resp.error}]"

    # ----------------------------------------------------------------
    # Read path
    # ----------------------------------------------------------------
    def load_entries(self) -> list[MemoryEntry]:
        with self._lock:
            return self._load_entries_unlocked()

    def _load_entries_unlocked(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        entries: list[MemoryEntry] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append(MemoryEntry.from_dict(d))
                except (json.JSONDecodeError, TypeError) as e:
                    log.warning(
                        "Skipping malformed memory-log line %d in %s: %r",
                        line_no, self.path, e,
                    )
        return entries

    def get_past_context(self, ticker: Optional[str] = None,
                          limit: int = 5) -> str:
        """Get recent resolved entries as context for future agents.

        Minor #9 fix: the in-memory entry list is now cached and only
        re-read from disk when the file's mtime changes. This avoids
        re-reading the entire JSONL file on every call (which was O(n)
        in file size, called once per symbol per cycle).
        """
        entries = self._get_cached_entries()
        resolved = [e for e in entries if e.status == "resolved"]
        if ticker:
            resolved = [e for e in resolved if e.ticker == ticker]
        resolved = resolved[-limit:]
        if not resolved:
            return ""
        lines = []
        for e in resolved:
            ret_str = f"{e.raw_return:.2%}" if e.raw_return is not None else "N/A"
            alpha_str = f"{e.alpha_return:.2%}" if e.alpha_return is not None else "N/A"
            lines.append(
                f"- [{e.trade_date} | {e.ticker} | {e.rating}] "
                f"Return: {ret_str}, Alpha: {alpha_str}. "
                f"Lesson: {e.reflection}"
            )
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Minor #9 fix: in-memory cache for entries, invalidated on file mtime.
    # ----------------------------------------------------------------
    def _get_cached_entries(self) -> list[MemoryEntry]:
        """Return cached entries, re-reading from disk only if the file
        has been modified since the last read."""
        import os as _os
        try:
            mtime = _os.path.getmtime(self.path) if self.path.exists() else 0
        except OSError:
            mtime = 0
        cache = getattr(self, "_entries_cache", None)
        cache_mtime = getattr(self, "_entries_cache_mtime", -1)
        if cache is not None and mtime == cache_mtime:
            return cache
        # File changed (or first read) — reload.
        with self._lock:
            entries = self._load_entries_unlocked()
        self._entries_cache = entries
        self._entries_cache_mtime = mtime
        return entries

    # ----------------------------------------------------------------
    def _trim(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Enforce max_entries by dropping the oldest resolved entries
        first, then oldest pending if still over budget. (L6)"""
        if not self.max_entries or len(entries) <= self.max_entries:
            return entries
        overflow = len(entries) - self.max_entries
        # Prefer dropping resolved entries (pending ones still need
        # to be reflected on) — but never drop more than exists.
        resolved_idx = [i for i, e in enumerate(entries) if e.status == "resolved"]
        drop = set(resolved_idx[:overflow])
        if len(drop) < overflow:
            remaining = overflow - len(drop)
            other_idx = [i for i in range(len(entries)) if i not in drop]
            drop.update(other_idx[:remaining])
        trimmed = [e for i, e in enumerate(entries) if i not in drop]
        log.info("Trimmed %d old memory entries (max_entries=%d)",
                  len(entries) - len(trimmed), self.max_entries)
        return trimmed

    def _write_all_unlocked(self, entries: list[MemoryEntry]) -> None:
        """Rewrite the entire log atomically (write to temp + rename)."""
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False))
                f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)
        # Minor #9 fix: update the in-memory cache so the next
        # get_past_context() call doesn't re-read the file.
        self._entries_cache = list(entries)
        try:
            self._entries_cache_mtime = os.path.getmtime(self.path)
        except OSError:
            pass

    # Kept for backward-compat call sites that may reference the old
    # private name.
    def _rewrite_all(self, entries: list[MemoryEntry]) -> None:
        with self._lock:
            self._write_all_unlocked(entries)