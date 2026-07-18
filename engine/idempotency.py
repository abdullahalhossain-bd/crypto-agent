"""engine.idempotency
=====================================================================
Day 26 — Idempotency layer.

Prevents duplicate trades from any of these scenarios:
  - Main loop crashes mid-cycle and the next cycle re-evaluates the
    same bar (because state wasn't persisted in time)
  - Strategy fires the same signal twice on the same bar timestamp
  - Operator restarts the bot mid-order and the recovery code re-sends

Approach: every order attempt is keyed by (symbol, action, bar_time,
strategy). We persist a hash of the key to `data/seen_orders.json`
before sending. On restart we reload the file and skip any key that's
already present.

The store is intentionally tiny (just a dict of hash → timestamp) so
it survives restarts and is easy to inspect.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

log = get_logger("engine.idempotency")


@dataclass
class IdempotencyStore:
    """Persistent seen-order store."""
    path: str
    _seen: dict[str, float] = None  # type: ignore[assignment]
    _lock: threading.Lock = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._seen = {}
        self._load()

    # ----------------------------------------------------------------
    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self._seen = json.load(f)
        except Exception as e:  # noqa: BLE001
            log.warning("idempotency load failed (%r) — starting empty", e)
            self._seen = {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._seen, f, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            log.error("idempotency save failed: %r", e)

    # ----------------------------------------------------------------
    @staticmethod
    def make_key(symbol: str, action: str, bar_time: Any,
                 strategy: str = "") -> str:
        """Build a deterministic key from the order context."""
        bt = ""
        if bar_time is not None:
            try:
                bt = bar_time.isoformat()
            except AttributeError:
                bt = str(bar_time)
        raw = f"{symbol}|{action}|{bt}|{strategy}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def seen(self, key: str) -> bool:
        with self._lock:
            return key in self._seen

    def mark_seen(self, key: str) -> None:
        with self._lock:
            self._seen[key] = time.time()
            self._save()

    def check_and_mark(self, key: str) -> bool:
        """Returns True if this is a NEW key (i.e. order should proceed).
        Returns False if we've already seen it (skip)."""
        with self._lock:
            if key in self._seen:
                return False
            self._seen[key] = time.time()
            self._save()
            return True

    def expire_older_than(self, seconds: float) -> int:
        """Drop entries older than `seconds` (housekeeping)."""
        cutoff = time.time() - seconds
        with self._lock:
            before = len(self._seen)
            self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}
            after = len(self._seen)
            if before != after:
                self._save()
            return before - after

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)
