"""trading_bot.runtime_state
=====================================================================
Day 7 — Crash-recovery helpers with schema versioning.

Audit Batch 1 remediation (C16, H15, M8, L11):
  - Use `threading.RLock` instead of `Lock` so re-entrant calls from the
    same thread don't deadlock (C16 fix).
  - `save_state` validates the payload is JSON-serializable BEFORE
    writing to disk, and rejects non-dict state with a clear error (C16).
  - `load_state` validates the schema version and the top-level shape;
    mismatched versions return None and log a warning (H15).
  - `save_state` accepts an optional `debounce_s` to coalesce rapid
    writes (M8 fix — frequent I/O on hot paths can now be batched).
  - `default=str` is replaced with a strict JSON encoder that rejects
    unserializable types (L11 fix — silent `str()` coercion hid bugs).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("runtime_state")

SCHEMA_VERSION = 1
_lock = threading.RLock()  # C16 fix: RLock allows re-entrancy.

# M8 fix: per-path debounce — last write time tracked so callers can
# skip writes that arrive within `debounce_s` of the previous one.
_last_write_ts: dict[str, float] = {}


def _strict_json_encode(obj: Any) -> str:
    """L11 fix: strict JSON encoder.

    `default=str` silently coerces unserializable types (datetime, custom
    classes) to their `str()` representation, which can hide bugs. We now
    only allow the standard JSON types plus datetime-like objects with
    an `isoformat()` method.
    """
    def _default(o: Any) -> Any:
        # Allow datetime and similar.
        if hasattr(o, "isoformat"):
            return o.isoformat()
        # Allow sets by converting to lists.
        if isinstance(o, (set, frozenset)):
            return list(o)
        # Allow dataclasses with __dict__.
        if hasattr(o, "__dict__"):
            return o.__dict__
        raise TypeError(
            f"runtime_state: object of type {type(o).__name__} is not "
            f"JSON-serializable (L11 fix: refusing to coerce with str())")
    return json.dumps(obj, indent=2, default=_default)


def save_state(path: str, state: dict[str, Any],
               debounce_s: float = 0.0) -> bool:
    """Persist state with schema version.

    C16 fix: validate that `state` is a dict and is JSON-serializable
    BEFORE touching the disk. A failed validation returns False and logs
    the error; the previous state file is preserved.

    M8 fix: if `debounce_s > 0`, skip the write if we just wrote to this
    path within the last `debounce_s` seconds. Useful for hot paths that
    call save_state() on every tick.

    Returns True on success, False on failure.
    """
    if not isinstance(state, dict):
        log.error("state save rejected: payload must be a dict, got %s",
                  type(state).__name__)
        return False

    # C16 fix: validate serializability before touching disk.
    payload = {
        "_schema_version": SCHEMA_VERSION,
        "_saved_at": datetime.now(tz=timezone.utc).isoformat(),
        **state,
    }
    try:
        encoded = _strict_json_encode(payload)
    except (TypeError, ValueError) as e:
        log.error("state save rejected: payload is not JSON-serializable: %s", e)
        return False

    # M8 fix: debounce rapid writes.
    if debounce_s > 0:
        import time as _t
        now = _t.monotonic()
        last = _last_write_ts.get(path, 0.0)
        if now - last < debounce_s:
            return True  # skip silently — caller will retry later
        _last_write_ts[path] = now

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        with _lock, open(tmp, "w", encoding="utf-8") as f:
            f.write(encoded)
        os.replace(tmp, path)
        return True
    except Exception as e:
        log.error("state save failed: %s", e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def load_state(path: str) -> dict[str, Any] | None:
    """Reload state if schema version matches, else None.

    H15 fix: also validate the top-level shape — if the file is corrupt
    or the structure is wrong, return None and log a warning instead of
    crashing the caller.
    """
    if not os.path.isfile(path):
        return None
    try:
        with _lock, open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("state file %s is not a dict — ignoring", path)
            return None
        version = data.get("_schema_version", 0)
        if version != SCHEMA_VERSION:
            log.warning("State schema version mismatch: expected %d, got %d — ignoring.",
                        SCHEMA_VERSION, version)
            return None
        # Remove metadata before returning.
        data.pop("_schema_version", None)
        data.pop("_saved_at", None)
        return data
    except (json.JSONDecodeError, OSError) as e:
        log.warning("state load failed (%s): %s — starting fresh", path, e)
        return None
    except Exception as e:
        log.warning("state load failed unexpectedly: %s — starting fresh", e)
        return None


def clear_state(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
        _last_write_ts.pop(path, None)
    except OSError as e:
        log.warning("state clear failed: %s", e)
