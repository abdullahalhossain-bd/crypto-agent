"""architecture/recovery_engine.py
=====================================================================
Recovery Engine + Snapshot Engine + Config Versioning (Improvements #15-17)
=====================================================================
Combines three tightly-coupled reliability systems:

1. SNAPSHOT ENGINE (#16)
   Periodically snapshots all bot state to disk so we can resume
   exactly where we left off after a crash. Snapshots include:
   - Open positions (with entry, SL, TP)
   - Portfolio state (equity, peak, drawdown)
   - Risk state (consecutive losses, cooldown timers)
   - Decision audit ring buffer
   - Recent equity curve

2. RECOVERY ENGINE (#15)
   On startup, detect if the previous session crashed (no clean
   shutdown marker). If so, restore from latest snapshot:
   - Reload positions into PortfolioManager
   - Reset risk state to snapshot values
   - Skip the warmup phase (already have indicators)
   - Replay any pending events
   - Transition state machine: RECOVERY → LIVE

3. CONFIG VERSIONING (#17)
   Every config change is versioned. Rollback to any prior version.
   - Auto-snapshot config on every change
   - Diff viewer between versions
   - Audit log of who changed what when

Usage:
    snap = SnapshotEngine(snapshot_dir="data/snapshots")
    snap.take_snapshot(portfolio=pm, risk_state=rs, cycle=42)

    recovery = RecoveryEngine(snapshot_dir="data/snapshots")
    if recovery.detect_crash():
        state = recovery.restore_latest()
        # state.portfolio, state.risk_state, state.cycle

    cfg_mgr = ConfigVersioning(config_path="config/config.yaml")
    cfg_mgr.snapshot("added new symbol XRPUSD")
    cfg_mgr.rollback("2024-03-15_14:30_v5")
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.recovery_engine")


# ======================================================================
# SNAPSHOT ENGINE
# ======================================================================
@dataclass
class BotSnapshot:
    """Serializable snapshot of entire bot state."""
    snapshot_id: str = ""
    timestamp: str = ""
    cycle: int = 0
    # Portfolio
    portfolio: Dict[str, Any] = field(default_factory=dict)
    # Risk state
    risk_state: Dict[str, Any] = field(default_factory=dict)
    # Open positions
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    # Equity
    equity: float = 0.0
    peak_equity: float = 0.0
    # State machine
    bot_state: str = "LIVE"
    # Config version
    config_version: str = ""
    # Metadata
    version: str = "1.0"
    notes: str = ""


class SnapshotEngine:
    """Periodic snapshot of bot state for crash recovery."""

    CLEAN_SHUTDOWN_MARKER = ".clean_shutdown"

    def __init__(self,
                 snapshot_dir: str = "data/snapshots",
                 max_snapshots: int = 100,
                 bus: Optional[EventBus] = None):
        self._dir = Path(snapshot_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max = max_snapshots
        self._bus = bus or get_bus()
        self._lock = threading.RLock()

    def take_snapshot(self,
                      portfolio: Any = None,
                      risk_state: Any = None,
                      open_positions: Optional[List[Dict]] = None,
                      cycle: int = 0,
                      equity: float = 0.0,
                      peak_equity: float = 0.0,
                      bot_state: str = "LIVE",
                      config_version: str = "",
                      notes: str = "") -> str:
        """Take a snapshot and persist to disk."""
        snap_id = f"snap_{int(time.time())}_{cycle}"
        snap = BotSnapshot(
            snapshot_id=snap_id,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            cycle=cycle,
            portfolio=_safe_dict(portfolio),
            risk_state=_safe_dict(risk_state),
            open_positions=open_positions or [],
            equity=equity,
            peak_equity=peak_equity,
            bot_state=bot_state,
            config_version=config_version,
            notes=notes,
        )
        path = self._dir / f"{snap_id}.json"
        with self._lock:
            with open(path, "w") as f:
                json.dump(asdict(snap), f, indent=2, default=str)
            self._prune_old()
        self._bus.emit(EventType.SNAPSHOT_TAKEN,
                      payload={"snapshot_id": snap_id, "path": str(path),
                              "cycle": cycle},
                      source="snapshot_engine")
        log.info("snapshot: saved %s (cycle=%d, equity=%.2f, positions=%d)",
                 snap_id, cycle, equity, len(open_positions or []))
        return snap_id

    def latest_snapshot(self) -> Optional[BotSnapshot]:
        """Load the most recent snapshot from disk."""
        with self._lock:
            files = sorted(self._dir.glob("snap_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                return None
            try:
                with open(files[0]) as f:
                    data = json.load(f)
                return BotSnapshot(**data)
            except Exception as e:  # noqa: BLE001
                log.warning("snapshot: failed to load %s: %r", files[0], e)
                return None

    def list_snapshots(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            files = sorted(self._dir.glob("snap_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
            out = []
            for f in files:
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    out.append({
                        "snapshot_id": d["snapshot_id"],
                        "timestamp": d["timestamp"],
                        "cycle": d["cycle"],
                        "equity": d["equity"],
                        "positions": len(d.get("open_positions", [])),
                    })
                except (KeyError, ValueError, TypeError) as e:
                    # H9 fix: log the full traceback (not just `e` which may
                    # be a tuple) so the root cause is visible. Include the
                    # filename so the operator can inspect/delete the bad file.
                    import traceback as _tb
                    log.warning("recovery_engine: malformed snapshot %s skipped: %r\n%s",
                                f.name, e, _tb.format_exc())
            return out

    def mark_clean_shutdown(self) -> None:
        """Write a marker file indicating the bot shut down cleanly."""
        marker = self._dir / self.CLEAN_SHUTDOWN_MARKER
        with open(marker, "w") as f:
            f.write(datetime.now(tz=timezone.utc).isoformat())

    def clear_clean_shutdown(self) -> None:
        marker = self._dir / self.CLEAN_SHUTDOWN_MARKER
        if marker.exists():
            marker.unlink()

    def _prune_old(self) -> None:
        """Keep only the last max_snapshots files."""
        files = sorted(self._dir.glob("snap_*.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[self._max:]:
            try:
                f.unlink()
            except OSError as e:
                # Phase 7: log prune failures — usually permission issues
                log.warning("recovery_engine: could not prune old snapshot %s: %r",
                           f.name, e)


# ======================================================================
# RECOVERY ENGINE
# ======================================================================
class RecoveryEngine:
    """Detects crashes and restores bot state from latest snapshot."""

    def __init__(self,
                 snapshot_engine: SnapshotEngine,
                 bus: Optional[EventBus] = None):
        self._snap = snapshot_engine
        self._bus = bus or get_bus()

    def detect_crash(self) -> bool:
        """Returns True if the previous session did NOT shut down cleanly."""
        marker = self._snap._dir / SnapshotEngine.CLEAN_SHUTDOWN_MARKER
        return not marker.exists()

    def restore_latest(self) -> Optional[BotSnapshot]:
        """Restore state from the latest snapshot.

        Emits SNAPSHOT_RESTORED event. Caller is responsible for
        applying the snapshot to the appropriate components.
        """
        snap = self._snap.latest_snapshot()
        if snap is None:
            log.warning("recovery: no snapshot found — cold start")
            return None
        self._bus.emit(EventType.SNAPSHOT_RESTORED,
                      payload={"snapshot_id": snap.snapshot_id,
                              "cycle": snap.cycle, "equity": snap.equity},
                      source="recovery_engine")
        self._bus.emit(EventType.RECOVERY_STARTED,
                      payload={"from_cycle": snap.cycle},
                      source="recovery_engine")
        log.info("recovery: restoring from %s (cycle=%d, equity=%.2f, %d positions)",
                 snap.snapshot_id, snap.cycle, snap.equity,
                 len(snap.open_positions))
        return snap

    def apply_snapshot(self,
                       snap: BotSnapshot,
                       portfolio: Any = None,
                       risk_state: Any = None) -> None:
        """Apply a snapshot to portfolio + risk state objects.

        FIX-RE-02: previously called portfolio.reset(capital=balance),
        which unconditionally reset peak_equity to the post-recovery
        balance — silently zeroing the drawdown high-water mark on every
        crash. Now uses restore_from_dict() (see PortfolioManager,
        FIX-RE-01) so balance, realized PnL, peak_equity, and max_drawdown
        all survive the recovery, and positions are re-added on top of
        that restored state rather than a freshly reset one.
        """
        if portfolio is not None and snap.portfolio:
            try:
                if hasattr(portfolio, "restore_from_dict"):
                    # FIX-RE-02: restores balance + drawdown history
                    # (peak_equity, max_drawdown) instead of resetting it.
                    portfolio.restore_from_dict(snap.portfolio)
                elif hasattr(portfolio, "reset"):
                    # Backward-compat fallback for a PortfolioManager
                    # implementation that hasn't picked up FIX-RE-01 yet.
                    log.warning("recovery: portfolio has no restore_from_dict() — "
                               "falling back to reset(), drawdown history will "
                               "NOT be preserved across this recovery")
                    portfolio.reset(capital=snap.portfolio.get("balance", snap.equity))

                # Re-add open positions on top of the restored balance/history
                for pos in snap.open_positions:
                    if hasattr(portfolio, "on_position_opened"):
                        portfolio.on_position_opened(
                            ticket=pos.get("ticket", 0),
                            symbol=pos.get("symbol", ""),
                            side=pos.get("side", "BUY"),
                            volume=pos.get("volume", 0),
                            entry_price=pos.get("entry_price", 0),
                            sl=pos.get("sl", 0),
                            tp=pos.get("tp", 0),
                            magic=pos.get("magic", 0),
                        )
            except Exception as e:  # noqa: BLE001
                log.warning("recovery: failed to apply portfolio: %r", e)

        if risk_state is not None and snap.risk_state:
            try:
                if hasattr(risk_state, "from_dict"):
                    risk_state.from_dict(snap.risk_state)
            except Exception as e:  # noqa: BLE001
                log.warning("recovery: failed to apply risk_state: %r", e)


# ======================================================================
# CONFIG VERSIONING
# ======================================================================
@dataclass
class ConfigVersion:
    version_id: str = ""
    timestamp: str = ""
    description: str = ""
    config_path: str = ""
    snapshot_path: str = ""
    hash: str = ""


class ConfigVersioning:
    """Versioned config management with rollback."""

    def __init__(self,
                 config_path: str = "config/config.yaml",
                 versions_dir: str = "data/config_versions",
                 max_versions: int = 50):
        self._config_path = Path(config_path)
        self._versions_dir = Path(versions_dir)
        self._versions_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._audit_log: List[Dict[str, Any]] = []
        # M4 fix: retention cap — old config snapshots are pruned so the
        # versions directory doesn't grow indefinitely (was unbounded).
        self._max_versions = int(max_versions)

    def snapshot(self, description: str = "") -> str:
        """Take a versioned snapshot of the current config file."""
        if not self._config_path.exists():
            log.warning("config_versioning: config file not found")
            return ""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_id = f"{ts}"
        snap_path = self._versions_dir / f"config_{version_id}.yaml"
        with self._lock:
            shutil.copy2(self._config_path, snap_path)
            import hashlib
            with open(snap_path, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
            self._audit_log.append({
                "version_id": version_id,
                "timestamp": ts,
                "description": description,
                "snapshot_path": str(snap_path),
                "hash": h,
            })
            # M4 fix: prune old snapshots beyond max_versions.
            self._prune_old_versions()
        log.info("config_versioning: snapshot %s — %s", version_id, description)
        return version_id

    def _prune_old_versions(self) -> int:
        """M4 fix: delete the oldest snapshot files beyond _max_versions.
        Returns the number of files pruned."""
        try:
            files = sorted(self._versions_dir.glob("config_*.yaml"),
                          key=lambda p: p.stat().st_mtime)
            pruned = 0
            while len(files) > self._max_versions:
                oldest = files.pop(0)
                try:
                    oldest.unlink()
                    pruned += 1
                except OSError:
                    pass
            if pruned:
                log.info("config_versioning: pruned %d old snapshots (max=%d)",
                         pruned, self._max_versions)
            return pruned
        except Exception as e:  # noqa: BLE001
            log.warning("config_versioning: prune failed: %r", e)
            return 0

    def list_versions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted(self._audit_log, key=lambda v: v["timestamp"],
                         reverse=True)[:limit]

    def rollback(self, version_id: str) -> bool:
        """Restore the config file from a previous version."""
        snap_path = self._versions_dir / f"config_{version_id}.yaml"
        if not snap_path.exists():
            log.warning("config_versioning: version %s not found", version_id)
            return False
        # Take a snapshot of current before rolling back
        self.snapshot(f"pre-rollback to {version_id}")
        with self._lock:
            shutil.copy2(snap_path, self._config_path)
        log.info("config_versioning: rolled back to %s", version_id)
        return True

    def diff(self, version_id: str) -> List[str]:
        """Show line-by-line diff between current and a previous version."""
        snap_path = self._versions_dir / f"config_{version_id}.yaml"
        if not snap_path.exists() or not self._config_path.exists():
            return []
        import difflib
        with open(snap_path) as f:
            old_lines = f.readlines()
        with open(self._config_path) as f:
            new_lines = f.readlines()
        return list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"config_{version_id}", tofile="current",
            lineterm="",
        ))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _safe_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of an object to a dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception as e:
            # Phase 7: log serialization failure — was silently swallowed
            log.debug("recovery_engine: to_dict() failed for %s: %r",
                     type(obj).__name__, e)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items()
                if not k.startswith("_") and _is_jsonable(v)}
    return {}


def _is_jsonable(v: Any) -> bool:
    try:
        json.dumps(v, default=str)
        return True
    except Exception:
        return False