"""factory.strategy_versioning
=====================================================================
Day 61-63 — Strategy Versioning System.

Every strategy has a full version history with:
  - SemVer version numbers (major.minor.patch)
  - Performance history per version
  - Decay tracking (rolling Sharpe per version)
  - Rollback capability (revert to previous version instantly)

Version semantics:
  - PATCH : parameter tuning, no signal logic change
  - MINOR : new feature/filter added, signal logic compatible
  - MAJOR : signal logic change → backtest must be re-run

Version states:
  - staging   : passed CI but not promoted
  - production: live in the strategy pool
  - retired   : auto-retired or manually retired
  - rolled_back: superseded by a rollback
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("factory.versioning")


@dataclass
class StrategyVersion:
    strategy_name: str
    version: str
    registered_at: float
    hypothesis: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "staging"          # staging | production | retired | rolled_back
    promoted_at: Optional[float] = None
    retired_at: Optional[float] = None
    retirement_reason: str = ""
    # Rolling performance (updated live)
    live_sharpe: Optional[float] = None
    live_drawdown: Optional[float] = None
    live_cycles: int = 0
    # Decay tracking
    decay_score: float = 1.0          # 1.0 = no decay, 0.0 = full decay

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyVersion":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ----------------------------------------------------------------------
class StrategyVersionStore:
    """Persistent store of strategy versions."""

    def __init__(self, path: str = "data/strategy_versions.json") -> None:
        self.path = path
        # strategy_name -> {version_string -> StrategyVersion}
        self._versions: dict[str, dict[str, StrategyVersion]] = {}
        self._load()

    # ----------------------------------------------------------------
    def register(self, version: StrategyVersion) -> None:
        d = self._versions.setdefault(version.strategy_name, {})
        d[version.version] = version
        self._save()
        log.info("Registered %s v%s (status=%s)",
                 version.strategy_name, version.version, version.status)

    def get(self, strategy_name: str,
            version: Optional[str] = None) -> Optional[StrategyVersion]:
        d = self._versions.get(strategy_name)
        if not d:
            return None
        if version is None:
            # Return the latest production or staging version
            prod = [v for v in d.values() if v.status == "production"]
            if prod:
                return max(prod, key=lambda v: v.registered_at)
            return max(d.values(), key=lambda v: v.registered_at)
        return d.get(version)

    def list_versions(self, strategy_name: str) -> list[StrategyVersion]:
        d = self._versions.get(strategy_name, {})
        return sorted(d.values(), key=lambda v: v.registered_at)

    def list_strategies(self) -> list[str]:
        return list(self._versions.keys())

    # ----------------------------------------------------------------
    def bump_version(self, strategy_name: str,
                     kind: str = "patch") -> str:
        """Return the next version string for a strategy."""
        existing = self._versions.get(strategy_name, {})
        if not existing:
            return "1.0.0"
        # Find highest existing version
        latest = max(existing.values(), key=lambda v: v.registered_at)
        parts = [int(x) for x in latest.version.split(".")]
        while len(parts) < 3:
            parts.append(0)
        major, minor, patch = parts[0], parts[1], parts[2]
        if kind == "major":
            major += 1
            minor = 0
            patch = 0
        elif kind == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1
        return f"{major}.{minor}.{patch}"

    # ----------------------------------------------------------------
    def promote(self, strategy_name: str, version: str) -> bool:
        v = self.get(strategy_name, version)
        if v is None:
            return False
        # Demote any currently-production version
        for other in self._versions.get(strategy_name, {}).values():
            if other.status == "production" and other.version != version:
                other.status = "rolled_back"
                log.info("Demoted %s v%s to rolled_back",
                         strategy_name, other.version)
        v.status = "production"
        v.promoted_at = time.time()
        self._save()
        log.info("PROMOTED %s v%s to production", strategy_name, version)
        return True

    def retire(self, strategy_name: str, version: str,
               reason: str = "") -> bool:
        v = self.get(strategy_name, version)
        if v is None:
            return False
        v.status = "retired"
        v.retired_at = time.time()
        v.retirement_reason = reason
        self._save()
        log.warning("RETIRED %s v%s reason=%s", strategy_name, version, reason)
        return True

    def rollback(self, strategy_name: str) -> Optional[str]:
        """Roll back to the previous production version."""
        versions = self.list_versions(strategy_name)
        prod_versions = [v for v in versions if v.status in ("rolled_back", "production")]
        if len(prod_versions) < 2:
            log.warning("No version to roll back to for %s", strategy_name)
            return None
        # Current production
        current = next((v for v in prod_versions if v.status == "production"), None)
        if current is None:
            return None
        # Promote the most recent rolled_back
        candidates = [v for v in prod_versions
                      if v.status == "rolled_back"
                      and v.registered_at < current.registered_at]
        if not candidates:
            return None
        target = max(candidates, key=lambda v: v.registered_at)
        current.status = "rolled_back"
        target.status = "production"
        target.promoted_at = time.time()
        self._save()
        log.info("ROLLBACK %s: v%s -> v%s",
                 strategy_name, current.version, target.version)
        return target.version

    # ----------------------------------------------------------------
    def update_live_metrics(self, strategy_name: str, version: str,
                            sharpe: float, drawdown: float,
                            cycles: int) -> None:
        v = self.get(strategy_name, version)
        if v is None:
            return
        v.live_sharpe = float(sharpe)
        v.live_drawdown = float(drawdown)
        v.live_cycles = int(cycles)
        self._save()

    def set_decay_score(self, strategy_name: str, version: str,
                        decay_score: float) -> None:
        v = self.get(strategy_name, version)
        if v is None:
            return
        v.decay_score = float(decay_score)
        self._save()

    # ----------------------------------------------------------------
    def production_strategies(self) -> list[StrategyVersion]:
        out = []
        for strat_versions in self._versions.values():
            for v in strat_versions.values():
                if v.status == "production":
                    out.append(v)
        return out

    # ----------------------------------------------------------------
    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        try:
            payload = {
                name: {ver: v.to_dict() for ver, v in d.items()}
                for name, d in self._versions.items()
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self.path)
        except Exception as e:  # noqa: BLE001
            log.error("version store save failed: %r", e)

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for name, d in payload.items():
                self._versions[name] = {
                    ver: StrategyVersion.from_dict(v) for ver, v in d.items()
                }
            log.info("Loaded %d strategies from version store",
                     len(self._versions))
        except Exception as e:  # noqa: BLE001
            log.warning("version store load failed: %r", e)
