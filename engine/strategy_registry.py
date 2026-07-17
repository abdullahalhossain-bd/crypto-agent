"""engine.strategy_registry
=====================================================================
Day 9 — Strategy registry with versioning and metadata tracking.

A registry decouples "what strategies exist" from "what strategies
should run on this symbol". The config file references strategies by
name; the registry looks up the class.

Usage:
    from engine.strategy_registry import REGISTRY
    REGISTRY.register(SmaCrossoverStrategy)
    cls = REGISTRY.get("sma_crossover", "2.0.0")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from engine.strategies.base import Strategy, StrategyMetadata
from engine.strategies.sma_cross import SmaCrossoverStrategy
from engine.strategies.breakout import BreakoutStrategy
from engine.strategies.mean_reversion import MeanReversionStrategy
from utils.logger import get_logger

log = get_logger("engine.strategy_registry")


@dataclass
class _RegistryEntry:
    cls: Type[Strategy]
    metadata: StrategyMetadata


class StrategyRegistry:
    """Central catalog of available strategies."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, _RegistryEntry]] = {}

    def register(self, strategy_cls: Type[Strategy]) -> Type[Strategy]:
        """Idempotently register a strategy class.

        The class MUST expose a `metadata: StrategyMetadata` attribute.
        Re-registering the same (name, version) overwrites silently.
        """
        meta: StrategyMetadata = getattr(strategy_cls, "metadata", None)
        if meta is None:
            raise TypeError(f"{strategy_cls.__name__} has no metadata attribute")
        if not isinstance(meta, StrategyMetadata):
            raise TypeError(f"{strategy_cls.__name__}.metadata must be StrategyMetadata")
        versions = self._entries.setdefault(meta.name, {})
        versions[meta.version] = _RegistryEntry(cls=strategy_cls, metadata=meta)
        log.debug("registered strategy %s v%s", meta.name, meta.version)
        return strategy_cls

    def unregister(self, name: str, version: Optional[str] = None) -> None:
        if name not in self._entries:
            return
        if version is None:
            del self._entries[name]
            log.debug("unregistered all versions of %s", name)
            return
        self._entries[name].pop(version, None)
        if not self._entries[name]:
            del self._entries[name]

    def get(self, name: str, version: Optional[str] = None) -> Type[Strategy]:
        """Look up a strategy class. Latest version if `version` is None."""
        if name not in self._entries or not self._entries[name]:
            raise KeyError(f"strategy not registered: {name}")
        if version is None:
            # Pick the highest semver-like version
            version = sorted(self._entries[name].keys())[-1]
        if version not in self._entries[name]:
            raise KeyError(f"strategy {name} v{version} not registered")
        return self._entries[name][version].cls

    def list_strategies(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, versions in self._entries.items():
            for v, entry in versions.items():
                m = entry.metadata
                out.append({
                    "name": m.name,
                    "version": m.version,
                    "author": m.author,
                    "description": m.description,
                    "min_bars": m.min_bars,
                    "tags": list(m.tags),
                    "regime_affinity": dict(m.regime_affinity),
                })
        return out

    def metadata(self, name: str, version: Optional[str] = None) -> StrategyMetadata:
        if name not in self._entries or not self._entries[name]:
            raise KeyError(name)
        if version is None:
            version = sorted(self._entries[name].keys())[-1]
        return self._entries[name][version].metadata

    def clear(self) -> None:
        self._entries.clear()


# ----------------------------------------------------------------------
# Global singleton
# ----------------------------------------------------------------------
REGISTRY = StrategyRegistry()

# Auto-register the built-in strategies on import
REGISTRY.register(SmaCrossoverStrategy)
REGISTRY.register(BreakoutStrategy)
REGISTRY.register(MeanReversionStrategy)
