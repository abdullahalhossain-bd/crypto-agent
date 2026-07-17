"""enhancements.news_calendar
=====================================================================
Day 158 — Economic news calendar integration.

Tracks upcoming high-impact economic events that may affect the
market. The trade quality scorer uses this to enforce news blackouts.

Event sources (pluggable):
  - Manual entry (operator adds events)
  - JSON file (periodic fetch from external provider)
  - Future: HTTP API (ForexFactory, Investing.com, etc.)

Each event has:
  - timestamp
  - symbol(s) affected (BTCUSD, ETHUSD, USD, etc.)
  - impact (LOW / MEDIUM / HIGH)
  - forecast / previous / actual
  - category (CPI, FOMC, NFP, etc.)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("enhancements.news")


class NewsImpact(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class NewsEvent:
    event_id: str
    timestamp: str           # ISO format
    symbol: str              # affected symbol or currency
    title: str
    impact: NewsImpact
    category: str = ""       # CPI, FOMC, NFP, etc.
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["impact"] = self.impact.value
        return d

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.timestamp.replace("Z", "+00:00"))


# ----------------------------------------------------------------------
class NewsCalendar:
    def __init__(self, path: str = "data/news_calendar.json") -> None:
        self.path = path
        self._events: list[NewsEvent] = []
        self._load()

    # ----------------------------------------------------------------
    def add_event(self, event: NewsEvent) -> None:
        self._events.append(event)
        self._save()
        log.info("News event added: %s @ %s (%s)",
                 event.title, event.timestamp, event.impact.value)

    def add_events(self, events: list[NewsEvent]) -> None:
        self._events.extend(events)
        self._save()

    def remove_event(self, event_id: str) -> bool:
        before = len(self._events)
        self._events = [e for e in self._events if e.event_id != event_id]
        if len(self._events) < before:
            self._save()
            return True
        return False

    # ----------------------------------------------------------------
    def upcoming(self, hours_ahead: int = 24,
                  impact_min: NewsImpact = NewsImpact.MEDIUM) -> list[NewsEvent]:
        """Return upcoming events within `hours_ahead`."""
        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        impact_order = {NewsImpact.LOW: 1, NewsImpact.MEDIUM: 2, NewsImpact.HIGH: 3}
        min_level = impact_order[impact_min]
        out = []
        for e in self._events:
            try:
                if e.dt < now or e.dt > cutoff:
                    continue
                if impact_order[e.impact] < min_level:
                    continue
                out.append(e)
            except Exception:  # noqa: BLE001
                continue
        out.sort(key=lambda e: e.timestamp)
        return out

    # ----------------------------------------------------------------
    def is_blackout(self, symbol: str, minutes_before: int = 15,
                      minutes_after: int = 15,
                      impact_min: NewsImpact = NewsImpact.HIGH) -> tuple[bool, Optional[NewsEvent]]:
        """Check if we're currently in a news blackout for `symbol`."""
        now = datetime.now(tz=timezone.utc)
        before = now - timedelta(minutes=minutes_before)
        after = now + timedelta(minutes=minutes_after)
        impact_order = {NewsImpact.LOW: 1, NewsImpact.MEDIUM: 2, NewsImpact.HIGH: 3}
        min_level = impact_order[impact_min]
        for e in self._events:
            try:
                if impact_order[e.impact] < min_level:
                    continue
                # Check symbol match (event symbol could be "USD" and we trade "BTCUSD")
                if not self._symbol_matches(symbol, e.symbol):
                    continue
                if before <= e.dt <= after:
                    return (True, e)
            except Exception:  # noqa: BLE001
                continue
        return (False, None)

    @staticmethod
    def _symbol_matches(traded: str, event_symbol: str) -> bool:
        """Check if event symbol matches traded symbol. E.g. 'USD'
        matches 'BTCUSD', 'ETHUSD', 'EURUSD'."""
        traded_upper = traded.upper()
        event_upper = event_symbol.upper()
        if event_upper in traded_upper:
            return True
        if traded_upper in event_upper:
            return True
        return False

    # ----------------------------------------------------------------
    def all_events(self) -> list[NewsEvent]:
        return list(self._events)

    def summary(self) -> dict[str, Any]:
        return {
            "n_events": len(self._events),
            "n_high_impact": sum(1 for e in self._events if e.impact == NewsImpact.HIGH),
            "n_upcoming_24h": len(self.upcoming(24)),
        }

    # ----------------------------------------------------------------
    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump([e.to_dict() for e in self._events], f, indent=2)
        except Exception as e:  # noqa: BLE001
            log.warning("news calendar save failed: %r", e)

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                d["impact"] = NewsImpact(d.get("impact", "MEDIUM"))
                self._events.append(NewsEvent(**d))
            log.info("News calendar loaded: %d events", len(self._events))
        except Exception as e:  # noqa: BLE001
            log.warning("news calendar load failed: %r", e)
