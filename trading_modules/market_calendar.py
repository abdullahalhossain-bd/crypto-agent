"""
Market Calendar Intelligence — holidays, expiries, rollovers
=============================================================

Tracks market events that affect liquidity and volatility:

    1. Bank holidays           — US, UK, EU, JP
    2. Half trading days       — day before Christmas, day after Thanksgiving
    3. Futures contract rollover — quarterly (Mar/Jun/Sep/Dec)
    4. Options expiry          — monthly (3rd Friday) + quarterly
    5. Month-end / quarter-end / year-end  — rebalancing flows
    6. CME crypto futures expiry            — last Friday of month
    7. Crypto-specific events  — BTC halving (every 4 years), major upgrades

Usage:
    from trading_modules.market_calendar import MarketCalendar
    cal = MarketCalendar()
    events = cal.get_events(date=datetime.now(timezone.utc))
    for e in events:
        print(f"{e.event_type}: {e.description} (impact={e.impact})")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    event_type: str            # "holiday" / "half_day" / "futures_expiry" / "options_expiry" / "rollover" / "month_end" / "quarter_end" / "year_end" / "crypto_event"
    description: str
    impact: str                # "high" / "medium" / "low"
    markets_affected: list[str]  # ["US", "UK", "crypto", ...]
    expected_volatility_change: float  # -1..+1 (positive = more vol)
    notes: str = ""


@dataclass
class CalendarResult:
    date: datetime
    events: list[CalendarEvent] = field(default_factory=list)
    is_trading_day: dict[str, bool] = field(default_factory=dict)  # market → open?
    liquidity_warning: str = "normal"  # "low" / "normal" / "high"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "events": [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                    "impact": e.impact,
                    "markets_affected": e.markets_affected,
                    "expected_volatility_change": e.expected_volatility_change,
                    "notes": e.notes,
                }
                for e in self.events
            ],
            "is_trading_day": self.is_trading_day,
            "liquidity_warning": self.liquidity_warning,
            "notes": self.notes,
        }


# ──────────────────────────────────────────────────────────────────────
# Static holiday definitions (simplified — fixed-date holidays)
# ──────────────────────────────────────────────────────────────────────
US_HOLIDAYS_2025_2026: dict[str, str] = {
    # 2025
    "2025-01-01": "New Year's Day",
    "2025-01-20": "MLK Day",
    "2025-02-17": "Presidents Day",
    "2025-04-18": "Good Friday",
    "2025-05-26": "Memorial Day",
    "2025-06-19": "Juneteenth",
    "2025-07-04": "Independence Day",
    "2025-09-01": "Labor Day",
    "2025-11-27": "Thanksgiving",
    "2025-12-25": "Christmas",
    # 2026
    "2026-01-01": "New Year's Day",
    "2026-01-19": "MLK Day",
    "2026-02-16": "Presidents Day",
    "2026-04-03": "Good Friday",
    "2026-05-25": "Memorial Day",
    "2026-06-19": "Juneteenth",
    "2026-07-03": "Independence Day (observed)",
    "2026-09-07": "Labor Day",
    "2026-11-26": "Thanksgiving",
    "2026-12-25": "Christmas",
}

UK_HOLIDAYS_2025_2026: dict[str, str] = {
    "2025-01-01": "New Year's Day",
    "2025-04-18": "Good Friday",
    "2025-04-21": "Easter Monday",
    "2025-05-05": "Early May Bank Holiday",
    "2025-05-26": "Spring Bank Holiday",
    "2025-08-25": "Summer Bank Holiday",
    "2025-12-25": "Christmas Day",
    "2025-12-26": "Boxing Day",
    "2026-01-01": "New Year's Day",
    "2026-04-03": "Good Friday",
    "2026-04-06": "Easter Monday",
    "2026-05-04": "Early May Bank Holiday",
    "2026-05-25": "Spring Bank Holiday",
    "2026-08-31": "Summer Bank Holiday",
    "2026-12-25": "Christmas Day",
    "2026-12-28": "Boxing Day (observed)",
}


class MarketCalendar:
    """Market calendar intelligence.

    Parameters:
        timezone: timezone for date calculations (default UTC)
    """

    # Futures rollover months (quarterly)
    FUTURES_ROLLOVER_MONTHS = [3, 6, 9, 12]

    def __init__(self, timezone: str = "UTC") -> None:
        self.timezone = timezone

    def get_events(self, date: datetime) -> CalendarResult:
        """Get all calendar events for a specific date."""
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        date_str = date.strftime("%Y-%m-%d")
        events: list[CalendarEvent] = []
        is_trading_day = {"US": True, "UK": True, "EU": True, "JP": True, "crypto": True}

        # ── US Holidays ───────────────────────────────────────────
        if date_str in US_HOLIDAYS_2025_2026:
            events.append(CalendarEvent(
                event_type="holiday",
                description=f"US: {US_HOLIDAYS_2025_2026[date_str]}",
                impact="high", markets_affected=["US", "stocks", "bonds"],
                expected_volatility_change=-0.5,
                notes="US markets closed",
            ))
            is_trading_day["US"] = False

        # ── UK Holidays ───────────────────────────────────────────
        if date_str in UK_HOLIDAYS_2025_2026:
            events.append(CalendarEvent(
                event_type="holiday",
                description=f"UK: {UK_HOLIDAYS_2025_2026[date_str]}",
                impact="medium", markets_affected=["UK", "FX"],
                expected_volatility_change=-0.3,
                notes="UK markets closed",
            ))
            is_trading_day["UK"] = False

        # ── Weekend ───────────────────────────────────────────────
        if date.weekday() >= 5:  # Saturday or Sunday
            events.append(CalendarEvent(
                event_type="weekend",
                description="Weekend",
                impact="high", markets_affected=["stocks", "bonds", "FX"],
                expected_volatility_change=-0.8,
                notes="Traditional markets closed (crypto still trades)",
            ))
            is_trading_day["US"] = False
            is_trading_day["UK"] = False
            is_trading_day["EU"] = False
            is_trading_day["JP"] = False

        # ── Half trading days ─────────────────────────────────────
        # Day after Thanksgiving (4th Thursday of November)
        if date.month == 11 and date.day >= 28 and date.day <= 30 and date.weekday() == 4:
            events.append(CalendarEvent(
                event_type="half_day",
                description="US: Day after Thanksgiving (half day)",
                impact="low", markets_affected=["US", "stocks"],
                expected_volatility_change=-0.2,
                notes="US markets close early (1:00 PM ET)",
            ))
        # Christmas Eve
        if date.month == 12 and date.day == 24:
            events.append(CalendarEvent(
                event_type="half_day",
                description="Christmas Eve (half day)",
                impact="medium", markets_affected=["US", "UK", "EU"],
                expected_volatility_change=-0.3,
                notes="Many markets close early",
            ))

        # ── Futures contract rollover ─────────────────────────────
        # 2nd Friday of rollover months
        if date.month in self.FUTURES_ROLLOVER_MONTHS:
            # Check if this is the 2nd Friday
            if date.weekday() == 4:  # Friday
                # Find which Friday of the month
                first_day = date.replace(day=1)
                first_friday = (4 - first_day.weekday()) % 7 + 1
                second_friday = first_friday + 7
                if date.day == second_friday:
                    events.append(CalendarEvent(
                        event_type="rollover",
                        description=f"Futures rollover ({date.strftime('%b')})",
                        impact="medium", markets_affected=["futures", "crypto"],
                        expected_volatility_change=0.2,
                        notes="Roll open positions to next contract",
                    ))

        # ── Options expiry (3rd Friday of each month) ─────────────
        if date.weekday() == 4:  # Friday
            first_day = date.replace(day=1)
            first_friday = (4 - first_day.weekday()) % 7 + 1
            third_friday = first_friday + 14
            if date.day == third_friday:
                impact = "high" if date.month in self.FUTURES_ROLLOVER_MONTHS else "medium"
                events.append(CalendarEvent(
                    event_type="options_expiry",
                    description=f"Monthly options expiry ({'triple witching' if date.month in self.FUTURES_ROLLOVER_MONTHS else 'standard'})",
                    impact=impact, markets_affected=["options", "stocks"],
                    expected_volatility_change=0.3,
                    notes="Options expire — increased volatility from hedging flows",
                ))

        # ── CME Crypto futures expiry (last Friday of month) ──────
        if date.weekday() == 4:  # Friday
            # Check if this is the last Friday of the month
            next_week = date + timedelta(days=7)
            if next_week.month != date.month:
                events.append(CalendarEvent(
                    event_type="futures_expiry",
                    description="CME BTC/ETH futures expiry",
                    impact="high", markets_affected=["crypto", "BTCUSD", "ETHUSD"],
                    expected_volatility_change=0.4,
                    notes="CME futures expire — potential price pinning",
                ))

        # ── Month-end / quarter-end / year-end ────────────────────
        # Last trading day of month
        tomorrow = date + timedelta(days=1)
        if tomorrow.month != date.month and date.weekday() < 5:
            events.append(CalendarEvent(
                event_type="month_end",
                description="Month-end rebalancing",
                impact="medium", markets_affected=["stocks", "bonds", "FX"],
                expected_volatility_change=0.2,
                notes="Institutional rebalancing flows",
            ))
            if date.month in [3, 6, 9, 12]:
                events.append(CalendarEvent(
                    event_type="quarter_end",
                    description="Quarter-end rebalancing",
                    impact="high", markets_affected=["stocks", "bonds"],
                    expected_volatility_change=0.3,
                    notes="Larger rebalancing flows — higher impact",
                ))
            if date.month == 12:
                events.append(CalendarEvent(
                    event_type="year_end",
                    description="Year-end rebalancing + tax-loss harvesting",
                    impact="high", markets_affected=["stocks", "bonds", "crypto"],
                    expected_volatility_change=0.4,
                    notes="Year-end flows can cause unusual moves",
                ))

        # ── Liquidity warning ─────────────────────────────────────
        if any(e.event_type in ("holiday", "weekend") for e in events):
            liquidity_warning = "low"
        elif any(e.event_type in ("half_day",) for e in events):
            liquidity_warning = "low"
        elif any(e.event_type in ("options_expiry", "futures_expiry", "triple_witching") for e in events):
            liquidity_warning = "high"
        else:
            liquidity_warning = "normal"

        notes: list[str] = []
        if not is_trading_day["US"]:
            notes.append("US markets closed")
        if events:
            notes.append(f"{len(events)} events: {', '.join(e.event_type for e in events)}")
        else:
            notes.append("No significant calendar events")

        return CalendarResult(
            date=date, events=events,
            is_trading_day=is_trading_day,
            liquidity_warning=liquidity_warning,
            notes=notes,
        )


__all__ = ["MarketCalendar", "CalendarEvent", "CalendarResult"]
