"""engine.candlestick.trade_quality
=====================================================================
Day 135 — Trade quality scorer (Sniper Trading).

The book advocates "Sniper Trading" — fewer but higher-quality setups.
Every potential trade gets a grade:

  A+   : exceptional setup, full size
  A    : strong setup, 75% size
  B    : acceptable setup, 50% size
  C    : marginal setup, 25% size (or skip)
  REJECT : no trade

Grading factors:
  - Confluence score (from confluence_engine)
  - Risk/reward ratio (must be >= 2:1 for A+)
  - Distance to nearest strong S/R (closer to entry = better)
  - Spread / cost (high spread = lower grade)
  - Time of day (low-liquidity hours = lower grade)
  - News blackout window (no trades around news — see NewsCalendar)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from utils.logger import get_logger

if TYPE_CHECKING:
    from enhancements.news_calendar import NewsEvent

log = get_logger("candlestick.quality")


class QualityGrade(str, Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    REJECT = "REJECT"


@dataclass
class TradeQuality:
    grade: QualityGrade
    score: float                  # 0-100
    size_multiplier: float        # 0-1, suggested position sizing
    reasons: list[str] = field(default_factory=list)
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade.value,
            "score": self.score,
            "size_multiplier": self.size_multiplier,
            "reasons": list(self.reasons),
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class TradeQualityScorer:
    def __init__(self,
                 min_rr_for_a_plus: float = 2.5,
                 min_rr_for_a: float = 2.0,
                 min_rr_for_b: float = 1.5,
                 min_rr_for_c: float = 1.0,
                 min_confluence_for_a_plus: float = 75.0,
                 news_blackout_minutes: int = 15,
                 # FIX (TQ-2): was a hardcoded (2, 22) crypto-specific window.
                 # Now configurable per instrument/asset-class so this scorer
                 # isn't silently wrong when used for FX pairs, whose
                 # low-liquidity windows (session rollover, Friday close,
                 # Sunday open) don't match crypto's late-UTC assumption.
                 low_liquidity_hours_utc: tuple[int, int] = (2, 22)) -> None:
        self.min_rr_a_plus = float(min_rr_for_a_plus)
        self.min_rr_a = float(min_rr_for_a)
        self.min_rr_b = float(min_rr_for_b)
        self.min_rr_c = float(min_rr_for_c)
        self.min_conf_a_plus = float(min_confluence_for_a_plus)
        self.news_blackout = int(news_blackout_minutes)
        self.low_liq_start, self.low_liq_end = low_liquidity_hours_utc

    # ----------------------------------------------------------------
    def score(
        self,
        confluence_score: float,
        risk_reward: float,
        spread_bps: float = 1.0,
        distance_to_sr_atr: float = 1.0,
        news_event_minutes_ago: int = 999,
        time_of_day_utc: tuple[int, int] | None = None,
        # FIX (TQ-1): NewsCalendar is now the single source of truth for
        # blackout windows (covers both *before* and *after* an event, with
        # impact filtering) instead of this module's own incomplete
        # "minutes since a past event only" check. `news_event_minutes_ago`
        # is kept for backward compatibility and is used only if
        # `news_blackout` is not supplied by the caller.
        news_blackout: Optional[tuple[bool, Optional["NewsEvent"]]] = None,
    ) -> TradeQuality:
        """Compute the trade quality grade.

        Args:
            news_blackout: result of `NewsCalendar.is_blackout(symbol, ...)`,
                i.e. `(is_blackout: bool, event: Optional[NewsEvent])`. When
                provided, this is authoritative and `news_event_minutes_ago`
                is ignored. When omitted, falls back to the legacy
                after-event-only check for backward compatibility.
        """
        reasons: list[str] = []
        components: dict[str, Any] = {
            "confluence_score": float(confluence_score),
            "risk_reward": float(risk_reward),
            "spread_bps": float(spread_bps),
            "distance_to_sr_atr": float(distance_to_sr_atr),
            "news_event_minutes_ago": int(news_event_minutes_ago),
        }

        # News blackout — prefer the authoritative NewsCalendar-derived result.
        if news_blackout is not None:
            is_blackout, event = news_blackout
            if is_blackout:
                reason = (f"news blackout: {event.title} ({event.impact.value})"
                           if event is not None else "news blackout")
                return TradeQuality(
                    grade=QualityGrade.REJECT, score=0.0, size_multiplier=0.0,
                    reasons=[reason], components=components,
                )
        elif news_event_minutes_ago < self.news_blackout:
            # Legacy fallback path (after-event-only). Logged so operators can
            # see when the incomplete path is still in use and migrate callers.
            log.debug("trade_quality: using legacy news_event_minutes_ago blackout "
                       "check — pass news_blackout=NewsCalendar.is_blackout(...) "
                       "for full before/after coverage")
            return TradeQuality(
                grade=QualityGrade.REJECT, score=0.0, size_multiplier=0.0,
                reasons=[f"news blackout ({news_event_minutes_ago}m < {self.news_blackout}m)"],
                components=components,
            )

        # Compute base score
        score = confluence_score
        # RR adjustment
        if risk_reward >= self.min_rr_a_plus:
            score += 10
            reasons.append(f"RR {risk_reward:.1f} >= {self.min_rr_a_plus}")
        elif risk_reward >= self.min_rr_a:
            score += 5
        elif risk_reward < self.min_rr_c:
            score -= 20
            reasons.append(f"RR {risk_reward:.1f} too low (< {self.min_rr_c})")

        # Spread penalty
        if spread_bps > 10:
            score -= 15
            reasons.append(f"high spread {spread_bps:.1f} bps")
        elif spread_bps > 5:
            score -= 5

        # S/R proximity bonus
        if distance_to_sr_atr < 0.5:
            score += 5
            reasons.append("near strong S/R")
        elif distance_to_sr_atr > 3.0:
            score -= 5

        # Time-of-day check (avoid low-liquidity hours) — window is now
        # configurable per instance instead of hardcoded (FIX TQ-2).
        if time_of_day_utc is not None:
            hour = time_of_day_utc[0]
            if hour < self.low_liq_start or hour >= self.low_liq_end:
                score -= 10
                reasons.append(f"low-liquidity hour {hour}:00")

        score = float(max(0.0, min(100.0, score)))

        # Grade
        if score >= 80 and risk_reward >= self.min_rr_a_plus:
            grade = QualityGrade.A_PLUS
            size = 1.0
        elif score >= 65 and risk_reward >= self.min_rr_a:
            grade = QualityGrade.A
            size = 0.75
        elif score >= 50 and risk_reward >= self.min_rr_b:
            grade = QualityGrade.B
            size = 0.50
        elif score >= 35 and risk_reward >= self.min_rr_c:
            grade = QualityGrade.C
            size = 0.25
        else:
            grade = QualityGrade.REJECT
            size = 0.0
            reasons.append("below quality threshold")

        return TradeQuality(
            grade=grade, score=score, size_multiplier=float(size),
            reasons=reasons, components=components,
        )