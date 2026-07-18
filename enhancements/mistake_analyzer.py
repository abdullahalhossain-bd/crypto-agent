"""enhancements.mistake_analyzer
=====================================================================
Systematic categorization of losing trades into specific mistake types.

Categories:
  - false_breakout    : SL hit within 4h of entry (breakout failed)
  - bad_timing        : SL hit during low-liquidity hours (00:00-06:00 UTC)
  - quick_reversal    : SL hit within 2h (entered against momentum)
  - no_tp1_hit        : Trade lost without ever reaching TP1
  - trailing_premature: Trailing stop triggered with <1.5% gain

Each mistake type has a specific recommendation that feeds back into
the trade quality scorer and entry style selector.

Inspired by Centina-Quant's MistakeAnalyzer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger

log = get_logger("enhancements.mistake_analyzer")


@dataclass
class MistakePattern:
    mistake_type: str
    count: int
    avg_loss_pct: float
    total_loss: float
    common_regime: str
    common_hour: int
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mistake_type": self.mistake_type,
            "count": self.count,
            "avg_loss_pct": round(self.avg_loss_pct, 4),
            "total_loss": round(self.total_loss, 2),
            "common_regime": self.common_regime,
            "common_hour": self.common_hour,
            "recommendation": self.recommendation,
        }


# ----------------------------------------------------------------------
# Mistake detection rules
# ----------------------------------------------------------------------
MISTAKE_RULES: dict[str, dict[str, Any]] = {
    "false_breakout": {
        "condition": lambda t: (
            t.get("exit_reason") == "sl" and t.get("duration_h", 99) < 4
        ),
        "recommendation": "Confirm breakout with 2 candles above resistance before entry",
    },
    "bad_timing": {
        "condition": lambda t: (
            t.get("exit_reason") == "sl"
            and t.get("entry_hour", 12) in [0, 1, 2, 3, 4, 5]
        ),
        "recommendation": "Avoid entries between 00:00-06:00 UTC (low liquidity)",
    },
    "quick_reversal": {
        "condition": lambda t: (
            t.get("exit_reason") == "sl" and t.get("duration_h", 99) < 2
        ),
        "recommendation": "Increase ATR multiplier for SL to 1.5x to survive noise",
    },
    "no_tp1_hit": {
        "condition": lambda t: (
            not t.get("tp1_hit", False) and t.get("pnl_pct", 0) < -1
        ),
        "recommendation": "Review entry quality score — ensure score >= 78 before entry",
    },
    "trailing_stop_premature": {
        "condition": lambda t: (
            t.get("exit_reason") == "trailing" and t.get("pnl_pct", 0) < 1.5
        ),
        "recommendation": "Widen trailing stop ATR multiplier from 1.0 to 1.5",
    },
    "overleveraged": {
        "condition": lambda t: (
            t.get("exit_reason") == "sl"
            and t.get("lots", 0) > 0.05
            and t.get("pnl_pct", 0) < -3
        ),
        "recommendation": "Reduce position size — use Kelly fractional 1/8 sizing",
    },
    "wrong_regime": {
        "condition": lambda t: (
            t.get("exit_reason") == "sl"
            and t.get("regime", "") in ("CHOPPY", "HIGH_VOL")
        ),
        "recommendation": "Skip trades in CHOPPY/HIGH_VOL regimes — enable regime filter",
    },
}


# ----------------------------------------------------------------------
class MistakeAnalyzer:
    """Analyzes losing trades to identify systematic mistake patterns."""

    def __init__(self) -> None:
        self._trades: list[dict[str, Any]] = []

    # ----------------------------------------------------------------
    def record_trade(self, trade: dict[str, Any]) -> None:
        """Record a closed trade for analysis.

        Expected keys in trade dict:
            - exit_reason: "sl" / "trailing" / "tp1" / "tp2" / "tp3" / "time"
            - duration_h: hours held
            - entry_hour: UTC hour of entry (0-23)
            - pnl_pct: PnL as percentage
            - pnl_usdt: PnL in USDT
            - tp1_hit: bool
            - lots: position size
            - regime: market regime at entry
        """
        self._trades.append(trade)

    # ----------------------------------------------------------------
    def analyze(self) -> list[MistakePattern]:
        """Analyze all recorded trades and return mistake patterns."""
        if not self._trades:
            return []

        # Categorize each losing trade
        mistake_counts: dict[str, list[dict]] = defaultdict(list)
        for trade in self._trades:
            if trade.get("pnl_pct", 0) >= 0:
                continue  # only analyze losers
            for mistake_type, rule in MISTAKE_RULES.items():
                try:
                    if rule["condition"](trade):
                        mistake_counts[mistake_type].append(trade)
                except Exception:  # noqa: BLE001
                    continue

        # Build patterns
        patterns: list[MistakePattern] = []
        for mistake_type, trades in mistake_counts.items():
            if not trades:
                continue
            losses = [float(t.get("pnl_pct", 0)) for t in trades]
            avg_loss = sum(losses) / len(losses) if losses else 0
            total_loss = sum(float(t.get("pnl_usdt", 0)) for t in trades)
            # Most common regime
            regimes = [t.get("regime", "unknown") for t in trades]
            common_regime = max(set(regimes), key=regimes.count) if regimes else "unknown"
            # Most common hour
            hours = [t.get("entry_hour", -1) for t in trades if t.get("entry_hour") is not None]
            common_hour = max(set(hours), key=hours.count) if hours else -1
            patterns.append(MistakePattern(
                mistake_type=mistake_type,
                count=len(trades),
                avg_loss_pct=avg_loss,
                total_loss=total_loss,
                common_regime=common_regime,
                common_hour=common_hour,
                recommendation=MISTAKE_RULES[mistake_type]["recommendation"],
            ))
        # Sort by count (most frequent first)
        patterns.sort(key=lambda p: p.count, reverse=True)
        return patterns

    # ----------------------------------------------------------------
    def get_recommendations(self) -> list[str]:
        """Get actionable recommendations based on mistake analysis."""
        patterns = self.analyze()
        return [f"{p.mistake_type} ({p.count}x): {p.recommendation}" for p in patterns]

    # ----------------------------------------------------------------
    def get_quality_adjustment(self) -> float:
        """Return a quality score adjustment based on recent mistakes.

        More frequent mistakes → lower quality threshold needed.
        Returns a value in [-20, 0] to subtract from the quality threshold.
        """
        patterns = self.analyze()
        if not patterns:
            return 0.0
        total_mistakes = sum(p.count for p in patterns)
        # Each 5 mistakes → -2 pts (max -20)
        return max(-20.0, -(total_mistakes / 5.0) * 2.0)

    # ----------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Summary of all mistakes."""
        patterns = self.analyze()
        return {
            "total_trades": len(self._trades),
            "losing_trades": sum(1 for t in self._trades if t.get("pnl_pct", 0) < 0),
            "mistake_patterns": [p.to_dict() for p in patterns],
            "recommendations": self.get_recommendations(),
            "quality_adjustment": self.get_quality_adjustment(),
        }

    # ----------------------------------------------------------------
    def clear(self) -> None:
        self._trades.clear()
