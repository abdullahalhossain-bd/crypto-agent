"""engine.candlestick.entry_style
=====================================================================
Day 136 — Entry style selector.

The book describes two entry styles:
  - Aggressive : enter on pattern confirmation (faster, more risk)
  - Conservative : wait for additional confirmation (slower, safer)

We add:
  - Adaptive : choose based on volatility regime

In high-volatility regimes, conservative entries avoid getting
stopped out on noise. In low-volatility regimes, aggressive entries
capture the move before it extends.

The selected style affects:
  - Number of confirmation bars required
  - Stop-loss distance (wider for aggressive, tighter for conservative)
  - Entry trigger (pattern close vs. pattern break)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils.logger import get_logger

log = get_logger("candlestick.entry_style")


class EntryStyle(str, Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


@dataclass
class EntryStyleDecision:
    style: EntryStyle
    confirmation_bars: int
    stop_atr_multiple: float
    entry_trigger: str        # "pattern_close" / "pattern_break" / "next_bar_open"
    reason: str = ""
    components: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style.value,
            "confirmation_bars": self.confirmation_bars,
            "stop_atr_multiple": self.stop_atr_multiple,
            "entry_trigger": self.entry_trigger,
            "reason": self.reason,
            "components": dict(self.components),
        }


# ----------------------------------------------------------------------
class EntryStyleSelector:
    def __init__(self,
                 high_vol_threshold: float = 1.8,
                 low_vol_threshold: float = 0.8,
                 aggressive_stop_atr: float = 1.5,
                 conservative_stop_atr: float = 2.5,
                 adaptive_default: EntryStyle = EntryStyle.CONSERVATIVE) -> None:
        self.high_vol_threshold = float(high_vol_threshold)
        self.low_vol_threshold = float(low_vol_threshold)
        self.aggressive_stop = float(aggressive_stop_atr)
        self.conservative_stop = float(conservative_stop_atr)
        self.adaptive_default = adaptive_default

    # ----------------------------------------------------------------
    def select(self, style: EntryStyle, atr_ratio: float = 1.0,
               market_state: str = "RANGE") -> EntryStyleDecision:
        """Return the entry style decision."""
        components: dict[str, Any] = {
            "requested_style": style.value,
            "atr_ratio": float(atr_ratio),
            "market_state": market_state,
        }

        if style == EntryStyle.ADAPTIVE:
            chosen = self._adaptive_choice(atr_ratio, market_state)
            reason = (f"adaptive: atr_ratio={atr_ratio:.2f}, "
                      f"market={market_state} -> {chosen.value}")
        else:
            chosen = style
            reason = f"explicit {style.value}"

        if chosen == EntryStyle.AGGRESSIVE:
            return EntryStyleDecision(
                style=EntryStyle.AGGRESSIVE,
                confirmation_bars=0,
                stop_atr_multiple=self.aggressive_stop,
                entry_trigger="pattern_close",
                reason=reason,
                components=components,
            )
        # Conservative
        return EntryStyleDecision(
            style=EntryStyle.CONSERVATIVE,
            confirmation_bars=1,
            stop_atr_multiple=self.conservative_stop,
            entry_trigger="next_bar_open",
            reason=reason,
            components=components,
        )

    # ----------------------------------------------------------------
    def _adaptive_choice(self, atr_ratio: float, market_state: str) -> EntryStyle:
        """In high vol → conservative; in low vol → aggressive."""
        if atr_ratio >= self.high_vol_threshold:
            return EntryStyle.CONSERVATIVE
        if atr_ratio <= self.low_vol_threshold:
            return EntryStyle.AGGRESSIVE
        # Medium vol: depends on market state
        if market_state == "TREND":
            return EntryStyle.AGGRESSIVE   # don't miss trend moves
        if market_state == "CHOPPY":
            return EntryStyle.CONSERVATIVE  # avoid whipsaws
        return self.adaptive_default   # RANGE
