"""architecture/regime_orchestrator.py
=====================================================================
Market Regime Orchestrator (Improvement #18)
=====================================================================
Detects the current market regime and adjusts strategy parameters,
position sizing, and active symbol universe dynamically.

Regimes Detected:
    - TREND_UP      — strong uptrend (ADX > 25, EMA stacked, price > SuperTrend)
    - TREND_DOWN    — strong downtrend
    - RANGE         — sideways, mean-reverting (ADX < 20, BBands tight)
    - HIGH_VOL      — high volatility (ATR% > 95th percentile)
    - LOW_VOL       — low volatility (ATR% < 5th percentile)
    - TRANSITION    — regime changing (recent BOS/CHoCH, regime flip)
    - CRISIS        — extreme vol + correlation spike (everything moving together)
    - CHOP          — noisy, no edge (low Sharpe potential)

Per-Regime Adjustments (configurable):
    | Regime      | Strategies Active         | Risk Per Trade | Symbols |
    |-------------|---------------------------|----------------|---------|
    | TREND_UP    | Trend, Momentum           | 2.0%           | All     |
    | TREND_DOWN  | Trend (short), Momentum   | 1.5%           | All     |
    | RANGE       | Mean Reversion            | 1.0%           | All     |
    | HIGH_VOL    | Reduce positions, hedge   | 0.5%           | Major   |
    | LOW_VOL     | Range strategies          | 1.5%           | All     |
    | TRANSITION  | Wait-and-see              | 0.0%           | None    |
    | CRISIS      | Defensive only (risk-off) | 0.0%           | None    |
    | CHOP        | Reduced activity          | 0.5%           | Major   |

Usage:
    orch = RegimeOrchestrator()
    regime = orch.detect(df_btc, df_eth, correlation_matrix)
    adjustments = orch.get_adjustments(regime)
    # adjustments = {"risk_per_trade": 0.015, "active_strategies": [...], ...}
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.regime_orchestrator")


class MarketRegime(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    TRANSITION = "transition"
    CRISIS = "crisis"
    CHOP = "chop"
    UNKNOWN = "unknown"


@dataclass
class RegimeAdjustments:
    """What should change based on the current regime?"""
    regime: MarketRegime = MarketRegime.UNKNOWN
    risk_per_trade: float = 0.02
    max_open_positions: int = 10
    active_strategies: List[str] = field(default_factory=lambda: ["momentum"])
    active_symbols: List[str] = field(default_factory=list)  # empty = all
    cooldown_s: float = 60.0
    min_confidence: float = 0.60
    description: str = ""


# Default adjustments per regime
DEFAULT_ADJUSTMENTS: Dict[MarketRegime, RegimeAdjustments] = {
    MarketRegime.TREND_UP: RegimeAdjustments(
        regime=MarketRegime.TREND_UP,
        risk_per_trade=0.025, max_open_positions=10,
        active_strategies=["trend", "momentum"],
        cooldown_s=45.0, min_confidence=0.60,
        description="Strong uptrend — full trend-following mode",
    ),
    MarketRegime.TREND_DOWN: RegimeAdjustments(
        regime=MarketRegime.TREND_DOWN,
        risk_per_trade=0.020, max_open_positions=8,
        active_strategies=["trend", "momentum"],
        cooldown_s=60.0, min_confidence=0.65,
        description="Downtrend — short-side focus",
    ),
    MarketRegime.RANGE: RegimeAdjustments(
        regime=MarketRegime.RANGE,
        risk_per_trade=0.015, max_open_positions=6,
        active_strategies=["mean_reversion"],
        cooldown_s=90.0, min_confidence=0.65,
        description="Range-bound — mean reversion only",
    ),
    MarketRegime.HIGH_VOL: RegimeAdjustments(
        regime=MarketRegime.HIGH_VOL,
        risk_per_trade=0.010, max_open_positions=4,
        active_strategies=["trend"],  # defensive only
        cooldown_s=180.0, min_confidence=0.75,
        description="High volatility — reduce exposure",
    ),
    MarketRegime.LOW_VOL: RegimeAdjustments(
        regime=MarketRegime.LOW_VOL,
        risk_per_trade=0.020, max_open_positions=8,
        active_strategies=["mean_reversion", "momentum"],
        cooldown_s=60.0, min_confidence=0.60,
        description="Low volatility — range + breakout",
    ),
    MarketRegime.TRANSITION: RegimeAdjustments(
        regime=MarketRegime.TRANSITION,
        risk_per_trade=0.005, max_open_positions=2,
        active_strategies=[],  # wait-and-see
        cooldown_s=300.0, min_confidence=0.85,
        description="Regime transition — wait and see",
    ),
    MarketRegime.CRISIS: RegimeAdjustments(
        regime=MarketRegime.CRISIS,
        risk_per_trade=0.000, max_open_positions=0,
        active_strategies=[],
        cooldown_s=3600.0, min_confidence=1.0,
        description="Crisis mode — risk off, no new trades",
    ),
    MarketRegime.CHOP: RegimeAdjustments(
        regime=MarketRegime.CHOP,
        risk_per_trade=0.008, max_open_positions=3,
        active_strategies=["mean_reversion"],
        cooldown_s=180.0, min_confidence=0.75,
        description="Choppy — minimal activity",
    ),
    MarketRegime.UNKNOWN: RegimeAdjustments(
        regime=MarketRegime.UNKNOWN,
        risk_per_trade=0.010, max_open_positions=3,
        active_strategies=["momentum"],
        cooldown_s=120.0, min_confidence=0.70,
        description="Unknown regime — cautious default",
    ),
}


class RegimeOrchestrator:
    """Detects market regime and orchestrates strategy adjustments."""

    def __init__(self,
                 bus: Optional[EventBus] = None,
                 atr_percentile_window: int = 100,
                 regime_history_size: int = 100,
                 crisis_correlation_threshold: float = 0.85,
                 crisis_atr_pct_threshold: float = 0.04,
                 min_regime_duration_s: float = 120.0,
                 regime_confirmation_cycles: int = 3):
        self._lock = threading.RLock()
        self._bus = bus or get_bus()
        self._atr_window: List[float] = []
        self._atr_window_size = atr_percentile_window
        self._regime_history: List[Dict[str, Any]] = []
        self._history_size = regime_history_size
        self._current_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_changed_at: float = time.time()
        # FIX-RO-03: previously hardcoded (0.85 / 0.04) inline in detect(),
        # independently of config.yaml's monitoring.risk.max_correlation —
        # now an explicit constructor parameter so callers wire it from the
        # same config value used elsewhere (single source of truth). Default
        # unchanged for backward compatibility with existing call sites.
        self._crisis_corr_threshold = crisis_correlation_threshold
        self._crisis_atr_pct_threshold = crisis_atr_pct_threshold
        # HYSTERESIS fix: the regime classifier was oscillating between
        # TRANSITION and HIGH_VOL on every cycle because each detect() call
        # could immediately change the regime. Now a new regime must be
        # observed for `regime_confirmation_cycles` consecutive cycles AND
        # the current regime must have been held for at least
        # `min_regime_duration_s` seconds before a switch is accepted.
        # Exception: CRISIS is always accepted immediately (safety first).
        self._min_regime_duration_s = float(min_regime_duration_s)
        self._confirmation_cycles = int(regime_confirmation_cycles)
        self._pending_regime: Optional[MarketRegime] = None
        self._pending_count: int = 0

    @property
    def current_regime(self) -> MarketRegime:
        with self._lock:
            return self._current_regime

    def time_in_regime(self) -> float:
        return time.time() - self._regime_changed_at

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def detect(self,
               df: pd.DataFrame,
               correlation_matrix: Optional[pd.DataFrame] = None,
               btc_dominance: Optional[float] = None) -> MarketRegime:
        """Detect the current market regime from price data.

        Combines multiple signals:
            - Trend strength (ADX, EMA stacking)
            - Volatility regime (ATR% percentile)
            - Correlation (crisis = high correlation)
            - Price action (SuperTrend direction)
        """
        if df is None or df.empty or len(df) < 50:
            return self._current_regime

        try:
            from utils.indicators import (
                adx, atr, atr_pct, ema, supertrend, bollinger_bands
            )
            # FIX: compute each indicator individually so one failure
            # doesn't kill all regime detection. Previously, if supertrend
            # raised (e.g., OHLCV validation), the entire try block failed
            # and the regime stayed UNKNOWN forever.
            price = float(df["close"].iloc[-1])
            try:
                adx_val = float(adx(df, 14).iloc[-1])
            except Exception:
                adx_val = 0.0
            try:
                atr_val = float(atr(df, 14).iloc[-1])
            except Exception:
                atr_val = 0.0
            try:
                atr_p = float(atr_pct(df, 14).iloc[-1])
            except Exception:
                atr_p = 0.0
            try:
                ema9 = float(ema(df["close"], 9).iloc[-1])
            except Exception:
                ema9 = price
            try:
                ema21 = float(ema(df["close"], 21).iloc[-1])
            except Exception:
                ema21 = price
            try:
                ema50 = float(ema(df["close"], 50).iloc[-1])
            except Exception:
                ema50 = price
            try:
                st = float(supertrend(df, 10, 3).iloc[-1])
            except Exception:
                st = price  # fallback: treat as no supertrend signal
            try:
                bb_upper, bb_mid, bb_lower, bb_width = bollinger_bands(df["close"], 20)
                bb_w = float(bb_width.iloc[-1])
            except Exception:
                bb_w = 0.0
        except Exception as e:  # noqa: BLE001
            log.warning("regime: indicator import failed: %r", e)
            return self._current_regime

        # Update ATR percentile window
        with self._lock:
            self._atr_window.append(atr_p)
            if len(self._atr_window) > self._atr_window_size:
                self._atr_window = self._atr_window[-self._atr_window_size:]

        # Crisis check: high correlation across many symbols
        is_crisis = False
        if correlation_matrix is not None and not correlation_matrix.empty:
            # Mean off-diagonal correlation
            mask = ~np.eye(len(correlation_matrix), dtype=bool)
            avg_corr = float(correlation_matrix.values[mask].mean())
            if avg_corr > self._crisis_corr_threshold and atr_p > self._crisis_atr_pct_threshold:
                is_crisis = True

        # ATR percentile — FIX-RO-01: this used to be
        #   np.percentile(window, 100 * (atr_p / max(window)))
        # which is not a percentile-rank computation at all — it fed a
        # made-up index into np.percentile() with no defined relationship
        # to where atr_p actually ranks in the distribution. The correct
        # computation is "what fraction of the historical window is below
        # the current value" (a percentile-of-score), computed directly
        # without needing scipy:
        atr_pctile = 0.5
        if len(self._atr_window) >= 20:
            window_arr = np.asarray(self._atr_window, dtype=float)
            atr_pctile = float(np.mean(window_arr < atr_p))

        # Regime determination (priority: crisis → high_vol → trend → range → chop)
        new_regime = self._current_regime

        if is_crisis:
            new_regime = MarketRegime.CRISIS
        elif atr_p > 0.05 and atr_pctile > 0.90:
            new_regime = MarketRegime.HIGH_VOL
        elif atr_p < 0.01 and atr_pctile < 0.10:
            new_regime = MarketRegime.LOW_VOL
        elif adx_val > 25 and ema9 > ema21 > ema50 and price > st:
            new_regime = MarketRegime.TREND_UP
        elif adx_val > 25 and ema9 < ema21 < ema50 and price < st:
            new_regime = MarketRegime.TREND_DOWN
        elif adx_val < 20 and bb_w < 0.02:
            new_regime = MarketRegime.RANGE
        elif adx_val < 18:
            new_regime = MarketRegime.CHOP
        else:
            new_regime = MarketRegime.TRANSITION

        # Detect regime change — with HYSTERESIS.
        # The previous code switched immediately on any detect() call, which
        # caused oscillation between TRANSITION and HIGH_VOL every cycle.
        # Now: a new regime must be observed for `confirmation_cycles`
        # consecutive cycles AND the current regime must have been held for
        # at least `min_regime_duration_s` before the switch is accepted.
        # CRISIS is always accepted immediately (safety override).
        with self._lock:
            if new_regime == self._current_regime:
                # Reset pending — the current regime is confirmed.
                self._pending_regime = None
                self._pending_count = 0
            elif new_regime == MarketRegime.CRISIS:
                # Safety override: CRISIS is always accepted immediately.
                old = self._current_regime
                prev_regime_entered_at = self._regime_changed_at
                self._current_regime = new_regime
                self._regime_changed_at = time.time()
                self._pending_regime = None
                self._pending_count = 0
                self._regime_history.append({
                    "from": old.value, "to": new_regime.value,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "duration_in_old_s": time.time() - prev_regime_entered_at,
                })
                log.warning("regime: %s → %s (CRISIS — immediate switch, ADX=%.1f, ATR%%=%.3f)",
                            old.value, new_regime.value, adx_val, atr_p)
                self._bus.emit(EventType.STATE_TRANSITION,
                              payload={"regime_from": old.value,
                                      "regime_to": new_regime.value},
                              source="regime_orchestrator")
            else:
                # Hysteresis: check minimum duration in current regime.
                time_in_current = time.time() - self._regime_changed_at
                if time_in_current < self._min_regime_duration_s:
                    # Too soon to switch — keep current regime, note pending.
                    if self._pending_regime != new_regime:
                        self._pending_regime = new_regime
                        self._pending_count = 1
                    log.debug("regime: %s detected but holding %s (min_duration=%.0fs, elapsed=%.0fs)",
                              new_regime.value, self._current_regime.value,
                              self._min_regime_duration_s, time_in_current)
                    new_regime = self._current_regime  # don't switch yet
                else:
                    # Duration OK — count confirmation cycles.
                    if self._pending_regime == new_regime:
                        self._pending_count += 1
                    else:
                        self._pending_regime = new_regime
                        self._pending_count = 1

                    if self._pending_count >= self._confirmation_cycles:
                        # Confirmed — switch.
                        old = self._current_regime
                        prev_regime_entered_at = self._regime_changed_at
                        self._current_regime = new_regime
                        self._regime_changed_at = time.time()
                        self._pending_regime = None
                        self._pending_count = 0
                        self._regime_history.append({
                            "from": old.value, "to": new_regime.value,
                            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                            "duration_in_old_s": time.time() - prev_regime_entered_at,
                        })
                        log.info("regime: %s → %s (confirmed after %d cycles, ADX=%.1f, ATR%%=%.3f, price=%.2f)",
                                 old.value, new_regime.value, self._confirmation_cycles,
                                 adx_val, atr_p, price)
                        self._bus.emit(EventType.STATE_TRANSITION,
                                      payload={"regime_from": old.value,
                                              "regime_to": new_regime.value},
                                      source="regime_orchestrator")
                    else:
                        log.debug("regime: %s pending (%d/%d confirmations)",
                                  new_regime.value, self._pending_count,
                                  self._confirmation_cycles)
                        new_regime = self._current_regime  # not confirmed yet

        return new_regime

    # ------------------------------------------------------------------
    # Adjustments
    # ------------------------------------------------------------------
    def get_adjustments(self,
                        regime: Optional[MarketRegime] = None) -> RegimeAdjustments:
        """Return the recommended adjustments for the current (or given) regime."""
        r = regime or self._current_regime
        return DEFAULT_ADJUSTMENTS.get(r, DEFAULT_ADJUSTMENTS[MarketRegime.UNKNOWN])

    def should_trade(self) -> bool:
        """Returns False if current regime forbids new trades."""
        adj = self.get_adjustments()
        return adj.risk_per_trade > 0 and adj.max_open_positions > 0

    def history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._regime_history[-last_n:])