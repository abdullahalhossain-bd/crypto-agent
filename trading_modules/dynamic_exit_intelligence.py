"""trading_modules/dynamic_exit_intelligence.py
=====================================================================
Dynamic Exit Intelligence (Principle #187)
=====================================================================
Exits are not static TP levels — they adapt to changing conditions.
This module monitors every open position and recommends exit actions.

Exit Triggers (6 types):
    1. TREND WEAKENING    — EMA slope flattening, ADX declining
    2. LIQUIDITY DROP     — spread widening, volume dropping
    3. MOMENTUM LOSS      — RSI/MACD divergence against position
    4. VOLATILITY CHANGE  — ATR spike (risk increased) or crush (no movement)
    5. STRUCTURE BREAK    — BoS/ChoCH against position direction
    6. PROFIT PROTECTION  — trailing stop, breakeven move, partial close

Exit Actions:
    - HOLD           — no action needed
    - TIGHTEN_STOP   — move stop closer
    - MOVE_TO_BREAKEVEN — move stop to entry price
    - TRAIL_STOP     — trail stop behind recent swing
    - PARTIAL_CLOSE  — close 50% of position
    - CLOSE_ALL      — exit entire position
    - REVERSE        — exit + open opposite (rare, high-conviction reversal)

Usage:
    exit_ai = DynamicExitIntelligence()

    # For each open position:
    recommendation = exit_ai.evaluate(
        position_side="BUY",
        entry_price=43250,
        current_price=43800,
        stop_loss=42500,
        take_profit=45000,
        df=df,
        r_multiple=1.2,  # current R multiple
    )
    # recommendation = {
    #     "action": "TRAIL_STOP",
    #     "new_stop": 43500,
    #     "reason": "Profit protection: trail behind recent swing",
    #     "urgency": "normal",
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.dynamic_exit_intelligence")


class ExitAction(str, Enum):
    HOLD = "HOLD"
    TIGHTEN_STOP = "TIGHTEN_STOP"
    MOVE_TO_BREAKEVEN = "MOVE_TO_BREAKEVEN"
    TRAIL_STOP = "TRAIL_STOP"
    PARTIAL_CLOSE = "PARTIAL_CLOSE"
    CLOSE_ALL = "CLOSE_ALL"
    REVERSE = "REVERSE"


class ExitUrgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    IMMEDIATE = "immediate"


@dataclass
class ExitRecommendation:
    """Dynamic exit recommendation for a position."""
    action: ExitAction = ExitAction.HOLD
    urgency: ExitUrgency = ExitUrgency.LOW
    new_stop: float = 0.0
    new_tp: float = 0.0
    close_pct: float = 0.0         # for PARTIAL_CLOSE (0-1)
    reason: str = ""

    # Triggered signals
    triggers: List[str] = field(default_factory=list)

    # Position info
    current_r: float = 0.0
    profit_usd: float = 0.0
    hold_time_bars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "urgency": self.urgency.value,
            "new_stop": round(self.new_stop, 5),
            "new_tp": round(self.new_tp, 5),
            "close_pct": round(self.close_pct, 3),
            "reason": self.reason,
            "triggers": self.triggers,
            "current_r": round(self.current_r, 2),
            "profit_usd": round(self.profit_usd, 2),
            "hold_time_bars": self.hold_time_bars,
        }


class DynamicExitIntelligence:
    """Monitors positions and recommends dynamic exits."""

    def __init__(self,
                 breakeven_r: float = 1.0,
                 trail_start_r: float = 1.5,
                 partial_close_r: float = 2.0,
                 max_hold_bars: int = 100,
                 max_adverse_r: float = -1.5):
        """Initialize exit intelligence.

        Args:
            breakeven_r: move stop to breakeven at this R
            trail_start_r: start trailing stop at this R
            partial_close_r: take partial profits at this R
            max_hold_bars: force exit after this many bars
            max_adverse_r: force exit if position goes against us by this R
        """
        self.breakeven_r = breakeven_r
        self.trail_start_r = trail_start_r
        self.partial_close_r = partial_close_r
        self.max_hold = max_hold_bars
        self.max_adverse = max_adverse_r

    def evaluate(self,
                 position_side: str,
                 entry_price: float,
                 current_price: float,
                 stop_loss: float,
                 take_profit: float,
                 df: pd.DataFrame,
                 r_multiple: float = 0.0,
                 hold_time_bars: int = 0,
                 spread_bps: float = 5.0) -> ExitRecommendation:
        """Evaluate a position and recommend exit action.

        Args:
            position_side: "BUY" or "SELL"
            entry_price: entry price
            current_price: current market price
            stop_loss: current stop loss
            take_profit: current take profit
            df: OHLCV DataFrame
            r_multiple: current R multiple of position
            hold_time_bars: bars since entry
            spread_bps: current spread

        Returns:
            ExitRecommendation with action + new levels
        """
        rec = ExitRecommendation(
            new_stop=stop_loss,
            new_tp=take_profit,
            current_r=r_multiple,
            hold_time_bars=hold_time_bars,
        )

        if df is None or df.empty or len(df) < 20:
            return rec

        triggers = []

        # === 1. Max adverse excursion — immediate exit ===
        if r_multiple <= self.max_adverse:
            rec.action = ExitAction.CLOSE_ALL
            rec.urgency = ExitUrgency.IMMEDIATE
            rec.reason = f"Max adverse excursion: {r_multiple:.1f}R"
            rec.triggers.append("max_adverse")
            return rec

        # === 2. Max hold time — force exit ===
        if hold_time_bars >= self.max_hold:
            rec.action = ExitAction.CLOSE_ALL
            rec.urgency = ExitUrgency.HIGH
            rec.reason = f"Max hold time: {hold_time_bars} bars"
            rec.triggers.append("max_hold")
            return rec

        # === 3. Trend weakening ===
        trend_weak = self._check_trend_weakening(df, position_side)
        if trend_weak:
            triggers.append("trend_weakening")

        # === 4. Liquidity drop ===
        liq_drop = self._check_liquidity_drop(df, spread_bps)
        if liq_drop:
            triggers.append("liquidity_drop")

        # === 5. Momentum loss ===
        mom_loss = self._check_momentum_loss(df, position_side)
        if mom_loss:
            triggers.append("momentum_loss")

        # === 6. Structure break ===
        struct_break = self._check_structure_break(df, position_side)
        if struct_break:
            triggers.append("structure_break")

        # === 7. Volatility spike ===
        vol_spike = self._check_volatility_spike(df)
        if vol_spike:
            triggers.append("volatility_spike")

        rec.triggers = triggers

        # === Profit protection (based on R) ===
        if r_multiple >= self.partial_close_r:
            # Partial close at 2R+
            rec.action = ExitAction.PARTIAL_CLOSE
            rec.close_pct = 0.5
            rec.urgency = ExitUrgency.NORMAL
            rec.reason = f"Take 50% profit at {r_multiple:.1f}R"
            # Trail the rest
            rec.new_stop = self._compute_trail_stop(df, position_side, entry_price, current_price)
            return rec

        if r_multiple >= self.trail_start_r:
            # Start trailing
            rec.action = ExitAction.TRAIL_STOP
            rec.new_stop = self._compute_trail_stop(df, position_side, entry_price, current_price)
            rec.urgency = ExitUrgency.NORMAL
            rec.reason = f"Trail stop at {r_multiple:.1f}R"
            return rec

        if r_multiple >= self.breakeven_r:
            # Move to breakeven
            rec.action = ExitAction.MOVE_TO_BREAKEVEN
            rec.new_stop = entry_price
            rec.urgency = ExitUrgency.LOW
            rec.reason = f"Move to breakeven at {r_multiple:.1f}R"
            return rec

        # === Negative R with triggers — tighten stop ===
        if triggers and r_multiple < 0:
            rec.action = ExitAction.TIGHTEN_STOP
            rec.urgency = ExitUrgency.HIGH
            # Tighten stop by 30%
            if position_side == "BUY":
                new_stop = current_price - (current_price - stop_loss) * 0.7
            else:
                new_stop = current_price + (stop_loss - current_price) * 0.7
            rec.new_stop = new_stop
            rec.reason = f"Tighten stop — triggers: {', '.join(triggers)}"
            return rec

        # === Multiple triggers on winning position — partial close ===
        if len(triggers) >= 2 and r_multiple > 0:
            rec.action = ExitAction.PARTIAL_CLOSE
            rec.close_pct = 0.5
            rec.urgency = ExitUrgency.NORMAL
            rec.reason = f"Partial close — {len(triggers)} exit signals"
            return rec

        # === All triggers — close ===
        if len(triggers) >= 3:
            rec.action = ExitAction.CLOSE_ALL
            rec.urgency = ExitUrgency.HIGH
            rec.reason = f"Close — {len(triggers)} exit signals"
            return rec

        # === HOLD ===
        rec.action = ExitAction.HOLD
        rec.urgency = ExitUrgency.LOW
        rec.reason = "No exit triggers — hold position"
        return rec

    # ------------------------------------------------------------------
    # Trigger detectors
    # ------------------------------------------------------------------
    def _check_trend_weakening(self, df: pd.DataFrame, side: str) -> bool:
        """EMA slope flattening."""
        close = df["close"]
        ema21 = close.ewm(span=21, adjust=False).mean()
        slope_now = float(ema21.iloc[-1] - ema21.iloc[-3])
        slope_before = float(ema21.iloc[-5] - ema21.iloc[-10])

        if side == "BUY":
            return slope_now < slope_before * 0.5 and slope_now > 0
        else:
            return slope_now > slope_before * 0.5 and slope_now < 0

    def _check_liquidity_drop(self, df: pd.DataFrame, spread_bps: float) -> bool:
        """Spread widening or volume dropping."""
        if spread_bps > 15:
            return True
        if "volume" in df:
            recent = float(df["volume"].tail(5).mean())
            avg = float(df["volume"].tail(20).mean())
            if recent < avg * 0.5:
                return True
        return False

    def _check_momentum_loss(self, df: pd.DataFrame, side: str) -> bool:
        """RSI/MACD divergence against position."""
        close = df["close"]
        if len(close) < 30:
            return False
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        rsi_before = float(rsi.iloc[-10]) if not pd.isna(rsi.iloc[-10]) else 50
        price_now = float(close.iloc[-1])
        price_before = float(close.iloc[-10])

        # Bearish divergence for longs
        if side == "BUY":
            return price_now > price_before and rsi_now < rsi_before - 5
        else:
            return price_now < price_before and rsi_now > rsi_before + 5

    def _check_structure_break(self, df: pd.DataFrame, side: str) -> bool:
        """BoS/ChoCH against position."""
        if len(df) < 20:
            return False
        close = df["close"]
        low = df["low"]
        high = df["high"]

        recent_low = float(low.tail(20).head(15).min())
        recent_high = float(high.tail(20).head(15).max())
        last_close = float(close.iloc[-1])

        if side == "BUY":
            # Structure break: price broke below recent low
            return last_close < recent_low * 0.998
        else:
            return last_close > recent_high * 1.002

    def _check_volatility_spike(self, df: pd.DataFrame) -> bool:
        """ATR spike — risk increased."""
        if len(df) < 30:
            return False
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        recent = float(atr.tail(5).mean())
        avg = float(atr.tail(20).mean())
        return recent > avg * 1.8  # 80% spike

    def _compute_trail_stop(self, df: pd.DataFrame, side: str,
                            entry: float, current: float) -> float:
        """Compute trailing stop based on recent swing."""
        if len(df) < 10:
            return entry
        if side == "BUY":
            # Trail below recent swing low
            return float(df["low"].tail(10).min())
        else:
            return float(df["high"].tail(10).max())
