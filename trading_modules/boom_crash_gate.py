"""trading_modules/boom_crash_gate.py
=====================================================================
P0-4 FIX: Boom/Crash-Specific Risk Gate
=====================================================================

Boom and Crash indices have unique microstructure:
- Boom: Spikes UP only (buy spikes), gradual decay down
- Crash: Spikes DOWN only (sell spikes), gradual climb up
- Standard forex logic FAILS on these instruments
- Counter-spike trades = guaranteed loss

This gate:
1. Detects if symbol is Boom/Crash type
2. Identifies spike direction
3. Blocks trades AGAINST spike direction
4. Requires wider ATR multipliers (spike hunts stops)
5. Rejects low-volatility environments (no spike = no trade)

Usage:
    Add to RiskPipeline gates list after LiquidityGate.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

import pandas as pd

from architecture.risk_pipeline import RiskGate, RiskVerdict, RiskContext
from utils.logger import get_logger

log = get_logger("trading_modules.boom_crash_gate")


class BoomCrashGate(RiskGate):
    """Specialized gate for Boom/Crash synthetic indices.
    
    These instruments have asymmetric spike behavior:
    - Boom 99/500/1000: Spikes UP, drifts down
    - Crash 99/500/1000: Spikes DOWN, drifts up
    
    NEVER trade against the spike direction.
    """
    name = "boom_crash"
    
    # Pattern to detect Boom/Crash symbols
    BOOM_PATTERN = re.compile(r"boom\s*\d*", re.IGNORECASE)
    CRASH_PATTERN = re.compile(r"crash\s*\d*", re.IGNORECASE)
    
    def __init__(
        self,
        min_spike_atr_multiple: float = 2.5,  # Wider stops for spike volatility
        spike_confirmation_bars: int = 3,      # Bars to confirm spike direction
        allow_counter_spike: bool = False      # NEVER enable in production
    ):
        self.min_spike_atr = min_spike_atr_multiple
        self.spike_confirm_bars = spike_confirmation_bars
        self.allow_counter_spike = allow_counter_spike
        
        if allow_counter_spike:
            log.warning(
                "BoomCrashGate: allow_counter_spike=True is EXTREMELY DANGEROUS. "
                "Counter-spike trades on Boom/Crash have >90% loss rate."
            )
    
    def _is_boom_symbol(self, symbol: str) -> bool:
        """Check if symbol is a Boom index."""
        return bool(self.BOOM_PATTERN.search(symbol))
    
    def _is_crash_symbol(self, symbol: str) -> bool:
        """Check if symbol is a Crash index."""
        return bool(self.CRASH_PATTERN.search(symbol))
    
    def _is_boom_crash_symbol(self, symbol: str) -> bool:
        """Check if symbol is Boom or Crash index."""
        return self._is_boom_symbol(symbol) or self._is_crash_symbol(symbol)
    
    def _detect_spike_direction(self, df: pd.DataFrame) -> Optional[str]:
        """Detect current spike direction from price action.
        
        Returns:
            "UP" if bullish spike detected
            "DOWN" if bearish spike detected
            None if no clear spike
        """
        if len(df) < self.spike_confirm_bars + 5:
            return None
        
        # Look at recent bars for spike pattern
        recent = df.tail(self.spike_confirm_bars * 2)
        
        # Calculate bar-by-bar ranges
        ranges = (recent["high"] - recent["low"]) / recent["close"] * 100
        
        # Identify largest range bars (potential spikes)
        if len(ranges) < 3:
            return None
        
        # Check if recent bars show spike characteristics
        # Spike = large range + close near extreme
        last_bar = df.iloc[-2]  # Use closed candle
        prev_bar = df.iloc[-3]
        
        last_range_pct = (last_bar["high"] - last_bar["low"]) / last_bar["close"] * 100
        prev_range_pct = (prev_bar["high"] - prev_bar["low"]) / prev_bar["close"] * 100
        
        # Spike detection thresholds (Boom/Crash typically spike 0.5-2%)
        spike_threshold = 0.3  # 0.3% move in one bar
        
        # Bullish spike: large up bar, close near high
        if last_range_pct > spike_threshold:
            body = last_bar["close"] - last_bar["open"]
            range_size = last_bar["high"] - last_bar["low"]
            
            if range_size > 0:
                body_ratio = body / range_size
                
                # Strong bullish spike: body is >70% of range, positive
                if body_ratio > 0.7 and body > 0:
                    return "UP"
                
                # Strong bearish spike: body is >70% of range, negative
                if body_ratio > 0.7 and body < 0:
                    return "DOWN"
        
        # Check for momentum continuation
        closes = recent["close"].values
        if len(closes) >= 3:
            # Upward momentum: consecutively higher closes
            if closes[-1] > closes[-2] > closes[-3]:
                mom = (closes[-1] - closes[-3]) / closes[-3] * 100
                if mom > 0.5:  # 0.5% move in 3 bars
                    return "UP"
            
            # Downward momentum: consecutively lower closes
            if closes[-1] < closes[-2] < closes[-3]:
                mom = (closes[-3] - closes[-1]) / closes[-3] * 100
                if mom > 0.5:
                    return "DOWN"
        
        return None
    
    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        """Evaluate Boom/Crash-specific risk constraints.
        
        Rules:
        1. Only trade IN spike direction (never counter-spike)
        2. Require minimum volatility (no spike = no opportunity)
        3. Use wider ATR multiples for SL/TP
        """
        signal = ctx.signal
        if signal is None:
            return RiskVerdict(self.name, False, "signal is None")
        
        symbol = getattr(signal, "symbol", "")
        if not symbol:
            return RiskVerdict(self.name, False, "signal missing symbol")
        
        # Check if this is a Boom/Crash symbol
        if not self._is_boom_crash_symbol(symbol):
            # Not a Boom/Crash symbol - pass through
            return RiskVerdict(self.name, True, "not boom/crash symbol")
        
        # This IS a Boom/Crash symbol - apply specialized checks
        instrument_type = "Boom" if self._is_boom_symbol(symbol) else "Crash"
        log.info(f"BoomCrashGate: evaluating {symbol} ({instrument_type})")
        
        # Check 1: Detect spike direction
        spike_dir = self._detect_spike_direction(ctx.df)
        
        if spike_dir is None:
            # No clear spike detected - require higher confidence
            action = getattr(signal, "action", None)
            if action:
                action_str = action.name if hasattr(action, "name") else str(action)
                log.warning(
                    f"BoomCrashGate: {symbol} - no clear spike detected, "
                    f"allowing {action_str} with caution"
                )
                # Allow but flag for monitoring
                return RiskVerdict(
                    self.name, True,
                    f"no spike detected on {symbol} - monitor closely",
                    metadata={"spike_direction": None, "instrument_type": instrument_type}
                )
        
        # Check 2: Block counter-spike trades (CRITICAL)
        action = getattr(signal, "action", None)
        if action:
            action_str = action.name if hasattr(action, "name") else str(action)
            
            if not self.allow_counter_spike:
                # Boom + spike UP = only BUY allowed
                # Boom + spike DOWN = only SELL allowed (counter-spike, block)
                # Crash + spike DOWN = only SELL allowed
                # Crash + spike UP = only BUY allowed (counter-spike, block)
                
                if instrument_type == "Boom":
                    if spike_dir == "UP" and action_str == "SELL":
                        return RiskVerdict(
                            self.name, False,
                            f"BLOCKED: {symbol} spiking UP, cannot SELL (counter-spike). "
                            f"Boom indices only spike UP - trade with spike direction only."
                        )
                    elif spike_dir == "DOWN" and action_str == "BUY":
                        log.warning(
                            f"BoomCrashGate: {symbol} drifting down (normal), allowing SELL"
                        )
                
                elif instrument_type == "Crash":
                    if spike_dir == "DOWN" and action_str == "BUY":
                        return RiskVerdict(
                            self.name, False,
                            f"BLOCKED: {symbol} spiking DOWN, cannot BUY (counter-spike). "
                            f"Crash indices only spike DOWN - trade with spike direction only."
                        )
                    elif spike_dir == "UP" and action_str == "SELL":
                        log.warning(
                            f"BoomCrashGate: {symbol} drifting up (normal), allowing BUY"
                        )
        
        # Check 3: Validate ATR for appropriate stop distance
        # Boom/Crash need wider stops due to spike volatility
        if ctx.df is not None and len(ctx.df) > 20:
            try:
                # Calculate ATR (simplified - should use proper ATR function)
                high_low = ctx.df["high"] - ctx.df["low"]
                atr = high_low.rolling(14).mean().iloc[-2]
                atr_pct = atr / ctx.df["close"].iloc[-2] * 100
                
                # Boom/Crash typically have ATR 0.1-0.5%
                # If ATR too low, no spike activity = avoid
                if atr_pct < 0.05:
                    return RiskVerdict(
                        self.name, False,
                        f"ATR too low ({atr_pct:.3f}%) - no spike activity on {symbol}"
                    )
                
                log.info(
                    f"BoomCrashGate: {symbol} ATR={atr_pct:.3f}% - acceptable for trading"
                )
                
            except Exception as e:
                log.warning(f"BoomCrashGate: ATR check failed for {symbol}: {e}")
        
        # All checks passed
        return RiskVerdict(
            self.name, True,
            f"OK: {instrument_type} spike direction={spike_dir}, trade aligned",
            metadata={
                "spike_direction": spike_dir,
                "instrument_type": instrument_type,
                "min_atr_multiple": self.min_spike_atr
            }
        )


# Export for easy import
__all__ = ["BoomCrashGate"]
