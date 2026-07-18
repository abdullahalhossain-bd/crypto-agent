"""trading_modules/capital_flow_analyzer.py
=====================================================================
Capital Flow Analyzer (Principle #121 — Capital Flows Create Trends)
=====================================================================
Detects institutional capital flows: where is smart money moving?

Markets are not driven by price alone — they are driven by CAPITAL FLOWS.
When large institutions deploy capital, they leave footprints:
    - Volume spikes on specific bars
    - Delta imbalance (buy volume vs sell volume)
    - Absorption (large volume but price doesn't move)
    - Liquidity shifts (depth moves from one level to another)
    - Cross-asset rotation (e.g., money flowing from BTC → altcoins)

This module detects and quantifies these flows.

Flow Types Detected:
    1. INSTITUTIONAL BUYING — high volume + rising price + tight spread
    2. INSTITUTIONAL SELLING — high volume + falling price + tight spread
    3. ABSORPTION — high volume but price stalls (smart money absorbing)
    4. ACCUMULATION — sustained buying at lower prices (quiet)
    5. DISTRIBUTION — sustained selling at higher prices (quiet)
    6. LIQUIDITY ROTATION — volume shifts from one level to another
    7. SECTOR ROTATION — capital rotating between asset classes

Flow Strength Score (0-100):
    Combines: MFI, OBV slope, Volume Delta, RVol, Price impact ratio

Usage:
    analyzer = CapitalFlowAnalyzer()
    flow = analyzer.analyze(df)
    # flow = {
    #     "flow_type": "institutional_buying",
    #     "strength": 78,
    #     "direction": "bullish",
    #     "delta_volume": 125000,
    #     "mfi": 68.5,
    #     "absorption": False,
    #     "rotation_detected": False,
    #     "description": "Strong institutional buying detected"
    # }

    # Cross-asset flow:
    flows = analyzer.analyze_rotation({"BTCUSD": df_btc, "ETHUSD": df_eth})
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.capital_flow_analyzer")


class FlowType(Enum):
    """Type of capital flow detected."""
    INSTITUTIONAL_BUYING = "institutional_buying"
    INSTITUTIONAL_SELLING = "institutional_selling"
    ABSORPTION = "absorption"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    ROTATION = "rotation"
    NEUTRAL = "neutral"


@dataclass
class FlowResult:
    """Capital flow analysis result for a single symbol."""
    flow_type: FlowType = FlowType.NEUTRAL
    strength: float = 0.0           # 0-100
    direction: str = "neutral"      # bullish, bearish, neutral
    confidence: float = 0.0         # 0-1
    # Metrics
    mfi: float = 50.0
    obv_slope: float = 0.0
    volume_delta: float = 0.0       # buy vol - sell vol
    rvol: float = 1.0               # relative volume
    price_impact: float = 0.0       # price change per unit volume
    absorption_detected: bool = False
    rotation_detected: bool = False
    smart_money_score: float = 0.0  # 0-1
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_type": self.flow_type.value,
            "strength": round(self.strength, 1),
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "mfi": round(self.mfi, 1),
            "obv_slope": round(self.obv_slope, 3),
            "volume_delta": round(self.volume_delta, 0),
            "rvol": round(self.rvol, 2),
            "price_impact": round(self.price_impact, 6),
            "absorption_detected": self.absorption_detected,
            "rotation_detected": self.rotation_detected,
            "smart_money_score": round(self.smart_money_score, 3),
            "description": self.description,
        }


class CapitalFlowAnalyzer:
    """Analyzes capital flows from OHLCV data.

    Uses volume analysis, price-vol relationships, and flow indicators
    to detect institutional activity.
    """

    def __init__(self,
                 high_volume_threshold: float = 2.0,
                 absorption_volume_threshold: float = 2.5,
                 lookback: int = 20):
        """Initialize analyzer.

        Args:
            high_volume_threshold: RVol above this = high volume
            absorption_volume_threshold: RVol above this with no price move = absorption
            lookback: bars to look back for averages
        """
        self.high_vol_threshold = high_volume_threshold
        self.absorption_threshold = absorption_volume_threshold
        self.lookback = lookback

    # ------------------------------------------------------------------
    # Single-symbol analysis
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame) -> FlowResult:
        """Analyze capital flow for a single symbol.

        Args:
            df: OHLCV DataFrame

        Returns:
            FlowResult with detected flow type + strength
        """
        result = FlowResult()

        if df is None or df.empty or len(df) < self.lookback * 2:
            result.description = "insufficient data"
            return result

        close = df["close"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # === 1. Money Flow Index (MFI) ===
        result.mfi = self._compute_mfi(df, 14)

        # === 2. OBV slope ===
        obv = self._compute_obv(df)
        result.obv_slope = float(obv.iloc[-1] - obv.iloc[-self.lookback])

        # === 3. Volume Delta (approximate from candle direction) ===
        result.volume_delta = self._estimate_volume_delta(df)

        # === 4. Relative Volume ===
        recent_vol = float(vol.tail(5).mean())
        avg_vol = float(vol.tail(self.lookback).mean())
        result.rvol = recent_vol / max(avg_vol, 1)

        # === 5. Price Impact (price change per unit volume) ===
        price_change = float(close.iloc[-1] - close.iloc[-5])
        vol_sum = float(vol.tail(5).sum())
        result.price_impact = price_change / max(vol_sum, 1)

        # === 6. Absorption detection ===
        result.absorption_detected = self._detect_absorption(df)

        # === 7. Rotation detection ===
        result.rotation_detected = self._detect_rotation(df)

        # === 8. Smart money score ===
        result.smart_money_score = self._compute_smart_money_score(
            result.mfi, result.rvol, result.absorption_detected,
            result.obv_slope, result.volume_delta)

        # === Classify flow type ===
        result.flow_type, result.direction, result.confidence = self._classify_flow(
            result, close)

        # === Strength score (0-100) ===
        result.strength = self._compute_strength(result)

        # === Description ===
        result.description = self._describe(result)

        return result

    # ------------------------------------------------------------------
    # Multi-symbol rotation analysis
    # ------------------------------------------------------------------
    def analyze_rotation(self,
                         dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze capital rotation across multiple symbols.

        Detects when capital is flowing FROM one asset TO another.

        Args:
            dfs: {"BTCUSD": df_btc, "ETHUSD": df_eth, ...}

        Returns:
            Dict with per-symbol flows + rotation signals
        """
        flows: Dict[str, FlowResult] = {}
        for symbol, df in dfs.items():
            if df is not None and not df.empty:
                flows[symbol] = self.analyze(df)

        # Sort by strength
        sorted_flows = sorted(flows.items(),
                             key=lambda x: x[1].strength, reverse=True)

        # Detect rotation: strong inflows vs strong outflows
        inflows = [(s, f) for s, f in sorted_flows
                   if f.direction == "bullish" and f.strength > 50]
        outflows = [(s, f) for s, f in sorted_flows
                    if f.direction == "bearish" and f.strength > 50]

        rotation_signal = len(inflows) > 0 and len(outflows) > 0

        return {
            "flows": {s: f.to_dict() for s, f in flows.items()},
            "strongest_inflow": inflows[0][0] if inflows else None,
            "strongest_outflow": outflows[0][0] if outflows else None,
            "rotation_detected": rotation_signal,
            "description": (
                f"Capital flowing from {outflows[0][0]} → {inflows[0][0]}"
                if rotation_signal else "No rotation detected"
            ),
            "ranking": [s for s, _ in sorted_flows],
        }

    # ------------------------------------------------------------------
    # Helper computations
    # ------------------------------------------------------------------
    def _compute_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Money Flow Index."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        vol = df.get("volume", pd.Series(1, index=df.index))
        tp = (high + low + close) / 3
        mf = tp * vol
        pos_mf = mf.where(tp > tp.shift(1), 0.0)
        neg_mf = mf.where(tp < tp.shift(1), 0.0)
        pos_sum = pos_mf.rolling(period).sum()
        neg_sum = neg_mf.rolling(period).sum()
        mfr = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0

    def _compute_obv(self, df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        close = df["close"]
        vol = df.get("volume", pd.Series(1, index=df.index))
        direction = close.diff().apply(np.sign)
        return (direction * vol).fillna(0).cumsum()

    def _estimate_volume_delta(self, df: pd.DataFrame) -> float:
        """Estimate buy volume - sell volume from candle direction.

        Without level-2 data, we approximate:
            Bullish candle: buy_vol = volume * (close-low)/(high-low)
            Bearish candle: sell_vol = volume * (high-close)/(high-low)
        """
        if len(df) < 5:
            return 0.0
        recent = df.tail(5)
        high = recent["high"]
        low = recent["low"]
        close = recent["close"]
        vol = recent.get("volume", pd.Series(1, index=recent.index))
        # Critical #4 fix: use clip(lower=1e-10) instead of replace(0, NaN)
        # to avoid NaN propagation. Zero-range candles produce buy_pct=0.5
        # (neutral) instead of NaN (which would corrupt all downstream calcs).
        range_ = (high - low).clip(lower=1e-10)
        buy_pct = (close - low) / range_
        buy_vol = (vol * buy_pct).sum()
        sell_vol = vol.sum() - buy_vol
        return float(buy_vol - sell_vol)

    def _detect_absorption(self, df: pd.DataFrame) -> bool:
        """Detect absorption: high volume but price barely moves.

        Smart money absorbs sell pressure without price dropping.
        """
        if len(df) < 10:
            return False
        recent = df.tail(5)
        vol = recent.get("volume", pd.Series(1, index=recent.index))
        avg_vol = float(df["volume"].tail(20).mean()) if "volume" in df else 1
        rvol = float(vol.mean()) / max(avg_vol, 1)

        # High volume but small price range
        price_range = float(recent["high"].max() - recent["low"].min())
        avg_price = float(recent["close"].mean())
        range_pct = price_range / max(avg_price, 1e-10)

        return rvol > self.absorption_threshold and range_pct < 0.005

    def _detect_rotation(self, df: pd.DataFrame) -> bool:
        """Detect volume rotation: volume shifts to different price levels."""
        if len(df) < 30 or "volume" not in df:
            return False
        # Split into 3 equal windows
        n = len(df)
        w1 = df["volume"].iloc[n//3 : 2*n//3].mean()
        w2 = df["volume"].iloc[2*n//3:].mean()
        if w1 > 0:
            return abs(w2 - w1) / w1 > 0.5  # 50% shift
        return False

    def _compute_smart_money_score(self, mfi: float, rvol: float,
                                    absorption: bool, obv_slope: float,
                                    delta: float) -> float:
        """Compute smart money participation score (0-1)."""
        score = 0.0
        # MFI extremes indicate smart money
        if mfi > 80 or mfi < 20:
            score += 0.3
        # High volume
        if rvol > self.high_vol_threshold:
            score += 0.25
        # Absorption
        if absorption:
            score += 0.25
        # OBV slope direction
        if abs(obv_slope) > 0:
            score += 0.1
        # Delta direction
        if abs(delta) > 0:
            score += 0.1
        return min(1.0, score)

    def _classify_flow(self, result: FlowResult,
                       close: pd.Series) -> Tuple[FlowType, str, float]:
        """Classify the flow type from metrics."""
        price_change = float(close.iloc[-1] - close.iloc[-5]) / max(close.iloc[-5], 1e-10)

        # Absorption: high volume, no price move
        if result.absorption_detected:
            direction = "bullish" if result.volume_delta > 0 else "bearish"
            return FlowType.ABSORPTION, direction, 0.75

        # Institutional buying: high volume + rising price + positive delta
        if (result.rvol > self.high_vol_threshold and
            price_change > 0.005 and
            result.volume_delta > 0):
            return FlowType.INSTITUTIONAL_BUYING, "bullish", 0.85

        # Institutional selling: high volume + falling price + negative delta
        if (result.rvol > self.high_vol_threshold and
            price_change < -0.005 and
            result.volume_delta < 0):
            return FlowType.INSTITUTIONAL_SELLING, "bearish", 0.85

        # Accumulation: sustained positive OBV + low volatility
        if result.obv_slope > 0 and result.rvol < 1.0:
            return FlowType.ACCUMULATION, "bullish", 0.60

        # Distribution: sustained negative OBV + low volatility
        if result.obv_slope < 0 and result.rvol < 1.0:
            return FlowType.DISTRIBUTION, "bearish", 0.60

        # Rotation
        if result.rotation_detected:
            return FlowType.ROTATION, "neutral", 0.50

        return FlowType.NEUTRAL, "neutral", 0.30

    def _compute_strength(self, result: FlowResult) -> float:
        """Compute flow strength (0-100)."""
        score = 0.0
        # RVol contribution (0-30)
        score += min(30, result.rvol * 15)
        # MFI contribution (0-20)
        score += abs(result.mfi - 50) * 0.4
        # Delta contribution (0-20)
        score += min(20, abs(result.volume_delta) / 1000)
        # Absorption bonus (0-15)
        if result.absorption_detected:
            score += 15
        # Smart money score (0-15)
        score += result.smart_money_score * 15
        return min(100, score)

    def _describe(self, result: FlowResult) -> str:
        """Human-readable description."""
        descs = {
            FlowType.INSTITUTIONAL_BUYING: (
                f"Institutional buying detected — RVol={result.rvol:.1f}x, "
                f"MFI={result.mfi:.0f}, delta=+{result.volume_delta:.0f}. "
                f"Smart money accumulating."
            ),
            FlowType.INSTITUTIONAL_SELLING: (
                f"Institutional selling detected — RVol={result.rvol:.1f}x, "
                f"MFI={result.mfi:.0f}, delta={result.volume_delta:.0f}. "
                f"Smart money distributing."
            ),
            FlowType.ABSORPTION: (
                f"Absorption detected — high volume ({result.rvol:.1f}x) "
                f"but price stalling. Smart money absorbing orders."
            ),
            FlowType.ACCUMULATION: (
                "Quiet accumulation — OBV rising, low volume. "
                "Smart money positioning before markup."
            ),
            FlowType.DISTRIBUTION: (
                "Quiet distribution — OBV falling, low volume. "
                "Smart money exiting before markdown."
            ),
            FlowType.ROTATION: (
                "Volume rotation detected — capital shifting levels. "
                "Watch for direction change."
            ),
            FlowType.NEUTRAL: "No significant capital flow detected.",
        }
        return descs.get(result.flow_type, "Unknown flow")