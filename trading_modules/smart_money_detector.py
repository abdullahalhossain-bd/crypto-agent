"""trading_modules/smart_money_detector.py
=====================================================================
Smart Money Detector (Principle #128)
=====================================================================
Detects institutional participation in the market.

Smart money leaves footprints:
    1. ABSORPTION — high volume but price doesn't move (they're absorbing)
    2. ICEBERG ORDERS — large orders split into small visible chunks
    3. LIQUIDITY POOLS — resting orders at key levels (stops)
    4. STOP HUNTS — price spikes to trigger stops, then reverses
    5. ORDER BLOCK PRESENCE — last opposite candle before strong move
    6. VOLUME CLUSTERS — unusual volume at specific price levels
    7. SQUEEZE PATTERNS — volatility compression before expansion

Smart Money Score (0-100):
    Higher = more institutional activity detected

Trading Rules:
    - If smart_money_score > 70: follow their direction
    - If absorption detected: expect reversal soon
    - If stop hunt detected: fade the spike (counter-trend)
    - If iceberg detected: be patient, they'll push price soon

Usage:
    detector = SmartMoneyDetector()
    result = detector.detect(df, spread_bps=2.5)
    if result.smart_money_score > 70:
        # Follow smart money direction
        if result.inferred_direction == "bullish":
            place_buy()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.smart_money_detector")


@dataclass
class SmartMoneyResult:
    """Smart money detection result."""
    smart_money_score: float = 0.0       # 0-100
    inferred_direction: str = "neutral"  # bullish, bearish, neutral
    confidence: float = 0.0              # 0-1

    # Detection flags
    absorption_detected: bool = False
    iceberg_detected: bool = False
    liquidity_pool_detected: bool = False
    stop_hunt_detected: bool = False
    order_block_detected: bool = False
    volume_cluster_detected: bool = False
    squeeze_detected: bool = False

    # Details
    absorption_level: float = 0.0        # price level where absorption occurred
    liquidity_above: float = 0.0         # liquidity pool above current price
    liquidity_below: float = 0.0         # liquidity pool below current price
    stop_hunt_direction: str = ""        # "up" or "down"
    order_block_high: float = 0.0
    order_block_low: float = 0.0
    cluster_price: float = 0.0           # where volume clustered
    cluster_volume: float = 0.0

    description: str = ""
    actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "smart_money_score": round(self.smart_money_score, 1),
            "inferred_direction": self.inferred_direction,
            "confidence": round(self.confidence, 3),
            "absorption_detected": self.absorption_detected,
            "iceberg_detected": self.iceberg_detected,
            "liquidity_pool_detected": self.liquidity_pool_detected,
            "stop_hunt_detected": self.stop_hunt_detected,
            "order_block_detected": self.order_block_detected,
            "volume_cluster_detected": self.volume_cluster_detected,
            "squeeze_detected": self.squeeze_detected,
            "absorption_level": self.absorption_level,
            "liquidity_above": self.liquidity_above,
            "liquidity_below": self.liquidity_below,
            "stop_hunt_direction": self.stop_hunt_direction,
            "order_block_high": self.order_block_high,
            "order_block_low": self.order_block_low,
            "cluster_price": self.cluster_price,
            "cluster_volume": round(self.cluster_volume, 0),
            "description": self.description,
            "actions": self.actions,
        }


class SmartMoneyDetector:
    """Detects institutional activity from OHLCV data.

    Without level-2 order book data, we approximate institutional
    activity using volume patterns, price-vol relationships, and
    candle structure analysis.
    """

    def __init__(self,
                 high_volume_threshold: float = 2.5,
                 absorption_range_pct: float = 0.003,
                 cluster_threshold: float = 3.0,
                 squeeze_period: int = 20):
        """Initialize detector."""
        self.high_vol_threshold = high_volume_threshold
        self.absorption_range_pct = absorption_range_pct
        self.cluster_threshold = cluster_threshold
        self.squeeze_period = squeeze_period

    def detect(self, df: pd.DataFrame,
               spread_bps: float = 5.0) -> SmartMoneyResult:
        """Detect smart money activity.

        Args:
            df: OHLCV DataFrame (need at least 50 bars)
            spread_bps: current bid-ask spread

        Returns:
            SmartMoneyResult with all detections
        """
        result = SmartMoneyResult()

        if df is None or df.empty or len(df) < 50:
            result.description = "insufficient data"
            return result

        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df.get("volume", pd.Series(1, index=df.index))
        open_ = df["open"]

        # === 1. Absorption detection ===
        result.absorption_detected, result.absorption_level = self._detect_absorption(df)

        # === 2. Iceberg order detection ===
        result.iceberg_detected = self._detect_iceberg(df)

        # === 3. Liquidity pools ===
        result.liquidity_pool_detected, result.liquidity_above, result.liquidity_below = \
            self._detect_liquidity_pools(df)

        # === 4. Stop hunt detection ===
        result.stop_hunt_detected, result.stop_hunt_direction = self._detect_stop_hunt(df)

        # === 5. Order block detection ===
        result.order_block_detected, result.order_block_high, result.order_block_low = \
            self._detect_order_block(df)

        # === 6. Volume cluster detection ===
        result.volume_cluster_detected, result.cluster_price, result.cluster_volume = \
            self._detect_volume_cluster(df)

        # === 7. Squeeze (volatility compression) ===
        result.squeeze_detected = self._detect_squeeze(df)

        # === Compute smart money score ===
        result.smart_money_score = self._compute_score(result)

        # === Infer direction ===
        result.inferred_direction, result.confidence = self._infer_direction(df, result)

        # === Description + actions ===
        result.description = self._describe(result)
        result.actions = self._actions(result)

        return result

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------
    def _detect_absorption(self, df: pd.DataFrame) -> tuple:
        """Detect absorption: high volume, small price range.

        Returns (detected: bool, level: float)
        """
        if len(df) < 20 or "volume" not in df:
            return False, 0.0

        recent = df.tail(5)
        avg_vol = float(df["volume"].tail(20).mean())
        recent_vol = float(recent["volume"].mean())
        rvol = recent_vol / max(avg_vol, 1)

        price_range = float(recent["high"].max() - recent["low"].min())
        avg_price = float(recent["close"].mean())
        range_pct = price_range / max(avg_price, 1e-10)

        if rvol > self.high_vol_threshold and range_pct < self.absorption_range_pct:
            return True, float(recent["close"].iloc[-1])
        return False, 0.0

    def _detect_iceberg(self, df: pd.DataFrame) -> bool:
        """Detect iceberg orders: many small trades but consistent direction.

        Approximation: small candle bodies + consistent direction + high tick volume
        """
        if len(df) < 10 or "volume" not in df:
            return False
        recent = df.tail(10)
        bodies = (recent["close"] - recent["open"]).abs()
        ranges = recent["high"] - recent["low"]
        body_ratio = bodies / ranges.replace(0, np.nan)
        # Small bodies (< 30% of range) but consistent direction
        avg_body_ratio = float(body_ratio.mean())
        direction = np.sign(recent["close"] - recent["open"])
        consistent = (direction == direction.iloc[0]).sum() >= 7
        avg_vol = float(df["volume"].tail(50).mean())
        recent_vol = float(recent["volume"].mean())
        rvol = recent_vol / max(avg_vol, 1)

        return avg_body_ratio < 0.3 and consistent and rvol > 1.2

    def _detect_liquidity_pools(self, df: pd.DataFrame) -> tuple:
        """Detect liquidity pools (areas of concentrated stops).

        Returns (detected, level_above, level_below)
        """
        if len(df) < 50:
            return False, 0.0, 0.0
        # Recent swing highs/lows
        high = df["high"]
        low = df["low"]
        close = df["close"]
        # Equal highs (resistance with stops above)
        recent_highs = high.tail(50).nlargest(5)
        eq_high = recent_highs.std() / max(recent_highs.mean(), 1e-10) < 0.002
        # Equal lows (support with stops below)
        recent_lows = low.tail(50).nsmallest(5)
        eq_low = recent_lows.std() / max(recent_lows.mean(), 1e-10) < 0.002

        detected = eq_high or eq_low
        above = float(recent_highs.mean()) if eq_high else float(high.tail(50).max())
        below = float(recent_lows.mean()) if eq_low else float(low.tail(50).min())
        return detected, above, below

    def _detect_stop_hunt(self, df: pd.DataFrame) -> tuple:
        """Detect stop hunt: spike beyond level, then reversal.

        Returns (detected, direction)
        """
        if len(df) < 20:
            return False, ""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        recent_high = high.tail(20).head(19).max()
        recent_low = low.tail(20).head(19).min()
        last_high = float(high.iloc[-1])
        last_low = float(low.iloc[-1])
        last_close = float(close.iloc[-1])

        # Stop hunt above
        if last_high > recent_high * 1.003 and last_close < recent_high:
            return True, "up"
        # Stop hunt below
        if last_low < recent_low * 0.997 and last_close > recent_low:
            return True, "down"
        return False, ""

    def _detect_order_block(self, df: pd.DataFrame) -> tuple:
        """Detect order block: last opposite candle before strong move.

        Returns (detected, ob_high, ob_low)
        """
        if len(df) < 5:
            return False, 0.0, 0.0
        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        # Look at last 3 candles
        for i in range(-3, -1):
            if abs(i) > len(df):
                continue
            # Bullish OB: bearish candle followed by strong bullish
            if close.iloc[i] < open_.iloc[i] and close.iloc[i + 1] > high.iloc[i]:
                if close.iloc[i + 1] > high.iloc[max(0, i - 2):i].max():
                    return True, float(high.iloc[i]), float(low.iloc[i])
            # Bearish OB
            if close.iloc[i] > open_.iloc[i] and close.iloc[i + 1] < low.iloc[i]:
                if close.iloc[i + 1] < low.iloc[max(0, i - 2):i].min():
                    return True, float(high.iloc[i]), float(low.iloc[i])
        return False, 0.0, 0.0

    def _detect_volume_cluster(self, df: pd.DataFrame) -> tuple:
        """Detect volume cluster: price level with unusually high volume.

        Returns (detected, price, volume)
        """
        if len(df) < 30 or "volume" not in df:
            return False, 0.0, 0.0
        # Bin by price and sum volume
        close = df["close"].tail(30)
        vol = df["volume"].tail(30)
        try:
            bins = pd.cut(close, bins=5)
            cluster = vol.groupby(bins).sum()
            avg = cluster.mean()
            max_cluster = cluster.max()
            max_bin = cluster.idxmax()
            if max_cluster > avg * self.cluster_threshold:
                # Get price level from bin
                mid_price = (max_bin.left + max_bin.right) / 2
                return True, float(mid_price), float(max_cluster)
        except Exception:
            pass
        return False, 0.0, 0.0

    def _detect_squeeze(self, df: pd.DataFrame) -> bool:
        """Detect volatility squeeze (BBands inside Keltner)."""
        if len(df) < 30:
            return False
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # BBands
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std

        # Keltner
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
        kelt_upper = sma + 1.5 * atr
        kelt_lower = sma - 1.5 * atr

        # Squeeze: BBands inside Keltner
        if pd.isna(bb_upper.iloc[-1]) or pd.isna(kelt_upper.iloc[-1]):
            return False
        return (bb_upper.iloc[-1] < kelt_upper.iloc[-1] and
                bb_lower.iloc[-1] > kelt_lower.iloc[-1])

    # ------------------------------------------------------------------
    # Score + direction
    # ------------------------------------------------------------------
    def _compute_score(self, r: SmartMoneyResult) -> float:
        """Compute smart money score (0-100)."""
        score = 0.0
        if r.absorption_detected:
            score += 25
        if r.iceberg_detected:
            score += 15
        if r.liquidity_pool_detected:
            score += 15
        if r.stop_hunt_detected:
            score += 20
        if r.order_block_detected:
            score += 15
        if r.volume_cluster_detected:
            score += 10
        if r.squeeze_detected:
            score += 10  # squeeze precedes smart money moves
        return min(100, score)

    def _infer_direction(self, df: pd.DataFrame,
                         r: SmartMoneyResult) -> tuple:
        """Infer smart money direction from detections.

        Returns (direction, confidence)
        """
        bull_signals = 0
        bear_signals = 0

        # Stop hunt direction tells us where stops were hunted
        # If stops hunted up, smart money is selling (bearish)
        # If stops hunted down, smart money is buying (bullish)
        if r.stop_hunt_detected:
            if r.stop_hunt_direction == "up":
                bear_signals += 2  # they pushed price up to sell
            else:
                bull_signals += 2  # they pushed price down to buy

        # Absorption at level: direction depends on whether price was going up or down
        if r.absorption_detected:
            recent_close = df["close"].iloc[-5:].values
            if recent_close[-1] > recent_close[0]:
                bear_signals += 1  # absorbed selling → reversal up?
            else:
                bull_signals += 1

        # Order block: bullish OB = expect up move
        if r.order_block_detected:
            close = df["close"].iloc[-1]
            open_ = df["open"].iloc[-2]
            if close > open_:
                bull_signals += 1
            else:
                bear_signals += 1

        # Squeeze: direction TBD, but smart money is positioning
        if r.squeeze_detected:
            # Use EMA to determine likely breakout direction
            ema = df["close"].ewm(span=20, adjust=False).mean()
            if df["close"].iloc[-1] > ema.iloc[-1]:
                bull_signals += 1
            else:
                bear_signals += 1

        if bull_signals > bear_signals:
            return "bullish", min(1.0, bull_signals / 5)
        elif bear_signals > bull_signals:
            return "bearish", min(1.0, bear_signals / 5)
        return "neutral", 0.3

    def _describe(self, r: SmartMoneyResult) -> str:
        """Human-readable description."""
        parts = []
        if r.absorption_detected:
            parts.append(f"Absorption at {r.absorption_level:.2f}")
        if r.iceberg_detected:
            parts.append("Iceberg orders")
        if r.liquidity_pool_detected:
            parts.append(f"Liquidity pools: above={r.liquidity_above:.2f}, below={r.liquidity_below:.2f}")
        if r.stop_hunt_detected:
            parts.append(f"Stop hunt {r.stop_hunt_direction}")
        if r.order_block_detected:
            parts.append(f"Order block: {r.order_block_low:.2f}-{r.order_block_high:.2f}")
        if r.volume_cluster_detected:
            parts.append(f"Volume cluster at {r.cluster_price:.2f}")
        if r.squeeze_detected:
            parts.append("Volatility squeeze")
        if not parts:
            return "No smart money activity detected"
        return f"Smart money score {r.smart_money_score:.0f}/100: " + "; ".join(parts)

    def _actions(self, r: SmartMoneyResult) -> List[str]:
        """Recommended actions."""
        actions = []
        if r.smart_money_score < 30:
            actions.append("No smart money activity — normal trading")
            return actions
        if r.stop_hunt_detected:
            actions.append(f"FADE the stop hunt (counter-trend): if hunted {r.stop_hunt_direction}, "
                          f"trade opposite direction")
        if r.absorption_detected:
            actions.append(f"Watch for reversal at {r.absorption_level:.2f} — absorption precedes reversal")
        if r.order_block_detected:
            actions.append(f"Order block active ({r.order_block_low:.2f}-{r.order_block_high:.2f}) — "
                          f"price likely to react here")
        if r.squeeze_detected:
            actions.append("Squeeze active — expect volatility expansion soon, prepare for breakout")
        if r.inferred_direction != "neutral":
            actions.append(f"Follow smart money: {r.inferred_direction}")
        return actions
