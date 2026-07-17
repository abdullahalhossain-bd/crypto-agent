"""
SMC Detector Module — Smart Money Concepts Detection
=====================================================

Detects institutional trading patterns using the smartmoneyconcepts library:
  - BOS (Break of Structure) — trend continuation signal
  - CHoCH (Change of Character) — trend reversal signal
  - Order Blocks — institutional order concentration zones
  - FVG (Fair Value Gap) — price refill targets
  - Liquidity Zones — stop-hunt target areas
  - Swing Highs/Lows — market structure mapping

This module wraps the smartmoneyconcepts library with a clean interface
suitable for integration into a multi-agent trading framework.

Dependencies:
    pip install smartmoneyconcepts pandas numpy

Usage:
    from smc_detector import SMCDetector

    detector = SMCDetector()
    result = detector.analyze(ohlcv_df, symbol="BTCUSDT")

    if result['bullish_bos']:
        print(f"Bullish BOS detected at {result['latest_bos']['level']}")
    if result['order_blocks']:
        print(f"{len(result['order_blocks'])} order blocks found")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SMCResult:
    """Container for all SMC analysis results."""
    symbol: str = ""
    # Market Structure
    swing_highs_lows: pd.DataFrame = field(default_factory=pd.DataFrame)
    bos: pd.DataFrame = field(default_factory=pd.DataFrame)  # Break of Structure
    choch: pd.DataFrame = field(default_factory=pd.DataFrame)  # Change of Character
    # Zones
    order_blocks: pd.DataFrame = field(default_factory=pd.DataFrame)
    fvg: pd.DataFrame = field(default_factory=pd.DataFrame)  # Fair Value Gap
    liquidity: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Summary signals
    bullish_bos: bool = False
    bearish_bos: bool = False
    bullish_choch: bool = False
    bearish_choch: bool = False
    current_trend: str = "unknown"  # bullish / bearish / ranging
    # Key levels
    nearest_demand_zone: Optional[float] = None
    nearest_supply_zone: Optional[float] = None
    nearest_fvg: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for LLM agent consumption."""
        return {
            "symbol": self.symbol,
            "current_trend": self.current_trend,
            "bullish_bos": self.bullish_bos,
            "bearish_bos": self.bearish_bos,
            "bullish_choch": self.bullish_choch,
            "bearish_choch": self.bearish_choch,
            "nearest_demand_zone": self.nearest_demand_zone,
            "nearest_supply_zone": self.nearest_supply_zone,
            "nearest_fvg": self.nearest_fvg,
            "order_block_count": len(self.order_blocks),
            "fvg_count": len(self.fvg),
            "liquidity_count": len(self.liquidity),
            "latest_bos": self._safe_last_row(self.bos),
            "latest_choch": self._safe_last_row(self.choch),
        }

    @staticmethod
    def _safe_last_row(df: pd.DataFrame) -> Optional[dict]:
        if df is None or df.empty:
            return None
        return df.iloc[-1].to_dict()


class SMCDetector:
    """
    Smart Money Concepts detector.

    Wraps the smartmoneyconcepts library to provide a unified interface
    for BOS/CHoCH/Order Block/FVG/Liquidity/Swing detection.

    Parameters:
        swing_length: Window for swing high/low detection (default: 10)
        close_break: Whether BOS/CHoCH requires closing-price break (default: True)
    """

    def __init__(self, swing_length: int = 10, close_break: bool = True):
        self.swing_length = swing_length
        self.close_break = close_break

    def analyze(self, df: pd.DataFrame, symbol: str = "") -> SMCResult:
        """
        Run full SMC analysis on OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                (case-insensitive, will be normalized)
            symbol: Trading symbol for context

        Returns:
            SMCResult with all detected SMC patterns
        """
        # Normalize column names to lowercase
        df = self._normalize_columns(df)

        if len(df) < 50:
            logger.warning(f"SMC analysis needs at least 50 bars, got {len(df)}")
            return SMCResult(symbol=symbol)

        result = SMCResult(symbol=symbol)

        try:
            # 1. Swing Highs/Lows — market structure foundation
            result.swing_highs_lows = self._detect_swing_highs_lows(df)

            # 2. BOS/CHoCH — structure breaks (needs swing_highs_lows)
            result.bos, result.choch = self._detect_bos_choch(df, result.swing_highs_lows)
            self._classify_structure_signals(result)

            # 3. Order Blocks — institutional zones (needs swing_highs_lows)
            result.order_blocks = self._detect_order_blocks(df, result.swing_highs_lows)

            # 4. FVG — Fair Value Gaps (doesn't need swing_highs_lows)
            result.fvg = self._detect_fvg(df)

            # 5. Liquidity — stop-hunt zones (needs swing_highs_lows)
            result.liquidity = self._detect_liquidity(df, result.swing_highs_lows)

            # 6. Derive key levels for confluence gate
            self._compute_nearest_zones(df, result)

            # 7. Determine current trend
            result.current_trend = self._determine_trend(result)

        except Exception as e:
            logger.error(f"SMC analysis failed for {symbol}: {e}", exc_info=True)

        return result

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV column names to lowercase."""
        df = df.copy()
        rename_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ('open', 'high', 'low', 'close', 'volume'):
                rename_map[col] = lower
        df.rename(columns=rename_map, inplace=True)

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        if 'volume' not in df.columns:
            df['volume'] = 0.0

        return df

    def _detect_swing_highs_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect swing highs and lows for market structure mapping."""
        from smartmoneyconcepts.smc import smc

        swing = smc.swing_highs_lows(
            df,
            swing_length=self.swing_length,
        )
        return swing if swing is not None else pd.DataFrame()

    def _detect_bos_choch(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH).

        BOS = trend continuation (price breaks previous high in uptrend)
        CHoCH = trend reversal (price breaks against the trend)
        """
        from smartmoneyconcepts.smc import smc

        if swing_highs_lows is None or swing_highs_lows.empty:
            return pd.DataFrame(), pd.DataFrame()

        bos_choch = smc.bos_choch(
            df,
            swing_highs_lows=swing_highs_lows,
            close_break=self.close_break,
        )

        if bos_choch is None or (hasattr(bos_choch, 'empty') and bos_choch.empty):
            return pd.DataFrame(), pd.DataFrame()

        # bos_choch returns a Series; convert to DataFrame for consistency
        if isinstance(bos_choch, pd.Series):
            bos_choch = bos_choch.to_frame(name='BOS/CHoCH')

        # Split into BOS and CHoCH
        bos_df = bos_choch[bos_choch.iloc[:, 0] == 'BOS'].copy() if len(bos_choch) > 0 else pd.DataFrame()
        choch_df = bos_choch[bos_choch.iloc[:, 0] == 'CHoCH'].copy() if len(bos_choch) > 0 else pd.DataFrame()

        return bos_df, choch_df

    def _detect_order_blocks(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks — zones where institutional orders are concentrated.

        Bullish OB: last down candle before a strong up move
        Bearish OB: last up candle before a strong down move
        """
        from smartmoneyconcepts.smc import smc

        if swing_highs_lows is None or swing_highs_lows.empty:
            return pd.DataFrame()

        ob = smc.ob(df, swing_highs_lows=swing_highs_lows)
        if ob is None:
            return pd.DataFrame()
        if isinstance(ob, pd.Series):
            ob = ob.to_frame(name='OB')
        return ob

    def _detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps — price imbalances that act as magnets.

        Bullish FVG: gap up where low(t) > high(t-2) — price likely to return
        Bearish FVG: gap down where high(t) < low(t-2) — price likely to return
        """
        from smartmoneyconcepts.smc import smc

        fvg = smc.fvg(df, join_consecutive=True)
        if fvg is None:
            return pd.DataFrame()
        if isinstance(fvg, pd.Series):
            fvg = fvg.to_frame(name='FVG')
        return fvg

    def _detect_liquidity(self, df: pd.DataFrame, swing_highs_lows: pd.DataFrame) -> pd.DataFrame:
        """
        Detect liquidity zones — areas where stop losses are clustered.

        These are magnets for smart money to hunt stops before reversing.
        """
        from smartmoneyconcepts.smc import smc

        if swing_highs_lows is None or swing_highs_lows.empty:
            return pd.DataFrame()

        liq = smc.liquidity(df, swing_highs_lows=swing_highs_lows)
        if liq is None:
            return pd.DataFrame()
        if isinstance(liq, pd.Series):
            liq = liq.to_frame(name='Liquidity')
        return liq

    def _classify_structure_signals(self, result: SMCResult) -> None:
        """Classify latest BOS/CHoCH as bullish or bearish."""
        if not result.bos.empty:
            last_bos = result.bos.iloc[-1]
            broken_level = str(last_bos.get('Broken', '')).lower()
            result.bullish_bos = 'bull' in broken_level
            result.bearish_bos = 'bear' in broken_level

        if not result.choch.empty:
            last_choch = result.choch.iloc[-1]
            broken_level = str(last_choch.get('Broken', '')).lower()
            result.bullish_choch = 'bull' in broken_level
            result.bearish_choch = 'bear' in broken_level

    def _compute_nearest_zones(self, df: pd.DataFrame, result: SMCResult) -> None:
        """Compute nearest demand/supply zones and FVG to current price."""
        if df.empty:
            return

        current_price = float(df['close'].iloc[-1])

        # Nearest demand zone (below current price) from Order Blocks
        if not result.order_blocks.empty:
            ob = result.order_blocks
            # Bullish OBs below price = demand
            if 'OB' in ob.columns or 'Top' in ob.columns:
                # Try to find bullish OBs below current price
                for _, row in ob.iterrows():
                    ob_high = float(row.get('Top', row.get('OB_High', 0)))
                    if ob_high < current_price:
                        if result.nearest_demand_zone is None or ob_high > result.nearest_demand_zone:
                            result.nearest_demand_zone = ob_high
                    else:
                        if result.nearest_supply_zone is None or ob_high < result.nearest_supply_zone:
                            result.nearest_supply_zone = ob_high

        # Nearest unfilled FVG
        if not result.fvg.empty:
            fvg = result.fvg
            for _, row in fvg.iterrows():
                fvg_high = float(row.get('Top', 0))
                fvg_low = float(row.get('Bottom', 0))
                fvg_type = str(row.get('FVG', '')).lower()
                if fvg_high == 0 and fvg_low == 0:
                    continue
                # Check if FVG is still unfilled (price hasn't returned)
                if 'bull' in fvg_type and fvg_low < current_price:
                    result.nearest_fvg = {
                        "type": "bullish",
                        "top": fvg_high,
                        "bottom": fvg_low,
                        "distance_pct": ((current_price - fvg_low) / current_price) * 100,
                    }
                    break
                elif 'bear' in fvg_type and fvg_high > current_price:
                    result.nearest_fvg = {
                        "type": "bearish",
                        "top": fvg_high,
                        "bottom": fvg_low,
                        "distance_pct": ((fvg_high - current_price) / current_price) * 100,
                    }
                    break

    def _determine_trend(self, result: SMCResult) -> str:
        """
        Determine current market trend from BOS/CHoCH sequence.

        Logic:
        - Latest CHoCH bullish → bullish (reversal)
        - Latest CHoCH bearish → bearish (reversal)
        - Latest BOS bullish → bullish (continuation)
        - Latest BOS bearish → bearish (continuation)
        - No structure breaks → ranging
        """
        if result.bullish_choch:
            return "bullish"
        if result.bearish_choch:
            return "bearish"
        if result.bullish_bos:
            return "bullish"
        if result.bearish_bos:
            return "bearish"
        return "ranging"

    def get_confluence_context(self, result: SMCResult, current_price: float) -> str:
        """
        Generate a text summary for LLM agent consumption.

        This provides a compact SMC context block that can be injected
        into analyst agent prompts.
        """
        lines = [f"## SMC Analysis for {result.symbol}"]

        # Trend
        lines.append(f"**Current Structure Trend**: {result.current_trend}")

        # Structure breaks
        if result.bullish_bos:
            lines.append("✅ Bullish BOS detected (trend continuation up)")
        if result.bearish_bos:
            lines.append("✅ Bearish BOS detected (trend continuation down)")
        if result.bullish_choch:
            lines.append("⚠️ Bullish CHoCH detected (potential reversal up)")
        if result.bearish_choch:
            lines.append("⚠️ Bearish CHoCH detected (potential reversal down)")

        # Zones
        if result.nearest_demand_zone:
            dist = ((current_price - result.nearest_demand_zone) / current_price) * 100
            lines.append(f"**Nearest Demand Zone**: {result.nearest_demand_zone:.2f} ({dist:.1f}% below)")

        if result.nearest_supply_zone:
            dist = ((result.nearest_supply_zone - current_price) / current_price) * 100
            lines.append(f"**Nearest Supply Zone**: {result.nearest_supply_zone:.2f} ({dist:.1f}% above)")

        if result.nearest_fvg:
            fvg = result.nearest_fvg
            lines.append(
                f"**Nearest FVG** ({fvg['type']}): "
                f"{fvg['bottom']:.2f} - {fvg['top']:.2f} "
                f"({fvg['distance_pct']:.1f}% away)"
            )

        # Counts
        lines.append(
            f"**Zones**: {len(result.order_blocks)} Order Blocks, "
            f"{len(result.fvg)} FVGs, "
            f"{len(result.liquidity)} Liquidity zones"
        )

        return "\n".join(lines)
