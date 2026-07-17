"""trading_modules/relative_strength_ranker.py
=====================================================================
Relative Strength Ranking Engine (Principle #122, #123)
=====================================================================
Ranks all tradable assets by RELATIVE STRENGTH — not absolute price.

Key Insight (Livermore #122):
    "Strong assets attract more capital."
    "Buy strength, sell weakness — not cheapness."

    Retail buys what's cheap (hoping it recovers).
    Institutions buy what's strong (because capital flows there).

How It Computes Strength:
    1. Price momentum (1, 5, 20, 50 bar returns)
    2. Trend alignment (price vs EMA20, EMA50, EMA200)
    3. Volume trend (RVol, OBV slope)
    4. Volatility-adjusted return (return / ATR)
    5. Outperformance vs benchmark (e.g., vs BTC or market index)
    6. Drawdown recovery (how quickly it bounces from lows)

Strength Score (0-100):
    Top 10%  → Priority long candidates
    Top 30%  → Watchlist
    Middle   → Neutral
    Bottom 30% → Watchlist (short candidates)
    Bottom 10% → Priority short candidates

Usage:
    ranker = RelativeStrengthRanker()

    # Rank multiple symbols
    ranking = ranker.rank({
        "BTCUSD": df_btc,
        "ETHUSD": df_eth,
        "EURUSD": df_eur,
        "GBPUSD": df_gbp,
    })
    # ranking = {
    #     "ranking": ["BTCUSD", "ETHUSD", "EURUSD", "GBPUSD"],
    #     "scores": {"BTCUSD": 85.3, "ETHUSD": 72.1, ...},
    #     "percentiles": {"BTCUSD": 0.95, ...},
    #     "top_candidates": ["BTCUSD", "ETHUSD"],
    #     "bottom_candidates": ["GBPUSD"],
    #     "benchmark": "BTCUSD",
    # }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.relative_strength_ranker")


class StrengthCategory(str, Enum):
    """Classification by strength percentile."""
    ELITE = "elite"            # top 10%
    STRONG = "strong"          # top 10-30%
    AVERAGE = "average"        # 30-70%
    WEAK = "weak"              # bottom 30-10%
    LAGGING = "lagging"        # bottom 10%


@dataclass
class StrengthScore:
    """Relative strength score for a single symbol."""
    symbol: str
    score: float = 0.0          # 0-100
    percentile: float = 0.5     # 0-1
    category: StrengthCategory = StrengthCategory.AVERAGE

    # Component scores
    momentum_5: float = 0.0     # 5-bar return %
    momentum_20: float = 0.0    # 20-bar return %
    momentum_50: float = 0.0    # 50-bar return %
    trend_alignment: float = 0.0  # -1 to +1 (price vs EMAs)
    volume_trend: float = 0.0   # RVol
    vol_adjusted_return: float = 0.0  # return / ATR
    vs_benchmark: float = 0.0   # outperformance vs benchmark
    drawdown_recovery: float = 0.0   # how quickly recovered from DD

    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 1),
            "percentile": round(self.percentile, 3),
            "category": self.category.value,
            "momentum_5": round(self.momentum_5, 3),
            "momentum_20": round(self.momentum_20, 3),
            "momentum_50": round(self.momentum_50, 3),
            "trend_alignment": round(self.trend_alignment, 3),
            "volume_trend": round(self.volume_trend, 2),
            "vol_adjusted_return": round(self.vol_adjusted_return, 3),
            "vs_benchmark": round(self.vs_benchmark, 3),
            "drawdown_recovery": round(self.drawdown_recovery, 3),
            "recommendation": self.recommendation,
        }


class RelativeStrengthRanker:
    """Ranks symbols by relative strength.

    Use this to decide WHICH symbol to trade when multiple signals exist.
    Always trade the STRONGEST symbol for longs, WEAKEST for shorts.
    """

    def __init__(self,
                 benchmark: str = "BTCUSD",
                 top_pct: float = 0.10,
                 bottom_pct: float = 0.10):
        """Initialize ranker.

        Args:
            benchmark: symbol to use as benchmark for outperformance
            top_pct: top percentile for "elite" classification
            bottom_pct: bottom percentile for "lagging" classification
        """
        self.benchmark = benchmark
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct

    # ------------------------------------------------------------------
    # Score single symbol
    # ------------------------------------------------------------------
    def score(self, symbol: str, df: pd.DataFrame,
              benchmark_df: Optional[pd.DataFrame] = None) -> StrengthScore:
        """Compute strength score for a single symbol.

        Args:
            symbol: symbol name
            df: OHLCV DataFrame
            benchmark_df: benchmark OHLCV (for outperformance calc)

        Returns:
            StrengthScore with all component scores
        """
        result = StrengthScore(symbol=symbol)

        if df is None or df.empty or len(df) < 50:
            result.recommendation = "insufficient data"
            return result

        close = df["close"]
        vol = df.get("volume", pd.Series(1, index=df.index))

        # === 1. Momentum (returns over different periods) ===
        result.momentum_5 = float((close.iloc[-1] - close.iloc[-5]) / max(close.iloc[-5], 1e-10) * 100)
        result.momentum_20 = float((close.iloc[-1] - close.iloc[-20]) / max(close.iloc[-20], 1e-10) * 100)
        if len(close) >= 50:
            result.momentum_50 = float((close.iloc[-1] - close.iloc[-50]) / max(close.iloc[-50], 1e-10) * 100)

        # === 2. Trend alignment ===
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(close.iloc[-1])

        alignment = 0.0
        if price > ema20:
            alignment += 0.33
        if ema20 > ema50:
            alignment += 0.33
        if price > ema50:
            alignment += 0.34
        # Make it -1 to +1
        result.trend_alignment = (alignment * 2) - 1

        # === 3. Volume trend ===
        recent_vol = float(vol.tail(10).mean())
        avg_vol = float(vol.tail(50).mean())
        result.volume_trend = recent_vol / max(avg_vol, 1)

        # === 4. Volatility-adjusted return ===
        # ATR
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        if atr > 0:
            result.vol_adjusted_return = result.momentum_20 / (atr / max(price, 1e-10) * 100)

        # === 5. Outperformance vs benchmark ===
        if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 20:
            bench_close = benchmark_df["close"]
            bench_return = float((bench_close.iloc[-1] - bench_close.iloc[-20]) /
                                 max(bench_close.iloc[-20], 1e-10) * 100)
            result.vs_benchmark = result.momentum_20 - bench_return

        # === 6. Drawdown recovery ===
        # How quickly did price recover from recent low?
        recent_low = float(low.tail(50).min())
        recent_high = float(high.tail(50).max())
        if recent_high > recent_low:
            # Position within range (0 = at low, 1 = at high)
            result.drawdown_recovery = (price - recent_low) / (recent_high - recent_low)

        # === Composite score (0-100) ===
        result.score = self._compute_composite_score(result)

        # === Recommendation ===
        result.recommendation = self._recommend(result)

        return result

    def _compute_composite_score(self, s: StrengthScore) -> float:
        """Compute composite strength score (0-100)."""
        score = 50.0  # base

        # Momentum (30 points max)
        score += np.clip(s.momentum_5 * 2, -10, 10)
        score += np.clip(s.momentum_20 * 1.5, -10, 10)
        score += np.clip(s.momentum_50 * 1, -10, 10)

        # Trend alignment (20 points)
        score += s.trend_alignment * 10

        # Volume trend (15 points)
        if s.volume_trend > 1.5:
            score += 7
        elif s.volume_trend > 1.0:
            score += 3

        # Vol-adjusted return (15 points)
        score += np.clip(s.vol_adjusted_return * 5, -7.5, 7.5)

        # Outperformance (10 points)
        score += np.clip(s.vs_benchmark * 0.5, -5, 5)

        # Drawdown recovery (10 points)
        score += s.drawdown_recovery * 10

        return max(0, min(100, score))

    def _recommend(self, s: StrengthScore) -> str:
        """Generate recommendation based on score."""
        if s.score >= 80:
            return f"ELITE strength ({s.score:.0f}) — priority LONG candidate"
        elif s.score >= 65:
            return f"STRONG ({s.score:.0f}) — watchlist for longs"
        elif s.score >= 35:
            return f"AVERAGE ({s.score:.0f}) — no directional edge"
        elif s.score >= 20:
            return f"WEAK ({s.score:.0f}) — watchlist for shorts"
        else:
            return f"LAGGING ({s.score:.0f}) — priority SHORT candidate"

    # ------------------------------------------------------------------
    # Rank multiple symbols
    # ------------------------------------------------------------------
    def rank(self,
             dfs: Dict[str, pd.DataFrame],
             benchmark: Optional[str] = None) -> Dict[str, Any]:
        """Rank multiple symbols by relative strength.

        Args:
            dfs: {symbol: DataFrame}
            benchmark: override default benchmark

        Returns:
            Dict with ranking, scores, percentiles, top/bottom candidates
        """
        bench_symbol = benchmark or self.benchmark
        benchmark_df = dfs.get(bench_symbol)

        # Score each symbol
        scores: Dict[str, StrengthScore] = {}
        for symbol, df in dfs.items():
            if df is not None and not df.empty:
                scores[symbol] = self.score(symbol, df, benchmark_df)

        if not scores:
            return {"ranking": [], "scores": {}, "top_candidates": [], "bottom_candidates": []}

        # Sort by score descending
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1].score, reverse=True)
        ranking = [s for s, _ in sorted_symbols]

        # Compute percentiles
        n = len(sorted_symbols)
        for i, (symbol, s) in enumerate(sorted_symbols):
            s.percentile = 1.0 - (i / max(n - 1, 1))  # 1.0 = strongest
            # Category
            if s.percentile >= 1 - self.top_pct:
                s.category = StrengthCategory.ELITE
            elif s.percentile >= 0.70:
                s.category = StrengthCategory.STRONG
            elif s.percentile >= 0.30:
                s.category = StrengthCategory.AVERAGE
            elif s.percentile >= self.bottom_pct:
                s.category = StrengthCategory.WEAK
            else:
                s.category = StrengthCategory.LAGGING

        # Top and bottom candidates
        top_candidates = [s for s, sc in sorted_symbols if sc.category in
                         (StrengthCategory.ELITE, StrengthCategory.STRONG)]
        bottom_candidates = [s for s, sc in sorted_symbols if sc.category in
                            (StrengthCategory.WEAK, StrengthCategory.LAGGING)]

        return {
            "ranking": ranking,
            "scores": {s: sc.to_dict() for s, sc in scores.items()},
            "top_candidates": top_candidates,
            "bottom_candidates": bottom_candidates,
            "benchmark": bench_symbol,
            "strongest": ranking[0] if ranking else None,
            "weakest": ranking[-1] if ranking else None,
            "spread": (scores[ranking[0]].score - scores[ranking[-1]].score) if len(ranking) >= 2 else 0,
        }
