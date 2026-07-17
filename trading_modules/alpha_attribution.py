"""
Alpha Attribution — where did profit come from?
=================================================

When a trade is profitable, it's critical to know WHY. Different sources
of alpha have different durability:

    - Trend alpha        — durable, lasts months/years
    - Momentum alpha     — semi-durable, weeks/months
    - Mean-reversion     — context-dependent
    - News alpha         — short-lived (minutes/hours)
    - Liquidity alpha    — durable if you have flow info
    - Volatility alpha   — regime-dependent
    - Carry alpha        — slow, durable

This module attributes each closed trade's PnL to one or more alpha
sources, using the market conditions at entry + exit time.

Usage:
    from trading_modules.alpha_attribution import AlphaAttribution
    attr = AlphaAttribution()
    result = attr.attribute(
        trade={"entry_price": 65000, "exit_price": 66000, "pnl": 100,
               "entry_time": ..., "exit_time": ..., "direction": "BUY"},
        market_data_at_entry=df_at_entry,
        market_data_at_exit=df_at_exit,
    )
    print(result.attributions)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    trade_pnl: float
    attributions: dict[str, float] = field(default_factory=dict)
    # {"trend": 40, "momentum": 30, "mean_reversion": -10, "news": 20, ...}
    dominant_source: str = "unknown"
    confidence: float = 0.0
    durability: str = "unknown"          # "durable" / "semi-durable" / "short-lived"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "trade_pnl": round(self.trade_pnl, 2),
            "attributions": {k: round(v, 2) for k, v in self.attributions.items()},
            "dominant_source": self.dominant_source,
            "confidence": round(self.confidence, 3),
            "durability": self.durability,
            "notes": self.notes,
        }


class AlphaAttribution:
    """Attribute a closed trade's PnL to alpha sources.

    Parameters:
        trend_lookback: bars for trend measurement (default 50)
        momentum_lookback: bars for momentum (default 20)
        mean_reversion_lookback: bars for MR (default 20)
        news_window_minutes: news effect window (default 60)
    """

    DURABILITY: dict[str, str] = {
        "trend": "durable",
        "momentum": "semi-durable",
        "mean_reversion": "context-dependent",
        "news": "short-lived",
        "liquidity": "durable",
        "volatility": "regime-dependent",
        "carry": "durable",
        "noise": "none",
    }

    def __init__(
        self, trend_lookback: int = 50, momentum_lookback: int = 20,
        mean_reversion_lookback: int = 20, news_window_minutes: int = 60,
    ) -> None:
        self.trend_lookback = trend_lookback
        self.momentum_lookback = momentum_lookback
        self.mean_reversion_lookback = mean_reversion_lookback
        self.news_window_minutes = news_window_minutes

    def attribute(
        self,
        trade: dict,
        market_data_at_entry: Optional[pd.DataFrame] = None,
        market_data_at_exit: Optional[pd.DataFrame] = None,
        had_news_event: bool = False,
    ) -> AttributionResult:
        """Attribute trade PnL to alpha sources.

        Args:
            trade: dict with at least pnl, entry_price, exit_price, direction
            market_data_at_entry: OHLCV df up to entry (for trend/momentum/MR)
            market_data_at_exit: OHLCV df up to exit
            had_news_event: was there a news event during the trade?
        """
        pnl = float(trade.get("pnl", 0))
        direction = trade.get("direction", "BUY").upper()

        attributions: dict[str, float] = {
            "trend": 0.0, "momentum": 0.0, "mean_reversion": 0.0,
            "news": 0.0, "liquidity": 0.0, "volatility": 0.0,
            "carry": 0.0, "noise": 0.0,
        }

        if abs(pnl) < 1e-10:
            attributions["noise"] = 0.0
            return AttributionResult(
                trade_pnl=pnl, attributions=attributions,
                dominant_source="noise", confidence=1.0, durability="none",
                notes=["zero PnL"],
            )

        # ── Trend alpha ───────────────────────────────────────────
        # How much of the move was in the direction of the prevailing trend?
        if market_data_at_entry is not None and len(market_data_at_entry) > self.trend_lookback:
            trend = self._trend_strength(market_data_at_entry, self.trend_lookback)
            # If trade direction aligned with trend, attribute some PnL to trend
            if direction == "BUY" and trend > 0:
                attributions["trend"] = pnl * 0.4 * abs(trend)
            elif direction == "SELL" and trend < 0:
                attributions["trend"] = pnl * 0.4 * abs(trend)
            else:
                attributions["trend"] = 0.0
        # ── Momentum alpha ────────────────────────────────────────
        if market_data_at_entry is not None and len(market_data_at_entry) > self.momentum_lookback:
            mom = self._momentum_strength(market_data_at_entry, self.momentum_lookback)
            if direction == "BUY" and mom > 0:
                attributions["momentum"] = pnl * 0.3 * abs(mom)
            elif direction == "SELL" and mom < 0:
                attributions["momentum"] = pnl * 0.3 * abs(mom)
            else:
                attributions["momentum"] = 0.0

        # ── Mean reversion alpha ──────────────────────────────────
        # If price was extended (z-score > 2) at entry and reverted
        if market_data_at_entry is not None and len(market_data_at_entry) > self.mean_reversion_lookback:
            z = self._zscore(market_data_at_entry["close"], self.mean_reversion_lookback)
            # MR trade: BUY when z < -2 (oversold), SELL when z > 2 (overbought)
            if direction == "BUY" and z < -1.5:
                attributions["mean_reversion"] = pnl * 0.3 * abs(z) / 3
            elif direction == "SELL" and z > 1.5:
                attributions["mean_reversion"] = pnl * 0.3 * abs(z) / 3
            else:
                attributions["mean_reversion"] = 0.0

        # ── News alpha ────────────────────────────────────────────
        if had_news_event:
            # News typically explains 20-40% of short-term moves
            attributions["news"] = pnl * 0.3
        # ── Liquidity alpha ───────────────────────────────────────
        # If volume was unusually low at entry (you front-ran the crowd)
        if market_data_at_entry is not None and "volume" in market_data_at_entry.columns:
            vols = market_data_at_entry["volume"].tail(20)
            vol_now = float(vols.iloc[-1])
            vol_avg = float(vols.iloc[:-1].mean()) if len(vols) > 1 else vol_now
            if vol_avg > 0 and vol_now < vol_avg * 0.7:
                attributions["liquidity"] = pnl * 0.15

        # ── Volatility alpha ──────────────────────────────────────
        # If volatility was expanding and you caught the breakout
        if market_data_at_entry is not None and len(market_data_at_entry) > 30:
            atr = self._atr(market_data_at_entry, 14)
            atr_now = float(atr.iloc[-1])
            atr_baseline = float(atr.rolling(30).mean().iloc[-1]) if len(atr) > 30 else atr_now
            if atr_baseline > 0 and atr_now > atr_baseline * 1.3:
                attributions["volatility"] = pnl * 0.2

        # ── Carry alpha ───────────────────────────────────────────
        # Long position held > 1 day = carry (funding rate in crypto)
        hold_minutes = trade.get("hold_minutes", 0)
        if hold_minutes > 1440:  # > 1 day
            attributions["carry"] = pnl * 0.1

        # ── Noise (residual) ──────────────────────────────────────
        explained = sum(attributions.values())
        attributions["noise"] = pnl - explained

        # Normalize to sum to pnl
        total = sum(attributions.values())
        if abs(total - pnl) > 0.01 and abs(total) > 0:
            scale = pnl / total
            attributions = {k: v * scale for k, v in attributions.items()}

        # ── Dominant source ───────────────────────────────────────
        if abs(pnl) > 0:
            dominant = max(attributions, key=lambda k: abs(attributions[k]) if attributions[k] != 0 else -1)
            confidence = abs(attributions[dominant]) / abs(pnl) if abs(pnl) > 0 else 0
        else:
            dominant = "noise"
            confidence = 0
        durability = self.DURABILITY.get(dominant, "unknown")

        notes = [
            f"dominant: {dominant} ({attributions[dominant]:+.2f} of {pnl:+.2f})",
            f"durability: {durability}",
        ]
        for k, v in attributions.items():
            if abs(v) > abs(pnl) * 0.05:  # only show significant attributions
                notes.append(f"  {k}: {v:+.2f}")

        return AttributionResult(
            trade_pnl=pnl, attributions=attributions,
            dominant_source=dominant, confidence=float(confidence),
            durability=durability, notes=notes,
        )

    @staticmethod
    def _trend_strength(df: pd.DataFrame, lookback: int) -> float:
        """Return -1..+1 trend strength."""
        closes = df["close"].tail(lookback)
        if len(closes) < 2:
            return 0
        # Linear regression slope
        x = np.arange(len(closes))
        try:
            slope = float(np.polyfit(x, closes, 1)[0])
            # Normalize by ATR
            atr = float((df["high"] - df["low"]).tail(lookback).mean())
            if atr > 0:
                return max(-1.0, min(1.0, slope * 10 / atr))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _momentum_strength(df: pd.DataFrame, lookback: int) -> float:
        """Return -1..+1 momentum (RSI-based)."""
        closes = df["close"].tail(lookback + 1)
        if len(closes) < 2:
            return 0
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).fillna(50).iloc[-1]
        return (rsi - 50) / 50  # -1..+1

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> float:
        if len(series) < window:
            return 0
        recent = series.tail(window)
        mu = float(recent.mean())
        sd = float(recent.std(ddof=0))
        if sd <= 0:
            return 0
        return float((float(series.iloc[-1]) - mu) / sd)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()


__all__ = ["AlphaAttribution", "AttributionResult"]
