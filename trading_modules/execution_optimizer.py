"""trading_modules/execution_optimizer.py
=====================================================================
Execution Optimizer (Principle #151, #155)
=====================================================================
Finds the best execution window: lowest spread, best liquidity, fastest
broker response. Also tracks execution quality to measure alpha leakage.

Why Execution Matters:
    Strategy × Execution = Result
    A perfect strategy with poor execution = loss.
    Slippage of 2 bps on every trade = 200 bps over 100 trades.

What It Optimizes:
    1. ORDER TIMING — when to place the order (wait for spread to narrow)
    2. ORDER TYPE — MARKET vs LIMIT vs ICEBERG
    3. ORDER SIZE — split large orders (TWAP/VWAP)
    4. BROKER ROUTING — fastest path to liquidity
    5. SLIPPAGE MINIMIZATION — limit orders at favorable levels

Execution Quality Tracking:
    - Expected price vs actual fill price (slippage)
    - Order latency (submission to fill)
    - Fill ratio (partial fills)
    - Spread at execution time

Usage:
    opt = ExecutionOptimizer()

    # Find best execution window:
    window = opt.find_best_window(symbol="BTCUSD", connector=conn)
    # window = {"score": 0.85, "wait_seconds": 12, "reason": "spread narrowing"}

    # Recommend order type:
    rec = opt.recommend_order(symbol="BTCUSD", side="BUY", volume=0.5,
                              urgency="normal", spread_bps=2.5)
    # rec = {"order_type": "LIMIT", "limit_offset_bps": 1.0, "time_in_force": "IOC"}

    # Record execution result:
    opt.record_execution(slippage_bps=1.2, latency_ms=85, fill_ratio=1.0)

    # Get quality score:
    quality = opt.execution_quality()
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("trading_bot.execution_optimizer")


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TWAP = "TWAP"
    VWAP = "VWAP"
    ICEBERG = "ICEBERG"


class Urgency(str, Enum):
    IMMEDIATE = "immediate"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    PATIENT = "patient"


@dataclass
class ExecutionWindow:
    """Best execution window assessment."""
    score: float = 0.0          # 0-1, higher = better window
    wait_seconds: float = 0.0   # how long to wait for optimal execution
    current_spread_bps: float = 0.0
    avg_spread_bps: float = 0.0
    liquidity_score: float = 0.0
    session_quality: float = 0.0
    recommendation: str = ""
    should_execute_now: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "wait_seconds": self.wait_seconds,
            "current_spread_bps": round(self.current_spread_bps, 2),
            "avg_spread_bps": round(self.avg_spread_bps, 2),
            "liquidity_score": round(self.liquidity_score, 3),
            "session_quality": round(self.session_quality, 3),
            "recommendation": self.recommendation,
            "should_execute_now": self.should_execute_now,
        }


@dataclass
class OrderRecommendation:
    """Order type + parameters recommendation."""
    order_type: OrderType = OrderType.MARKET
    limit_offset_bps: float = 0.0  # for LIMIT orders, offset from mid
    time_in_force: str = "IOC"      # GTC, IOC, FOK, DAY
    volume_split: int = 1           # split into N child orders
    split_interval_s: float = 0.0   # interval between splits
    max_slippage_bps: float = 5.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_type": self.order_type.value,
            "limit_offset_bps": self.limit_offset_bps,
            "time_in_force": self.time_in_force,
            "volume_split": self.volume_split,
            "split_interval_s": self.split_interval_s,
            "max_slippage_bps": self.max_slippage_bps,
            "reason": self.reason,
        }


@dataclass
class ExecutionRecord:
    """Record of a completed execution."""
    timestamp: float
    symbol: str
    side: str
    expected_price: float
    actual_price: float
    slippage_bps: float
    latency_ms: float
    fill_ratio: float
    spread_at_execution: float


class ExecutionOptimizer:
    """Finds the best execution window + tracks execution quality."""

    def __init__(self,
                 good_spread_bps: float = 3.0,
                 max_slippage_bps: float = 5.0,
                 max_latency_ms: float = 500.0,
                 history_size: int = 500):
        """Initialize optimizer.

        Args:
            good_spread_bps: spread below this = good liquidity
            max_slippage_bps: max acceptable slippage
            max_latency_ms: max acceptable order latency
            history_size: how many execution records to keep
        """
        self.good_spread = good_spread_bps
        self.max_slippage = max_slippage_bps
        self.max_latency = max_latency_ms
        self._lock = threading.RLock()
        self._history: Deque[ExecutionRecord] = deque(maxlen=history_size)
        self._spread_history: Dict[str, Deque[float]] = {}

    # ------------------------------------------------------------------
    # Find best execution window
    # ------------------------------------------------------------------
    def find_best_window(self,
                         symbol: str,
                         spread_bps: float,
                         session: str = "off_hours",
                         minutes_to_news: float = 999,
                         orderbook_depth_usd: float = 1_000_000) -> ExecutionWindow:
        """Assess current execution window quality.

        Args:
            symbol: trading symbol
            spread_bps: current bid-ask spread
            session: current trading session
            minutes_to_news: minutes until next high-impact news
            orderbook_depth_usd: depth at ±1%

        Returns:
            ExecutionWindow with score + recommendation
        """
        window = ExecutionWindow(
            current_spread_bps=spread_bps,
            avg_spread_bps=self._avg_spread(symbol, spread_bps),
        )

        # === Spread score (0-1) ===
        if spread_bps < self.good_spread:
            spread_score = 1.0
        elif spread_bps < self.good_spread * 2:
            spread_score = 0.7
        elif spread_bps < self.good_spread * 3:
            spread_score = 0.4
        else:
            spread_score = 0.2

        # === Liquidity score ===
        if orderbook_depth_usd > 5_000_000:
            liq_score = 1.0
        elif orderbook_depth_usd > 1_000_000:
            liq_score = 0.7
        elif orderbook_depth_usd > 250_000:
            liq_score = 0.4
        else:
            liq_score = 0.2

        # === Session score ===
        session_scores = {
            "london": 1.0, "new_york": 1.0, "overlap": 0.9,
            "asia": 0.5, "off_hours": 0.2,
        }
        sess_score = session_scores.get(session, 0.3)

        # === News penalty ===
        news_penalty = 1.0
        if minutes_to_news < 15:
            news_penalty = 0.1
        elif minutes_to_news < 30:
            news_penalty = 0.4
        elif minutes_to_news < 60:
            news_penalty = 0.7

        # === Composite ===
        window.liquidity_score = liq_score
        window.session_quality = sess_score
        window.score = (
            spread_score * 0.35 +
            liq_score * 0.30 +
            sess_score * 0.20 +
            news_penalty * 0.15
        )

        # === Wait recommendation ===
        if spread_bps > self.good_spread * 1.5 and spread_score < 0.7:
            window.wait_seconds = 30  # wait for spread to narrow
            window.recommendation = f"Wait {window.wait_seconds:.0f}s — spread {spread_bps:.1f}bps above average"
            window.should_execute_now = False
        elif minutes_to_news < 30:
            window.wait_seconds = (minutes_to_news + 5) * 60
            window.recommendation = f"Wait for news ({minutes_to_news:.0f}min)"
            window.should_execute_now = False
        elif window.score > 0.7:
            window.recommendation = "Execute now — optimal window"
            window.should_execute_now = True
        else:
            window.recommendation = "Marginal window — proceed with caution"
            window.should_execute_now = window.score > 0.5

        return window

    # ------------------------------------------------------------------
    # Recommend order type
    # ------------------------------------------------------------------
    def recommend_order(self,
                        symbol: str, side: str, volume: float,
                        urgency: Urgency = Urgency.NORMAL,
                        spread_bps: float = 5.0,
                        volume_threshold: float = 5.0) -> OrderRecommendation:
        """Recommend order type + parameters.

        Args:
            symbol: trading symbol
            side: "BUY" or "SELL"
            volume: order volume in lots
            urgency: how quickly must this fill?
            spread_bps: current spread
            volume_threshold: above this, split the order

        Returns:
            OrderRecommendation with type + params
        """
        rec = OrderRecommendation()

        # === Order type selection ===
        if urgency == Urgency.IMMEDIATE:
            rec.order_type = OrderType.MARKET
            rec.time_in_force = "IOC"
            rec.reason = "Immediate urgency — market order, accept slippage"
        elif urgency == Urgency.HIGH:
            rec.order_type = OrderType.LIMIT
            rec.limit_offset_bps = spread_bps / 2  # mid + half spread
            rec.time_in_force = "IOC"
            rec.reason = "High urgency — limit at mid, IOC if not filled"
        elif urgency == Urgency.NORMAL:
            if spread_bps > self.good_spread:
                rec.order_type = OrderType.LIMIT
                rec.limit_offset_bps = spread_bps / 4  # favorable side
                rec.time_in_force = "IOC"
                rec.reason = f"Normal urgency, wide spread ({spread_bps:.1f}bps) — limit at favorable price"
            else:
                rec.order_type = OrderType.MARKET
                rec.time_in_force = "IOC"
                rec.reason = "Normal urgency, tight spread — market order OK"
        elif urgency == Urgency.LOW:
            rec.order_type = OrderType.LIMIT
            rec.limit_offset_bps = spread_bps  # at favorable side of spread
            rec.time_in_force = "GTC"
            rec.reason = "Low urgency — limit order, wait for fill"
        else:  # PATIENT
            rec.order_type = OrderType.LIMIT
            rec.limit_offset_bps = spread_bps * 2  # beyond spread, hoping for fill
            rec.time_in_force = "GTC"
            rec.reason = "Patient — deep limit order, wait for favorable fill"

        # === Volume splitting ===
        if volume > volume_threshold:
            rec.order_type = OrderType.TWAP if urgency in (Urgency.NORMAL, Urgency.LOW) else rec.order_type
            rec.volume_split = max(2, int(volume / volume_threshold))
            rec.split_interval_s = 30 if urgency == Urgency.NORMAL else 60
            rec.reason += f" — split into {rec.volume_split} child orders (large volume)"

        # === Max slippage ===
        rec.max_slippage_bps = self.max_slippage

        return rec

    # ------------------------------------------------------------------
    # Record execution
    # ------------------------------------------------------------------
    def record_execution(self,
                         symbol: str, side: str,
                         expected_price: float, actual_price: float,
                         latency_ms: float, fill_ratio: float = 1.0,
                         spread_at_execution: float = 0.0) -> None:
        """Record an execution for quality tracking."""
        slippage = abs(actual_price - expected_price) / max(expected_price, 1e-10) * 10000
        record = ExecutionRecord(
            timestamp=time.time(),
            symbol=symbol, side=side,
            expected_price=expected_price, actual_price=actual_price,
            slippage_bps=slippage, latency_ms=latency_ms,
            fill_ratio=fill_ratio, spread_at_execution=spread_at_execution,
        )
        with self._lock:
            self._history.append(record)

        # Track spread
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=100)
        self._spread_history[symbol].append(spread_at_execution)

    # ------------------------------------------------------------------
    # Execution quality metrics
    # ------------------------------------------------------------------
    def execution_quality(self, last_n: int = 50) -> Dict[str, Any]:
        """Compute execution quality metrics."""
        with self._lock:
            recent = list(self._history)[-last_n:]

        if not recent:
            return {"status": "no_data"}

        slippages = [r.slippage_bps for r in recent]
        latencies = [r.latency_ms for r in recent]
        fills = [r.fill_ratio for r in recent]

        avg_slippage = sum(slippages) / len(slippages)
        avg_latency = sum(latencies) / len(latencies)
        avg_fill = sum(fills) / len(fills)
        max_slippage = max(slippages)
        max_latency = max(latencies)

        # Quality score (0-1)
        slip_score = max(0, 1 - avg_slippage / self.max_slippage)
        lat_score = max(0, 1 - avg_latency / self.max_latency)
        fill_score = avg_fill
        quality_score = (slip_score + lat_score + fill_score) / 3

        # Alpha leakage: how much are we losing to execution?
        # Assuming 1 lot per trade, slippage in bps
        alpha_leakage_bps = avg_slippage  # bps per trade

        return {
            "total_executions": len(recent),
            "avg_slippage_bps": round(avg_slippage, 2),
            "max_slippage_bps": round(max_slippage, 2),
            "avg_latency_ms": round(avg_latency, 0),
            "max_latency_ms": round(max_latency, 0),
            "avg_fill_ratio": round(avg_fill, 3),
            "quality_score": round(quality_score, 3),
            "slippage_score": round(slip_score, 3),
            "latency_score": round(lat_score, 3),
            "fill_score": round(fill_score, 3),
            "alpha_leakage_bps_per_trade": round(alpha_leakage_bps, 2),
            "estimated_daily_leakage_usd": round(alpha_leakage_bps * len(recent) / 10000 * 10000, 2),
            "status": "excellent" if quality_score > 0.8 else
                     "good" if quality_score > 0.6 else
                     "poor" if quality_score > 0.4 else "critical",
        }

    # ------------------------------------------------------------------
    # Helper: average spread
    # ------------------------------------------------------------------
    def _avg_spread(self, symbol: str, current: float) -> float:
        """Get average spread for a symbol."""
        history = self._spread_history.get(symbol)
        if not history:
            return current
        return sum(history) / len(history)
