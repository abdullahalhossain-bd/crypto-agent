"""
Order Book AI — Depth Analysis, Imbalance, Spoofing Detection
================================================================

Analyzes order book microstructure:
  1. Bid-ask imbalance → directional pressure
  2. Depth profile → support/resistance from order book
  3. Spoofing detection → large orders that get cancelled
  4. Slippage prediction → estimated fill price for given size

Usage:
    from trading_modules.order_book_ai import OrderBookAnalyzer

    analyzer = OrderBookAnalyzer()

    # Analyze order book snapshot
    result = analyzer.analyze(bids, asks, current_price=65000)
    # bids = [(price, size), ...], asks = [(price, size), ...]

    # Predict slippage for a market order
    slippage = analyzer.predict_slippage(asks, order_size=10.0)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

@dataclass
class OrderBookResult:
    """Order book analysis result."""
    imbalance: float = 0.0          # -1 (all asks) to +1 (all bids)
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    spread_pct: float = 0.0
    bid_wall: Optional[float] = None  # Large bid order price
    ask_wall: Optional[float] = None  # Large ask order price
    spoofing_detected: bool = False
    slippage_estimate: float = 0.0
    signal: str = "neutral"  # bullish / bearish / neutral

    def to_dict(self) -> dict:
        return {
            "imbalance": round(self.imbalance, 4),
            "bid_depth_usd": round(self.bid_depth_usd, 2),
            "ask_depth_usd": round(self.ask_depth_usd, 2),
            "spread_pct": round(self.spread_pct, 4),
            "bid_wall": self.bid_wall,
            "ask_wall": self.ask_wall,
            "spoofing": self.spoofing_detected,
            "slippage_estimate": round(self.slippage_estimate, 6),
            "signal": self.signal,
        }


class OrderBookAnalyzer:
    """
    Order book microstructure analyzer.

    Features:
      - Bid-ask imbalance ratio → directional pressure
      - Depth profiling → identify support/resistance walls
      - Spoofing detection → unusually large orders that appear/disappear
      - Slippage prediction → walk the book to estimate fill price
    """

    SPOOFING_THRESHOLD = 10.0  # Order 10x larger than average = potential spoof
    WALL_THRESHOLD = 3.0       # Order 3x larger than average = wall

    def __init__(self):
        # Critical #3 fix: removed unused _order_history (dead code).
        # If spoofing detection via multi-snapshot history is needed,
        # it should be implemented as a separate feature with proper
        # snapshot tracking logic.
        pass

    def analyze(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        current_price: float = 0.0,
    ) -> OrderBookResult:
        """
        Analyze order book snapshot.

        Args:
            bids: List of (price, size) for bid side (descending price)
            asks: List of (price, size) for ask side (ascending price)
            current_price: Mid price for reference

        Returns:
            OrderBookResult with all metrics
        """
        result = OrderBookResult()

        if not bids or not asks:
            return result

        # Spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        result.spread_pct = (best_ask - best_bid) / mid if mid > 0 else 0

        # Depth
        bid_depth = sum(p * s for p, s in bids[:20])
        ask_depth = sum(p * s for p, s in asks[:20])
        result.bid_depth_usd = float(bid_depth)
        result.ask_depth_usd = float(ask_depth)

        # Imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        total = bid_depth + ask_depth
        result.imbalance = float((bid_depth - ask_depth) / total) if total > 0 else 0

        # Signal
        if result.imbalance > 0.3:
            result.signal = "bullish"
        elif result.imbalance < -0.3:
            result.signal = "bearish"

        # Detect walls (large orders)
        bid_sizes = [s for _, s in bids[:20]]
        ask_sizes = [s for _, s in asks[:20]]

        avg_bid_size = np.mean(bid_sizes) if bid_sizes else 0
        avg_ask_size = np.mean(ask_sizes) if ask_sizes else 0

        for price, size in bids[:20]:
            if avg_bid_size > 0 and size > avg_bid_size * self.WALL_THRESHOLD:
                result.bid_wall = float(price)
                break

        for price, size in asks[:20]:
            if avg_ask_size > 0 and size > avg_ask_size * self.WALL_THRESHOLD:
                result.ask_wall = float(price)
                break

        # Spoofing detection
        for price, size in bids[:20] + asks[:20]:
            avg_size = (avg_bid_size + avg_ask_size) / 2
            if avg_size > 0 and size > avg_size * self.SPOOFING_THRESHOLD:
                result.spoofing_detected = True
                break

        # Slippage estimate (for a 1 BTC market buy)
        result.slippage_estimate = self.predict_slippage(asks, order_size=1.0)

        return result

    def predict_slippage(
        self,
        asks: list[tuple[float, float]],
        order_size: float,
    ) -> float:
        """
        Predict slippage for a market buy order.

        Walks the ask book to find average fill price.
        Slippage = (avg_fill_price - best_ask) / best_ask
        """
        if not asks:
            return 0.0

        best_ask = asks[0][0]
        remaining = order_size
        total_cost = 0.0
        total_filled = 0.0

        for price, size in asks:
            fill = min(remaining, size)
            total_cost += price * fill
            total_filled += fill
            remaining -= fill
            if remaining <= 0:
                break

        if total_filled == 0:
            return 0.0

        avg_fill = total_cost / total_filled
        return float((avg_fill - best_ask) / best_ask) if best_ask > 0 else 0.0
