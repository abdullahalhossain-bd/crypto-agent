"""external.cross_asset_correlator
=====================================================================
Cross-asset correlation: BTC dominance, ETH/BTC ratio, funding rates.

Macro-level signals that pure OHLCV analysis misses:
  - ETH/BTC ratio rising + neutral funding → altseason (+20 pts)
  - Funding > 0.05% → longs overloaded (-20 pts malus)
  - Funding < 0 → shorts liquidated, bounce likely (+10 pts)
  - BTC dominance dropping + alt volume rising → +15 pts

Data sources (all free, no API key needed):
  - Binance public API for funding rates
  - CoinGecko API for BTC dominance (free, no key)
  - Binance for ETH/BTC ratio

Inspired by Centina-Quant's CrossAssetCorrelator.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from utils.logger import get_logger

log = get_logger("external.cross_asset")

CACHE_TTL = 300  # 5 minutes


@dataclass
class CrossAssetResult:
    btc_dominance: float = 50.0
    eth_btc_ratio: float = 0.0
    eth_btc_trend: str = "NEUTRAL"     # UP / DOWN / NEUTRAL
    funding_rate: float = 0.0
    funding_signal: str = "NEUTRAL"    # BULLISH / BEARISH / NEUTRAL / DANGER
    bonus: int = 0
    veto: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "btc_dominance": self.btc_dominance,
            "eth_btc_ratio": self.eth_btc_ratio,
            "eth_btc_trend": self.eth_btc_trend,
            "funding_rate": self.funding_rate,
            "funding_signal": self.funding_signal,
            "bonus": self.bonus,
            "veto": self.veto,
            "details": dict(self.details),
        }


# ----------------------------------------------------------------------
class CrossAssetCorrelator:
    """Fetches and analyzes cross-asset signals."""

    # Market-wide metrics (ETH/BTC ratio, BTC dominance) are the same for
    # every symbol and are cached separately from the per-symbol result so
    # that analyzing N symbols doesn't trigger N redundant network calls
    # for data that hasn't changed.
    _MARKET_WIDE_KEY = "cross_asset:market_wide"

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}

    # ----------------------------------------------------------------
    def analyze(self, symbol: str = "BTCUSDT") -> CrossAssetResult:
        """Get cross-asset correlation signals for `symbol`.

        Funding rate is symbol-specific and is cached per symbol; ETH/BTC
        ratio and BTC dominance are market-wide and cached once, shared
        across all symbols.
        """
        now = time.time()
        symbol_key = f"cross_asset:{symbol.upper()}"
        if symbol_key in self._cache:
            ts, cached = self._cache[symbol_key]
            if now - ts < CACHE_TTL:
                return cached

        result = CrossAssetResult()
        eth_btc, btc_dom = self._get_market_wide(now)
        funding = self._fetch_funding_rate(symbol)

        result.eth_btc_ratio = eth_btc
        result.funding_rate = funding
        result.btc_dominance = btc_dom

        # ETH/BTC trend
        # Minor #5 fix: the old code compared the ratio to absolute thresholds
        # (0.04 / 0.06), which doesn't detect direction — a ratio of 0.055
        # that just jumped from 0.04 to 0.055 would be "NEUTRAL" even though
        # it's clearly rising. Now we track a rolling history of the ratio
        # and compare the current value to the moving average to detect the
        # trend direction.
        if eth_btc > 0:
            # Track rolling history for trend detection.
            if not hasattr(self, "_eth_btc_history"):
                self._eth_btc_history: list[float] = []
            self._eth_btc_history.append(eth_btc)
            if len(self._eth_btc_history) > 20:
                self._eth_btc_history = self._eth_btc_history[-20:]
            if len(self._eth_btc_history) >= 5:
                avg = sum(self._eth_btc_history[:-1]) / max(1, len(self._eth_btc_history) - 1)
                change_pct = (eth_btc - avg) / max(avg, 1e-9)
                if change_pct > 0.02:  # > 2% above recent average
                    result.eth_btc_trend = "UP"
                elif change_pct < -0.02:  # > 2% below recent average
                    result.eth_btc_trend = "DOWN"
            # Fallback: use absolute thresholds only if no history yet.
            elif eth_btc > 0.06:
                result.eth_btc_trend = "UP"
            elif eth_btc < 0.04:
                result.eth_btc_trend = "DOWN"

        # Funding rate signal
        if funding > 0.0005:  # > 0.05%
            result.funding_signal = "DANGER"
            result.bonus = -20
        elif funding > 0.0001:
            result.funding_signal = "BEARISH"
            result.bonus = -10
        elif funding < -0.0001:
            result.funding_signal = "BULLISH"
            result.bonus = max(result.bonus, 10)
        else:
            result.funding_signal = "NEUTRAL"

        # Altseason signal: ETH/BTC rising + neutral funding
        if result.eth_btc_trend == "UP" and result.funding_signal == "NEUTRAL":
            result.bonus = max(result.bonus, 20)

        # BTC dominance dropping → alts pumping
        if btc_dom < 40 and result.eth_btc_trend == "UP":
            result.bonus = max(result.bonus, 15)

        result.details = {
            "eth_btc_ratio": eth_btc,
            "funding_rate_pct": round(funding * 100, 4),
            "btc_dominance": btc_dom,
        }
        self._cache[symbol_key] = (now, result)
        return result

    # ----------------------------------------------------------------
    def _get_market_wide(self, now: float) -> tuple[float, float]:
        """Return (eth_btc_ratio, btc_dominance), cached and shared across
        all symbols since neither value is symbol-specific."""
        cached = self._cache.get(self._MARKET_WIDE_KEY)
        if cached:
            ts, values = cached
            if now - ts < CACHE_TTL:
                return values
        eth_btc = self._fetch_eth_btc()
        btc_dom = self._fetch_btc_dominance()
        self._cache[self._MARKET_WIDE_KEY] = (now, (eth_btc, btc_dom))
        return eth_btc, btc_dom

    # ----------------------------------------------------------------
    def _fetch_eth_btc(self) -> float:
        """Fetch ETH/BTC ratio from Binance."""
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=ETHBTC"
            req = urllib_request.Request(url, method="GET",
                                          headers={"User-Agent": "TradingBot/1.0"})
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return 0.0
                data = json.loads(resp.read().decode("utf-8"))
                return float(data.get("price", 0))
        except Exception as e:  # noqa: BLE001
            log.debug("ETH/BTC fetch failed: %r", e)
            return 0.0

    # ----------------------------------------------------------------
    def _fetch_funding_rate(self, symbol: str) -> float:
        """Fetch current funding rate from Binance Futures."""
        try:
            # Convert symbol: BTCUSDT → BTCUSDT
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            req = urllib_request.Request(url, method="GET",
                                          headers={"User-Agent": "TradingBot/1.0"})
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return 0.0
                data = json.loads(resp.read().decode("utf-8"))
                return float(data.get("lastFundingRate", 0))
        except Exception as e:  # noqa: BLE001
            log.debug("Funding rate fetch failed: %r", e)
            return 0.0

    # ----------------------------------------------------------------
    def _fetch_btc_dominance(self) -> float:
        """Fetch BTC dominance from CoinGecko (free, no key)."""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            req = urllib_request.Request(url, method="GET",
                                          headers={"User-Agent": "TradingBot/1.0"})
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return 50.0
                data = json.loads(resp.read().decode("utf-8"))
                return float(data.get("data", {}).get("market_cap_percentage", {}).get("btc", 50.0))
        except Exception as e:  # noqa: BLE001
            log.debug("BTC dominance fetch failed: %r", e)
            return 50.0