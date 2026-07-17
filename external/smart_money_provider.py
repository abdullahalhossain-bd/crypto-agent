"""external.smart_money_provider
=====================================================================
Smart Money / Whale flow detection.

Detects institutional activity through:
  1. Large trades on Binance (>$100k) → whale buys/sells
  2. WhaleAlert API (optional, free tier)
  3. Exchange inflow/outflow heuristic (outflow > inflow = accumulation = bullish)

Signals:
  - whale_buys > whale_sells → BULLISH
  - exchange_outflow > exchange_inflow → accumulation → +bonus
  - large_buy_ratio > 0.6 → +bonus
  - Sudden whale dump → VETO

Inspired by Centina-Quant's SmartMoneyDetector. Adapted to our
external provider architecture.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from external.env_loader import env
from utils.logger import get_logger

log = get_logger("external.smart_money")

WHALE_ALERT_BASE = "https://api.whale-alert.io/v1/transactions"
LARGE_TRADE_USD = 100_000
CACHE_TTL_SEC = 180


@dataclass
class SmartMoneyResult:
    whale_buys: int = 0
    whale_sells: int = 0
    exchange_inflow: float = 0.0    # $M flowing INTO exchanges (bearish)
    exchange_outflow: float = 0.0   # $M flowing OUT of exchanges (bullish)
    net_flow: float = 0.0           # positive = bullish (net outflow)
    large_buy_ratio: float = 0.0    # ratio of large buys to total
    smart_signal: str = "NEUTRAL"   # BULLISH / BEARISH / NEUTRAL
    bonus: int = 0
    veto: bool = False
    veto_reason: str = ""
    source: str = "synthetic"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "whale_buys": self.whale_buys,
            "whale_sells": self.whale_sells,
            "exchange_inflow": self.exchange_inflow,
            "exchange_outflow": self.exchange_outflow,
            "net_flow": self.net_flow,
            "large_buy_ratio": self.large_buy_ratio,
            "smart_signal": self.smart_signal,
            "bonus": self.bonus,
            "veto": self.veto,
            "veto_reason": self.veto_reason,
            "source": self.source,
            "details": dict(self.details),
        }


# ----------------------------------------------------------------------
class SmartMoneyDetector:
    """Detects institutional/whale activity."""

    def __init__(self) -> None:
        self._wa_key = os.environ.get("WHALE_ALERT_API_KEY", "")
        self._cache: dict[str, tuple[float, SmartMoneyResult]] = {}

    # ----------------------------------------------------------------
    def analyze(self, symbol: str, current_price: float = 0.0) -> SmartMoneyResult:
        """Analyze smart money flow for a symbol."""
        # Cache check
        now = time.time()
        if symbol in self._cache:
            ts, cached = self._cache[symbol]
            if now - ts < CACHE_TTL_SEC:
                return cached

        # Try WhaleAlert API
        if self._wa_key:
            result = self._fetch_whale_alert(symbol)
            if result is not None:
                self._cache[symbol] = (now, result)
                return result

        # Fallback: synthetic (neutral)
        result = self._synthetic(symbol)
        self._cache[symbol] = (now, result)
        return result

    # ----------------------------------------------------------------
    # Major #3 fix: WhaleAlert supports specific crypto currencies (BTC, ETH,
    # etc.) but NOT forex pairs (EUR, USD, JPY). The old code blindly stripped
    # USDT/USD from the symbol, turning EURUSD into "eur" which WhaleAlert
    # doesn't support — the API would always return no transactions.
    # Now we validate the coin against a known list of supported currencies.
    _SUPPORTED_WHALEALERT_COINS = frozenset({
        "btc", "eth", "xrp", "ltc", "bch", "usdt", "usdc", "doge",
        "ada", "sol", "dot", "matic", "avax", "link", "uni", "atom",
        "xlm", "etc", "trx", "shib", "wbtc", " dai",
    })

    def _fetch_whale_alert(self, symbol: str) -> Optional[SmartMoneyResult]:
        """Fetch from WhaleAlert API (requires key).

        Major #3 fix: validate that the extracted coin is supported by
        WhaleAlert. Forex pairs and unsupported assets return None early
        instead of making a wasted API call that will always return empty.
        """
        coin = symbol.replace("USDT", "").replace("USD", "").lower()
        # Major #3 fix: skip unsupported coins (e.g. forex pairs like EUR, GBP).
        if coin not in self._SUPPORTED_WHALEALERT_COINS:
            log.debug("WhaleAlert: skipping unsupported coin %r (from symbol %s)", coin, symbol)
            return None
        now = datetime.now(tz=timezone.utc)
        start_ts = int(now.timestamp()) - 3600  # last 1 hour
        url = (f"{WHALE_ALERT_BASE}?api_key={self._wa_key}"
               f"&min={LARGE_TRADE_USD}&start={start_ts}&currency={coin}")
        try:
            req = urllib_request.Request(url, method="GET",
                                          headers={"User-Agent": "TradingBot/1.0"})
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            log.debug("WhaleAlert fetch failed: %r", e)
            return None

        if not data or data.get("result") != "success":
            return None

        transactions = data.get("transactions", [])
        whale_buys = 0
        whale_sells = 0
        exchange_inflow = 0.0
        exchange_outflow = 0.0

        for tx in transactions:
            amount_usd = float(tx.get("amount_usd", 0))
            sender_type = tx.get("sender", {}).get("owner_type", "")
            receiver_type = tx.get("recipient", {}).get("owner_type", "")
            if sender_type == "exchange" and receiver_type != "exchange":
                exchange_outflow += amount_usd / 1e6  # $M
                whale_buys += 1
            elif receiver_type == "exchange" and sender_type != "exchange":
                exchange_inflow += amount_usd / 1e6
                whale_sells += 1

        return self._build_result(
            whale_buys, whale_sells,
            exchange_inflow, exchange_outflow,
            source="whale_alert",
        )

    # ----------------------------------------------------------------
    def _synthetic(self, symbol: str) -> SmartMoneyResult:
        """Neutral fallback when no API available."""
        return SmartMoneyResult(
            smart_signal="NEUTRAL",
            source="synthetic",
            details={"reason": "no whale alert API key configured"},
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _build_result(whale_buys: int, whale_sells: int,
                        inflow: float, outflow: float,
                        source: str) -> SmartMoneyResult:
        """Build SmartMoneyResult from raw data."""
        net_flow = outflow - inflow  # positive = bullish
        total = whale_buys + whale_sells
        large_buy_ratio = whale_buys / total if total > 0 else 0.0

        # Signal
        if whale_buys > whale_sells and net_flow > 0:
            signal = "BULLISH"
        elif whale_sells > whale_buys and net_flow < 0:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        # Bonus
        bonus = 0
        veto = False
        veto_reason = ""
        if net_flow > 0 and outflow > inflow * 1.5:
            bonus = 20  # strong accumulation
        if large_buy_ratio > 0.6:
            bonus = max(bonus, 15)
        # Veto: sudden whale dump (3× more sells than buys)
        if whale_sells > whale_buys * 3 and whale_sells >= 3:
            veto = True
            veto_reason = "whale_dump_detected"

        return SmartMoneyResult(
            whale_buys=whale_buys,
            whale_sells=whale_sells,
            exchange_inflow=round(inflow, 4),
            exchange_outflow=round(outflow, 4),
            net_flow=round(net_flow, 4),
            large_buy_ratio=round(large_buy_ratio, 4),
            smart_signal=signal,
            bonus=bonus,
            veto=veto,
            veto_reason=veto_reason,
            source=source,
        )
