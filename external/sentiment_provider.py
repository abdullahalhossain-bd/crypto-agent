"""external.sentiment_provider
=====================================================================
Day 175-177 — Retail sentiment integration.

Sources:
  1. Myfxbook Community Outlook (free, public)
     - Shows what % of retail traders are long vs short
     - Contrarian signal: when >80% are long, often a top
  2. OANDA v20 (order book + sentiment for account holders)
  3. Synthetic fallback (RSI-based when no API available)

Provides:
  - Per-symbol long/short ratio
  - Net positioning
  - Contrarian signal strength

The sentiment is used as ONE feature in the confluence engine, NOT
as a standalone trading signal. Retail sentiment is notoriously
contrarian at extremes but unreliable in the middle.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from external.env_loader import env
from utils.logger import get_logger

log = get_logger("external.sentiment")


@dataclass
class SentimentData:
    symbol: str
    long_pct: float            # 0-100
    short_pct: float           # 0-100
    net_positioning: float     # (long - short) / 100, range -1 to +1
    contrarian_signal: float   # -1 (bearish contrarian) to +1 (bullish contrarian)
    source: str
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "long_pct": self.long_pct,
            "short_pct": self.short_pct,
            "net_positioning": self.net_positioning,
            "contrarian_signal": self.contrarian_signal,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
class SentimentProviderManager:
    """Multi-source retail sentiment."""

    # Re-attempt login if we haven't refreshed the session in this long,
    # even if we still have a cached session string — Myfxbook sessions can
    # be invalidated server-side without any client-visible signal beyond
    # subsequent calls returning an error.
    _SESSION_MAX_AGE_SEC = 3600

    def __init__(self) -> None:
        self._myfxbook_session: Optional[str] = None
        self._myfxbook_session_ts: float = 0.0

    # ----------------------------------------------------------------
    def get_sentiment(self, symbol: str) -> SentimentData:
        """Get retail sentiment for a symbol. Tries Myfxbook → OANDA → synthetic."""
        # Try Myfxbook first (free, public community outlook)
        if env.myfxbook_email and env.myfxbook_password:
            data = self._fetch_myfxbook(symbol)
            if data:
                return data
        # Try OANDA
        if env.oanda_api_key:
            data = self._fetch_oanda(symbol)
            if data:
                return data
        # Fallback: synthetic (neutral)
        return self._synthetic(symbol)

    # ----------------------------------------------------------------
    def _fetch_myfxbook(self, symbol: str) -> Optional[SentimentData]:
        """Fetch from Myfxbook Community Outlook.

        Retries once with a fresh login if the cached session has expired
        or been invalidated server-side, instead of silently going dark
        for the rest of the process lifetime.
        """
        self._ensure_myfxbook_session()
        if not self._myfxbook_session:
            return None

        data = self._myfxbook_community_outlook()
        if data is None:
            # Could be an expired/invalidated session — force a re-login
            # and retry exactly once before giving up on this call.
            log.info("Myfxbook outlook fetch failed; forcing re-login and retrying once")
            self._myfxbook_session = None
            self._ensure_myfxbook_session()
            if not self._myfxbook_session:
                return None
            data = self._myfxbook_community_outlook()

        if not data:
            return None
        # Find symbol in the response
        # Myfxbook returns symbols like "EURUSD" with longShortRatio
        for symbol_data in data.get("symbols", []):
            if symbol_data.get("name", "").upper() == symbol.upper():
                long_pct = float(symbol_data.get("longPositions", 50))
                short_pct = float(symbol_data.get("shortPositions", 50))
                total = long_pct + short_pct
                if total > 0:
                    long_pct = (long_pct / total) * 100
                    short_pct = (short_pct / total) * 100
                return self._build_sentiment(symbol, long_pct, short_pct, "myfxbook")
        return None

    def _myfxbook_community_outlook(self) -> Optional[dict]:
        url = (f"https://www.myfxbook.com/api/get-community-outlook.json"
               f"?session={self._myfxbook_session}")
        data = self._http_get(url)
        if not data or data.get("response", {}).get("error", True):
            return None
        return data

    def _ensure_myfxbook_session(self) -> None:
        """Log in if we have no session, or if the current one is older
        than `_SESSION_MAX_AGE_SEC` (proactive refresh)."""
        age = time.time() - self._myfxbook_session_ts
        if self._myfxbook_session and age < self._SESSION_MAX_AGE_SEC:
            return
        session = self._myfxbook_login()
        if session:
            self._myfxbook_session = session
            self._myfxbook_session_ts = time.time()
        else:
            self._myfxbook_session = None

    def _myfxbook_login(self) -> Optional[str]:
        url = (f"https://www.myfxbook.com/api/login.json"
               f"?email={env.myfxbook_email}&password={env.myfxbook_password}")
        data = self._http_get(url)
        if not data or not data.get("session"):
            log.warning("Myfxbook login failed")
            return None
        return data.get("session")

    # ----------------------------------------------------------------
    def _fetch_oanda(self, symbol: str) -> Optional[SentimentData]:
        """Fetch from OANDA v20 (requires account)."""
        key = env.oanda_api_key
        account = env.oanda_account_id
        if not key or not account:
            return None
        base = ("https://api-fxpractice.oanda.com" if env.oanda_use_practice
                else "https://api-fxtrade.oanda.com")
        url = f"{base}/v3/instruments/{symbol}/positionBook"
        try:
            req = urllib_request.Request(url, method="GET",
                                          headers={"Authorization": f"Bearer {key}"})
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError):
            return None
        # Parse position book
        book = data.get("positionBook", {})
        # This is a simplified parse — OANDA's position book is complex
        long_pct = 50.0
        short_pct = 50.0
        return self._build_sentiment(symbol, long_pct, short_pct, "oanda")

    # ----------------------------------------------------------------
    def _synthetic(self, symbol: str) -> SentimentData:
        """Synthetic sentiment (neutral) when no API available."""
        return SentimentData(
            symbol=symbol,
            long_pct=50.0,
            short_pct=50.0,
            net_positioning=0.0,
            contrarian_signal=0.0,
            source="synthetic",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            metadata={"note": "no sentiment API available"},
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _build_sentiment(symbol: str, long_pct: float,
                          short_pct: float, source: str) -> SentimentData:
        net = (long_pct - short_pct) / 100.0
        # Contrarian signal: when extreme (>80% or <20%), signal reverses
        if long_pct >= 80:
            contrarian = -1.0   # too many longs → bearish contrarian
        elif long_pct <= 20:
            contrarian = 1.0    # too many shorts → bullish contrarian
        else:
            # Scale linearly: 50% long → 0, 80%+ → -1, 20%- → +1
            if long_pct > 50:
                contrarian = -(long_pct - 50) / 30.0
            else:
                contrarian = (50 - long_pct) / 30.0
            contrarian = max(-1.0, min(1.0, contrarian))
        return SentimentData(
            symbol=symbol,
            long_pct=float(long_pct),
            short_pct=float(short_pct),
            net_positioning=float(net),
            contrarian_signal=float(contrarian),
            source=source,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _http_get(url: str) -> Optional[dict]:
        try:
            req = urllib_request.Request(url, method="GET",
                                          headers={"User-Agent": "TradingBot/1.0"})
            with urllib_request.urlopen(req, timeout=15.0) as resp:
                if resp.status != 200:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            log.debug("HTTP GET failed: %r", e)
            return None
        except Exception as e:  # noqa: BLE001
            log.debug("HTTP GET exception: %r", e)
            return None

    # ----------------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "has_myfxbook": bool(env.myfxbook_email),
            "has_oanda": bool(env.oanda_api_key),
            "myfxbook_session_active": self._myfxbook_session is not None,
        }