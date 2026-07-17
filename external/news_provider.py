"""external.news_provider
=====================================================================
Day 171-174 — News + economic calendar integration.

Sources:
  1. FRED (Federal Reserve Economic Data) — FREE, unlimited
     CPI, Unemployment, Treasury Yields, GDP, FOMC rates
  2. NewsAPI.org — 100 req/day free, financial news + sentiment
  3. ForexFactory (web scrape) — economic calendar, no key needed

Provides:
  - Upcoming economic events (for news blackout in trade_quality)
  - Historical economic data (CPI etc. for macro features)
  - Financial news headlines with simple sentiment scoring

This integrates with:
  - enhancements/news_calendar.py (feeds events into the calendar)
  - engine/candlestick/trade_quality.py (news blackout enforcement)
  - ml/feature_store.py (macro features as additional ML inputs)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from external.env_loader import env
from utils.logger import get_logger

log = get_logger("external.news")


class NewsImpact(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class EconomicEvent:
    event_id: str
    timestamp: str
    symbol: str           # affected currency/symbol
    title: str
    impact: NewsImpact
    category: str = ""
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "title": self.title,
            "impact": self.impact.value,
            "category": self.category,
            "forecast": self.forecast,
            "previous": self.previous,
            "actual": self.actual,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass
class NewsArticle:
    title: str
    description: str
    url: str
    published_at: str
    source: str
    sentiment_score: float = 0.0    # -1 to +1
    sentiment_label: str = "neutral"
    relevant_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "url": self.url,
            "published_at": self.published_at,
            "source": self.source,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "relevant_symbols": list(self.relevant_symbols),
        }


# ----------------------------------------------------------------------
class NewsProviderManager:
    """Multi-source news + economic calendar."""

    def __init__(self) -> None:
        self._fred_cache: dict[str, Any] = {}

    # ----------------------------------------------------------------
    # FRED — Federal Reserve Economic Data
    # ----------------------------------------------------------------
    def fred_series(self, series_id: str) -> Optional[list[dict[str, Any]]]:
        """Fetch a FRED time series (e.g. 'CPIAUCSL' = CPI)."""
        key = env.fred_api_key
        if not key:
            log.debug("FRED key not configured")
            return None
        # Check cache (1 hour TTL)
        cache_key = f"fred_{series_id}"
        cached = self._fred_cache.get(cache_key)
        if cached and cached["ts"] > datetime.now(tz=timezone.utc).timestamp() - 3600:
            return cached["data"]
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={series_id}&api_key={key}&file_type=json"
               f"&limit=100&sort_order=desc")
        data = self._http_get(url)
        if not data or "observations" not in data:
            return None
        obs = data["observations"]
        result = [{"date": o["date"], "value": float(o["value"])}
                  for o in obs if o["value"] != "."]
        self._fred_cache[cache_key] = {
            "ts": datetime.now(tz=timezone.utc).timestamp(),
            "data": result,
        }
        return result

    def fred_macro_features(self) -> dict[str, Optional[float]]:
        """Fetch current values of key macro indicators."""
        series = {
            "CPIAUCSL": "cpi",            # Consumer Price Index
            "UNRATE": "unemployment",       # Unemployment Rate
            "DGS10": "treasury_10y",       # 10-Year Treasury
            "DGS2": "treasury_2y",         # 2-Year Treasury
            "FEDFUNDS": "fed_funds_rate",  # Federal Funds Rate
            "GDP": "gdp",                  # GDP
            "VIXCLS": "vix",               # VIX
        }
        out: dict[str, Optional[float]] = {}
        for series_id, name in series.items():
            data = self.fred_series(series_id)
            if data and len(data) > 0:
                out[name] = float(data[0]["value"])
            else:
                out[name] = None
        return out

    # ----------------------------------------------------------------
    # NewsAPI.org — financial news
    # ----------------------------------------------------------------
    def newsapi_headlines(self, query: str = "crypto OR bitcoin OR forex",
                            n: int = 10) -> list[NewsArticle]:
        key = env.newsapi_api_key
        if not key:
            log.debug("NewsAPI key not configured")
            return []
        # NewsAPI free tier: 1-day delay, dev license
        from_date = (datetime.now(tz=timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        url = (f"https://newsapi.org/v2/everything"
               f"?q={query}&from={from_date}&sortBy=publishedAt"
               f"&pageSize={n}&apiKey={key}&language=en")
        data = self._http_get(url)
        if not data or data.get("status") != "ok":
            return []
        articles = []
        for a in data.get("articles", [])[:n]:
            article = NewsArticle(
                title=a.get("title", ""),
                description=a.get("description", "") or "",
                url=a.get("url", ""),
                published_at=a.get("publishedAt", ""),
                source=a.get("source", {}).get("name", ""),
            )
            article.sentiment_score = self._simple_sentiment(article.title + " " + article.description)
            article.sentiment_label = self._sentiment_label(article.sentiment_score)
            article.relevant_symbols = self._extract_symbols(article.title + " " + article.description)
            articles.append(article)
        return articles

    # ----------------------------------------------------------------
    # Economic Calendar (synthetic when no API available)
    # ----------------------------------------------------------------
    def upcoming_events(self, hours_ahead: int = 24) -> list[EconomicEvent]:
        """Get upcoming high-impact economic events."""
        events: list[EconomicEvent] = []
        # Try TradingEconomics API if key available
        if env.tradingeconomics_api_key:
            events = self._fetch_tradingeconomics(hours_ahead)
        # Try Tradermade if available
        if not events and env.tradermade_api_key:
            events = self._fetch_tradermade(hours_ahead)
        # If no API, return empty (trade_quality will not enforce blackout)
        return events

    def _fetch_tradingeconomics(self, hours_ahead: int) -> list[EconomicEvent]:
        key = env.tradingeconomics_api_key
        url = (f"https://api.tradingeconomics.com/calendar"
               f"?c={key}&f=json")
        data = self._http_get(url)
        if not data or not isinstance(data, list):
            return []
        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        events = []
        for e in data[:50]:
            try:
                ts = datetime.fromisoformat(e.get("date", "").replace("Z", "+00:00"))
                if ts < now or ts > cutoff:
                    continue
                impact_str = e.get("importance", "").lower()
                impact = (NewsImpact.HIGH if impact_str == "high"
                          else NewsImpact.MEDIUM if impact_str == "medium"
                          else NewsImpact.LOW)
                events.append(EconomicEvent(
                    event_id=f"te_{e.get('eventId', '')}",
                    timestamp=ts.isoformat(),
                    symbol=e.get("country", ""),
                    title=e.get("event", ""),
                    impact=impact,
                    category=e.get("category", ""),
                    forecast=str(e.get("forecast", "")),
                    previous=str(e.get("previous", "")),
                    actual=str(e.get("actual", "")),
                    source="tradingeconomics",
                ))
            except Exception:  # noqa: BLE001
                continue
        return events

    def _fetch_tradermade(self, hours_ahead: int) -> list[EconomicEvent]:
        key = env.tradermade_api_key
        url = (f"https://marketdata.tradermade.com/api/v1/calendar"
               f"?api_key={key}&currency=USD,EUR,GBP,JPY")
        data = self._http_get(url)
        if not data:
            return []
        # Tradermade returns different format
        events = []
        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        for e in data.get("data", [])[:50]:
            try:
                ts = datetime.fromisoformat(e.get("date", "").replace("Z", "+00:00"))
                if ts < now or ts > cutoff:
                    continue
                impact_str = e.get("impact", "").lower()
                impact = (NewsImpact.HIGH if impact_str == "high"
                          else NewsImpact.MEDIUM if impact_str == "medium"
                          else NewsImpact.LOW)
                events.append(EconomicEvent(
                    event_id=f"tm_{e.get('event_id', '')}",
                    timestamp=ts.isoformat(),
                    symbol=e.get("currency", ""),
                    title=e.get("event", ""),
                    impact=impact,
                    category=e.get("category", ""),
                    forecast=str(e.get("forecast", "")),
                    previous=str(e.get("previous", "")),
                    actual=str(e.get("actual", "")),
                    source="tradermade",
                ))
            except Exception:  # noqa: BLE001
                continue
        return events

    # ----------------------------------------------------------------
    # Sentiment scoring (simple keyword-based)
    # ----------------------------------------------------------------
    POSITIVE_KEYWORDS = {
        "bullish", "surge", "rally", "gain", "rise", "soar", "breakout",
        "uptrend", "support", "buy", "long", "optimistic", "growth",
        "rally", "boost", "jump", "climb", "recovery",
    }
    NEGATIVE_KEYWORDS = {
        "bearish", "crash", "plunge", "drop", "fall", "decline", "breakdown",
        "downtrend", "resistance", "sell", "short", "pessimistic", "recession",
        "loss", "slide", "tumble", "fear", "panic", "risk",
    }

    @classmethod
    def _simple_sentiment(cls, text: str) -> float:
        """Simple keyword-based sentiment: -1 to +1."""
        text_lower = text.lower()
        words = set(re.findall(r"\b\w+\b", text_lower))
        pos = len(words & cls.POSITIVE_KEYWORDS)
        neg = len(words & cls.NEGATIVE_KEYWORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return float((pos - neg) / total)

    @staticmethod
    def _sentiment_label(score: float) -> str:
        if score > 0.2:
            return "positive"
        if score < -0.2:
            return "negative"
        return "neutral"

    @staticmethod
    def _extract_symbols(text: str) -> list[str]:
        """Extract trading symbols from text."""
        symbols = []
        text_upper = text.upper()
        for sym in ["BTC", "ETH", "EUR", "USD", "GBP", "JPY", "XAU", "OIL"]:
            if sym in text_upper:
                symbols.append(sym)
        return symbols

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
            "has_fred": bool(env.fred_api_key),
            "has_newsapi": bool(env.newsapi_api_key),
            "has_tradingeconomics": bool(env.tradingeconomics_api_key),
            "has_tradermade": bool(env.tradermade_api_key),
            "fred_cache_size": len(self._fred_cache),
        }
