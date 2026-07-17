"""architecture/context_providers.py
=====================================================================
Market Context Providers — wires external/ news + sentiment + macro
into a single cached facade consumed by the LLM augmentation layer.

Co-Founder Audit (production hardening):
  - external/news_provider.py, external/sentiment_provider.py, and the
    other external/ providers were 100% isolated — never imported by
    the live path. The LLM graph's analysts (NewsAnalyst, SentimentAnalyst)
    expected context fields (news_items, macro, retail_sentiment,
    news_sentiment) but got empty dicts because nothing populated them.
  - This module bridges that gap. It builds a single MarketContextProvider
    that wraps each external/ provider, caches results with per-source
    TTLs, and exposes a get_context(symbol) method that returns the
    exact dict shape the analysts expect.

FIELD MAPPING (verified against agents/analysts.py):
  NewsAnalyst.analyze() reads:
    - context["news_items"]  → list[dict] with title/source/sentiment_label
    - context["macro"]       → dict of macro indicators
  SentimentAnalyst.analyze() reads:
    - context["social_sentiment"]  → dict (no provider yet → empty)
    - context["news_sentiment"]    → dict (computed from news_items)
    - context["retail_sentiment"]  → dict (from SentimentProviderManager)
  FundamentalsAnalyst.analyze() reads:
    - context (generic — passed whole to LLM)
  TechnicalAnalyst.analyze() reads:
    - context (generic — uses df directly)

CACHE STRATEGY:
  - news_headlines: 10 min TTL (news doesn't change minute-to-minute)
  - macro_indicators: 1 hour TTL (CPI/unemployment updates daily at most)
  - retail_sentiment: 5 min TTL per symbol (positioning shifts slowly)
  - upcoming_events: 30 min TTL (economic calendar updates infrequently)
  Without caching, the LLM augmentation layer would fire ~100 HTTP calls
  per cycle (one per symbol). With caching, it fires 1-2 per cycle.

FAIL-SAFE:
  - Any provider exception → returns empty dict for that field
  - Missing API keys → provider auto-disabled, returns empty
  - Network timeout → returns empty, logged at DEBUG
  - The bot NEVER blocks on external data; LLM analysts degrade
    gracefully to "no recent news available" and base analysis on
    the OHLCV data alone.

USAGE:
  from architecture.context_providers import MarketContextProvider
  ctx_provider = MarketContextProvider(config)
  context = ctx_provider.get_context(symbol="BTCUSD")
  # context now has: news_items, macro, retail_sentiment, news_sentiment,
  # upcoming_events, social_sentiment (empty), timestamp, symbol
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger("trading_bot.architecture.context_providers")


# ----------------------------------------------------------------------
# Cache entry
# ----------------------------------------------------------------------
class _CacheEntry:
    __slots__ = ("data", "expires_at")

    def __init__(self, data: Any, ttl_s: float):
        self.data = data
        self.expires_at = time.time() + ttl_s

    def is_fresh(self) -> bool:
        return time.time() < self.expires_at


# ----------------------------------------------------------------------
# MarketContextProvider — the single facade
# ----------------------------------------------------------------------
class MarketContextProvider:
    """Single entry point for all market context data.

    Wraps external/ providers with caching, fail-safe error handling,
    and the exact dict shape the LLM analysts expect.
    """

    def __init__(self, config: Dict[str, Any]):
        self._cfg = config or {}
        news_cfg = self._cfg.get("news", {})
        sent_cfg = self._cfg.get("sentiment", {})

        # Feature toggles
        self._news_enabled = bool(news_cfg.get("enabled", True))
        self._sentiment_enabled = bool(sent_cfg.get("enabled", True))
        self._macro_enabled = bool(news_cfg.get("macro_enabled", True))
        self._events_enabled = bool(news_cfg.get("events_enabled", True))

        # Cache TTLs (seconds)
        self._news_ttl = float(news_cfg.get("cache_ttl_s", 600))      # 10 min
        self._macro_ttl = float(news_cfg.get("macro_cache_ttl_s", 3600))  # 1 hour
        self._sentiment_ttl = float(sent_cfg.get("cache_ttl_s", 300))  # 5 min
        self._events_ttl = float(news_cfg.get("events_cache_ttl_s", 1800))  # 30 min

        # Limits
        self._max_headlines = int(news_cfg.get("max_headlines", 10))
        self._news_query = news_cfg.get("query", "crypto OR bitcoin OR forex")

        # Per-source caches (key → _CacheEntry)
        # News + macro are symbol-independent (same for all symbols in a cycle)
        self._news_cache: Optional[_CacheEntry] = None
        self._macro_cache: Optional[_CacheEntry] = None
        self._events_cache: Optional[_CacheEntry] = None
        # Sentiment is per-symbol
        self._sentiment_cache: Dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()

        # Lazy-initialized providers (None = not yet tried)
        self._news_provider = None
        self._sentiment_provider = None
        self._news_provider_tried = False
        self._sentiment_provider_tried = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_context(self, symbol: str) -> Dict[str, Any]:
        """Build the full context dict for one symbol.

        Returns a dict with keys matching what agents/analysts.py expects:
          - news_items: list[dict] with title/source/sentiment_label
          - macro: dict of macro indicators
          - retail_sentiment: dict with long_pct/short_pct/contrarian_signal
          - news_sentiment: dict with avg score/label/article_count
          - social_sentiment: dict (empty — no provider yet)
          - upcoming_events: list[dict] of high-impact economic events
          - timestamp: ISO timestamp
          - symbol: the symbol string
        """
        ctx: Dict[str, Any] = {
            "symbol": symbol,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        # News + macro + events are symbol-independent (fetch once per cycle)
        ctx["news_items"] = self._get_news_headlines()
        ctx["macro"] = self._get_macro_indicators()
        ctx["upcoming_events"] = self._get_upcoming_events()
        ctx["news_sentiment"] = self._compute_news_sentiment(ctx["news_items"])

        # Sentiment is per-symbol
        ctx["retail_sentiment"] = self._get_retail_sentiment(symbol)

        # Social sentiment — no provider wired yet (future: Twitter/Reddit)
        ctx["social_sentiment"] = {}

        return ctx

    def status(self) -> Dict[str, Any]:
        """Status for diagnostics — visible via llm_status()."""
        return {
            "news_enabled": self._news_enabled,
            "sentiment_enabled": self._sentiment_enabled,
            "macro_enabled": self._macro_enabled,
            "events_enabled": self._events_enabled,
            "news_provider_active": self._news_provider is not None,
            "sentiment_provider_active": self._sentiment_provider is not None,
            "news_cache_fresh": self._news_cache.is_fresh() if self._news_cache else False,
            "macro_cache_fresh": self._macro_cache.is_fresh() if self._macro_cache else False,
            "events_cache_fresh": self._events_cache.is_fresh() if self._events_cache else False,
            "sentiment_cache_size": len(self._sentiment_cache),
        }

    # ------------------------------------------------------------------
    # News headlines (symbol-independent)
    # ------------------------------------------------------------------
    def _get_news_headlines(self) -> List[Dict[str, Any]]:
        if not self._news_enabled:
            return []
        # Check cache
        with self._cache_lock:
            if self._news_cache and self._news_cache.is_fresh():
                return self._news_cache.data
        # Fetch
        provider = self._get_news_provider()
        if provider is None:
            return []
        try:
            articles = provider.newsapi_headlines(
                query=self._news_query, n=self._max_headlines)
            items = [a.to_dict() for a in articles]
        except Exception as e:
            log.debug("context_providers: news fetch failed: %r", e)
            items = []
        # Cache even empty results so we don't hammer the API on every cycle
        with self._cache_lock:
            self._news_cache = _CacheEntry(items, self._news_ttl)
        return items

    # ------------------------------------------------------------------
    # Macro indicators (symbol-independent)
    # ------------------------------------------------------------------
    def _get_macro_indicators(self) -> Dict[str, Any]:
        if not self._macro_enabled:
            return {}
        with self._cache_lock:
            if self._macro_cache and self._macro_cache.is_fresh():
                return self._macro_cache.data
        provider = self._get_news_provider()
        if provider is None:
            return {}
        try:
            macro = provider.fred_macro_features()
            # Filter out None values — analysts handle missing data better
            # than None placeholders
            macro = {k: v for k, v in macro.items() if v is not None}
        except Exception as e:
            log.debug("context_providers: macro fetch failed: %r", e)
            macro = {}
        with self._cache_lock:
            self._macro_cache = _CacheEntry(macro, self._macro_ttl)
        return macro

    # ------------------------------------------------------------------
    # Upcoming economic events (symbol-independent)
    # ------------------------------------------------------------------
    def _get_upcoming_events(self) -> List[Dict[str, Any]]:
        if not self._events_enabled:
            return []
        with self._cache_lock:
            if self._events_cache and self._events_cache.is_fresh():
                return self._events_cache.data
        provider = self._get_news_provider()
        if provider is None:
            return []
        try:
            events = provider.upcoming_events(hours_ahead=24)
            # Only include HIGH and MEDIUM impact — LOW impact noise
            items = [e.to_dict() for e in events
                     if e.impact.value in ("HIGH", "MEDIUM")]
        except Exception as e:
            log.debug("context_providers: events fetch failed: %r", e)
            items = []
        with self._cache_lock:
            self._events_cache = _CacheEntry(items, self._events_ttl)
        return items

    # ------------------------------------------------------------------
    # Retail sentiment (per-symbol)
    # ------------------------------------------------------------------
    def _get_retail_sentiment(self, symbol: str) -> Dict[str, Any]:
        if not self._sentiment_enabled:
            return {}
        with self._cache_lock:
            entry = self._sentiment_cache.get(symbol)
            if entry and entry.is_fresh():
                return entry.data
        provider = self._get_sentiment_provider()
        if provider is None:
            return {}
        try:
            data = provider.get_sentiment(symbol)
            result = data.to_dict()
        except Exception as e:
            log.debug("context_providers: sentiment fetch failed for %s: %r",
                     symbol, e)
            result = {}
        with self._cache_lock:
            self._sentiment_cache[symbol] = _CacheEntry(result, self._sentiment_ttl)
            # Cap per-symbol cache to last 50 symbols
            if len(self._sentiment_cache) > 50:
                oldest = next(iter(self._sentiment_cache))
                del self._sentiment_cache[oldest]
        return result

    # ------------------------------------------------------------------
    # Compute aggregated news sentiment from headlines
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_news_sentiment(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not news_items:
            return {"score": 0.0, "label": "neutral", "article_count": 0}
        scores = []
        for item in news_items:
            score = item.get("sentiment_score", 0.0)
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                continue
        if not scores:
            return {"score": 0.0, "label": "neutral",
                    "article_count": len(news_items)}
        avg = sum(scores) / len(scores)
        if avg > 0.2:
            label = "positive"
        elif avg < -0.2:
            label = "negative"
        else:
            label = "neutral"
        return {
            "score": round(avg, 3),
            "label": label,
            "article_count": len(news_items),
            "sample_size": len(scores),
        }

    # ------------------------------------------------------------------
    # Lazy provider initialization
    # ------------------------------------------------------------------
    def _get_news_provider(self):
        if self._news_provider_tried:
            return self._news_provider
        self._news_provider_tried = True
        try:
            from external.news_provider import NewsProviderManager
            self._news_provider = NewsProviderManager()
            stats = self._news_provider.stats
            active_sources = [k for k, v in stats.items()
                              if k.startswith("has_") and v]
            if active_sources:
                log.info("context_providers: news provider active (sources: %s)",
                         active_sources)
            else:
                log.info("context_providers: news provider constructed but no "
                         "API keys set — news/macro/events will be empty. "
                         "Set FRED_API_KEY, NEWSAPI_KEY, TRADINGECONOMICS_KEY, "
                         "or TRADERMADE_KEY in .env to enable.")
                # Keep the provider — it may still be useful for the
                # synthetic fallback in sentiment_provider
        except Exception as e:
            log.warning("context_providers: news provider init failed: %r", e)
            self._news_provider = None
        return self._news_provider

    def _get_sentiment_provider(self):
        if self._sentiment_provider_tried:
            return self._sentiment_provider
        self._sentiment_provider_tried = True
        try:
            from external.sentiment_provider import SentimentProviderManager
            self._sentiment_provider = SentimentProviderManager()
            stats = self._sentiment_provider.stats
            active_sources = [k for k, v in stats.items()
                              if k.startswith("has_") and v]
            if active_sources:
                log.info("context_providers: sentiment provider active (sources: %s)",
                         active_sources)
            else:
                log.info("context_providers: sentiment provider constructed but "
                         "no API keys set — retail_sentiment will use synthetic "
                         "fallback (neutral). Set MYFXBOOK_EMAIL/PASSWORD or "
                         "OANDA_API_KEY in .env to enable real data.")
        except Exception as e:
            log.warning("context_providers: sentiment provider init failed: %r", e)
            self._sentiment_provider = None
        return self._sentiment_provider
