"""enhancements.rss_news_collector
=====================================================================
Inspired by OpenAlice's RSS news collector.

Periodic RSS feed fetcher that ingests news into a local store.
Unlike the news_provider (which fetches on-demand), this runs on a
timer and builds a local archive.

Features:
  - Configurable RSS feeds (crypto news, financial news, etc.)
  - Deduplication via content hash
  - In-flight guard (no overlapping fetches)
  - Persists to JSONL file
  - Simple keyword sentiment scoring (reuses news_provider logic)

Default feeds (crypto-focused):
  - CoinDesk RSS
  - Bitcoin Magazine
  - CryptoSlate
  - Reuters Crypto
  - RSS is the cheapest reliable news source — no API key needed.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from xml.etree import ElementTree as ET

from utils.logger import get_logger

log = get_logger("enhancements.rss_collector")


# ----------------------------------------------------------------------
DEFAULT_FEEDS: list[dict[str, str]] = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/",
     "category": "crypto"},
    {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/.rss/full/",
     "category": "crypto"},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/",
     "category": "crypto"},
    {"name": "Cointelegraph", "url": "https://cointelegraph.com/rss",
     "category": "crypto"},
]


@dataclass
class NewsItem:
    title: str
    description: str
    url: str
    published_at: str
    source: str
    category: str = ""
    dedup_key: str = ""
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    relevant_symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class RSSNewsCollector:
    """Periodic RSS feed collector with deduplication."""

    POSITIVE_KEYWORDS = {
        "bullish", "surge", "rally", "gain", "rise", "soar", "breakout",
        "uptrend", "support", "buy", "long", "optimistic", "growth",
        "boost", "jump", "climb", "recovery", "adoption",
    }
    NEGATIVE_KEYWORDS = {
        "bearish", "crash", "plunge", "drop", "fall", "decline", "breakdown",
        "downtrend", "resistance", "sell", "short", "pessimistic", "recession",
        "loss", "slide", "tumble", "fear", "panic", "hack", "ban", "lawsuit",
    }

    def __init__(self,
                 feeds: Optional[list[dict]] = None,
                 store_path: str = "data/rss_news.jsonl",
                 interval_s: int = 300) -> None:
        self.feeds = feeds or DEFAULT_FEEDS
        self.store_path = store_path
        self.interval_s = int(interval_s)
        self._seen_keys: set[str] = set()
        self._timer: Optional[threading.Timer] = None
        self._fetch_in_flight: Optional[threading.Event] = None
        self._running = False
        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        self._load_seen()

    # ----------------------------------------------------------------
    def start(self) -> None:
        """Start periodic collection."""
        if self._running:
            return
        self._running = True
        # Initial fetch
        threading.Thread(target=self._fetch_all, daemon=True).start()
        # Schedule periodic
        self._schedule_next()

    def stop(self) -> None:
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self.interval_s, self._fetch_cycle)
        self._timer.daemon = True
        self._timer.start()

    def _fetch_cycle(self) -> None:
        try:
            self._fetch_all()
        except Exception as e:  # noqa: BLE001
            log.warning("RSS fetch cycle error: %r", e)
        self._schedule_next()

    # ----------------------------------------------------------------
    def fetch_once(self) -> dict[str, int]:
        """Fetch all feeds once. Returns {total, new}."""
        return self._fetch_all()

    def _fetch_all(self) -> dict[str, int]:
        """Fetch all feeds. In-flight guard prevents overlapping runs."""
        if self._fetch_in_flight is not None:
            log.debug("RSS fetch already in flight — skipping")
            return {"total": 0, "new": 0}
        self._fetch_in_flight = threading.Event()
        total = 0
        new_count = 0
        for feed in self.feeds:
            try:
                items = self._fetch_feed(feed)
                for item in items:
                    total += 1
                    if item.dedup_key not in self._seen_keys:
                        self._seen_keys.add(item.dedup_key)
                        self._persist(item)
                        new_count += 1
            except Exception as e:  # noqa: BLE001
                log.warning("RSS feed %s failed: %r", feed.get("name", "?"), e)
        self._fetch_in_flight = None
        if new_count:
            log.info("RSS collected: %d new / %d total", new_count, total)
        return {"total": total, "new": new_count}

    # ----------------------------------------------------------------
    def _fetch_feed(self, feed: dict) -> list[NewsItem]:
        url = feed["url"]
        name = feed.get("name", "unknown")
        category = feed.get("category", "")
        try:
            req = urllib_request.Request(url, method="GET",
                                          headers={"User-Agent": "TradingBot/1.0"})
            with urllib_request.urlopen(req, timeout=15.0) as resp:
                if resp.status != 200:
                    return []
                content = resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as e:
            log.debug("RSS fetch %s failed: %r", name, e)
            return []
        # Parse XML
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            log.debug("RSS parse %s failed: %r", name, e)
            return []
        items: list[NewsItem] = []
        # RSS 2.0: /rss/channel/item
        # Atom: /feed/entry
        for item_elem in root.iter("item"):
            title = self._text(item_elem, "title")
            desc = self._text(item_elem, "description")
            link = self._text(item_elem, "link")
            pub = self._text(item_elem, "pubDate")
            if not title:
                continue
            # Dedup key
            dedup = hashlib.sha256(
                (title + link).encode("utf-8")
            ).hexdigest()[:16]
            # Sentiment
            sent = self._sentiment(title + " " + desc)
            label = "positive" if sent > 0.2 else ("negative" if sent < -0.2 else "neutral")
            # Symbols
            syms = self._extract_symbols(title + " " + desc)
            items.append(NewsItem(
                title=title, description=desc, url=link,
                published_at=pub, source=name, category=category,
                dedup_key=dedup,
                sentiment_score=float(sent), sentiment_label=label,
                relevant_symbols=syms,
            ))
        # Also try Atom entries
        for entry_elem in root.iter("entry"):
            title = self._text(entry_elem, "title")
            desc = self._text(entry_elem, "summary") or self._text(entry_elem, "content")
            link = ""
            for link_elem in entry_elem.findall("link"):
                link = link_elem.get("href", "")
                if link:
                    break
            pub = self._text(entry_elem, "updated") or self._text(entry_elem, "published")
            if not title:
                continue
            dedup = hashlib.sha256(
                (title + link).encode("utf-8")
            ).hexdigest()[:16]
            sent = self._sentiment(title + " " + desc)
            label = "positive" if sent > 0.2 else ("negative" if sent < -0.2 else "neutral")
            syms = self._extract_symbols(title + " " + desc)
            items.append(NewsItem(
                title=title, description=desc, url=link,
                published_at=pub, source=name, category=category,
                dedup_key=dedup,
                sentiment_score=float(sent), sentiment_label=label,
                relevant_symbols=syms,
            ))
        return items

    # ----------------------------------------------------------------
    @staticmethod
    def _text(parent: ET.Element, tag: str) -> str:
        elem = parent.find(tag)
        if elem is not None and elem.text:
            return elem.text.strip()
        return ""

    @classmethod
    def _sentiment(cls, text: str) -> float:
        text_lower = text.lower()
        words = set(text_lower.split())
        pos = len(words & cls.POSITIVE_KEYWORDS)
        neg = len(words & cls.NEGATIVE_KEYWORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return float((pos - neg) / total)

    @staticmethod
    def _extract_symbols(text: str) -> list[str]:
        text_upper = text.upper()
        out = []
        for sym in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX",
                     "DOT", "LINK", "MATIC", "UNI", "USD", "EUR"]:
            if sym in text_upper:
                out.append(sym)
        return list(set(out))

    # ----------------------------------------------------------------
    def _persist(self, item: NewsItem) -> None:
        try:
            with open(self.store_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("RSS persist failed: %r", e)

    def _load_seen(self) -> None:
        if not os.path.isfile(self.store_path):
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "dedup_key" in d:
                            self._seen_keys.add(d["dedup_key"])
                    except Exception:  # noqa: BLE001
                        continue
            log.info("RSS loaded %d seen keys", len(self._seen_keys))
        except Exception as e:  # noqa: BLE001
            log.warning("RSS load failed: %r", e)

    # ----------------------------------------------------------------
    def query(self, symbol: Optional[str] = None,
                category: Optional[str] = None,
                limit: int = 50) -> list[dict[str, Any]]:
        """Query the local archive."""
        if not os.path.isfile(self.store_path):
            return []
        out: list[dict] = []
        try:
            with open(self.store_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return []
        for line in reversed(lines):
            try:
                d = json.loads(line)
                if symbol and symbol.upper() not in str(d.get("relevant_symbols", [])).upper():
                    continue
                if category and d.get("category") != category:
                    continue
                out.append(d)
                if len(out) >= limit:
                    break
            except Exception:  # noqa: BLE001
                continue
        return out

    # ----------------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "n_feeds": len(self.feeds),
            "n_seen": len(self._seen_keys),
            "running": self._running,
            "store_path": self.store_path,
        }
