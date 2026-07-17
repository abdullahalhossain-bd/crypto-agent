"""external.polymarket
=====================================================================
Polymarket prediction-market data vendor.

Inspired by TradingAgents' polymarket module.

Surfaces live, market-implied probabilities for forward-looking events
(Fed decisions, recession, elections, geopolitics, crypto) — a complement
to news (what happened) and FRED macro data (where things stand):
what the crowd actually prices to happen next.

Uses Polymarket's public Gamma API (https://gamma-api.polymarket.com) —
NO KEY, NO AUTH required.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from utils.logger import get_logger

log = get_logger("external.polymarket")

GAMMA_BASE = "https://gamma-api.polymarket.com"
REQUEST_TIMEOUT = 30
DEFAULT_LIMIT = 6


# ----------------------------------------------------------------------
def _request(path: str, params: dict) -> Optional[dict]:
    """HTTP GET to Polymarket Gamma API."""
    try:
        from urllib.parse import urlencode
        url = f"{GAMMA_BASE}/{path}?{urlencode(params)}"
        req = urllib_request.Request(url, method="GET",
                                      headers={"User-Agent": "TradingBot/1.0"})
        with urllib_request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as e:
        log.debug("Polymarket request failed: %r", e)
        return None
    except Exception as e:  # noqa: BLE001
        log.debug("Polymarket exception: %r", e)
        return None


def _parse_json_list(value) -> list:
    """Gamma encodes outcomes/outcomePrices as JSON-string arrays."""
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _is_forward_looking(market: dict, now: datetime) -> bool:
    """Keep only open markets that resolve in the future."""
    if market.get("closed"):
        return False
    end_date = market.get("endDate")
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            if end_dt < now:
                return False
        except Exception:  # noqa: BLE001
            pass
    return True


# ----------------------------------------------------------------------
def get_prediction_markets(limit: int = DEFAULT_LIMIT,
                            tag: Optional[str] = None) -> list[dict[str, Any]]:
    """Fetch live prediction markets ranked by volume.

    Args:
        limit: max markets to return
        tag: optional filter (e.g. "crypto", "politics", "economics")

    Returns list of dicts with:
        - question: market question
        - outcomes: list of outcome labels
        - probabilities: list of implied probabilities (0-1)
        - volume: traded volume
        - end_date: resolution date
        - url: market URL
    """
    now = datetime.now(tz=timezone.utc)
    params: dict[str, Any] = {
        "limit": limit * 3,  # over-fetch to compensate for filtering
        "active": "true",
        "closed": "false",
        "order": "volume",
        "ascending": "false",
    }
    if tag:
        params["tag"] = tag
    data = _request("/markets", params)
    if not data or not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for market in data:
        if not _is_forward_looking(market, now):
            continue
        outcomes = _parse_json_list(market.get("outcomes", []))
        prices = _parse_json_list(market.get("outcomePrices", []))
        if not outcomes or not prices:
            continue
        # Parse probabilities
        probs = []
        for p in prices:
            try:
                probs.append(float(p))
            except (ValueError, TypeError):
                probs.append(0.0)
        out.append({
            "question": market.get("question", ""),
            "outcomes": outcomes,
            "probabilities": probs,
            "volume": float(market.get("volume", 0) or 0),
            "liquidity": float(market.get("liquidity", 0) or 0),
            "end_date": market.get("endDate", ""),
            "url": f"https://polymarket.com/event/{market.get('slug', '')}",
            "source": "polymarket",
        })
        if len(out) >= limit:
            break
    return out


def get_crypto_markets(limit: int = 5) -> list[dict[str, Any]]:
    """Fetch crypto-related prediction markets."""
    return get_prediction_markets(limit=limit, tag="crypto")


def format_markets_for_prompt(markets: list[dict]) -> str:
    """Format markets as plaintext for LLM prompt injection."""
    if not markets:
        return "No prediction market data available."
    lines = ["Prediction market-implied probabilities:"]
    for m in markets[:6]:
        lines.append(f"\n  Q: {m['question']}")
        for outcome, prob in zip(m["outcomes"], m["probabilities"]):
            lines.append(f"    {outcome}: {prob:.0%}")
        if m["volume"] > 0:
            lines.append(f"    Volume: ${m['volume']:,.0f}")
    return "\n".join(lines)
