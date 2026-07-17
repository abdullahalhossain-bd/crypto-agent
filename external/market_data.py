"""external.market_data
=====================================================================
Day 166-170 — Multi-source market data manager.

Sources (in priority order, configurable via PREFERRED_DATA_SOURCE):
  1. Twelve Data  (800 req/day free, 5-year history)  ← recommended
  2. Alpha Vantage (25 req/day, FX_INTRADAY now premium)
  3. Polygon.io   (5 req/min free, end-of-day)
  4. MT5 (if connected)

All sources return the canonical OHLCV schema used by engine.data_feed.
If the preferred source fails, falls back to the next.

This is an ALTERNATIVE to MT5 for environments where MT5 isn't
available (Linux VPS). The bot can run fully on Twelve Data.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import pandas as pd

from external.env_loader import env
from utils.logger import get_logger

log = get_logger("external.market_data")


@dataclass
class MarketDataResult:
    source: str
    symbol: str
    timeframe: str
    df: pd.DataFrame
    n_bars: int
    success: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_bars": self.n_bars,
            "success": self.success,
            "error": self.error,
        }


# ----------------------------------------------------------------------
class MarketDataManager:
    """Multi-source market data with automatic fallback."""

    # Map our timeframes to provider-specific formats
    TIMEFRAME_MAP = {
        "twelve_data": {"M1": "1min", "M5": "5min", "M15": "15min",
                          "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"},
        "alpha_vantage": {"M1": "1", "M5": "5", "M15": "15", "M30": "30",
                            "H1": "60", "D1": "Daily"},
        "polygon": {"M1": "1", "M5": "5", "M15": "15", "M30": "30",
                      "H1": "1 hour", "D1": "1 day"},
    }

    # ISO-4217 currency codes Alpha Vantage's FX_INTRADAY endpoint actually
    # supports. Any "...USD"-suffixed symbol whose base is NOT one of these
    # is treated as crypto (DIGITAL_CURRENCY_DAILY), not FX. This matters
    # because major crypto pairs (BTCUSD, ETHUSD, ...) are also 6 characters
    # and contain "USD", so a length/substring check alone misclassifies them.
    _FX_CURRENCY_CODES = {
        "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "SEK", "NOK",
        "DKK", "PLN", "TRY", "MXN", "ZAR", "SGD", "HKD", "CNH", "THB",
    }

    def __init__(self, preferred: Optional[str] = None) -> None:
        self.preferred = preferred or env.preferred_data_source
        self._request_counts: dict[str, int] = {"twelve_data": 0,
                                                  "alpha_vantage": 0,
                                                  "polygon": 0}

    # ----------------------------------------------------------------
    def fetch(self, symbol: str, timeframe: str,
              count: int = 500) -> MarketDataResult:
        """Fetch OHLCV from preferred source, fall back on failure."""
        sources = self._source_priority()
        for source in sources:
            if source == "mt5":
                continue  # handled elsewhere
            try:
                df = self._fetch_from(source, symbol, timeframe, count)
                if df is not None and not df.empty:
                    self._request_counts[source] = self._request_counts.get(source, 0) + 1
                    return MarketDataResult(
                        source=source, symbol=symbol, timeframe=timeframe,
                        df=df, n_bars=len(df), success=True,
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("market data source %s failed: %r", source, e)
        return MarketDataResult(
            source="none", symbol=symbol, timeframe=timeframe,
            df=pd.DataFrame(), n_bars=0, success=False,
            error="all sources failed",
        )

    # ----------------------------------------------------------------
    def _source_priority(self) -> list[str]:
        """Return source priority list."""
        all_sources = ["twelve_data", "alpha_vantage", "polygon"]
        if self.preferred in all_sources:
            # Move preferred to front
            rest = [s for s in all_sources if s != self.preferred]
            return [self.preferred] + rest
        return all_sources

    # ----------------------------------------------------------------
    def _fetch_from(self, source: str, symbol: str,
                      timeframe: str, count: int) -> Optional[pd.DataFrame]:
        if source == "twelve_data":
            return self._fetch_twelve_data(symbol, timeframe, count)
        if source == "alpha_vantage":
            return self._fetch_alpha_vantage(symbol, timeframe, count)
        if source == "polygon":
            return self._fetch_polygon(symbol, timeframe, count)
        return None

    # ----------------------------------------------------------------
    def _fetch_twelve_data(self, symbol: str, timeframe: str,
                             count: int) -> Optional[pd.DataFrame]:
        key = env.twelve_data_api_key
        if not key:
            return None
        interval = self.TIMEFRAME_MAP["twelve_data"].get(timeframe, "15min")
        # Convert crypto symbols (BTCUSD → BTC/USD)
        td_symbol = self._convert_symbol_twelve_data(symbol)
        url = (f"{env.twelve_data_base_url}/time_series"
               f"?symbol={td_symbol}&interval={interval}"
               f"&outputsize={count}&apikey={key}&format=JSON")
        data = self._http_get(url)
        if not data or "values" not in data:
            return None
        values = data["values"]
        # Reverse (Twelve Data returns newest first)
        values = list(reversed(values))
        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.rename(columns={
            "datetime": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
        })
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0.0
        return df[["time", "open", "high", "low", "close", "volume"]]

    # ----------------------------------------------------------------
    def _fetch_alpha_vantage(self, symbol: str, timeframe: str,
                               count: int) -> Optional[pd.DataFrame]:
        key = env.alpha_vantage_api_key
        if not key:
            return None
        interval = self.TIMEFRAME_MAP["alpha_vantage"].get(timeframe)
        if not interval:
            return None
        # FX_INTRADAY for forex pairs, DIGITAL_CURRENCY for crypto.
        # A 6-char "...USD" symbol is ambiguous (EURUSD is FX, BTCUSD/ETHUSD
        # are crypto) — length/substring alone is not enough. Only route to
        # FX_INTRADAY if the base currency is a real ISO FX code.
        base_sym = symbol[:3].upper() if len(symbol) >= 6 else ""
        is_fx_pair = (len(symbol) == 6 and symbol.upper().endswith("USD")
                      and base_sym in self._FX_CURRENCY_CODES)
        if is_fx_pair:
            from_sym = symbol[:3]
            to_sym = symbol[3:]
            url = (f"{env.alpha_vantage_base_url}?function=FX_INTRADAY"
                   f"&from_symbol={from_sym}&to_symbol={to_sym}"
                   f"&interval={interval}min&outputsize=full&apikey={key}")
        else:
            # Crypto
            url = (f"{env.alpha_vantage_base_url}?function=DIGITAL_CURRENCY_DAILY"
                   f"&symbol={symbol.replace('USD','')}&market=USD&apikey={key}")
        data = self._http_get(url)
        if not data:
            return None
        # Parse Time Series
        ts_key = next((k for k in data if "Time Series" in k), None)
        if not ts_key:
            return None
        ts = data[ts_key]
        rows = []
        for dt, values in ts.items():
            row = {"time": pd.to_datetime(dt, utc=True)}
            for k, v in values.items():
                field = k.split(". ")[-1].lower()
                row[field] = float(v)
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("time").tail(count)
        if "volume" not in df.columns:
            df["volume"] = 0.0
        return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    # ----------------------------------------------------------------
    def _fetch_polygon(self, symbol: str, timeframe: str,
                         count: int) -> Optional[pd.DataFrame]:
        key = env.polygon_api_key
        if not key:
            return None
        # Polygon uses different symbol format
        poly_symbol = self._convert_symbol_polygon(symbol)
        multiplier, unit = self._polygon_timeframe(timeframe)
        # End date = today, start date = N bars back
        end = datetime.now(tz=timezone.utc)
        # Rough calculation of start date
        days_back = max(1, count // 24) * 2  # buffer
        start = end - pd.Timedelta(days=days_back)
        url = (f"https://api.polygon.io/v2/aggs/ticker/X:{poly_symbol}"
               f"/range/{multiplier}/{unit}/{start.strftime('%Y-%m-%d')}"
               f"/{end.strftime('%Y-%m-%d')}"
               f"?adjusted=true&sort=asc&limit={count}&apiKey={key}")
        data = self._http_get(url)
        if not data or "results" not in data:
            return None
        results = data["results"]
        df = pd.DataFrame(results)
        df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                                  "c": "close", "v": "volume"})
        return df[["time", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    # ----------------------------------------------------------------
    @staticmethod
    def _convert_symbol_twelve_data(symbol: str) -> str:
        """BTCUSD → BTC/USD, ETHUSD → ETH/USD, EURUSD → EUR/USD."""
        if "USD" in symbol and not symbol.startswith("USD"):
            base = symbol.replace("USD", "")
            return f"{base}/USD"
        return symbol

    @staticmethod
    def _convert_symbol_polygon(symbol: str) -> str:
        """BTCUSD → BTCUSD (Polygon uses concatenated for crypto)."""
        return symbol

    @staticmethod
    def _polygon_timeframe(tf: str) -> tuple[int, str]:
        mapping = {"M1": (1, "minute"), "M5": (5, "minute"),
                    "M15": (15, "minute"), "M30": (30, "minute"),
                    "H1": (1, "hour"), "H4": (4, "hour"),
                    "D1": (1, "day")}
        return mapping.get(tf, (15, "minute"))

    # ----------------------------------------------------------------
    @staticmethod
    def _http_get(url: str) -> Optional[dict]:
        try:
            req = urllib_request.Request(url, method="GET")
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
            "preferred_source": self.preferred,
            "request_counts": dict(self._request_counts),
            "available_sources": {
                "twelve_data": bool(env.twelve_data_api_key),
                "alpha_vantage": bool(env.alpha_vantage_api_key),
                "polygon": bool(env.polygon_api_key),
            },
        }