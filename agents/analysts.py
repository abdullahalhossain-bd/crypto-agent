"""agents.analysts
=====================================================================
Analyst Team — LLM-powered specialized analysts.

Each analyst produces a markdown report:
  - FundamentalsAnalyst : on-chain metrics, tokenomics, project health
  - NewsAnalyst          : global news + macro indicators impact
  - SentimentAnalyst     : social media + news sentiment aggregation
  - TechnicalAnalyst     : price-action + indicator-based analysis

Each analyst takes (symbol, market_data, context) and returns a report
string. The LLM is invoked via external.llm_provider with fallback.

CRITICAL: Analysts produce ANALYSIS, not trading decisions. The bull/
bear researchers and trader use these reports as INPUT context.

FIXES (Batch 2 audit):
  - C10: `_call_llm` now retries up to 3 times with exponential backoff
    (0.5s, 1.0s, 2.0s) on LLM failure instead of returning a single
    error string. If all retries fail, the error text is clearly marked.
  - C19/H8/H13/H14: `_validate_df` checks that df is not None, not empty,
    and has a 'close' column before any indicator access. TechnicalAnalyst
    checks ATR series length before `.iloc[-1]`. FundamentalsAnalyst
    checks 'volume' column existence before `.tail(30).mean()`.
  - H2: `AnalystTeam` now caches reports keyed by (symbol, last_bar_time)
    so repeated calls within the same bar don't waste LLM tokens.
  - M1: `AnalystTeam.__init__` raises `ValueError` if a selected analyst
    name doesn't exist, instead of silently skipping it.
  - M9: `NewsAnalyst` validates that each news item is a dict with at
    least a 'title' key; malformed items are skipped with a debug log.
  - M20: each analyst class has a `temperature` class attribute that
    controls its LLM call temperature (was hard-coded 0.3 for all).
  - L2: `FundamentalsAnalyst.metadata` now stores avg_vol, latest_vol,
    and a volume_ratio (latest / avg) for downstream feature engineering.
  - L10: `TechnicalAnalyst` rounds indicator values to 4 decimal places
    in the prompt text (was printing full float precision).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from external.llm_provider import LLMProvider, LLMMessage
from utils.indicators import (
    sma, ema, rsi, macd, bbands, atr, obv, vwap, rvol,
)
from utils.logger import get_logger

log = get_logger("agents.analysts")

# C10 fix: retry config for LLM calls.
_LLM_MAX_RETRIES = 3
_LLM_BASE_DELAY_S = 0.5


@dataclass
class AnalystReport:
    analyst: str
    symbol: str
    report: str
    success: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyst": self.analyst,
            "symbol": self.symbol,
            "report": self.report,
            "success": self.success,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


# ----------------------------------------------------------------------
def _validate_df(df: pd.DataFrame, require_cols: tuple[str, ...] = ("close",)) -> None:
    """C19/H8 fix: validate df before indicator access. Raises ValueError."""
    if df is None:
        raise ValueError("df is None")
    if df.empty:
        raise ValueError("df is empty")
    missing = [c for c in require_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df missing required columns: {missing}")


# ----------------------------------------------------------------------
class BaseAnalyst:
    """Base class for all analysts.

    M20 fix: each subclass sets a `temperature` class attribute.
    """

    name: str = "base"
    temperature: float = 0.3

    def __init__(self, llm: Optional[LLMProvider] = None) -> None:
        self.llm = llm or LLMProvider()

    def analyze(self, symbol: str, df: pd.DataFrame,
                 context: Optional[dict] = None) -> AnalystReport:
        raise NotImplementedError

    def _call_llm(self, system_prompt: str, user_prompt: str,
                    max_tokens: int = 600) -> str:
        """C10 fix: retry on LLM failure with exponential backoff.

        M2 FIX (Chief AI Architect Audit): on failure, return an empty string
        instead of an error message. Downstream agents (BullBearDebate, etc.)
        were treating error text as real analysis and "debating" an error
        message. Now they'll see an empty report and skip it.
        """
        last_error = ""
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                resp = self.llm.chat(
                    messages=[
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=user_prompt),
                    ],
                    max_tokens=max_tokens,
                    temperature=self.temperature,  # M20 fix
                )
                if resp.success:
                    return resp.text
                last_error = resp.error
                log.warning("LLM call failed for %s (attempt %d/%d): %s",
                            self.name, attempt + 1, _LLM_MAX_RETRIES, resp.error)
            except Exception as e:  # noqa: BLE001
                last_error = str(e)
                log.warning("LLM call raised for %s (attempt %d/%d): %r",
                            self.name, attempt + 1, _LLM_MAX_RETRIES, e)
            # Exponential backoff: 0.5s, 1.0s, 2.0s
            if attempt < _LLM_MAX_RETRIES - 1:
                delay = _LLM_BASE_DELAY_S * (2 ** attempt)
                time.sleep(delay)
        # M2 FIX: return empty string, NOT error text. Downstream agents
        # check for empty reports and skip them. Error text was being
        # debated as if it were real analysis.
        log.warning("LLM %s unavailable after %d attempts: %s — returning empty report",
                    self.name, _LLM_MAX_RETRIES, last_error)
        return ""


# ----------------------------------------------------------------------
class FundamentalsAnalyst(BaseAnalyst):
    """Analyzes crypto fundamentals: on-chain metrics, tokenomics."""

    name = "fundamentals"
    temperature = 0.4  # M20 fix

    def analyze(self, symbol: str, df: pd.DataFrame,
                 context: Optional[dict] = None) -> AnalystReport:
        context = context or {}
        # C19 fix: validate df.
        _validate_df(df, require_cols=("close",))

        # H14 fix: check 'volume' column existence before access.
        volume_data = ""
        metadata: dict[str, Any] = {}
        if "volume" in df.columns and len(df) >= 30:
            try:
                avg_vol = float(df["volume"].tail(30).mean())
                latest_vol = float(df["volume"].iloc[-1])
                volume_data = (
                    f"30-day avg volume: {avg_vol:,.0f}\n"
                    f"Latest volume: {latest_vol:,.0f}\n"
                    f"Volume ratio (latest/avg): {latest_vol / avg_vol:.2f}x"
                )
                # L2 fix: store more metadata for downstream feature engineering.
                metadata = {
                    "avg_volume": avg_vol,
                    "latest_volume": latest_vol,
                    "volume_ratio": (latest_vol / avg_vol) if avg_vol > 0 else 1.0,
                }
            except Exception as e:  # noqa: BLE001
                log.warning("FundamentalsAnalyst volume calc failed: %r", e)

        # Safely get the latest bar time.
        latest_time = str(df["time"].iloc[-1]) if "time" in df.columns else "latest"

        system_prompt = (
            "You are a fundamentals analyst specializing in cryptocurrency markets. "
            "Analyze on-chain metrics, tokenomics, network activity, and project health. "
            "Provide a comprehensive report with specific, actionable insights. "
            "Include a markdown table at the end summarizing key points."
        )
        user_prompt = (
            f"Analyze fundamentals for {symbol} as of {latest_time}.\n\n"
            f"Available data:\n{volume_data}\n\n"
            f"Context: {context}\n\n"
            "Write a 3-5 paragraph fundamentals report covering:\n"
            "1. Network activity and adoption trends\n"
            "2. Tokenomics (supply, distribution, utility)\n"
            "3. Development activity and ecosystem health\n"
            "4. Competitive position\n"
            "5. Key risks and red flags"
        )
        report_text = self._call_llm(system_prompt, user_prompt)
        return AnalystReport(
            analyst=self.name, symbol=symbol, report=report_text,
            metadata=metadata,
        )


# ----------------------------------------------------------------------
class NewsAnalyst(BaseAnalyst):
    """Analyzes global news + macro indicators impact."""

    name = "news"
    temperature = 0.3  # M20 fix

    def analyze(self, symbol: str, df: pd.DataFrame,
                 context: Optional[dict] = None) -> AnalystReport:
        context = context or {}
        _validate_df(df, require_cols=("close",))  # C19 fix

        news_items = context.get("news_items", [])
        macro_data = context.get("macro", {})

        # M9 fix: validate news item structure; skip malformed items.
        clean_news: list[dict] = []
        for n in news_items[:10]:
            if isinstance(n, dict) and n.get("title"):
                clean_news.append(n)
            else:
                log.debug("NewsAnalyst: skipping malformed news item: %r", n)

        news_summary = "\n".join(
            f"- [{n.get('source', '')}] {n.get('title', '')} "
            f"(sentiment: {n.get('sentiment_label', 'neutral')})"
            for n in clean_news
        ) or "No recent news available."

        macro_summary = "\n".join(
            f"- {k}: {v}" for k, v in macro_data.items() if v is not None
        ) or "No macro data available."

        system_prompt = (
            "You are a news analyst monitoring global events and macroeconomic indicators. "
            "Interpret how news and macro data impact the cryptocurrency market. "
            "Be objective and distinguish between factual events and speculative interpretations."
        )
        user_prompt = (
            f"Analyze news impact for {symbol}.\n\n"
            f"Recent news:\n{news_summary}\n\n"
            f"Macro indicators:\n{macro_summary}\n\n"
            "Write a 3-5 paragraph report covering:\n"
            "1. Key news events and their market impact\n"
            "2. Macro environment (rates, inflation, risk sentiment)\n"
            "3. Regulatory developments\n"
            "4. Upcoming catalysts or risks\n"
            "5. Overall news sentiment: positive/negative/neutral"
        )
        report_text = self._call_llm(system_prompt, user_prompt)
        return AnalystReport(
            analyst=self.name, symbol=symbol, report=report_text,
            metadata={"n_news_items": len(clean_news)},
        )


# ----------------------------------------------------------------------
class SentimentAnalyst(BaseAnalyst):
    """Aggregates social media + news sentiment."""

    name = "sentiment"
    temperature = 0.3  # M20 fix

    def analyze(self, symbol: str, df: pd.DataFrame,
                 context: Optional[dict] = None) -> AnalystReport:
        context = context or {}
        _validate_df(df, require_cols=("close",))  # C19 fix

        social_sentiment = context.get("social_sentiment", {})
        news_sentiment = context.get("news_sentiment", {})
        retail_sentiment = context.get("retail_sentiment", {})

        system_prompt = (
            "You are a sentiment analyst aggregating social media chatter, news sentiment, "
            "and retail positioning into a single sentiment read. "
            "Gauge short-term market mood. Note: extreme retail sentiment is often contrarian."
        )
        user_prompt = (
            f"Aggregate sentiment for {symbol}.\n\n"
            f"Social media sentiment: {social_sentiment}\n"
            f"News sentiment: {news_sentiment}\n"
            f"Retail positioning: {retail_sentiment}\n\n"
            "Write a 2-4 paragraph report covering:\n"
            "1. Overall sentiment read (bullish/bearish/neutral)\n"
            "2. Social media buzz volume and direction\n"
            "3. Contrarian signals (if retail is extremely one-sided)\n"
            "4. Sentiment momentum (improving or deteriorating?)"
        )
        report_text = self._call_llm(system_prompt, user_prompt, max_tokens=400)
        return AnalystReport(
            analyst=self.name, symbol=symbol, report=report_text,
        )


# ----------------------------------------------------------------------
class TechnicalAnalyst(BaseAnalyst):
    """Utilizes technical indicators to detect patterns and forecast."""

    name = "technical"
    temperature = 0.2  # M20 fix — more deterministic for indicator analysis

    def analyze(self, symbol: str, df: pd.DataFrame,
                 context: Optional[dict] = None) -> AnalystReport:
        context = context or {}
        _validate_df(df, require_cols=("close",))  # C19/H8 fix

        close = df["close"]
        indicators: dict[str, Any] = {}
        if len(close) >= 20:
            indicators["sma_20"] = float(sma(close, 20).iloc[-1])
        if len(close) >= 50:
            indicators["sma_50"] = float(sma(close, 50).iloc[-1])
        if len(close) >= 200:
            indicators["sma_200"] = float(sma(close, 200).iloc[-1])
        if len(close) >= 15:
            indicators["rsi_14"] = float(rsi(close, 14).iloc[-1])
        if len(close) >= 26:
            # C19 fix: macd() returns a tuple (macd_line, signal_line, histogram),
            # NOT a dict — the old code crashed here with TypeError.
            macd_line, signal_line, histogram = macd(close)
            indicators["macd"] = float(macd_line.iloc[-1])
            indicators["macd_signal"] = float(signal_line.iloc[-1])
            indicators["macd_histogram"] = float(histogram.iloc[-1])
        if len(close) >= 20:
            # C19 fix: bbands() returns (upper, middle, lower, width) tuple.
            bb_upper, bb_middle, bb_lower, _bb_width = bbands(close)
            indicators["bb_upper"] = float(bb_upper.iloc[-1])
            indicators["bb_lower"] = float(bb_lower.iloc[-1])
        # H13 fix: check ATR series length before .iloc[-1].
        if {"high", "low", "close"}.issubset(df.columns) and len(df) >= 15:
            atr_series = atr(df, 14)
            if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                indicators["atr_14"] = float(atr_series.iloc[-1])
        if "volume" in df.columns:
            if len(df) >= 21:
                # C19 fix: rvol takes a DataFrame, not a Series.
                indicators["rvol_20"] = float(rvol(df, 20).iloc[-1])
            # C19 fix: obv takes a DataFrame, not (close, volume).
            indicators["obv"] = float(obv(df).iloc[-1])

        latest_close = float(close.iloc[-1])
        indicators["latest_close"] = latest_close

        # L10 fix: round indicator values to 4 decimal places for the prompt.
        ind_text = "\n".join(f"- {k}: {v:.4f}" for k, v in indicators.items())

        system_prompt = (
            "You are a technical analyst using price-action and indicators to detect "
            "trading patterns and forecast short-term price movements. "
            "Anchor your analysis in the computed indicators — do not confabulate numbers. "
            "The indicator values below are GROUND TRUTH."
        )
        user_prompt = (
            f"Perform technical analysis for {symbol}.\n\n"
            f"Verified indicator values (ground truth):\n{ind_text}\n\n"
            "Write a 3-5 paragraph report covering:\n"
            "1. Trend analysis (based on SMA alignment + price position)\n"
            "2. Momentum (RSI, MACD histogram)\n"
            "3. Volatility (ATR, Bollinger Band position)\n"
            "4. Volume confirmation (RVOL, OBV trend)\n"
            "5. Key levels to watch (support/resistance from indicators)"
        )
        report_text = self._call_llm(system_prompt, user_prompt)
        return AnalystReport(
            analyst=self.name, symbol=symbol, report=report_text,
            metadata=indicators,
        )


# ----------------------------------------------------------------------
class AnalystTeam:
    """Orchestrates all analysts.

    H2 fix: caches reports keyed by (symbol, last_bar_time) so repeated
    calls within the same bar don't waste LLM tokens.
    M1 fix: raises ValueError if a selected analyst name doesn't exist.
    """

    def __init__(self, llm: Optional[LLMProvider] = None,
                 selected: Optional[list[str]] = None) -> None:
        self.llm = llm or LLMProvider()
        self.analysts: dict[str, BaseAnalyst] = {
            "fundamentals": FundamentalsAnalyst(self.llm),
            "news": NewsAnalyst(self.llm),
            "sentiment": SentimentAnalyst(self.llm),
            "technical": TechnicalAnalyst(self.llm),
        }
        self.selected = selected or list(self.analysts.keys())
        # M1 fix: validate selection.
        invalid = [s for s in self.selected if s not in self.analysts]
        if invalid:
            raise ValueError(
                f"Unknown analyst(s): {invalid}. "
                f"Valid options: {list(self.analysts.keys())}")
        # H2 fix: cache keyed by (symbol, last_bar_time).
        # M6 FIX (Chief AI Architect Audit): add threading.Lock to prevent
        # race condition when _run_analysts_parallel accesses cache from
        # multiple threads concurrently.
        import threading as _threading
        self._cache: dict[tuple[str, str], dict[str, AnalystReport]] = {}
        self._cache_lock = _threading.Lock()

    def run(self, symbol: str, df: pd.DataFrame,
             context: Optional[dict] = None) -> dict[str, AnalystReport]:
        """Run all selected analysts. Returns {analyst_name: AnalystReport}.

        H2 fix: if df has a 'time' column, the cache key includes the
        last bar's time so repeated calls within the same bar return
        cached results instead of re-calling the LLM.

        Critical #2 fix: cache key now includes a hash of the context dict
        so that the same symbol+bar with different context (news, sentiment)
        produces fresh reports instead of returning stale cached results.
        """
        # H2 fix: cache lookup (Critical #2: now context-aware).
        # M6 FIX: use lock for cache access to prevent race condition.
        cache_key = self._cache_key(symbol, df, context)
        with self._cache_lock:
            if cache_key is not None and cache_key in self._cache:
                cached = self._cache[cache_key]
                log.info("AnalystTeam: cache hit for %s @ %s (ctx hash=%s)",
                         symbol, cache_key[1], cache_key[2])
                return cached

        results: dict[str, AnalystReport] = {}
        for name in self.selected:
            try:
                results[name] = self.analysts[name].analyze(symbol, df, context)
                log.info("Analyst %s completed for %s", name, symbol)
            except Exception as e:  # noqa: BLE001
                results[name] = AnalystReport(
                    analyst=name, symbol=symbol, report="",
                    success=False, error=str(e),
                )
                log.warning("Analyst %s failed: %r", name, e)

        # H2 fix: cache store.
        # M6 FIX: use lock for cache store too.
        if cache_key is not None:
            with self._cache_lock:
                self._cache[cache_key] = results
                # Limit cache to last 100 entries to avoid unbounded growth.
                if len(self._cache) > 100:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
        return results

    @staticmethod
    def _cache_key(symbol: str, df: pd.DataFrame,
                    context: Optional[dict] = None) -> Optional[tuple[str, str, str]]:
        """Build a cache key from (symbol, last_bar_time, context_hash).

        Critical #2 fix: the context dict (news, sentiment, macro) is now
        hashed into the cache key. Without this, the same symbol+bar with
        different context would return stale cached reports — a classic
        cache-invalidation bug.

        Returns None if df has no 'time' column (caching disabled).
        """
        if df is None or df.empty or "time" not in df.columns:
            return None
        try:
            bar_time = str(df["time"].iloc[-1])
            # Critical #2: hash the context so different news/sentiment
            # data invalidates the cache.
            ctx_hash = "empty"
            if context:
                import hashlib
                import json
                try:
                    ctx_str = json.dumps(context, sort_keys=True, default=str)
                    ctx_hash = hashlib.md5(ctx_str.encode()).hexdigest()[:8]
                except (TypeError, ValueError):
                    ctx_hash = "unhashable"
            return (symbol, bar_time, ctx_hash)
        except Exception:  # noqa: BLE001
            return None
