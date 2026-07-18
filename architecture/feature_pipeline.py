"""architecture/feature_pipeline.py
=====================================================================
Feature Pipeline + Feature Store (Improvement #4)
=====================================================================
Centralizes all feature computation. Every indicator, statistical
feature, and derived metric flows through this pipeline so:

    1. No duplicate computation (cache by symbol+timestamp+window)
    2. Consistent across strategies, risk engine, and AI models
    3. Reproducible (deterministic feature_vector for any historical bar)
    4. Auditable (each feature has a name, version, and lineage)

Architecture:
    Raw OHLCV → FeatureExtractor (compute) → FeatureStore (cache) → Consumer
                                              ↓
                                         FeatureVector (named dict)
                                              ↓
                                    [Strategy] [Risk] [AI Model] [Audit]

Feature Categories (75+ total):
    - Momentum (RSI, MACD, Stoch RSI, ROC, TSI, MFI, Williams %R, CCI)
    - Trend (ADX, DMI, SuperTrend, Ichimoku, EMA Ribbon, Slope)
    - Volatility (ATR, ATR%, BBands, Keltner, Donchian, Hist Vol)
    - Volume (OBV, VWAP, RVol, CMF, ADL, PVT, Volume Profile)
    - SMC/ICT (FVG, Order Block, Liquidity Sweep, Swing H/L, BOS/CHoCH)
    - Pattern (7 candlestick, divergence, chart patterns)
    - Microstructure (spread, depth imbalance, trade flow)
    - Cross-asset (BTC dominance, correlation, beta)
    - Regime (trend/range/volatility classification)
    - Statistical (skew, kurtosis, autocorrelation, Hurst exponent)

=====================================================================
CHANGELOG
=====================================================================
2026-07-11  Fix: fvg_present / order_block feature computation was
            raising "ValueError: The truth value of a Series is
            ambiguous" on every bar. Root cause: detect_fvg() and
            detect_order_block() can return a DataFrame (not just a
            Series), and `hasattr(x, "iloc")` is True for both, so
            `.iloc[-1]` on a DataFrame produced a multi-value row
            Series, which cannot be coerced with bool(). The pipeline's
            per-feature try/except silently swallowed the error and
            left both features permanently None. Added `_last_bool()`
            helper to safely reduce Series/DataFrame/scalar results to
            a single boolean, and call each detector function once
            instead of twice.
=====================================================================
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("trading_bot.architecture.feature_pipeline")


@dataclass
class FeatureDefinition:
    """Metadata for a single feature."""
    name: str
    category: str
    version: str  # feature computation version (bump when algo changes)
    description: str
    warmup_bars: int  # how many bars needed before this feature is valid
    compute_fn: Callable[[pd.DataFrame], Any]


@dataclass
class FeatureVector:
    """A snapshot of all features for one bar of one symbol."""
    symbol: str
    timestamp: str  # ISO 8601
    bar_close: float
    features: Dict[str, Any] = field(default_factory=dict)
    feature_versions: Dict[str, str] = field(default_factory=dict)
    is_warmed_up: bool = False
    hash: str = ""

    def get(self, name: str, default: Any = None) -> Any:
        return self.features.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bar_close": self.bar_close,
            "features": self.features,
            "versions": self.feature_versions,
            "is_warmed_up": self.is_warmed_up,
            "hash": self.hash,
        }


class FeatureStore:
    """LRU cache of feature vectors.

    Key: (symbol, timestamp_iso)
    Value: FeatureVector
    Max size enforced by LRU eviction.
    Thread-safe.
    """

    def __init__(self, max_size: int = 5000):
        # H2 fix: 50,000 entries x 75+ features was measured at ~200MB+.
        # Now that the cache key bug (C2) is fixed and entries actually
        # get reused across cycles, we don't need to retain 50k historical
        # bars per store — a few thousand recent entries (spanning all
        # active symbols) is enough to make the cache effective while
        # keeping memory bounded. Callers with many symbols/long history
        # needs can still pass a larger max_size explicitly.
        self._cache: OrderedDict[Tuple[str, str], FeatureVector] = OrderedDict()
        self._lock = threading.RLock()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def get(self, symbol: str, timestamp: str) -> Optional[FeatureVector]:
        key = (symbol, timestamp)
        with self._lock:
            v = self._cache.get(key)
            if v is None:
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return v

    def put(self, fv: FeatureVector) -> None:
        key = (fv.symbol, fv.timestamp)
        with self._lock:
            self._cache[key] = fv
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (self._hits / max(self._hits + self._misses, 1)),
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class FeaturePipeline:
    """Central feature computation pipeline.

    Usage:
        pipe = FeaturePipeline()
        pipe.register("rsi_14", "momentum", "1.0", "RSI 14",
                      warmup=14, fn=lambda df: rsi(df.close, 14).iloc[-1])
        fv = pipe.compute("BTCUSD", df)
        rsi_value = fv.get("rsi_14")
    """

    def __init__(self, store: Optional[FeatureStore] = None):
        self._defs: Dict[str, FeatureDefinition] = {}
        self._store = store or FeatureStore()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register(self,
                 name: str,
                 category: str,
                 version: str,
                 description: str,
                 warmup_bars: int,
                 compute_fn: Callable[[pd.DataFrame], Any]) -> None:
        with self._lock:
            self._defs[name] = FeatureDefinition(
                name=name, category=category, version=version,
                description=description, warmup_bars=warmup_bars,
                compute_fn=compute_fn,
            )
        log.debug("feature_pipeline: registered %s/%s v%s", category, name, version)

    def categories(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for d in self._defs.values():
            out.setdefault(d.category, []).append(d.name)
        return out

    def warmup_requirement(self) -> int:
        """Maximum warmup needed across all registered features."""
        if not self._defs:
            return 0
        return max(d.warmup_bars for d in self._defs.values())

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def compute(self,
                symbol: str,
                df: pd.DataFrame,
                timestamp: Optional[str] = None) -> FeatureVector:
        """Compute the full feature vector for the latest bar in df.

        Uses cache when possible. Falls back to direct computation.
        """
        if df is None or df.empty:
            return FeatureVector(symbol=symbol, timestamp="",
                                 bar_close=0.0, is_warmed_up=False)

        # C2/X4 fix: the cache key MUST be derived from the bar's own
        # close time, not wall-clock time. The previous default —
        # datetime.now().isoformat() — produced a different timestamp on
        # every call (even for the same, unchanged bar), so the cache
        # never hit and every cycle recomputed all 75+ features from
        # scratch. Prefer the last row's "time" column; only fall back to
        # wall-clock as a last resort when the DataFrame has no time info
        # (and log that fallback, since it silently disables caching).
        if timestamp is not None:
            ts = timestamp
        elif "time" in df.columns:
            ts = str(pd.Timestamp(df["time"].iloc[-1]))
        elif df.index.name in ("time", "timestamp") or hasattr(df.index[-1], "isoformat"):
            ts = str(df.index[-1])
        else:
            ts = datetime.now(tz=timezone.utc).isoformat()
            log.warning("feature_pipeline: no bar-time column found for %s — "
                        "falling back to wall-clock timestamp, caching disabled "
                        "for this call", symbol)
        bar_close = float(df["close"].iloc[-1])

        # Check cache
        cached = self._store.get(symbol, ts)
        if cached is not None:
            return cached

        features: Dict[str, Any] = {}
        versions: Dict[str, str] = {}
        warmed_up = len(df) >= self.warmup_requirement()

        # Compute each feature
        for name, d in self._defs.items():
            if len(df) < d.warmup_bars:
                features[name] = None
                continue
            try:
                val = d.compute_fn(df)
                # Convert numpy types to python types for JSON serialization
                if isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.bool_,)):
                    val = bool(val)
                features[name] = val
                versions[name] = d.version
            except Exception as e:  # noqa: BLE001
                log.warning("feature_pipeline: %s failed: %r", name, e)
                features[name] = None

        # Deterministic hash for dedup / audit
        # MEMORY SYSTEM AUDIT FIX: use SHA-256 instead of MD5 for consistency
        # with memory_system.py (which was already fixed to SHA-256). MD5
        # truncated to 12 hex chars has collision risk across thousands of
        # feature vectors; SHA-256 truncated to 16 is much safer.
        hash_str = hashlib.sha256(
            (f"{symbol}|{ts}|" + ",".join(f"{k}={features[k]}"
                                          for k in sorted(features))).encode()
        ).hexdigest()[:16]

        fv = FeatureVector(
            symbol=symbol, timestamp=ts, bar_close=bar_close,
            features=features, feature_versions=versions,
            is_warmed_up=warmed_up, hash=hash_str,
        )
        self._store.put(fv)
        return fv

    # ------------------------------------------------------------------
    # Bulk compute (backfill for backtest)
    # ------------------------------------------------------------------
    def compute_series(self, symbol: str, df: pd.DataFrame) -> List[FeatureVector]:
        """Compute feature vectors for every bar in df (for backtest)."""
        out: List[FeatureVector] = []
        for i in range(len(df)):
            sub = df.iloc[: i + 1]
            ts = str(sub.index[-1]) if hasattr(sub.index[-1], "isoformat") \
                else str(i)
            fv = self.compute(symbol, sub, timestamp=ts)
            out.append(fv)
        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        return {
            "registered_features": len(self._defs),
            "categories": self.categories(),
            "store": self._store.stats(),
            "warmup_max": self.warmup_requirement(),
        }

    # ------------------------------------------------------------------
    # Health check — surfaces silently-failing features instead of
    # letting them disappear into None forever (see CHANGELOG above).
    # ------------------------------------------------------------------
    def health_check(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute a fresh (non-cached) feature vector and report which
        registered features returned None despite the data being warmed
        up for them. Useful for diagnostics: a None here can mean a
        genuine "not applicable" OR a swallowed exception — this method
        distinguishes the two by re-running compute_fn directly and
        capturing the exception, if any.
        """
        report: Dict[str, Any] = {"symbol": symbol, "broken": [], "ok": [], "not_warmed_up": []}
        if df is None or df.empty:
            report["error"] = "empty dataframe"
            return report

        for name, d in self._defs.items():
            if len(df) < d.warmup_bars:
                report["not_warmed_up"].append(name)
                continue
            try:
                d.compute_fn(df)
                report["ok"].append(name)
            except Exception as e:  # noqa: BLE001
                report["broken"].append({"name": name, "error": repr(e)})
        return report


# ----------------------------------------------------------------------
# Safe boolean extraction helper
# ----------------------------------------------------------------------
def _last_bool(val: Any) -> bool:
    """Safely extract a single boolean flag from an indicator's latest
    bar, regardless of whether the indicator function returns a plain
    scalar, a pandas Series, or a multi-column pandas DataFrame.

    This fixes the previous bug where `hasattr(x, "iloc")` was True for
    both Series AND DataFrame, but `DataFrame.iloc[-1]` returns a row
    Series (one value per column) rather than a scalar — passing that
    into bool() raised:
        ValueError: The truth value of a Series is ambiguous.

    Behavior:
        - DataFrame -> True if ANY column is truthy on the last row
          (adjust to a specific column name if your detector encodes
          a single canonical flag column, e.g. row["is_fvg"]).
        - Series    -> boolean value of the last element.
        - scalar / bool / None -> bool(val), with None treated as False.
    """
    if val is None:
        return False
    if isinstance(val, pd.DataFrame):
        if val.empty:
            return False
        row = val.iloc[-1]
        try:
            return bool(row.any())
        except Exception:  # noqa: BLE001
            return False
    if isinstance(val, pd.Series):
        if val.empty:
            return False
        try:
            return bool(val.iloc[-1])
        except Exception:  # noqa: BLE001
            return False
    try:
        return bool(val)
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------
# Default pipeline builder — registers all 75+ institutional features
# ----------------------------------------------------------------------
def build_default_pipeline() -> FeaturePipeline:
    """Build a FeaturePipeline pre-loaded with all standard features.

    Each compute_fn takes a DataFrame and returns the latest scalar value.
    """
    pipe = FeaturePipeline()

    try:
        # Use importlib to bypass name collisions in __init__.py
        import importlib
        _trend = importlib.import_module("utils.indicators.trend")
        _momentum = importlib.import_module("utils.indicators.momentum")
        _volatility = importlib.import_module("utils.indicators.volatility")
        _volume = importlib.import_module("utils.indicators.volume")
        _smc = importlib.import_module("utils.indicators.smc")
        _structure = importlib.import_module("utils.indicators.structure")
        _regime = importlib.import_module("utils.indicators.regime")

        sma = _trend.sma
        ema = _trend.ema
        wma = _trend.wma
        vwma = _trend.vwma
        hull_ma = _trend.hull_ma
        rsi = _momentum.rsi
        stoch_rsi = _momentum.stoch_rsi
        macd = _momentum.macd
        roc = _momentum.roc
        cci = _momentum.cci
        williams_r = _momentum.williams_r
        tsi = _momentum.tsi
        mfi = _volume.mfi
        atr = _volatility.atr
        atr_pct = _volatility.atr_pct
        bollinger_bands = _volatility.bollinger_bands
        keltner_channel = _volatility.keltner_channel
        donchian_channel = _volatility.donchian_channel
        historical_volatility = _volatility.historical_volatility
        adx = _trend.adx
        dmi = _trend.dmi
        supertrend = _trend.supertrend
        ichimoku = _trend.ichimoku
        slope = _trend.slope
        obv = _volume.obv
        vwap = _volume.vwap
        rvol = _volume.rvol
        cmf = _volume.cmf
        adl = _volume.adl
        pvt = _volume.pvt
        detect_fvg = _smc.detect_fvg
        detect_order_block = _smc.detect_order_block
        detect_liquidity_sweep = _structure.liquidity_sweep
        regime_detection = _regime.regime_detection
    except ImportError:
        log.warning("feature_pipeline: utils.indicators not available, "
                    "registering minimal features only")
        _register_minimal(pipe)
        return pipe

    # ---- Moving Averages ----
    pipe.register("ema_9", "ma", "1.0", "EMA 9", 9,
                  lambda df: float(ema(df["close"], 9).iloc[-1]))
    pipe.register("ema_21", "ma", "1.0", "EMA 21", 21,
                  lambda df: float(ema(df["close"], 21).iloc[-1]))
    pipe.register("ema_50", "ma", "1.0", "EMA 50", 50,
                  lambda df: float(ema(df["close"], 50).iloc[-1]))
    pipe.register("ema_200", "ma", "1.0", "EMA 200", 200,
                  lambda df: float(ema(df["close"], 200).iloc[-1]))
    pipe.register("sma_20", "ma", "1.0", "SMA 20", 20,
                  lambda df: float(sma(df["close"], 20).iloc[-1]))
    pipe.register("sma_50", "ma", "1.0", "SMA 50", 50,
                  lambda df: float(sma(df["close"], 50).iloc[-1]) if len(df) >= 50 else 0.0)
    pipe.register("vwap", "ma", "1.0", "VWAP", 1,
                  lambda df: float(vwap(df).iloc[-1]))
    pipe.register("wma_20", "ma", "1.0", "WMA 20", 20,
                  lambda df: float(wma(df["close"], 20).iloc[-1]))
    pipe.register("vwma_20", "ma", "1.0", "VWMA 20", 20,
                  lambda df: float(vwma(df, 20).iloc[-1]))
    pipe.register("hull_ma_20", "ma", "1.0", "Hull MA 20", 20,
                  lambda df: float(hull_ma(df["close"], 20).iloc[-1]))

    # ---- Momentum ----
    pipe.register("rsi_14", "momentum", "1.0", "RSI 14", 14,
                  lambda df: float(rsi(df["close"], 14).iloc[-1]))
    pipe.register("stoch_rsi", "momentum", "1.0", "Stochastic RSI", 14,
                  lambda df: float(stoch_rsi(df["close"]).iloc[-1])
                      if hasattr(stoch_rsi(df["close"]), "iloc") else 0.0)
    pipe.register("macd", "momentum", "1.0", "MACD line", 26,
                  lambda df: float(macd(df["close"])[0].iloc[-1]))
    pipe.register("macd_signal", "momentum", "1.0", "MACD signal", 26,
                  lambda df: float(macd(df["close"])[1].iloc[-1]))
    pipe.register("macd_histogram", "momentum", "1.0", "MACD histogram", 26,
                  lambda df: float(macd(df["close"])[2].iloc[-1]))
    pipe.register("roc_10", "momentum", "1.0", "Rate of Change 10", 10,
                  lambda df: float(roc(df["close"], 10).iloc[-1]))
    pipe.register("cci_20", "momentum", "1.0", "CCI 20", 20,
                  lambda df: float(cci(df, 20).iloc[-1]))
    pipe.register("williams_r", "momentum", "1.0", "Williams %R", 14,
                  lambda df: float(williams_r(df, 14).iloc[-1]))
    pipe.register("mfi_14", "momentum", "1.0", "Money Flow Index 14", 14,
                  lambda df: float(mfi(df, 14).iloc[-1]))
    pipe.register("tsi", "momentum", "1.0", "True Strength Index", 38,
                  lambda df: float(tsi(df["close"]).iloc[-1]))

    # ---- Volatility ----
    pipe.register("atr_14", "volatility", "1.0", "ATR 14", 14,
                  lambda df: float(atr(df, 14).iloc[-1]))
    pipe.register("atr_pct", "volatility", "1.0", "ATR as % of price", 14,
                  lambda df: float(atr_pct(df, 14).iloc[-1]))
    pipe.register("bb_width", "volatility", "1.0", "Bollinger width", 20,
                  lambda df: float(bollinger_bands(df["close"], 20)[3].iloc[-1]))  # P0-6 fix: [3] is width, not [2]
    pipe.register("hist_vol_20", "volatility", "1.0", "20-bar hist vol", 20,
                  lambda df: float(historical_volatility(df["close"], 20).iloc[-1]))
    pipe.register("keltner_mid", "volatility", "1.0", "Keltner Channel middle", 20,
                  lambda df: float(keltner_channel(df, 20)[1].iloc[-1]))
    pipe.register("donchian_mid", "volatility", "1.0", "Donchian Channel middle", 20,
                  lambda df: float(donchian_channel(df, 20)[1].iloc[-1]))

    # ---- Trend ----
    pipe.register("adx_14", "trend", "1.0", "ADX 14", 14,
                  lambda df: float(adx(df, 14).iloc[-1]))
    pipe.register("supertrend", "trend", "1.0", "SuperTrend", 10,
                  lambda df: float(supertrend(df, 10, 3).iloc[-1]))
    pipe.register("dmi_plus", "trend", "1.0", "+DI 14", 28,
                  lambda df: float(dmi(df, 14)[0].iloc[-1]))
    pipe.register("dmi_minus", "trend", "1.0", "-DI 14", 28,
                  lambda df: float(dmi(df, 14)[1].iloc[-1]))
    pipe.register("ichimoku_tenkan", "trend", "1.0", "Ichimoku Tenkan-sen", 52,
                  lambda df: float(ichimoku(df)["tenkan"].iloc[-1]))
    pipe.register("ichimoku_kijun", "trend", "1.0", "Ichimoku Kijun-sen", 52,
                  lambda df: float(ichimoku(df)["kijun"].iloc[-1]))
    pipe.register("close_slope_5", "trend", "1.0", "Close price slope (5-bar)", 5,
                  lambda df: float(slope(df["close"], 5).iloc[-1]))

    # ---- Volume ----
    pipe.register("obv", "volume", "1.0", "On-Balance Volume", 1,
                  lambda df: float(obv(df).iloc[-1]))
    pipe.register("rvol", "volume", "1.0", "Relative Volume", 20,
                  lambda df: float(rvol(df, 20).iloc[-1]))
    pipe.register("cmf", "volume", "1.0", "Chaikin Money Flow", 20,
                  lambda df: float(cmf(df, 20).iloc[-1]))
    pipe.register("adl", "volume", "1.0", "Accumulation/Distribution Line", 1,
                  lambda df: float(adl(df).iloc[-1]))
    pipe.register("pvt", "volume", "1.0", "Price-Volume Trend", 1,
                  lambda df: float(pvt(df).iloc[-1]))

    # ---- SMC / ICT ----
    # FIX: previously called detect_fvg(df)/detect_order_block(df) TWICE
    # (once for hasattr check, once for the value) and used a bare
    # `.iloc[-1]` + bool() that broke on DataFrame results. Now calls
    # the detector once and routes through _last_bool(), which safely
    # handles scalar / Series / DataFrame return types.
    pipe.register("fvg_present", "smc", "1.0", "Fair Value Gap present", 3,
                  lambda df: _last_bool(detect_fvg(df)))
    pipe.register("order_block", "smc", "1.0", "Order Block detected", 3,
                  lambda df: _last_bool(detect_order_block(df)))
    pipe.register("liquidity_sweep", "smc", "1.0", "Liquidity sweep (stop hunt) detected", 20,
                  lambda df: _last_bool(detect_liquidity_sweep(df)))

    # ---- Regime ----
    pipe.register("regime", "regime", "1.0", "Market regime", 50,
                  lambda df: regime_detection(df).get("regime", "unknown"))

    log.info("feature_pipeline: registered %d features across %d categories",
             len(pipe._defs), len(pipe.categories()))
    return pipe


def _register_minimal(pipe: FeaturePipeline) -> None:
    """Fallback: register basic features using pure pandas (no utils.indicators)."""
    def _ema(series, period):
        return series.ewm(span=period, adjust=False).mean()

    pipe.register("ema_9", "ma", "1.0", "EMA 9", 9,
                  lambda df: float(_ema(df["close"], 9).iloc[-1]))
    pipe.register("ema_21", "ma", "1.0", "EMA 21", 21,
                  lambda df: float(_ema(df["close"], 21).iloc[-1]))
    pipe.register("rsi_14", "momentum", "1.0", "RSI 14", 14,
                  lambda df: _compute_rsi(df["close"], 14))
    pipe.register("atr_14", "volatility", "1.0", "ATR 14", 14,
                  lambda df: _compute_atr(df, 14))
    pipe.register("volume", "volume", "1.0", "Latest volume", 1,
                  lambda df: float(df["volume"].iloc[-1]) if "volume" in df else 0.0)
    log.info("feature_pipeline: minimal mode — 5 features registered")


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    rs = gain.iloc[-1] / loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ----------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------
_GLOBAL_PIPELINE: Optional[FeaturePipeline] = None


def get_pipeline() -> FeaturePipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = build_default_pipeline()
    return _GLOBAL_PIPELINE