"""utils/indicators/registry.py
=====================================================================
Indicator Registry + Engine (Improvements #16, #17, #18, #19)
=====================================================================
Orchestrates all indicators: dependency management, caching, warm-up
validation, batch execution, and AI-ready output.

Features:
    - IndicatorResult: AI-ready output with name, value, confidence, metadata
    - IndicatorRegistry: registers + looks up indicator functions
    - IndicatorEngine: batch calculate_all() with diagnostics
    - FeatureMetadata: timestamp, lookback, warmup, valid, source, version
    - Multi-timeframe support: same indicator on M1/M5/M15/H1/H4/D1

Usage:
    from utils.indicators.registry import IndicatorEngine

    engine = IndicatorEngine()
    results = engine.calculate_all(df)
    # results = {"ema_20": IndicatorResult(...), "rsi_14": ..., ...}
    print(engine.diagnostics.stats())
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.indicators.caching import IndicatorCache, get_global_cache
from utils.indicators.diagnostics import Diagnostics, get_diagnostics


# ----------------------------------------------------------------------
# IndicatorResult — AI-ready output
# ----------------------------------------------------------------------
@dataclass
class FeatureMetadata:
    """Metadata attached to every indicator result."""
    timestamp: str = ""
    lookback: int = 0
    warmup: int = 0
    valid: bool = False  # False during warmup or on NaN
    source: str = ""     # which module computed it
    latency_ms: float = 0.0
    version: str = "1.0"


@dataclass
class IndicatorResult:
    """AI-ready indicator output.

    Fields:
        name: indicator name (e.g., "rsi_14")
        value: latest scalar value
        normalized: value scaled to [0, 1] or [-1, 1]
        confidence: how reliable is this reading [0, 1]
        valid: False during warmup or on NaN
        timestamp: ISO 8601 UTC
        metadata: FeatureMetadata
    """
    name: str
    value: Any = None
    normalized: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    metadata: FeatureMetadata = field(default_factory=FeatureMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "normalized": self.normalized,
            "confidence": self.confidence,
            "valid": self.valid,
            "timestamp": self.timestamp,
            "metadata": asdict(self.metadata),
        }


# ----------------------------------------------------------------------
# Risk Features (Improvement #18)
# ----------------------------------------------------------------------
@dataclass
class RiskFeatures:
    """Pre-computed risk metrics for the risk engine."""
    atr_stop_distance: float = 0.0
    volatility_risk: float = 0.0      # 0-1, higher = more risky
    trend_strength: float = 0.0       # ADX-like
    pullback_strength: float = 0.0    # how deep is the current pullback
    breakout_strength: float = 0.0    # how strong is the breakout
    exhaustion_score: float = 0.0     # 0-1, higher = more likely reversal


# ----------------------------------------------------------------------
# Indicator Registry
# ----------------------------------------------------------------------
class IndicatorRegistry:
    """Registry of all available indicator functions.

    Each entry: (name, function, category, warmup_bars, default_params, normalizer)
    """

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    def register(self,
                 name: str,
                 fn: Callable,
                 category: str,
                 warmup_bars: int = 50,
                 default_params: Optional[Dict[str, Any]] = None,
                 normalizer: Optional[Callable[[Any], float]] = None) -> None:
        """Register an indicator function."""
        self._registry[name] = {
            "fn": fn,
            "category": category,
            "warmup": warmup_bars,
            "params": default_params or {},
            "normalizer": normalizer,
        }

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._registry.get(name)

    def list_names(self, category: Optional[str] = None) -> List[str]:
        if category is None:
            return sorted(self._registry.keys())
        return sorted(n for n, v in self._registry.items()
                      if v["category"] == category)

    def categories(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for name, v in self._registry.items():
            out.setdefault(v["category"], []).append(name)
        return out

    def count(self) -> int:
        return len(self._registry)


# ----------------------------------------------------------------------
# Indicator Engine — batch execution + diagnostics
# ----------------------------------------------------------------------
class IndicatorEngine:
    """Batch indicator engine with diagnostics + warmup tracking.

    Usage:
        engine = IndicatorEngine()
        results = engine.calculate_all(df)
        # results = {"ema_20": IndicatorResult(...), ...}
        diag = engine.diagnostics.stats()
    """

    def __init__(self,
                 cache: Optional[IndicatorCache] = None,
                 diagnostics: Optional[Diagnostics] = None):
        self.cache = cache or get_global_cache()
        self.diagnostics = diagnostics or get_diagnostics()
        self.registry = IndicatorRegistry()
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default indicators."""
        # Import submodules using importlib to bypass name collisions
        # (utils.indicators.__init__ exports a `momentum` function that
        # shadows the `momentum` module when accessed as attribute)
        import importlib
        trend = importlib.import_module("utils.indicators.trend")
        momentum = importlib.import_module("utils.indicators.momentum")
        volatility = importlib.import_module("utils.indicators.volatility")
        volume = importlib.import_module("utils.indicators.volume")
        smc = importlib.import_module("utils.indicators.smc")
        structure = importlib.import_module("utils.indicators.structure")
        candles = importlib.import_module("utils.indicators.candles")
        statistics = importlib.import_module("utils.indicators.statistics")
        features = importlib.import_module("utils.indicators.features")

        # Trend
        self.registry.register("ema_9", lambda df: trend.ema(df["close"], 9),
                              "trend", 9)
        self.registry.register("ema_21", lambda df: trend.ema(df["close"], 21),
                              "trend", 21)
        self.registry.register("ema_50", lambda df: trend.ema(df["close"], 50),
                              "trend", 50)
        self.registry.register("ema_200", lambda df: trend.ema(df["close"], 200),
                              "trend", 200)
        self.registry.register("sma_20", lambda df: trend.sma(df["close"], 20),
                              "trend", 20)
        self.registry.register("wma_20", lambda df: trend.wma(df["close"], 20),
                              "trend", 20)
        self.registry.register("hull_ma", lambda df: trend.hull_ma(df["close"], 20),
                              "trend", 20)
        self.registry.register("vwma", lambda df: trend.vwma(df),
                              "trend", 20)
        self.registry.register("dema", lambda df: trend.dema(df["close"], 20),
                              "trend", 20)
        self.registry.register("tema", lambda df: trend.tema(df["close"], 20),
                              "trend", 20)
        self.registry.register("zlema", lambda df: trend.zlema(df["close"], 20),
                              "trend", 20)
        self.registry.register("kama", lambda df: trend.kama(df["close"], 10),
                              "trend", 10)
        self.registry.register("alma", lambda df: trend.alma(df["close"], 9),
                              "trend", 9)
        self.registry.register("t3", lambda df: trend.t3_ma(df["close"], 5),
                              "trend", 5)
        self.registry.register("supertrend", lambda df: trend.supertrend(df, 10, 3),
                              "trend", 30)
        self.registry.register("ichimoku", lambda df: trend.ichimoku(df),
                              "trend", 52)
        self.registry.register("adx", lambda df: trend.adx(df, 14),
                              "trend", 28)
        self.registry.register("trend_score", lambda df: trend.trend_score(df["close"], 20),
                              "trend", 40)

        # Momentum
        self.registry.register("rsi_14", lambda df: momentum.rsi(df["close"], 14),
                              "momentum", 14,
                              normalizer=lambda v: (v - 50) / 50)
        self.registry.register("stoch_rsi", lambda df: momentum.stoch_rsi(df["close"]),
                              "momentum", 28,
                              normalizer=lambda v: (v - 50) / 50)
        self.registry.register("stochastic_k", lambda df: momentum.stochastic(df)[0],
                              "momentum", 14,
                              normalizer=lambda v: (v - 50) / 50)
        self.registry.register("macd", lambda df: momentum.macd(df["close"])[0],
                              "momentum", 26)
        self.registry.register("macd_signal", lambda df: momentum.macd(df["close"])[1],
                              "momentum", 35)
        self.registry.register("macd_hist", lambda df: momentum.macd(df["close"])[2],
                              "momentum", 35)
        self.registry.register("ppo", lambda df: momentum.ppo(df["close"])[0],
                              "momentum", 26)
        self.registry.register("roc_10", lambda df: momentum.roc(df["close"], 10),
                              "momentum", 10)
        self.registry.register("momentum", lambda df: momentum.momentum(df["close"], 10),
                              "momentum", 10)
        self.registry.register("trix", lambda df: momentum.trix(df["close"]),
                              "momentum", 45)
        self.registry.register("tsi", lambda df: momentum.tsi(df["close"]),
                              "momentum", 38)
        self.registry.register("cci", lambda df: momentum.cci(df, 20),
                              "momentum", 20,
                              normalizer=lambda v: max(-1, min(1, v / 200)))
        self.registry.register("williams_r", lambda df: momentum.williams_r(df, 14),
                              "momentum", 14,
                              normalizer=lambda v: (v + 50) / 50)
        self.registry.register("ultimate_osc", lambda df: momentum.ultimate_oscillator(df),
                              "momentum", 28,
                              normalizer=lambda v: (v - 50) / 50)
        self.registry.register("awesome_osc", lambda df: momentum.awesome_oscillator(df),
                              "momentum", 34)

        # Volatility
        self.registry.register("atr_14", lambda df: volatility.atr(df, 14),
                              "volatility", 14)
        self.registry.register("atr_pct", lambda df: volatility.atr_pct(df, 14),
                              "volatility", 14)
        self.registry.register("natr", lambda df: volatility.natr(df, 14),
                              "volatility", 14)
        self.registry.register("bb_upper", lambda df: volatility.bollinger_bands(df["close"])[0],
                              "volatility", 20)
        self.registry.register("bb_lower", lambda df: volatility.bollinger_bands(df["close"])[2],
                              "volatility", 20)
        self.registry.register("bb_width", lambda df: volatility.bollinger_width(df["close"]),
                              "volatility", 20)
        self.registry.register("bb_pct_b", lambda df: volatility.bollinger_pct_b(df["close"]),
                              "volatility", 20)
        self.registry.register("keltner_mid", lambda df: volatility.keltner_channel(df)[1],
                              "volatility", 20)
        self.registry.register("donchian_mid", lambda df: volatility.donchian_channel(df)[1],
                              "volatility", 20)
        self.registry.register("chaikin_vol", lambda df: volatility.chaikin_volatility(df),
                              "volatility", 20)
        self.registry.register("stddev", lambda df: volatility.stddev(df["close"], 20),
                              "volatility", 20)
        self.registry.register("hist_vol", lambda df: volatility.historical_volatility(df["close"]),
                              "volatility", 20)

        # Volume
        self.registry.register("obv", lambda df: volume.obv(df),
                              "volume", 1)
        self.registry.register("vwap", lambda df: volume.vwap(df),
                              "volume", 1)
        self.registry.register("cmf", lambda df: volume.cmf(df, 20),
                              "volume", 20,
                              normalizer=lambda v: max(-1, min(1, v * 5)))
        self.registry.register("mfi", lambda df: volume.mfi(df, 14),
                              "momentum", 14,
                              normalizer=lambda v: (v - 50) / 50)
        self.registry.register("adl", lambda df: volume.adl(df),
                              "volume", 1)
        self.registry.register("eom", lambda df: volume.ease_of_movement(df),
                              "volume", 14)
        self.registry.register("vol_osc", lambda df: volume.volume_oscillator(df),
                              "volume", 20)
        self.registry.register("force_index", lambda df: volume.force_index(df),
                              "volume", 13)
        self.registry.register("nvi", lambda df: volume.negative_volume_index(df),
                              "volume", 1)
        self.registry.register("pvi", lambda df: volume.positive_volume_index(df),
                              "volume", 1)
        self.registry.register("rvol", lambda df: volume.rvol(df, 20),
                              "volume", 20,
                              normalizer=lambda v: min(1, v / 3))

        # AI Features
        self.registry.register("ema_distance", lambda df: features.ema_distance(df["close"]),
                              "ai_feature", 20,
                              normalizer=lambda v: max(-1, min(1, v / 5)))
        self.registry.register("price_position", lambda df: features.price_position(df),
                              "ai_feature", 20)
        self.registry.register("rsi_norm", lambda df: features.rsi_normalized(df["close"]),
                              "ai_feature", 14)
        self.registry.register("momentum_score", lambda df: features.momentum_score(df["close"]),
                              "ai_feature", 10)
        self.registry.register("volatility_score", lambda df: features.volatility_score(df),
                              "ai_feature", 100)
        self.registry.register("candle_body_pct", lambda df: features.candle_body_pct(df),
                              "ai_feature", 1)
        self.registry.register("upper_wick_pct", lambda df: features.upper_wick_pct(df),
                              "ai_feature", 1)
        self.registry.register("lower_wick_pct", lambda df: features.lower_wick_pct(df),
                              "ai_feature", 1)
        self.registry.register("daily_range_pct", lambda df: features.daily_range_pct(df),
                              "ai_feature", 1)
        self.registry.register("gap_pct", lambda df: features.gap_pct(df),
                              "ai_feature", 1)

    # ------------------------------------------------------------------
    # Calculate single
    # ------------------------------------------------------------------
    def calculate(self, name: str, df: pd.DataFrame) -> IndicatorResult:
        """Calculate a single indicator by name."""
        entry = self.registry.get(name)
        if entry is None:
            return IndicatorResult(name=name, valid=False,
                                   metadata=FeatureMetadata(valid=False,
                                                           source="not_found"))

        with self.diagnostics.track(name) as metric:
            t0 = time.time()
            try:
                result = entry["fn"](df)
                elapsed_ms = (time.time() - t0) * 1000

                # Extract latest scalar value
                if isinstance(result, pd.Series):
                    value = float(result.iloc[-1]) if len(result) > 0 else None
                elif isinstance(result, (tuple, list)):
                    value = float(result[0].iloc[-1]) if len(result[0]) > 0 else None
                elif isinstance(result, dict):
                    value = result
                else:
                    value = result

                # Compute normalized
                normalized = 0.0
                if entry["normalizer"] is not None and value is not None:
                    try:
                        normalized = float(entry["normalizer"](value))
                    except Exception:
                        normalized = 0.0

                # Validity check
                valid = (value is not None
                         and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value)))
                         and len(df) >= entry["warmup"])

                # Confidence (simple heuristic: based on warmup progress + value validity)
                if not valid:
                    confidence = 0.0
                else:
                    warmup_progress = min(1.0, len(df) / max(entry["warmup"] * 2, 1))
                    confidence = warmup_progress

                meta = FeatureMetadata(
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                    lookback=len(df),
                    warmup=entry["warmup"],
                    valid=valid,
                    source=entry["category"],
                    latency_ms=elapsed_ms,
                    version="1.0",
                )
                return IndicatorResult(
                    name=name, value=value, normalized=normalized,
                    confidence=confidence, valid=valid, metadata=meta,
                )
            except Exception as e:
                self.diagnostics.record_warning(f"{name} failed: {e}")
                return IndicatorResult(name=name, valid=False,
                                       metadata=FeatureMetadata(valid=False,
                                                               source=f"error: {e}"))

    # ------------------------------------------------------------------
    # Batch calculate all
    # ------------------------------------------------------------------
    def calculate_all(self, df: pd.DataFrame,
                      categories: Optional[List[str]] = None) -> Dict[str, IndicatorResult]:
        """Calculate all registered indicators. Returns dict of IndicatorResult.

        categories: filter to specific categories (e.g., ["trend", "momentum"])
        """
        results: Dict[str, IndicatorResult] = {}
        names = self.registry.list_names()
        if categories:
            # Filter by category
            cat_map = self.registry.categories()
            allowed = set()
            for cat in categories:
                allowed.update(cat_map.get(cat, []))
            names = [n for n in names if n in allowed]

        for name in names:
            results[name] = self.calculate(name, df)
        return results

    # ------------------------------------------------------------------
    # Multi-timeframe
    # ------------------------------------------------------------------
    def calculate_multi_timeframe(self,
                                  dfs: Dict[str, pd.DataFrame],
                                  indicators: Optional[List[str]] = None) -> Dict[str, Dict[str, IndicatorResult]]:
        """Calculate indicators on multiple timeframes.

        dfs: {"M5": df_m5, "M15": df_m15, "H1": df_h1, ...}
        Returns: {"M5": {"rsi_14": IndicatorResult, ...}, "M15": {...}, ...}
        """
        out: Dict[str, Dict[str, IndicatorResult]] = {}
        for tf, df in dfs.items():
            if df is None or df.empty:
                continue
            if indicators:
                out[tf] = {name: self.calculate(name, df) for name in indicators}
            else:
                out[tf] = self.calculate_all(df)
        return out

    # ------------------------------------------------------------------
    # Risk features
    # ------------------------------------------------------------------
    def calculate_risk_features(self, df: pd.DataFrame) -> RiskFeatures:
        """Compute the 6 risk features for the risk engine."""
        from utils.indicators.volatility import atr, atr_pct
        from utils.indicators.trend import adx, trend_score
        from utils.indicators import features

        try:
            atr_val = float(atr(df, 14).iloc[-1])
        except Exception:
            atr_val = 0.0
        try:
            atr_p = float(atr_pct(df, 14).iloc[-1])
        except Exception:
            atr_p = 0.0
        try:
            adx_val = float(adx(df, 14).iloc[-1])
        except Exception:
            adx_val = 0.0
        try:
            ts = float(trend_score(df["close"], 20).iloc[-1])
        except Exception:
            ts = 0.0

        # Pullback strength: distance from recent high
        try:
            recent_high = df["high"].rolling(20).max().iloc[-1]
            close = float(df["close"].iloc[-1])
            pullback = (recent_high - close) / max(recent_high, 1e-10)
        except Exception:
            pullback = 0.0

        # Breakout strength: how far above recent high
        try:
            recent_high_50 = df["high"].rolling(50).max().iloc[-1]
            breakout = (close - recent_high_50) / max(recent_high_50, 1e-10)
            breakout = max(0, breakout)
        except Exception:
            breakout = 0.0

        # Exhaustion: RSI extreme + long trend
        try:
            from utils.indicators.momentum import rsi
            rsi_val = float(rsi(df["close"], 14).iloc[-1])
            exhaustion = max(0, (rsi_val - 70) / 30) if rsi_val > 70 else \
                        max(0, (30 - rsi_val) / 30) if rsi_val < 30 else 0
        except Exception:
            exhaustion = 0.0

        return RiskFeatures(
            atr_stop_distance=atr_val * 1.5,
            volatility_risk=min(1.0, atr_p / 0.05),
            trend_strength=adx_val / 100,
            pullback_strength=min(1.0, pullback * 20),
            breakout_strength=min(1.0, breakout * 20),
            exhaustion_score=exhaustion,
        )

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        cats = self.registry.categories()
        return {
            "total_indicators": self.registry.count(),
            "categories": {c: len(v) for c, v in cats.items()},
            "diagnostics": self.diagnostics.stats(),
            "cache_size": self.cache.size(),
            "cache_hit_rate": self.cache.stats.hit_rate,
        }
