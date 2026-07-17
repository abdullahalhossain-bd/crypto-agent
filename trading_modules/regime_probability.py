"""
Regime Probability — soft regime classification
================================================

Instead of a single hard regime label, this module computes a
probability distribution over all possible regimes:

    P(trending_up) = 0.45
    P(trending_down) = 0.10
    P(ranging) = 0.30
    P(high_vol_breakout) = 0.10
    P(low_vol_dead) = 0.05

This is critical for position sizing — when regime is uncertain, reduce size.

Uses a soft-max over regime scores + optional HMM smoothing from v5.7.

Usage:
    from trading_modules.regime_probability import RegimeProbability
    rp = RegimeProbability()
    result = rp.compute(df_m15)
    print(f"Dominant: {result.dominant_regime} ({result.dominant_prob:.1%})")
    print(f"Entropy: {result.entropy:.2f} (high = uncertain)")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegimeProbResult:
    probabilities: dict[str, float]    # regime → probability
    dominant_regime: str
    dominant_prob: float               # 0..1
    entropy: float                     # 0..log(N) — higher = more uncertain
    confidence: float                  # 1 - normalized_entropy
    regime_history: list[str] = field(default_factory=list)  # last N regimes
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "dominant_regime": self.dominant_regime,
            "dominant_prob": round(self.dominant_prob, 3),
            "entropy": round(self.entropy, 3),
            "confidence": round(self.confidence, 3),
            "regime_history": self.regime_history,
            "notes": self.notes,
        }


class RegimeProbability:
    """Soft regime classification with probability distribution.

    Combines:
        - ADX + EMA stack for trend probability
        - ATR ratio for volatility regime
        - Kaufman efficiency for trending vs ranging
        - Optional HMM smoothing (from v5.7 quant_factors)

    Parameters:
        adx_period: ADX lookback (default 14)
        atr_period: ATR lookback (default 14)
        ema_fast: fast EMA (default 20)
        ema_slow: slow EMA (default 50)
        temperature: soft-max temperature (default 1.0; lower = sharper)
        use_hmm: if True, smooth probabilities with HMM (default False)
    """

    REGIMES = ["trending_up", "trending_down", "ranging",
               "high_vol_breakout", "low_vol_dead", "choppy"]

    def __init__(
        self, adx_period: int = 14, atr_period: int = 14,
        ema_fast: int = 20, ema_slow: int = 50,
        temperature: float = 1.0, use_hmm: bool = False,
    ) -> None:
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.temperature = temperature
        self.use_hmm = use_hmm

    def compute(self, df: pd.DataFrame) -> RegimeProbResult:
        if df is None or len(df) < max(self.ema_slow + 10, 60):
            return RegimeProbResult(
                probabilities={r: 1.0 / len(self.REGIMES) for r in self.REGIMES},
                dominant_regime="unknown", dominant_prob=0,
                entropy=float(np.log(len(self.REGIMES))),
                confidence=0.0,
                notes=["insufficient data"],
            )
        # ── Compute indicators ────────────────────────────────────
        close = df["close"]
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
        adx = self._adx(df, self.adx_period)
        atr = self._atr(df, self.atr_period)
        atr_baseline = atr.rolling(50).mean()
        eff = self._efficiency_ratio(close, 20)

        adx_now = float(adx.iloc[-1]) if not adx.empty else 0
        atr_now = float(atr.iloc[-1]) if not atr.empty else 0
        atr_base = float(atr_baseline.iloc[-1]) if not atr_baseline.empty and not np.isnan(atr_baseline.iloc[-1]) else atr_now
        atr_ratio = atr_now / atr_base if atr_base > 0 else 1.0
        eff_now = float(eff)
        ema_diff = float(ema_f.iloc[-1] - ema_s.iloc[-1])
        ema_diff_pct = ema_diff / float(ema_s.iloc[-1]) if float(ema_s.iloc[-1]) != 0 else 0

        # ── Compute raw scores for each regime ────────────────────
        scores: dict[str, float] = {}
        # Trending up
        scores["trending_up"] = (
            0.4 * max(0, adx_now / 50) +           # ADX strength
            0.3 * max(0, ema_diff_pct * 100) +     # EMA separation (positive = up)
            0.3 * max(0, eff_now if ema_diff > 0 else 0)
        )
        # Trending down
        scores["trending_down"] = (
            0.4 * max(0, adx_now / 50) +
            0.3 * max(0, -ema_diff_pct * 100) +
            0.3 * max(0, eff_now if ema_diff < 0 else 0)
        )
        # Ranging
        scores["ranging"] = (
            0.5 * max(0, 1 - adx_now / 25) +       # low ADX = ranging
            0.3 * max(0, 1 - abs(ema_diff_pct) * 100) +
            0.2 * max(0, 1 - eff_now)
        )
        # High vol breakout
        scores["high_vol_breakout"] = (
            0.5 * max(0, atr_ratio - 1.5) +
            0.3 * max(0, adx_now / 40) +
            0.2 * max(0, eff_now)
        )
        # Low vol dead
        scores["low_vol_dead"] = (
            0.7 * max(0, 1 - atr_ratio / 0.7) +    # ATR well below baseline
            0.3 * max(0, 1 - adx_now / 20)
        )
        # Choppy
        scores["choppy"] = (
            0.4 * max(0, atr_ratio - 1.2) +        # high ATR
            0.4 * max(0, 1 - eff_now) +             # low efficiency
            0.2 * max(0, 1 - adx_now / 25)
        )

        # ── Soft-max to convert scores → probabilities ────────────
        score_values = np.array([scores[r] for r in self.REGIMES])
        # Scale by temperature
        scaled = score_values / max(self.temperature, 0.01)
        # Soft-max
        exp_scores = np.exp(scaled - scaled.max())  # numerical stability
        probs = exp_scores / exp_scores.sum()
        probabilities = {r: float(p) for r, p in zip(self.REGIMES, probs)}

        # ── Optional HMM smoothing ────────────────────────────────
        if self.use_hmm:
            try:
                from .quant_factors import hmm_regime
                returns = close.pct_change().dropna().to_numpy()
                if len(returns) > 50:
                    hmm = hmm_regime(returns, n_states=3, n_iter=20)
                    # HMM labels: 0/1/2 → bear/sideways/bull
                    label_map = {0: "trending_down", 1: "ranging", 2: "trending_up"}
                    hmm_label = label_map.get(hmm.current_state, "ranging")
                    # Boost the HMM-predicted regime by 20%
                    if hmm_label in probabilities:
                        probabilities[hmm_label] *= 1.2
                        # Renormalize
                        total = sum(probabilities.values())
                        probabilities = {k: v / total for k, v in probabilities.items()}
            except Exception as e:
                logger.warning(f"HMM smoothing failed: {e}")

        # ── Dominant regime + entropy ─────────────────────────────
        dominant = max(probabilities, key=probabilities.get)
        dominant_prob = probabilities[dominant]
        # Entropy: -sum(p × log(p))
        entropy = float(-sum(p * np.log(p + 1e-10) for p in probabilities.values()))
        max_entropy = float(np.log(len(self.REGIMES)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        confidence = 1.0 - normalized_entropy

        # ── Regime history (last 5 dominant regimes) ──────────────
        regime_history: list[str] = []
        # Compute regime for last 5 bars
        for i in range(-5, 0):
            if abs(i) <= len(df):
                sub_df = df.iloc[:i] if i < 0 else df
                if len(sub_df) > 60:
                    try:
                        sub_result = self._quick_regime(sub_df)
                        regime_history.append(sub_result)
                    except Exception:
                        pass

        notes = [
            f"ADX={adx_now:.1f} ATR_ratio={atr_ratio:.2f} eff={eff_now:.2f} ema_diff={ema_diff_pct:.3%}",
            f"dominant={dominant} ({dominant_prob:.1%}) entropy={entropy:.2f} confidence={confidence:.1%}",
        ]
        if self.use_hmm:
            notes.append("HMM smoothing applied")

        return RegimeProbResult(
            probabilities=probabilities,
            dominant_regime=dominant,
            dominant_prob=float(dominant_prob),
            entropy=entropy,
            confidence=float(confidence),
            regime_history=regime_history,
            notes=notes,
        )

    def _quick_regime(self, df: pd.DataFrame) -> str:
        """Quick regime label for a sub-dataframe (for history)."""
        close = df["close"]
        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()
        adx = self._adx(df, self.adx_period)
        adx_now = float(adx.iloc[-1]) if not adx.empty else 0
        ema_diff = float(ema_f.iloc[-1] - ema_s.iloc[-1])
        if adx_now > 25 and ema_diff > 0:
            return "trending_up"
        if adx_now > 25 and ema_diff < 0:
            return "trending_down"
        return "ranging"

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        prev_close = c.shift(1)
        tr = pd.concat([
            (h - l).abs(),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()

    def _adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        up = h.diff(); down = -l.diff()
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        tr = pd.concat([
            (h - l).abs(),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int) -> float:
        if len(close) < window:
            return 0.0
        recent = close.tail(window)
        net = abs(recent.iloc[-1] - recent.iloc[0])
        gross = recent.diff().abs().sum()
        if gross == 0:
            return 0.0
        return float(net / gross)


__all__ = ["RegimeProbability", "RegimeProbResult"]
