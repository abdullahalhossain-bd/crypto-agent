"""validation.shadow_live_engine
=====================================================================
Day 91-95 — Long-duration shadow tracking with outcome matching.

Runs alongside the live system (or in pure shadow mode) and records
EVERY trade decision alongside the actual market outcome over a
configurable horizon (e.g. 5, 15, 30 bars). After enough samples,
we can answer the only question that matters:

    "If we had taken every trade the system suggested, would we
     have made money AFTER costs?"

This is the bridge between backtest and live. Backtests assume fills;
shadow-live MEASURES what would have happened.

Outputs:
  - Per-trade outcome (win/loss/pnl after estimated costs)
  - Aggregate expectancy with confidence interval
  - Signal stability over time (does edge decay during the test?)
  - Per-regime breakdown (does edge persist across regimes?)
  - Per-strategy breakdown (which strategies actually work?)

Statistical rigour:
  - Bootstrap confidence intervals on expectancy
  - t-test against zero (is the edge real or luck?)
  - Sample-size requirement (refuse to call edge "proven" with < N trades)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("validation.shadow_live")


# ----------------------------------------------------------------------
@dataclass
class ShadowOutcome:
    """One shadow decision plus its realised outcome."""
    decision_id: str
    ts: str
    symbol: str
    timeframe: str
    action: str                # BUY / SELL
    strength: float
    entry_price: float
    predicted_lots: float
    ml_confidence: Optional[float] = None
    regime: Optional[str] = None
    strategy: Optional[str] = None
    # Outcome (filled in later when horizon elapses)
    exit_price: Optional[float] = None
    holding_bars: Optional[int] = None
    pnl_pct: Optional[float] = None          # raw pnl as fraction
    pnl_after_costs_pct: Optional[float] = None
    cost_bps: Optional[float] = None
    max_adverse_pct: Optional[float] = None  # max drawdown during hold
    max_favourable_pct: Optional[float] = None
    outcome_status: str = "pending"          # pending / realised / expired
    realised_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class ShadowLiveEngine:
    """Long-duration shadow tracking with outcome matching."""

    def __init__(
        self,
        storage_path: str = "data/shadow_outcomes.jsonl",
        default_horizon_bars: int = 5,
        cost_bps_per_trade: float = 7.0,    # round-trip commission + slippage estimate
        min_sample_size: int = 100,          # refuse to call edge "proven" below this
        bootstrap_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        self.storage_path = storage_path
        self.default_horizon = int(default_horizon_bars)
        self.cost_bps = float(cost_bps_per_trade)
        self.min_samples = int(min_sample_size)
        self.bootstrap_iters = int(bootstrap_iterations)
        self.confidence = float(confidence_level)
        # Pending decisions keyed by (symbol, decision_id)
        self._pending: dict[str, ShadowOutcome] = {}
        # Realised outcomes kept in memory for fast analysis
        self._realised: list[ShadowOutcome] = []
        self._load()

    # ----------------------------------------------------------------
    # Record a new shadow decision
    # ----------------------------------------------------------------
    def record_decision(
        self,
        symbol: str,
        timeframe: str,
        action: str,
        strength: float,
        entry_price: float,
        predicted_lots: float = 0.0,
        ml_confidence: Optional[float] = None,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
        horizon_bars: Optional[int] = None,
    ) -> ShadowOutcome:
        import uuid
        oid = uuid.uuid4().hex[:16]
        # Critical #2 fix: store the decision timestamp so match_outcomes
        # can find the correct bar position instead of using the last
        # `holding_bars` bars (which introduces severe lookahead bias).
        now_ts = datetime.now(tz=timezone.utc).isoformat()
        outcome = ShadowOutcome(
            decision_id=oid,
            ts=now_ts,
            symbol=symbol, timeframe=timeframe, action=action,
            strength=float(strength), entry_price=float(entry_price),
            predicted_lots=float(predicted_lots),
            ml_confidence=ml_confidence, regime=regime, strategy=strategy,
            holding_bars=horizon_bars or self.default_horizon,
        )
        # Minor #10 fix: check for duplicate decision_id.
        if oid in self._pending:
            log.warning("shadow_live: duplicate decision_id %s — overwriting", oid)
        self._pending[oid] = outcome
        self._persist(outcome)
        log.debug("shadow decision recorded: %s %s %s @ %.5f",
                  action, symbol, oid, entry_price)
        return outcome

    # ----------------------------------------------------------------
    # Match pending decisions against actual market data
    # ----------------------------------------------------------------
    def match_outcomes(self, bars_by_symbol: dict[str, list[dict[str, Any]]]) -> int:
        """Walk through `bars_by_symbol` and resolve pending decisions.

        Each bar dict must have: {time, open, high, low, close}.
        Returns the number of outcomes resolved.

        Critical #2 fix: the old code used `bars[-horizon:]` for ALL
        pending decisions — meaning every decision was evaluated on the
        same final window regardless of when it was made. This is severe
        lookahead bias. Now we find the bar closest to the decision's
        timestamp and use the SUBSEQUENT `holding_bars` bars for outcome
        evaluation.
        """
        resolved = 0
        for oid, outcome in list(self._pending.items()):
            bars = bars_by_symbol.get(outcome.symbol)
            if not bars:
                continue
            n = len(bars)
            horizon = outcome.holding_bars or self.default_horizon
            if n < horizon + 1:
                continue
            entry = outcome.entry_price
            if entry <= 0:
                continue

            # Critical #2 fix: find the bar index closest to the decision
            # timestamp, then use the SUBSEQUENT `horizon` bars.
            decision_ts = outcome.ts
            start_idx = 0
            try:
                # Parse the decision timestamp
                dec_dt = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
                # Find the first bar whose time is >= decision time
                for i, bar in enumerate(bars):
                    bar_time = bar.get("time")
                    if bar_time is None:
                        continue
                    # Handle both string and datetime bar times
                    if isinstance(bar_time, str):
                        bar_dt = datetime.fromisoformat(bar_time.replace("Z", "+00:00"))
                    elif hasattr(bar_time, "isoformat"):
                        bar_dt = bar_time
                    else:
                        continue
                    if bar_dt >= dec_dt:
                        start_idx = i
                        break
                else:
                    # Decision is after all bars — not enough data yet
                    continue
            except (ValueError, TypeError):
                # Fallback: if timestamp parsing fails, use bars after
                # the first half (rough heuristic, better than last N)
                start_idx = n // 2

            # Check we have enough bars after the decision bar
            if start_idx + horizon >= n:
                continue  # Not enough future bars yet

            # Use the bars AFTER the decision bar (no lookahead)
            window = bars[start_idx + 1: start_idx + 1 + horizon]
            if not window:
                continue
            exit_bar = window[-1]
            exit_price = float(exit_bar["close"])
            # PnL
            if outcome.action == "BUY":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry
            # Max adverse / favourable
            if outcome.action == "BUY":
                adverse = (min(float(b["low"]) for b in window) - entry) / entry
                favourable = (max(float(b["high"]) for b in window) - entry) / entry
            else:
                adverse = (entry - max(float(b["high"]) for b in window)) / entry
                favourable = (entry - min(float(b["low"]) for b in window)) / entry
            # Costs (round-trip bps)
            cost_pct = self.cost_bps / 10_000.0
            pnl_after_costs = pnl_pct - cost_pct
            # Update outcome
            outcome.exit_price = exit_price
            outcome.pnl_pct = float(pnl_pct)
            outcome.pnl_after_costs_pct = float(pnl_after_costs)
            outcome.cost_bps = self.cost_bps
            outcome.max_adverse_pct = float(adverse)
            outcome.max_favourable_pct = float(favourable)
            outcome.outcome_status = "realised"
            outcome.realised_at = datetime.now(tz=timezone.utc).isoformat()
            self._persist(outcome)
            self._realised.append(outcome)
            del self._pending[oid]
            resolved += 1
        if resolved:
            log.info("shadow outcomes resolved: %d (pending=%d, realised=%d)",
                     resolved, len(self._pending), len(self._realised))
        return resolved

    # ----------------------------------------------------------------
    # Statistical analysis
    # ----------------------------------------------------------------
    def analyse(self) -> dict[str, Any]:
        """Run statistical analysis on realised outcomes."""
        if not self._realised:
            return {
                "n_samples": 0,
                "status": "insufficient_data",
                "message": f"need >= {self.min_samples} realised outcomes",
            }
        pnls = np.array([o.pnl_after_costs_pct or 0.0 for o in self._realised])
        n = len(pnls)
        mean = float(pnls.mean())
        std = float(pnls.std()) if n > 1 else 0.0
        win_rate = float((pnls > 0).mean())

        # Bootstrap confidence interval on mean
        ci_low, ci_high = self._bootstrap_ci(pnls)

        # t-test against zero (is edge statistically real?)
        t_stat, p_value = self._t_test(pnls)

        # Per-strategy breakdown
        per_strategy = self._per_strategy_breakdown()
        per_regime = self._per_regime_breakdown()
        per_symbol = self._per_symbol_breakdown()

        # Edge stability over time (compare first half vs second half)
        stability = self._stability_analysis(pnls)

        # Verdict
        edge_proven = (
            n >= self.min_samples
            and mean > 0
            and ci_low > 0                      # 95% CI excludes zero
            and p_value < (1 - self.confidence) # significant
            and stability["decayed"] is False
        )

        return {
            "n_samples": int(n),
            "min_required": self.min_samples,
            "expectancy_pct": mean,
            "std_pct": std,
            "win_rate": win_rate,
            "profit_factor": self._profit_factor(pnls),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level": self.confidence,
            "t_statistic": t_stat,
            "p_value": p_value,
            "edge_statistically_significant": p_value < (1 - self.confidence),
            "ci_excludes_zero": ci_low > 0,
            "edge_proven": edge_proven,
            "stability": stability,
            "per_strategy": per_strategy,
            "per_regime": per_regime,
            "per_symbol": per_symbol,
            "status": "proven" if edge_proven else (
                "suggestive" if mean > 0 and n >= self.min_samples // 2
                else "insufficient_data"
            ),
        }

    # ----------------------------------------------------------------
    # Statistical helpers
    # ----------------------------------------------------------------
    def _bootstrap_ci(self, pnls: np.ndarray) -> tuple[float, float]:
        if len(pnls) < 2:
            return (0.0, 0.0)
        rng = np.random.default_rng(42)
        means = np.empty(self.bootstrap_iters)
        for i in range(self.bootstrap_iters):
            sample = rng.choice(pnls, size=len(pnls), replace=True)
            means[i] = sample.mean()
        alpha = 1.0 - self.confidence
        ci_low = float(np.percentile(means, 100 * alpha / 2))
        ci_high = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return ci_low, ci_high

    @staticmethod
    def _t_test(pnls: np.ndarray) -> tuple[float, float]:
        if len(pnls) < 2:
            return (0.0, 1.0)
        from scipy import stats
        t, p = stats.ttest_1samp(pnls, 0.0)
        return float(t), float(p)

    @staticmethod
    def _profit_factor(pnls: np.ndarray) -> float:
        gross_w = float(pnls[pnls > 0].sum())
        gross_l = float(-pnls[pnls < 0].sum())
        if gross_l <= 0:
            return float("inf") if gross_w > 0 else 0.0
        return gross_w / gross_l

    def _stability_analysis(self, pnls: np.ndarray) -> dict[str, Any]:
        """Compare first half vs second half to detect decay."""
        if len(pnls) < 20:
            return {"decayed": False, "reason": "too few samples"}
        mid = len(pnls) // 2
        first_half = pnls[:mid]
        second_half = pnls[mid:]
        m1, m2 = float(first_half.mean()), float(second_half.mean())
        decayed = m2 < m1 * 0.5 and m1 > 0   # edge more than halved
        return {
            "first_half_expectancy": m1,
            "second_half_expectancy": m2,
            "decay_pct": float((m1 - m2) / m1) if m1 != 0 else 0.0,
            "decayed": bool(decayed),
        }

    def _per_strategy_breakdown(self) -> dict[str, dict[str, Any]]:
        out: dict[str, list[float]] = {}
        for o in self._realised:
            if o.strategy is None:
                continue
            out.setdefault(o.strategy, []).append(o.pnl_after_costs_pct or 0.0)
        return {s: self._summarise(p) for s, p in out.items()}

    def _per_regime_breakdown(self) -> dict[str, dict[str, Any]]:
        out: dict[str, list[float]] = {}
        for o in self._realised:
            if o.regime is None:
                continue
            out.setdefault(o.regime, []).append(o.pnl_after_costs_pct or 0.0)
        return {r: self._summarise(p) for r, p in out.items()}

    def _per_symbol_breakdown(self) -> dict[str, dict[str, Any]]:
        out: dict[str, list[float]] = {}
        for o in self._realised:
            out.setdefault(o.symbol, []).append(o.pnl_after_costs_pct or 0.0)
        return {s: self._summarise(p) for s, p in out.items()}

    @staticmethod
    def _summarise(pnls: list[float]) -> dict[str, Any]:
        if not pnls:
            return {}
        arr = np.array(pnls)
        return {
            "n": int(len(arr)),
            "expectancy": float(arr.mean()),
            "std": float(arr.std()) if len(arr) > 1 else 0.0,
            "win_rate": float((arr > 0).mean()),
            "profit_factor": (
                float(arr[arr > 0].sum() / -arr[arr < 0].sum())
                if (arr < 0).any() and arr[arr < 0].sum() != 0
                else float("inf") if (arr > 0).any() else 0.0
            ),
        }

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------
    def _persist(self, outcome: ShadowOutcome) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(outcome.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("shadow outcome persist failed: %r", e)

    def _load(self) -> None:
        if not os.path.isfile(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        outcome = ShadowOutcome(**d)
                        if outcome.outcome_status == "pending":
                            self._pending[outcome.decision_id] = outcome
                        else:
                            self._realised.append(outcome)
                    except Exception:  # noqa: BLE001
                        continue
            log.info("shadow outcomes loaded: %d pending, %d realised",
                     len(self._pending), len(self._realised))
        except Exception as e:  # noqa: BLE001
            log.warning("shadow outcomes load failed: %r", e)

    # ----------------------------------------------------------------
    @property
    def n_pending(self) -> int:
        return len(self._pending)

    @property
    def n_realised(self) -> int:
        return len(self._realised)
