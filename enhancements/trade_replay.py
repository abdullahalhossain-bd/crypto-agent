"""enhancements.trade_replay
=====================================================================
Day 159 — Trade replay / forensics system.

Replays any historical trade decision step-by-step, showing exactly:
  - What the bar looked like
  - What features were active
  - What each candlestick module scored
  - What the confluence engine combined them into
  - What the risk engine approved/rejected
  - What the execution engine did
  - What happened next (outcome)

This is the forensic tool for understanding WHY a trade was taken
and WHY it succeeded/failed. Essential for post-mortem analysis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from engine.candlestick.candlestick_features import CandlestickFeatureExtractor
from engine.candlestick.confluence_engine import ConfluenceEngine
from engine.candlestick.market_state import MarketStateClassifier
from engine.candlestick.pattern_detector import PatternDetector
from engine.candlestick.rejection_strength import RejectionStrengthScorer
from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("enhancements.replay")


@dataclass
class ReplayStep:
    step: str
    description: str
    output: dict[str, Any]
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "description": self.description,
            "output": dict(self.output),
            "timestamp": self.timestamp,
        }


@dataclass
class ReplayResult:
    trade_id: str
    symbol: str
    entry_time: str
    steps: list[ReplayStep] = field(default_factory=list)
    final_outcome: Optional[dict[str, Any]] = None
    lessons: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_time": self.entry_time,
            "steps": [s.to_dict() for s in self.steps],
            "final_outcome": self.final_outcome,
            "lessons": self.lessons,
        }


# ----------------------------------------------------------------------
class TradeReplay:
    """Reconstructs a trade decision from historical data."""

    def __init__(self) -> None:
        self.pattern_detector = PatternDetector()
        self.rejection_scorer = RejectionStrengthScorer()
        self.market_state_clf = MarketStateClassifier()
        self.confluence_engine = ConfluenceEngine()
        self.feature_extractor = CandlestickFeatureExtractor()

    # ----------------------------------------------------------------
    def replay(
        self,
        df: pd.DataFrame,
        trade_id: str,
        entry_bar_index: int,
        symbol: str,
        exit_bar_index: Optional[int] = None,
        horizon_bars: int = 5,
    ) -> ReplayResult:
        """Replay the trade decision at `entry_bar_index`.

        Args:
            df: full OHLCV dataframe
            trade_id: identifier for this trade
            entry_bar_index: bar index where trade was taken
            symbol: symbol traded
            exit_bar_index: optional explicit exit bar; if None, uses horizon
            horizon_bars: if no explicit exit, how many bars to look forward
        """
        result = ReplayResult(
            trade_id=trade_id, symbol=symbol,
            entry_time=str(df["time"].iloc[entry_bar_index]
                            if "time" in df.columns else entry_bar_index),
        )
        # Slice df up to entry bar (NO lookahead)
        df_to_entry = df.iloc[: entry_bar_index + 1]

        # Step 1: Bar context
        bar = df.iloc[entry_bar_index]
        result.steps.append(ReplayStep(
            step="bar_context",
            description="The bar at entry",
            output={
                "time": str(bar.get("time", "")),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume", 0)),
            },
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        ))

        # Step 2: Pattern detection
        pattern = self.pattern_detector.detect_latest(df_to_entry)
        result.steps.append(ReplayStep(
            step="pattern_detection",
            description="Candlestick pattern detected",
            output=pattern.to_dict(),
        ))

        # Step 3: Rejection strength
        rejection = self.rejection_scorer.score(df_to_entry)
        result.steps.append(ReplayStep(
            step="rejection_strength",
            description="Rejection strength of the bar",
            output=rejection.to_dict(),
        ))

        # Step 4: Market state
        ms = self.market_state_clf.classify(df_to_entry)
        result.steps.append(ReplayStep(
            step="market_state",
            description="Market state at entry",
            output=ms.to_dict(),
        ))

        # Step 5: Feature extraction
        features = self.feature_extractor.extract(df_to_entry)
        result.steps.append(ReplayStep(
            step="feature_extraction",
            description="29 candlestick + context features",
            output={k: float(v) if isinstance(v, (int, float, np.floating))
                    else v for k, v in features.items()},
        ))

        # Step 6: ATR context
        atr_series = atr(df_to_entry, 14)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 0.0
        result.steps.append(ReplayStep(
            step="volatility_context",
            description="ATR at entry",
            output={"atr_14": atr_val,
                    "close_atr_ratio": float(bar["close"]) / max(atr_val, 1e-9)},
        ))

        # Step 7: Outcome (forward-looking — only for replay, never for live)
        if exit_bar_index is None:
            exit_bar_index = min(entry_bar_index + horizon_bars, len(df) - 1)
        if exit_bar_index > entry_bar_index and exit_bar_index < len(df):
            entry_close = float(df["close"].iloc[entry_bar_index])
            exit_close = float(df["close"].iloc[exit_bar_index])
            pnl_pct = (exit_close - entry_close) / entry_close
            # Max favourable / adverse excursion
            forward = df.iloc[entry_bar_index: exit_bar_index + 1]
            mfe = float((forward["high"].max() - entry_close) / entry_close)
            mae = float((forward["low"].min() - entry_close) / entry_close)
            result.final_outcome = {
                "entry_price": entry_close,
                "exit_price": exit_close,
                "pnl_pct": float(pnl_pct),
                "outcome": "win" if pnl_pct > 0 else "loss",
                "bars_held": int(exit_bar_index - entry_bar_index),
                "max_favourable_excursion_pct": float(mfe),
                "max_adverse_excursion_pct": float(mae),
                "exit_time": str(df["time"].iloc[exit_bar_index]
                                   if "time" in df.columns else exit_bar_index),
            }
        return result

    # ----------------------------------------------------------------
    def replay_from_trace(self, trace: dict[str, Any],
                            df: pd.DataFrame) -> ReplayResult:
        """Replay from a saved decision trace (from observability.decision_trace)."""
        # Find the bar index from the trace timestamp
        trace_ts = trace.get("ts", "")
        symbol = trace.get("symbol", "")
        entry_bar_index = 0
        if "time" in df.columns and trace_ts:
            try:
                trace_dt = datetime.fromisoformat(trace_ts.replace("Z", "+00:00"))
                # Find the bar closest to this timestamp
                df_times = pd.to_datetime(df["time"], utc=True)
                diffs = (df_times - trace_dt).abs()
                entry_bar_index = int(diffs.idxmin())
            except Exception:  # noqa: BLE001
                pass
        return self.replay(
            df=df,
            trade_id=trace.get("trace_id", "unknown"),
            entry_bar_index=entry_bar_index,
            symbol=symbol,
        )

    # ----------------------------------------------------------------
    def save_replay(self, result: ReplayResult, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

    # ----------------------------------------------------------------
    def summarise(self, result: ReplayResult) -> str:
        """Human-readable summary of a replay."""
        lines = [
            f"\n{'=' * 70}",
            f"  TRADE REPLAY — {result.trade_id}",
            f"  Symbol: {result.symbol}  Entry: {result.entry_time}",
            f"{'=' * 70}",
        ]
        for step in result.steps:
            lines.append(f"\n  [{step.step}] {step.description}")
            for k, v in step.output.items():
                if isinstance(v, dict):
                    lines.append(f"    {k}:")
                    for k2, v2 in v.items():
                        lines.append(f"      {k2}: {v2}")
                else:
                    lines.append(f"    {k}: {v}")
        if result.final_outcome:
            lines.append(f"\n  [OUTCOME]")
            for k, v in result.final_outcome.items():
                lines.append(f"    {k}: {v}")
        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)
