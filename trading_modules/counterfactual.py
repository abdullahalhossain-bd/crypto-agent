"""
Counterfactual AI — What-If Analysis
=====================================

Answers: "What would have happened if we didn't take this trade?"
         "What if we used a different stop loss?"
         "What if we entered 1 bar later?"

Methods:
  1. Trade counterfactual: simulate alternative scenarios
  2. Feature counterfactual: "What if RSI was 35 instead of 28?"
  3. Strategy counterfactual: "What if we used mean reversion instead?"

Usage:
    from trading_modules.counterfactual import CounterfactualAnalyzer

    cf = CounterfactualAnalyzer()

    # What if we didn't take this trade?
    result = cf.analyze_trade_counterfactual(
        entry_price=65000,
        stop_loss=63500,
        take_profit=68000,
        df=df,  # OHLCV data after entry
    )
    # → {"actual_pnl": 300, "counterfactual_pnl": 0, "opportunity_cost": 300}

    # What if SL was tighter?
    result = cf.analyze_stop_loss_alternative(
        entry=65000, original_sl=63500, alternative_sl=64200, df=df
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    scenario: str
    actual_outcome: float = 0.0
    counterfactual_outcome: float = 0.0
    difference: float = 0.0
    alternative_params: dict = field(default_factory=dict)
    interpretation: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "actual": round(self.actual_outcome, 4),
            "counterfactual": round(self.counterfactual_outcome, 4),
            "difference": round(self.difference, 4),
            "interpretation": self.interpretation,
        }


class CounterfactualAnalyzer:
    """
    What-if analysis for trading decisions.

    Simulates alternative scenarios to answer:
      - Was this trade worth taking? (vs doing nothing)
      - Was the stop loss optimal? (tighter vs wider)
      - Was the entry timing optimal? (earlier vs later)
      - Was the position size optimal? (bigger vs smaller)
    """

    def analyze_trade_counterfactual(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        df: pd.DataFrame,
        direction: str = "long",
        position_size: float = 1.0,
    ) -> CounterfactualResult:
        """
        What would have happened if we DIDN'T take this trade?

        Compare actual trade outcome vs doing nothing (0 PnL).
        """
        # Simulate the actual trade
        actual_pnl = self._simulate_trade(
            entry_price, stop_loss, take_profit, df, direction, position_size
        )

        # Counterfactual: no trade = 0 PnL
        cf_pnl = 0.0

        result = CounterfactualResult(
            scenario="no_trade",
            actual_outcome=actual_pnl,
            counterfactual_outcome=cf_pnl,
            difference=actual_pnl - cf_pnl,
        )

        if actual_pnl > 0:
            result.interpretation = f"Trade was profitable (+{actual_pnl:.2f}) — taking it was correct"
        elif actual_pnl < 0:
            result.interpretation = f"Trade was losing ({actual_pnl:.2f}) — not trading would have been better"
        else:
            result.interpretation = "Trade broke even — no difference"

        return result

    def analyze_stop_loss_alternative(
        self,
        entry_price: float,
        original_sl: float,
        alternative_sl: float,
        take_profit: float,
        df: pd.DataFrame,
        direction: str = "long",
        position_size: float = 1.0,
    ) -> CounterfactualResult:
        """
        What if we used a different stop loss?
        """
        original_pnl = self._simulate_trade(
            entry_price, original_sl, take_profit, df, direction, position_size
        )

        alternative_pnl = self._simulate_trade(
            entry_price, alternative_sl, take_profit, df, direction, position_size
        )

        result = CounterfactualResult(
            scenario="alternative_stop_loss",
            actual_outcome=original_pnl,
            counterfactual_outcome=alternative_pnl,
            difference=alternative_pnl - original_pnl,
            alternative_params={"original_sl": original_sl, "alternative_sl": alternative_sl},
        )

        if alternative_pnl > original_pnl:
            result.interpretation = f"Alternative SL ({alternative_sl}) would have been better by {result.difference:.2f}"
        elif alternative_pnl < original_pnl:
            result.interpretation = f"Original SL ({original_sl}) was better by {-result.difference:.2f}"
        else:
            result.interpretation = "Both SL levels produced same result"

        return result

    def analyze_entry_timing(
        self,
        original_entry_bar: int,
        df: pd.DataFrame,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        direction: str = "long",
        position_size: float = 1.0,
        bars_offset: int = 1,
    ) -> CounterfactualResult:
        """
        What if we entered N bars earlier or later?
        """
        close = df['close']

        # Original entry
        entry_price = float(close.iloc[original_entry_bar])
        original_pnl = self._simulate_trade(
            entry_price,
            entry_price * (1 - stop_loss_pct) if direction == "long" else entry_price * (1 + stop_loss_pct),
            entry_price * (1 + take_profit_pct) if direction == "long" else entry_price * (1 - take_profit_pct),
            df.iloc[original_entry_bar:],
            direction,
            position_size,
        )

        # Alternative entry (N bars later)
        alt_bar = min(original_entry_bar + bars_offset, len(df) - 1)
        alt_entry = float(close.iloc[alt_bar])
        alt_pnl = self._simulate_trade(
            alt_entry,
            alt_entry * (1 - stop_loss_pct) if direction == "long" else alt_entry * (1 + stop_loss_pct),
            alt_entry * (1 + take_profit_pct) if direction == "long" else alt_entry * (1 - take_profit_pct),
            df.iloc[alt_bar:],
            direction,
            position_size,
        )

        result = CounterfactualResult(
            scenario=f"entry_{bars_offset}_bars_later",
            actual_outcome=original_pnl,
            counterfactual_outcome=alt_pnl,
            difference=alt_pnl - original_pnl,
            alternative_params={"original_bar": original_entry_bar, "alternative_bar": alt_bar},
        )

        if alt_pnl > original_pnl:
            result.interpretation = f"Waiting {bars_offset} bars would have been better by {result.difference:.2f}"
        elif alt_pnl < original_pnl:
            result.interpretation = f"Entering at bar {original_entry_bar} was better by {-result.difference:.2f}"
        else:
            result.interpretation = "Entry timing made no difference"

        return result

    def _simulate_trade(
        self,
        entry: float,
        stop_loss: float,
        take_profit: float,
        df: pd.DataFrame,
        direction: str,
        position_size: float,
    ) -> float:
        """Simulate a trade and return PnL."""
        if direction == "long":
            for _, row in df.iterrows():
                if row['low'] <= stop_loss:
                    return (stop_loss - entry) * position_size
                if row['high'] >= take_profit:
                    return (take_profit - entry) * position_size
            # No SL/TP hit — exit at last close
            return (float(df['close'].iloc[-1]) - entry) * position_size
        else:
            for _, row in df.iterrows():
                if row['high'] >= stop_loss:
                    return (entry - stop_loss) * position_size
                if row['low'] <= take_profit:
                    return (entry - take_profit) * position_size
            return (entry - float(df['close'].iloc[-1])) * position_size
