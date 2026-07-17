"""engine.strategy_runner
=====================================================================
Day 10 — Strategy runner.

Runs every strategy in the pool against the same OHLCV DataFrame and
returns a `SignalPool` (dict[strategy_name -> Signal]) plus an
aggregated consensus signal.

Aggregation rules:
  - Each strategy's signal contributes its strength, signed by action.
  - The consensus action is the sign of the weighted sum.
  - The consensus strength is the magnitude of the sum, clamped to [0,1].
  - Conflicting weak signals collapse to HOLD.

The runner is intentionally synchronous (the GIL + pandas make
threading pointless for CPU-bound indicator work). Strategies are
isolated: each gets its own copy of the dataframe so a buggy strategy
cannot corrupt the stream for others.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from engine.signals import Action, Signal
from engine.strategies.base import Strategy
from utils.logger import get_logger

log = get_logger("engine.strategy_runner")


# ----------------------------------------------------------------------
@dataclass
class SignalPool:
    """All signals produced for one bar across the strategy pool."""
    symbol: str
    timeframe: str
    bar_time: Optional[Any]
    signals: dict[str, Signal] = field(default_factory=dict)
    # strategies that raised — they get a HOLD signal in the pool
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def actionable(self) -> dict[str, Signal]:
        return {name: s for name, s in self.signals.items()
                if s.is_actionable}

    @property
    def consensus(self) -> Signal:
        """Aggregate the pool into a single consensus signal.

        Vote = Σ strength * sign(action). Consensus action is the sign
        of the vote. Consensus strength is |vote| / N, clamped to [0,1].
        """
        if not self.signals:
            return Signal.hold(self.symbol, self.timeframe, reason="empty pool")
        votes = []
        for s in self.signals.values():
            if s.action == Action.BUY:
                votes.append(s.strength)
            elif s.action == Action.SELL:
                votes.append(-s.strength)
            else:
                votes.append(0.0)
        n = max(1, len(votes))
        net = sum(votes) / n
        if abs(net) < 0.15:
            return Signal.hold(self.symbol, self.timeframe,
                               reason=f"weak consensus net={net:.2f}")
        action = Action.BUY if net > 0 else Action.SELL
        # Use the most recent price/bar_time available
        sample = next(iter(self.signals.values()))
        return Signal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            action=action,
            strength=float(min(1.0, abs(net))),
            price=sample.price,
            bar_time=sample.bar_time,
            meta={"votes": votes, "n_strategies": n,
                  "strategy_consensus": True},
        )


# ----------------------------------------------------------------------
class StrategyRunner:
    """Runs a pool of strategies on the same dataframe."""

    def __init__(self, strategies: list[Strategy],
                 parallel: bool = False,
                 max_workers: int = 4) -> None:
        if not strategies:
            raise ValueError("StrategyRunner needs at least one strategy")
        self.strategies = list(strategies)
        self.parallel = bool(parallel)
        self.max_workers = int(max_workers)
        # Sanity: ensure no duplicate (name, symbol, timeframe)
        seen = set()
        for s in self.strategies:
            key = (s.metadata.name, s.symbol, s.timeframe)
            if key in seen:
                raise ValueError(f"duplicate strategy in pool: {key}")
            seen.add(key)

    # ----------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> SignalPool:
        """Generate a signal from every strategy in the pool."""
        if df.empty:
            raise ValueError("empty df passed to StrategyRunner")
        symbol = self.strategies[0].symbol
        timeframe = self.strategies[0].timeframe
        bar_time = df["time"].iloc[-1] if "time" in df.columns else None

        pool = SignalPool(symbol=symbol, timeframe=timeframe, bar_time=bar_time)

        if self.parallel and len(self.strategies) > 1:
            self._run_parallel(df, pool)
        else:
            self._run_sequential(df, pool)

        log.debug("pool %s/%s — %d signals (%d actionable, %d errors)",
                  symbol, timeframe, len(pool.signals),
                  len(pool.actionable), len(pool.errors))
        return pool

    # ----------------------------------------------------------------
    def _run_sequential(self, df: pd.DataFrame, pool: SignalPool) -> None:
        for strat in self.strategies:
            self._run_one(strat, df, pool)

    def _run_parallel(self, df: pd.DataFrame, pool: SignalPool) -> None:
        # Each strategy gets its own copy so parallel mutation is safe.
        copies = [df.copy() for _ in self.strategies]
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(strat.generate_signal, c): strat
                for strat, c in zip(self.strategies, copies)
            }
            for fut in as_completed(futures):
                strat = futures[fut]
                try:
                    pool.signals[strat.metadata.name] = fut.result()
                except (KeyboardInterrupt, SystemExit):
                    raise  # C9 fix: propagate shutdown signals
                except Exception as e:  # noqa: BLE001
                    pool.errors[strat.metadata.name] = repr(e)
                    pool.signals[strat.metadata.name] = Signal.hold(
                        strat.symbol, strat.timeframe, reason=f"error:{e!r}"
                    )

    @staticmethod
    def _run_one(strat: Strategy, df: pd.DataFrame, pool: SignalPool) -> None:
        # C9 fix: re-raise KeyboardInterrupt / SystemExit so shutdown
        # signals propagate cleanly instead of being swallowed by the
        # broad Exception handler below.
        try:
            sig = strat.generate_signal(df.copy())
            if not isinstance(sig, Signal):
                raise TypeError(f"strategy {strat.metadata.name} returned {type(sig)}")
            pool.signals[strat.metadata.name] = sig
        except (KeyboardInterrupt, SystemExit):
            raise  # C9 fix: don't swallow shutdown signals
        except Exception as e:  # noqa: BLE001
            pool.errors[strat.metadata.name] = repr(e)
            pool.signals[strat.metadata.name] = Signal.hold(
                strat.symbol, strat.timeframe, reason=f"error:{e!r}"
            )
            log.warning("strategy %s raised: %r", strat.metadata.name, e)
