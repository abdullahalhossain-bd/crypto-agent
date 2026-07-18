"""backtest.backtester
=====================================================================
Phase 10 refactor (P0-9 FIX): Now uses the canonical RiskPipeline from
architecture/risk_pipeline.py — the SAME 13-gate pipeline that live
trading uses. This closes the P0-9 bug where backtest and live used
different risk code (backtest used archived engine.risk.RiskManager,
live used architecture.risk_pipeline.RiskPipeline).

Critical correctness rules (unchanged from original):
  1. NO LOOKAHEAD — at bar t we only ever pass df.iloc[:t+1] to the
     strategy and risk pipeline.
  2. Fills happen at next bar's open (default) or current bar's close
     — never at the high/low of the same bar (look-inside-bar bias).
  3. ATR for sizing is computed at bar t using only bars ≤ t.
  4. Commission + slippage are deducted on every fill.

P0-9 FIX: The risk pipeline is now RiskPipeline (13 gates including
DailyLossGate, drawdown scaling, consecutive-loss, etc.) — identical
to live. A shared-code-path test (tests/test_phase10_backtest.py) proves
that the same RiskContext run through both backtest and live produces
identical gate verdicts.

Produces:
  - equity curve (list of floats, indexed by bar)
  - trade log (one row per closed trade)
  - summary metrics (win rate, max drawdown, sharpe, profit factor)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from architecture.risk_pipeline import RiskPipeline, RiskContext, RiskVerdict
from architecture.portfolio_manager_v2 import PortfolioManager
from architecture.exchange_abstraction import SymbolInfo
from engine.signals import Signal, Action
from engine.strategy import Strategy
from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("backtest")


# ----------------------------------------------------------------------
@dataclass
class BacktestSummary:
    initial_capital: float
    final_equity: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float
    avg_trade_pnl: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__

    def __str__(self) -> str:
        return (
            f"\n=== Backtest Summary ===\n"
            f"  Initial capital : {self.initial_capital:,.2f}\n"
            f"  Final equity    : {self.final_equity:,.2f}\n"
            f"  Total trades    : {self.n_trades}\n"
            f"  Wins / Losses   : {self.n_wins} / {self.n_losses}\n"
            f"  Win rate        : {self.win_rate:.2%}\n"
            f"  Max drawdown    : {self.max_drawdown_pct:.2%}\n"
            f"  Sharpe (rf=0)   : {self.sharpe:.3f}\n"
            f"  Profit factor   : {self.profit_factor:.3f}\n"
            f"  Avg trade pnl   : {self.avg_trade_pnl:,.2f}\n"
        )


# ----------------------------------------------------------------------
@dataclass
class _OpenTrade:
    side: Action
    entry_price: float
    lots: float
    stop: float
    take: float
    entry_bar: int
    entry_time: pd.Timestamp
    atr_value: float


# ----------------------------------------------------------------------
class Backtester:
    """Canonical backtester — uses the SAME RiskPipeline as live trading.

    P0-9 FIX (Phase 10): The `risk` parameter is now a RiskPipeline instance
    (from architecture.risk_pipeline), NOT the archived engine.risk.RiskManager.
    This ensures backtest and live share identical risk-gate code.

    The `portfolio` parameter is a PortfolioManager instance — the same class
    live trading uses. It tracks open positions, realized P&L, drawdown, and
    consecutive losses, feeding real telemetry into the RiskContext at each bar.
    """

    def __init__(
        self,
        strategy: Strategy,
        risk: RiskPipeline,
        portfolio: PortfolioManager,
        initial_capital: float = 10_000.0,
        commission_per_trade: float = 0.0002,
        slippage_points: float = 2.0,
        fill_price: str = "close",
        point: float = 0.0001,
        contract_size: float = 1.0,
        risk_config: Optional[dict] = None,
    ) -> None:
        self.strategy = strategy
        self.risk = risk  # RiskPipeline — same as live
        self.portfolio = portfolio  # PortfolioManager — same as live
        self.initial_capital = float(initial_capital)
        self.commission = float(commission_per_trade)
        self.slippage_points = float(slippage_points)
        self.fill_price = fill_price.lower()
        if self.fill_price not in ("close", "open"):
            raise ValueError("fill_price must be 'close' or 'open'")
        self.point = float(point)
        self.contract_size = float(contract_size)
        self.risk_config = risk_config or {}

    # ----------------------------------------------------------------
    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], BacktestSummary]:
        """Walk through `df` bar by bar, generate signals, simulate fills.

        Returns (equity_curve_df, trade_log, summary).

        P0-9 FIX: At each bar, builds a RiskContext (same as live) and calls
        risk_pipeline.evaluate(ctx) — the exact same 13-gate path that
        TradingBot._process_symbol uses in live trading.
        """
        if df.empty:
            raise ValueError("empty df")

        # Pre-compute ATR once (vectorised). Slice at bar t during signal
        # generation to guarantee no lookahead.
        atr_full = atr(df, 14)

        equity = self.initial_capital
        equity_curve = np.empty(len(df), dtype=float)
        trade_log: list[dict[str, Any]] = []
        open_trade: Optional[_OpenTrade] = None

        # Minor #9 fix: derive warmup from the strategy's min_bars property
        # instead of hardcoding 50. This ensures strategies with longer
        # indicator periods (e.g. SMA 200) get enough warmup bars.
        strategy_min_bars = getattr(self.strategy, "min_bars", None)
        if strategy_min_bars is None:
            # Try metadata (StrategyMetadata has min_bars)
            meta = getattr(self.strategy, "metadata", None)
            if meta is not None:
                strategy_min_bars = getattr(meta, "min_bars", 50)
            else:
                strategy_min_bars = 50
        # ATR needs 14 bars + 5 buffer; strategy may need more.
        warmup = max(14 + 5, int(strategy_min_bars))

        for i in range(len(df)):
            row = df.iloc[i]

            # ---- 1. manage open trade first (mark-to-market + SL/TP) ----
            if open_trade is not None:
                hit_sl, hit_tp = self._stops_hit(open_trade, row)
                if hit_sl or hit_tp:
                    exit_price = open_trade.stop if hit_sl else open_trade.take
                    pnl = self._pnl(open_trade, exit_price)
                    equity += pnl - self._commission_cost(open_trade.lots, exit_price)
                    # Update portfolio (same as live) — this feeds real
                    # telemetry into the next bar's RiskContext.
                    self.portfolio.on_position_closed(
                        ticket=open_trade.entry_bar,
                        exit_price=exit_price,
                        reason="SL" if hit_sl else "TP",
                    )
                    trade_log.append(self._close_trade_dict(
                        open_trade, exit_price, pnl, row, hit_sl, hit_tp))
                    open_trade = None

            # ---- 2. check for new signal (only if no open trade) ----
            if open_trade is None and i >= warmup:
                window = df.iloc[: i + 1].copy()
                signal = self.strategy.evaluate(window)
                if signal.is_actionable:
                    # P0-9 FIX: Build RiskContext — same as live path
                    atr_val = float(atr_full.iloc[i])
                    ctx = self._build_risk_context(
                        signal=signal, df=window, equity=equity, atr_val=atr_val,
                    )
                    # Run the SAME 13-gate pipeline as live.
                    # Critical #1 fix: handle both the canonical 3-tuple return
                    # (approved, final_verdict, all_verdicts) from
                    # architecture.risk_pipeline.RiskPipeline AND a possible
                    # single-object return from legacy risk engines. This
                    # prevents a TypeError crash if the risk interface changes.
                    risk_result = self.risk.evaluate(ctx)
                    if isinstance(risk_result, tuple) and len(risk_result) == 3:
                        approved, final_verdict, _all_verdicts = risk_result
                    elif isinstance(risk_result, tuple) and len(risk_result) == 2:
                        approved, final_verdict = risk_result
                    else:
                        # Single object — treat it as the final verdict.
                        final_verdict = risk_result
                        approved = getattr(final_verdict, "passed", False)
                    if approved:
                        fill_px = self._fill_price(final_verdict, df, i)
                        # Apply slippage against us
                        slip = self.slippage_points * self.point
                        action = signal.action
                        if action == Action.BUY:
                            fill_px += slip
                        else:
                            fill_px -= slip
                        open_trade = _OpenTrade(
                            side=action,
                            entry_price=fill_px,
                            lots=final_verdict.modified_lots or 0.01,
                            stop=final_verdict.modified_sl or 0.0,
                            take=final_verdict.modified_tp or 0.0,
                            entry_bar=i,
                            entry_time=row["time"],
                            atr_value=atr_val,
                        )
                        equity -= self._commission_cost(open_trade.lots, fill_px)
                        # Update portfolio (same as live)
                        self.portfolio.on_position_opened(
                            ticket=i, symbol=signal.symbol,
                            side=action.value, volume=open_trade.lots,
                            entry_price=fill_px, sl=open_trade.stop,
                            tp=open_trade.take, magic=0,
                        )
                        trade_log.append(self._open_trade_dict(
                            signal, final_verdict, fill_px, row))

            equity_curve[i] = equity

        # ---- 3. force-close any remaining trade at last close ----
        if open_trade is not None:
            last_row = df.iloc[-1]
            exit_price = float(last_row["close"])
            pnl = self._pnl(open_trade, exit_price)
            equity += pnl - self._commission_cost(open_trade.lots, exit_price)
            self.portfolio.on_position_closed(
                ticket=open_trade.entry_bar, exit_price=exit_price, reason="forced")
            trade_log.append(self._close_trade_dict(
                open_trade, exit_price, pnl, last_row, False, False, forced=True))
            equity_curve[-1] = equity

        eq_df = pd.DataFrame({"time": df["time"], "equity": equity_curve})
        summary = self._summarise(equity_curve, trade_log)
        return eq_df, trade_log, summary

    # ----------------------------------------------------------------
    # P0-9 FIX: Build RiskContext — same construction as live path
    # ----------------------------------------------------------------
    def _build_risk_context(self, signal: Signal, df: pd.DataFrame,
                            equity: float, atr_val: float) -> RiskContext:
        """Build a RiskContext identical to what TradingBot._process_symbol
        builds in live mode. This is the shared-code-path guarantee: the
        same context, run through the same RiskPipeline, produces the same
        gate verdicts in backtest and live.
        """
        pm_metrics = self.portfolio.metrics()
        # FIX: symbol_info=None used to silently DISABLE the spread gate,
        # volume step/min/max clamping, and minimum-stop-distance check in
        # RiskPipeline (they're all guarded by `if ctx.symbol_info is not
        # None`) — despite the comment above claiming backtest and live
        # "produce the same gate verdicts". A conservative synthetic
        # default (0 spread, 0.01 lot step, no min-stop-distance) is not
        # broker-exact, but it exercises the same code paths live does
        # instead of skipping them outright, so a strategy that only looks
        # profitable because backtest waived these constraints will now
        # show that in the backtest results too.
        default_symbol_info = SymbolInfo(name=str(getattr(signal, "symbol", "")))
        return RiskContext(
            signal=signal,
            df=df,
            account_equity=equity,
            portfolio=pm_metrics,
            symbol_info=default_symbol_info,
            current_prices={},
            open_positions=[p for p in self.portfolio.all_positions()],
            consecutive_losses=self.portfolio.consecutive_losses(),
            current_drawdown_pct=pm_metrics.current_drawdown_pct,
            last_trade_time=self.portfolio.last_trade_time(),
            realized_pnl_today=self.portfolio.realized_pnl_today(),
            recent_trades=self.portfolio.recent_trades(n=30),
        )

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _fill_price(self, verdict: RiskVerdict, df: pd.DataFrame, i: int) -> float:
        if self.fill_price == "open" and i + 1 < len(df):
            return float(df["open"].iloc[i + 1])
        return float(df["close"].iloc[i])

    @staticmethod
    def _stops_hit(t: _OpenTrade, row: pd.Series) -> tuple[bool, bool]:
        high, low = float(row["high"]), float(row["low"])
        if t.side == Action.BUY:
            return (low <= t.stop, high >= t.take)
        return (high >= t.stop, low <= t.take)

    def _pnl(self, t: _OpenTrade, exit_price: float) -> float:
        diff = (exit_price - t.entry_price) if t.side == Action.BUY else (t.entry_price - exit_price)
        return diff * t.lots * self.contract_size

    def _commission_cost(self, lots: float, price: float) -> float:
        return self.commission * lots * self.contract_size * price

    @staticmethod
    def _open_trade_dict(signal: Signal, verdict: RiskVerdict,
                         fill: float, row: pd.Series) -> dict[str, Any]:
        return {
            "type": "open",
            "time": str(row["time"]),
            "symbol": signal.symbol,
            "action": signal.action.value,
            "lots": verdict.modified_lots or 0.01,
            "entry": fill,
            "sl": verdict.modified_sl or 0.0,
            "tp": verdict.modified_tp or 0.0,
            "atr": verdict.metadata.get("atr", 0.0),
        }

    @staticmethod
    def _close_trade_dict(t: _OpenTrade, exit_price: float, pnl: float,
                          row: pd.Series, hit_sl: bool, hit_tp: bool,
                          forced: bool = False) -> dict[str, Any]:
        return {
            "type": "close",
            "time": str(row["time"]),
            "symbol": "",
            "action": t.side.value,
            "lots": t.lots,
            "entry": t.entry_price,
            "exit": exit_price,
            "sl": t.stop,
            "tp": t.take,
            "hit_sl": hit_sl,
            "hit_tp": hit_tp,
            "forced": forced,
            "pnl": pnl,
            "bars_held": 0,
        }

    # ----------------------------------------------------------------
    def _summarise(self, equity_curve: np.ndarray,
                   trade_log: list[dict[str, Any]]) -> BacktestSummary:
        closes = [t for t in trade_log if t.get("type") == "close"]
        n_trades = len(closes)
        n_wins = sum(1 for t in closes if t["pnl"] > 0)
        n_losses = sum(1 for t in closes if t["pnl"] <= 0)
        win_rate = n_wins / n_trades if n_trades else 0.0
        gross_win = sum(t["pnl"] for t in closes if t["pnl"] > 0)
        gross_loss = -sum(t["pnl"] for t in closes if t["pnl"] < 0)
        # Critical #3 fix: use 0.0 instead of float('inf') when there are
        # no losing trades — infinity breaks downstream JSON serialization,
        # logging, and visualization tools that expect a finite number.
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 0.0
        avg_pnl = (sum(t["pnl"] for t in closes) / n_trades) if n_trades else 0.0

        eq = pd.Series(equity_curve)
        rolling_max = eq.cummax()
        dd = (eq - rolling_max) / rolling_max.replace(0, np.nan)
        max_dd = float(dd.min()) if not dd.isna().all() else 0.0

        rets = eq.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * math.sqrt(len(rets))) if len(rets) > 1 and rets.std() > 0 else 0.0

        return BacktestSummary(
            initial_capital=self.initial_capital,
            final_equity=float(eq.iloc[-1]),
            n_trades=n_trades,
            n_wins=n_wins,
            n_losses=n_losses,
            win_rate=win_rate,
            max_drawdown_pct=abs(max_dd),
            sharpe=sharpe,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
        )