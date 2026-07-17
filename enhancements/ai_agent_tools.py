"""enhancements.ai_agent_tools
=====================================================================
Inspired by OpenAlice's tool factory pattern.

Wraps trading operations as AI-agent-callable tools. Each tool is a
thin shell with a Zod-like schema (we use Python type hints + docstrings)
that delegates to the actual trading/risk/data layer.

An AI agent (LLM) can call these tools to:
  - search_bars(query)          → find K-line sources for a symbol
  - calculate_quant(script)     → run TA script (sma, rsi, macd, etc.)
  - snapshot(symbol, as_of)     → point-in-time market snapshot
  - simulate(symbol, entry, exit) → what-if backtest
  - stage_order(symbol, side, lots) → stage a trade (NOT execute)
  - commit_orders(message)      → commit staged orders
  - push_orders()               → execute committed orders (through guards)
  - get_positions()             → current open positions
  - get_account()               → account info
  - search_news(query)          → financial news search

CRITICAL: These tools NEVER bypass the guard pipeline. The AI can
stage and commit, but push() runs through guards + risk engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from enhancements.as_of_snapshot import AsOfSnapshot
from enhancements.sector_rotation import SectorRotationAnalyzer
from enhancements.trade_simulator import TradeSimulator
from enhancements.trading_as_git import (
    TradingGit, Operation, OperationAction,
)
from utils.indicators import (
    sma, ema, rsi, macd, bbands, atr, obv, mfi, vwap, rvol,
    cci, williams_r, roc, zscore, slope, highest, lowest,
)
from utils.logger import get_logger

log = get_logger("enhancements.ai_agent_tools")


@dataclass
class ToolResult:
    tool: str
    success: bool
    result: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }


# ----------------------------------------------------------------------
class AIAgentTools:
    """AI-agent-callable trading + analysis tools."""

    def __init__(
        self,
        data_fetcher: Optional[Callable[[str, str, int], pd.DataFrame]] = None,
        trading_git: Optional[TradingGit] = None,
        positions_getter: Optional[Callable[[], list[dict]]] = None,
        account_getter: Optional[Callable[[], dict]] = None,
        news_searcher: Optional[Callable[[str, int], list[dict]]] = None,
    ) -> None:
        self.data_fetcher = data_fetcher
        self.trading_git = trading_git or TradingGit()
        self.positions_getter = positions_getter
        self.account_getter = account_getter
        self.news_searcher = news_searcher
        self.snapshot = AsOfSnapshot()
        self.simulator = TradeSimulator()
        self.sector_analyzer = SectorRotationAnalyzer()

    # ----------------------------------------------------------------
    # Data / analysis tools
    # ----------------------------------------------------------------
    def search_bars(self, query: str, limit: int = 20) -> ToolResult:
        """Find K-line sources for a symbol."""
        if not self.data_fetcher:
            return ToolResult("search_bars", False,
                                error="no data fetcher configured")
        try:
            # In a real system, this would search across multiple sources
            # Here we just return the query as a "barId"
            return ToolResult("search_bars", True, {
                "candidates": [{"barId": query, "source": "default"}],
                "count": 1,
            })
        except Exception as e:  # noqa: BLE001
            return ToolResult("search_bars", False, error=str(e))

    def calculate_quant(self, script: str, symbol: str = "",
                          timeframe: str = "M15", count: int = 500) -> ToolResult:
        """Run a TA script over K-lines.

        Script is a simple expression like:
            s = bars("BTCUSD", "M15", 500)
            sma(s.close, 50) - sma(s.close, 200)

        For safety, we only support a subset of operations.
        """
        if not self.data_fetcher or not symbol:
            return ToolResult("calculate_quant", False,
                                error="data_fetcher and symbol required")
        try:
            df = self.data_fetcher(symbol, timeframe, count)
            if df.empty:
                return ToolResult("calculate_quant", False, error="no data")
            close = df["close"]
            # Evaluate a safe subset
            result: dict[str, Any] = {}
            # Pre-compute common indicators
            # Major #4 fix: compute ATR once, reuse — was called twice.
            _atr_series = atr(df, 14) if len(df) >= 15 else None
            _atr_val = None
            if _atr_series is not None and len(_atr_series) > 0:
                _last_atr = _atr_series.iloc[-1]
                if not pd.isna(_last_atr):
                    _atr_val = float(_last_atr)
            indicators = {
                "sma_20": float(sma(close, 20).iloc[-1]) if len(close) >= 20 else None,
                "sma_50": float(sma(close, 50).iloc[-1]) if len(close) >= 50 else None,
                "sma_200": float(sma(close, 200).iloc[-1]) if len(close) >= 200 else None,
                "ema_12": float(ema(close, 12).iloc[-1]) if len(close) >= 12 else None,
                "ema_26": float(ema(close, 26).iloc[-1]) if len(close) >= 26 else None,
                "rsi_14": float(rsi(close, 14).iloc[-1]) if len(close) >= 15 else None,
                "atr_14": _atr_val,
            }
            macd_df = macd(close)
            indicators["macd"] = float(macd_df["macd"].iloc[-1])
            indicators["macd_signal"] = float(macd_df["signal"].iloc[-1])
            indicators["macd_histogram"] = float(macd_df["histogram"].iloc[-1])
            bb = bbands(close)
            indicators["bbands"] = {
                "upper": float(bb["upper"].iloc[-1]),
                "middle": float(bb["middle"].iloc[-1]),
                "lower": float(bb["lower"].iloc[-1]),
            }
            if "volume" in df.columns:
                indicators["obv"] = float(obv(close, df["volume"]).iloc[-1])
                indicators["mfi_14"] = float(mfi(df, 14).iloc[-1]) if len(df) >= 15 else None
                indicators["vwap"] = float(vwap(df).iloc[-1])
                indicators["rvol_20"] = float(rvol(df["volume"], 20).iloc[-1]) \
                    if len(df) >= 21 else None
            result["indicators"] = indicators
            result["latest_close"] = float(close.iloc[-1])
            result["n_bars"] = len(close)
            return ToolResult("calculate_quant", True, result)
        except Exception as e:  # noqa: BLE001
            return ToolResult("calculate_quant", False, error=str(e))

    def take_snapshot(self, symbol: str, as_of: Optional[str] = None,
                       interval: str = "1d") -> ToolResult:
        """Point-in-time market snapshot."""
        if not self.data_fetcher:
            return ToolResult("snapshot", False, error="no data fetcher")
        try:
            df = self.data_fetcher(symbol, interval, 200)
            if df.empty:
                return ToolResult("snapshot", False, error="no data")
            result = self.snapshot.take(df, symbol, as_of, interval)
            return ToolResult("snapshot", True, result.to_dict())
        except Exception as e:  # noqa: BLE001
            return ToolResult("snapshot", False, error=str(e))

    def simulate_trade(self, symbol: str, entry_date: str,
                         exit_rule: dict, as_of: Optional[str] = None,
                         direction: str = "long") -> ToolResult:
        """What-if backtest: enter at entry_date, exit per rule."""
        if not self.data_fetcher:
            return ToolResult("simulate", False, error="no data fetcher")
        try:
            df = self.data_fetcher(symbol, "1d", 500)
            if df.empty:
                return ToolResult("simulate", False, error="no data")
            result = self.simulator.simulate(
                df, symbol, entry_date, exit_rule,
                as_of=as_of, direction=direction,
            )
            return ToolResult("simulate", True, result.to_dict())
        except Exception as e:  # noqa: BLE001
            return ToolResult("simulate", False, error=str(e))

    def sector_rotation(self, symbols: Optional[list[str]] = None) -> ToolResult:
        """Compute sector rotation table."""
        if not self.data_fetcher:
            return ToolResult("sector_rotation", False, error="no data fetcher")
        try:
            from enhancements.sector_rotation import CRYPTO_SECTORS
            sectors = ([{"symbol": s, "sector": s, "name": s} for s in symbols]
                        if symbols else CRYPTO_SECTORS)
            ohlcv: dict[str, pd.DataFrame] = {}
            for s in sectors:
                df = self.data_fetcher(s["symbol"], "1d", 200)
                if not df.empty:
                    ohlcv[s["symbol"]] = df
            analyzer = SectorRotationAnalyzer(sectors)
            rows = analyzer.compute(ohlcv)
            summary = analyzer.summary(rows)
            return ToolResult("sector_rotation", True, {
                "rows": [r.to_dict() for r in rows],
                "summary": summary,
            })
        except Exception as e:  # noqa: BLE001
            return ToolResult("sector_rotation", False, error=str(e))

    # ----------------------------------------------------------------
    # Trading tools (stage → commit → push)
    # ----------------------------------------------------------------
    def stage_order(self, symbol: str, side: str, lots: float,
                      price: float = 0.0, stop_loss: float = 0.0,
                      take_profit: float = 0.0) -> ToolResult:
        """Stage a trade order (does NOT execute)."""
        try:
            result = self.trading_git.stage_place_order(
                symbol=symbol, side=side, lots=lots, price=price,
                stop_loss=stop_loss, take_profit=take_profit,
            )
            return ToolResult("stage_order", True, result)
        except Exception as e:  # noqa: BLE001
            return ToolResult("stage_order", False, error=str(e))

    def stage_close(self, ticket: int) -> ToolResult:
        """Stage a position close."""
        try:
            result = self.trading_git.stage_close_position(ticket)
            return ToolResult("stage_close", True, result)
        except Exception as e:  # noqa: BLE001
            return ToolResult("stage_close", False, error=str(e))

    def commit(self, message: str) -> ToolResult:
        """Commit staged orders."""
        try:
            result = self.trading_git.commit(message)
            return ToolResult("commit", True, result)
        except Exception as e:  # noqa: BLE001
            return ToolResult("commit", False, error=str(e))

    def push(self) -> ToolResult:
        """Execute committed orders through the guard pipeline."""
        try:
            result = self.trading_git.push()
            return ToolResult("push", True, result)
        except Exception as e:  # noqa: BLE001
            return ToolResult("push", False, error=str(e))

    def reject(self, reason: str) -> ToolResult:
        """Reject the pending commit."""
        try:
            result = self.trading_git.reject(reason)
            return ToolResult("reject", True, result)
        except Exception as e:  # noqa: BLE001
            return ToolResult("reject", False, error=str(e))

    def trading_status(self) -> ToolResult:
        """Get trading-as-git status."""
        try:
            return ToolResult("trading_status", True, self.trading_git.status())
        except Exception as e:  # noqa: BLE001
            return ToolResult("trading_status", False, error=str(e))

    # ----------------------------------------------------------------
    # Account / position tools
    # ----------------------------------------------------------------
    def get_positions(self) -> ToolResult:
        if not self.positions_getter:
            return ToolResult("get_positions", True, [])
        try:
            return ToolResult("get_positions", True, self.positions_getter())
        except Exception as e:  # noqa: BLE001
            return ToolResult("get_positions", False, error=str(e))

    def get_account(self) -> ToolResult:
        if not self.account_getter:
            return ToolResult("get_account", False, error="no account getter")
        try:
            return ToolResult("get_account", True, self.account_getter())
        except Exception as e:  # noqa: BLE001
            return ToolResult("get_account", False, error=str(e))

    # ----------------------------------------------------------------
    # News tool
    # ----------------------------------------------------------------
    def search_news(self, query: str, limit: int = 10) -> ToolResult:
        if not self.news_searcher:
            return ToolResult("search_news", True, [])
        try:
            return ToolResult("search_news", True, self.news_searcher(query, limit))
        except Exception as e:  # noqa: BLE001
            return ToolResult("search_news", False, error=str(e))

    # ----------------------------------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool catalog for AI agent discovery."""
        return [
            {"name": "search_bars", "description": "Find K-line sources for a symbol",
             "args": {"query": "str", "limit": "int=20"}},
            {"name": "calculate_quant", "description": "Run technical indicators on a symbol",
             "args": {"symbol": "str", "timeframe": "str=M15", "count": "int=500"}},
            {"name": "take_snapshot", "description": "Point-in-time market snapshot (no lookahead)",
             "args": {"symbol": "str", "as_of": "str=None", "interval": "str=1d"}},
            {"name": "simulate_trade", "description": "What-if backtest with exit rule",
             "args": {"symbol": "str", "entry_date": "str", "exit_rule": "dict",
                       "as_of": "str=None", "direction": "str=long"}},
            {"name": "sector_rotation", "description": "Cross-sectional sector momentum",
             "args": {"symbols": "list=None"}},
            {"name": "stage_order", "description": "Stage a trade (does NOT execute)",
             "args": {"symbol": "str", "side": "str", "lots": "float",
                       "price": "float=0", "stop_loss": "float=0", "take_profit": "float=0"}},
            {"name": "stage_close", "description": "Stage a position close",
             "args": {"ticket": "int"}},
            {"name": "commit", "description": "Commit staged orders with a message",
             "args": {"message": "str"}},
            {"name": "push", "description": "Execute committed orders (through guards)",
             "args": {}},
            {"name": "reject", "description": "Reject the pending commit",
             "args": {"reason": "str"}},
            {"name": "trading_status", "description": "Get trading-as-git status",
             "args": {}},
            {"name": "get_positions", "description": "Current open positions",
             "args": {}},
            {"name": "get_account", "description": "Account info",
             "args": {}},
            {"name": "search_news", "description": "Financial news search",
             "args": {"query": "str", "limit": "int=10"}},
        ]
