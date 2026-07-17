"""architecture/exchange_abstraction.py
=====================================================================
Exchange Abstraction Layer (Improvement #5)
=====================================================================
Provides a unified interface to multiple brokers/exchanges so the bot
can trade on MT5, Binance, Bybit, OKX, Deribit, etc. without changing
strategy or risk code.

Architecture:
    Strategy/Risk/Execution code
        ↓ uses
    ExchangeInterface (abstract)
        ↓ implemented by
    [MT5Adapter] [BinanceAdapter] [BybitAdapter] [OKXAdapter] [DeribitAdapter]
        ↓ talks to
    [Broker APIs]

Key Methods (all adapters must implement):
    connect() / disconnect() / is_connected()
    account_info() → AccountInfo
    fetch_candles(symbol, timeframe, count) → DataFrame
    symbol_tick(symbol) → TickInfo
    symbol_info(symbol) → SymbolInfo
    get_symbols_by_pattern(patterns) → List[str]
    positions(symbol?) → List[Position]
    place_order(req) → OrderResult
    modify_order(ticket, sl, tp) → OrderResult
    close_order(ticket, symbol, volume, side) → OrderResult
    cancel_order(ticket) → OrderResult
    pending_orders() → List[Order]
    order_history(limit) → List[Order]

Why this matters:
    - Strategy code never imports MetaTrader5 directly
    - Switching brokers = swap one line in config.yaml
    - Multi-broker arbitrage becomes possible
    - Paper trading = use the same interface with a PaperAdapter
    - Backtest = use BacktestAdapter with historical data
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.exchange_abstraction")


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class Timeframe(str, Enum):
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


# ----------------------------------------------------------------------
# Data Transfer Objects (DTOs)
# ----------------------------------------------------------------------
@dataclass
class AccountInfo:
    login: int = 0
    server: str = ""
    currency: str = "USD"
    leverage: int = 100
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0  # equity/margin * 100
    profit: float = 0.0
    connected: bool = False


@dataclass
class TickInfo:
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: float = 0.0
    time: str = ""
    spread: float = 0.0  # in points

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass
class SymbolInfo:
    name: str
    description: str = ""
    currency_base: str = ""
    currency_profit: str = ""
    currency_margin: str = ""
    digits: int = 5
    point: float = 0.00001
    spread: int = 0
    contract_size: float = 1.0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01
    stops_level: int = 0  # minimum SL/TP distance in points
    tick_value: float = 1.0
    tick_size: float = 0.00001
    visible: bool = True
    trade_mode: str = "FULL"  # FULL, LONGONLY, SHORTONLY, DISABLED


@dataclass
class Position:
    ticket: int
    symbol: str
    type: OrderSide  # BUY or SELL
    volume: float
    open_price: float
    current_price: float
    sl: float = 0.0
    tp: float = 0.0
    profit: float = 0.0
    swap: float = 0.0
    commission: float = 0.0
    open_time: str = ""
    magic: int = 0
    comment: str = ""


@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    volume: float = 0.01
    price: float = 0.0  # for limit/stop orders
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    comment: str = ""
    deviation: int = 20
    filling_mode: str = "IOC"  # IOC, FOK, RETURN


@dataclass
class OrderResult:
    ok: bool = False
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""
    error_code: int = 0
    error_desc: str = ""
    latency_ms: float = 0.0


# ----------------------------------------------------------------------
# Abstract interface
# ----------------------------------------------------------------------
class ExchangeInterface(ABC):
    """All exchange adapters must implement this interface."""

    name: str = "abstract"

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def account_info(self) -> AccountInfo: ...

    @abstractmethod
    def fetch_candles(self,
                      symbol: str,
                      timeframe: Timeframe,
                      count: int = 500) -> pd.DataFrame: ...

    @abstractmethod
    def symbol_tick(self, symbol: str) -> TickInfo: ...

    @abstractmethod
    def symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abstractmethod
    def get_symbols_by_pattern(self, patterns: List[str]) -> List[str]: ...

    @abstractmethod
    def positions(self, symbol: Optional[str] = None) -> List[Position]: ...

    @abstractmethod
    def place_order(self, req: OrderRequest) -> OrderResult: ...

    @abstractmethod
    def modify_order(self, ticket: int, sl: float, tp: float) -> OrderResult: ...

    @abstractmethod
    def close_order(self, ticket: int, symbol: str,
                    volume: float, side: OrderSide) -> OrderResult: ...

    @abstractmethod
    def pending_orders(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def order_history(self, limit: int = 100) -> List[Dict[str, Any]]: ...


# ----------------------------------------------------------------------
# MT5 Adapter (wraps the existing MT5Connector)
# ----------------------------------------------------------------------
class MT5Adapter(ExchangeInterface):
    """Adapter that wraps brokers/mt5_connector.MT5Connector."""

    name = "mt5"

    def __init__(self,
                 login: int, password: str, server: str,
                 terminal_path: str = "",
                 timeout_ms: int = 5000,
                 reconnect_attempts: int = 5,
                 reconnect_delay_s: float = 2.0,
                 bus: Optional[EventBus] = None,
                 candle_cache_ttl_s: float = 900.0,
                 order_max_retries: int = 3,
                 modify_max_retries: int = 2):
        from brokers.mt5_connector import MT5Connector  # lazy import
        self._conn = MT5Connector(
            login=login, password=password, server=server,
            terminal_path=terminal_path, timeout_ms=timeout_ms,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay_s=reconnect_delay_s,
        )
        self._bus = bus or get_bus()
        self._connected = False
        # PERF/CORRECTNESS FIX: the MetaTrader5 Python package is NOT
        # thread-safe — it talks to the terminal over a single IPC
        # channel. integration.py calls fetch_candles/symbol_tick/etc.
        # from an 8-worker ThreadPoolExecutor; without this lock those
        # calls race on the same channel. In the best case the terminal
        # just serializes them anyway (which is why threading wasn't
        # actually reducing cycle time — see the CYCLE 1 log showing
        # ~35s for ~100 IPC-bound calls despite 8 workers). In the worst
        # case, concurrent access can return corrupted/interleaved data.
        # An RLock here makes the serialization explicit and safe instead
        # of accidental and unverified.
        import threading
        self._ipc_lock = threading.RLock()

        # PERF FIX (the big one): fetch_candles(symbol, "M15", 500) was
        # being called fresh — ALL 500 bars, full IPC round-trip — for
        # every symbol, EVERY cycle, even though the bot polls every 5s
        # while M15 bars only close every 900s. Measured cost: ~325ms per
        # call x 100 symbols (serialized) = the ~32s cycle time. 499 of
        # those 500 bars are almost always identical to the previous
        # cycle's fetch. We cache the last response per (symbol,
        # timeframe) and only re-hit MT5 when the cache is older than
        # candle_cache_ttl_s. Trade-off: the in-progress (still-forming)
        # last bar can be up to candle_cache_ttl_s stale — acceptable for
        # M15 strategies that act on closed bars; lower the TTL (or set
        # to 0 to disable) if your strategy needs live intra-bar data.
        self._candle_cache_ttl_s = float(candle_cache_ttl_s)
        self._candle_cache: Dict[tuple, tuple] = {}  # (symbol, tf, count) -> (df, fetched_at)
        self._candle_cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        # H17 fix: retry counts are now configurable (were hardcoded)
        self._order_max_retries = int(order_max_retries)
        self._modify_max_retries = int(modify_max_retries)


    def connect(self) -> bool:
        try:
            self._conn.connect()
            self._connected = True
            self._bus.emit(EventType.MT5_RECONNECT,
                           payload={"adapter": self.name},
                           source="mt5_adapter")
            return True
        except Exception as e:  # noqa: BLE001
            self._bus.emit(EventType.MT5_DISCONNECT,
                           payload={"error": str(e)},
                           source="mt5_adapter")
            return False

    def disconnect(self) -> None:
        try:
            self._conn.disconnect()
        finally:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def account_info(self) -> AccountInfo:
        if not self._connected:
            return AccountInfo(connected=False)
        try:
            with self._ipc_lock:
                info = self._conn.account_info()
            return AccountInfo(
                login=int(getattr(info, "login", 0)),
                server=str(getattr(info, "server", "")),
                currency=str(getattr(info, "currency", "USD")),
                leverage=int(getattr(info, "leverage", 100)),
                balance=float(getattr(info, "balance", 0)),
                equity=float(getattr(info, "equity", 0)),
                margin=float(getattr(info, "margin", 0)),
                free_margin=float(getattr(info, "margin_free", 0)),
                margin_level=float(getattr(info, "margin_level", 0)),
                profit=float(getattr(info, "profit", 0)),
                connected=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("mt5_adapter: account_info failed: %r", e)
            return AccountInfo(connected=False)

    def fetch_candles(self, symbol, timeframe, count=500):
        # Accept both Timeframe enum and plain string ("M15", "H1", etc.)
        tf_str = timeframe.value if hasattr(timeframe, "value") else str(timeframe)

        # PERF FIX: serve from cache if fresh enough — see __init__ note.
        cache_key = (symbol, tf_str, count)
        if self._candle_cache_ttl_s > 0:
            # C11 fix: read, freshness-check, copy, and hit-count all happen
            # under a single lock acquisition so the cached DataFrame can't
            # be replaced/invalidated between the check and the copy.
            with self._candle_cache_lock:
                cached = self._candle_cache.get(cache_key)
                if cached is not None:
                    df_cached, fetched_at = cached
                    if time.time() - fetched_at < self._candle_cache_ttl_s:
                        self._cache_hits += 1
                        return df_cached.copy()

        with self._candle_cache_lock:
            self._cache_misses += 1

        with self._ipc_lock:
            df = self._conn.fetch_candles(symbol, tf_str, count, as_dataframe=True)

        if self._candle_cache_ttl_s > 0 and df is not None and not df.empty:
            with self._candle_cache_lock:
                self._candle_cache[cache_key] = (df, time.time())

        return df

    def cache_stats(self) -> Dict[str, Any]:
        """Diagnostic: candle-cache hit/miss counts, for verifying the
        cache is actually absorbing repeat fetches (independent of
        main.py's log-frequency filter, which can hide scan cycles)."""
        with self._candle_cache_lock:
            hits, misses = self._cache_hits, self._cache_misses
        total = hits + misses
        return {
            "hits": hits, "misses": misses,
            "hit_rate": (hits / total) if total else 0.0,
            "cached_entries": len(self._candle_cache),
            "ttl_s": self._candle_cache_ttl_s,
        }

    def symbol_tick(self, symbol: str) -> TickInfo:
        with self._ipc_lock:
            tick = self._conn.symbol_tick(symbol)
        bid = float(getattr(tick, "bid", 0))
        ask = float(getattr(tick, "ask", 0))
        return TickInfo(
            symbol=symbol, bid=bid, ask=ask,
            last=float(getattr(tick, "last", 0)),
            volume=float(getattr(tick, "volume", 0)),
            time=str(getattr(tick, "time", "")),
            spread=ask - bid,
        )

    def symbol_info(self, symbol: str) -> SymbolInfo:
        with self._ipc_lock:
            info = self._conn.symbol_info(symbol)
        return SymbolInfo(
            name=symbol,
            description=str(getattr(info, "description", "")),
            currency_base=str(getattr(info, "currency_base", "")),
            currency_profit=str(getattr(info, "currency_profit", "")),
            currency_margin=str(getattr(info, "currency_margin", "")),
            digits=int(getattr(info, "digits", 5)),
            point=float(getattr(info, "point", 0.00001)),
            spread=int(getattr(info, "spread", 0)),
            contract_size=float(getattr(info, "trade_contract_size", 1.0)),
            volume_min=float(getattr(info, "volume_min", 0.01)),
            volume_max=float(getattr(info, "volume_max", 100.0)),
            volume_step=float(getattr(info, "volume_step", 0.01)),
            stops_level=int(getattr(info, "trade_stops_level", 0)),
            tick_value=float(getattr(info, "trade_tick_value", 1.0)),
            tick_size=float(getattr(info, "trade_tick_size", 0.00001)),
        )

    def get_symbols_by_pattern(self, patterns: List[str]) -> List[str]:
        # C1 fix: this call was not covered by _ipc_lock, allowing it to
        # race with fetch_candles/symbol_tick/place_order/positions on the
        # same (non-thread-safe) MT5 IPC channel.
        with self._ipc_lock:
            return self._conn.get_symbols_by_pattern(patterns)

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        with self._ipc_lock:
            raw = self._conn.positions(symbol=symbol) or []
        out = []
        for p in raw:
            side = OrderSide.BUY if int(getattr(p, "type", 0)) == 0 else OrderSide.SELL
            out.append(Position(
                ticket=int(getattr(p, "ticket", 0)),
                symbol=str(getattr(p, "symbol", "")),
                type=side,
                volume=float(getattr(p, "volume", 0)),
                open_price=float(getattr(p, "price_open", 0)),
                current_price=float(getattr(p, "price_current", 0)),
                sl=float(getattr(p, "sl", 0)),
                tp=float(getattr(p, "tp", 0)),
                profit=float(getattr(p, "profit", 0)),
                swap=float(getattr(p, "swap", 0)),
                commission=float(getattr(p, "commission", 0)),
                open_time=str(getattr(p, "time", "")),
                magic=int(getattr(p, "magic", 0)),
                comment=str(getattr(p, "comment", "")),
            ))
        return out

    def _build_market_request(self, req: OrderRequest) -> dict:
        """Build an MT5 order_send request dict for a market deal.

        P0-5 FIX (Phase 3): This is now called by place_order() which then
        routes through MT5Connector.send_request() — the retry/fresh-price/
        retcode-classification logic lives there and must not be bypassed.
        Previously place_order() called mt5.order_send() directly, which
        meant a requote (retcode 10004) became a hard failure instead of
        being retried with a refreshed price.

        Major #3 fix: validate SL/TP against the broker's stops_level
        (minimum distance from current price). If SL or TP is too close,
        adjust them outward to the minimum distance before sending. This
        prevents broker rejections that would otherwise fail the order
        after all risk gates have passed.
        """
        from brokers.mt5_connector import mt5  # for the ORDER_* constants only
        tick = self.symbol_tick(req.symbol)
        price = tick.ask if req.side == OrderSide.BUY else tick.bid

        # Major #3 fix: validate SL/TP against broker stops_level.
        sl = float(req.sl) if req.sl else 0.0
        tp = float(req.tp) if req.tp else 0.0
        try:
            info = self.symbol_info(req.symbol)
            stops_level = getattr(info, "stops_level", 0)
            point = getattr(info, "point", 0.00001)
            min_dist = stops_level * point
            if min_dist > 0:
                if sl > 0 and abs(price - sl) < min_dist:
                    if req.side == OrderSide.BUY:
                        sl = price - min_dist
                    else:
                        sl = price + min_dist
                    log.warning("MT5Adapter: SL adjusted to stops_level min dist for %s: sl=%.5f",
                                req.symbol, sl)
                if tp > 0 and abs(tp - price) < min_dist:
                    if req.side == OrderSide.BUY:
                        tp = price + min_dist
                    else:
                        tp = price - min_dist
                    log.warning("MT5Adapter: TP adjusted to stops_level min dist for %s: tp=%.5f",
                                req.symbol, tp)
        except Exception as e:  # noqa: BLE001
            log.debug("MT5Adapter: stops_level validation skipped: %r", e)

        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": req.symbol,
            "volume": float(req.volume),
            "type": mt5.ORDER_TYPE_BUY if req.side == OrderSide.BUY
                    else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": int(req.deviation),
            "magic": int(req.magic),
            "comment": req.comment or "ai_bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    def place_order(self, req: OrderRequest) -> OrderResult:
        """Place a market order via MT5Connector.send_request (with retry).

        P0-5 FIX (Phase 3): Routes through MT5Connector.send_request() so
        transient retcodes (requote 10004, timeout 10035, connection 10031)
        are retried with a fresh price up to max_retries, while permanent
        rejections (insufficient funds, invalid stops, market closed) fail
        fast. Previously this method called mt5.order_send() directly and
        bypassed all resilience logic — the worst possible place to skip it.
        """
        t0 = time.time()
        try:
            request = self._build_market_request(req)
            # send_request returns a dict (MT5Connector._result_to_dict)
            with self._ipc_lock:
                result = self._conn.send_request(request, max_retries=self._order_max_retries)
            latency = (time.time() - t0) * 1000

            from brokers.mt5_connector import mt5
            retcode = int(result.get("retcode", 0))
            ok = (retcode == mt5.TRADE_RETCODE_DONE or retcode == mt5.TRADE_RETCODE_DONE_PARTIAL)  # P0-4 fix

            order_result = OrderResult(
                ok=ok,
                ticket=int(result.get("order", 0)),
                price=float(result.get("price", 0.0)),
                volume=float(result.get("volume", req.volume)),
                comment=str(result.get("comment", "")),
                error_code=retcode,
                error_desc=str(result.get("comment", "")),
                latency_ms=latency,
            )

            # Emit explicit event so the bot can never silently place or
            # fail to place an order without the rest of the system knowing.
            # ORDER_FILLED = success (MT5 retcode DONE), ORDER_REJECTED = failure.
            self._bus.emit(
                EventType.ORDER_FILLED if ok else EventType.ORDER_REJECTED,
                payload={
                    "symbol": req.symbol, "side": req.side.value,
                    "volume": req.volume, "ok": ok,
                    "ticket": order_result.ticket,
                    "price": order_result.price,
                    "retcode": retcode,
                    "comment": order_result.comment,
                    "latency_ms": latency,
                },
                source="mt5_adapter",
            )
            return order_result
        except Exception as e:  # noqa: BLE001
            latency = (time.time() - t0) * 1000
            log.error("MT5Adapter.place_order failed: %r", e)
            self._bus.emit(
                EventType.ORDER_REJECTED,
                payload={"symbol": req.symbol, "side": req.side.value,
                         "volume": req.volume, "ok": False,
                         "error": str(e), "latency_ms": latency},
                source="mt5_adapter",
            )
            return OrderResult(ok=False, comment=str(e), latency_ms=latency)

    def modify_order(self, ticket: int, sl: float, tp: float) -> OrderResult:
        """Modify SL/TP on an open position via TRADE_ACTION_SLTP.

        P0-5 FIX (Phase 3): Also routes through send_request for consistency,
        though SLTP modifications are less commonly requoted.
        """
        try:
            from brokers.mt5_connector import mt5
            pos = next((p for p in self.positions() if p.ticket == ticket), None)
            if pos is None:
                return OrderResult(ok=False, comment="position not found")
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": int(ticket),
                "sl": float(sl), "tp": float(tp),
            }
            with self._ipc_lock:
                result = self._conn.send_request(request, max_retries=self._modify_max_retries)
            retcode = int(result.get("retcode", 0))
            ok = (retcode == mt5.TRADE_RETCODE_DONE)
            return OrderResult(
                ok=ok, ticket=int(ticket),
                comment=str(result.get("comment", "")),
                error_code=retcode,
                error_desc=str(result.get("comment", "")),
            )
        except Exception as e:  # noqa: BLE001
            log.error("MT5Adapter.modify_order failed: %r", e)
            return OrderResult(ok=False, comment=str(e))

    def close_order(self, ticket: int, symbol: str,
                    volume: float, side: OrderSide) -> OrderResult:
        """Close a position by sending the opposite-side market order."""
        opposite_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
        req = OrderRequest(
            symbol=symbol, side=opposite_side,
            volume=volume, magic=0, comment=f"close_{ticket}",
        )
        return self.place_order(req)

    def pending_orders(self) -> List[Dict[str, Any]]:
        try:
            from brokers.mt5_connector import mt5
            with self._ipc_lock:
                return [vars(o) for o in mt5.orders_get() or []]
        except Exception:  # noqa: BLE001
            return []

    def order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            from brokers.mt5_connector import mt5
            from datetime import datetime, timedelta
            from_t = datetime.now() - timedelta(days=30)
            with self._ipc_lock:
                return [vars(d) for d in mt5.history_deals_get(from_t, datetime.now()) or []][:limit]
        except Exception:  # noqa: BLE001
            return []


# ----------------------------------------------------------------------
# Paper Trading Adapter (in-memory simulation)
# ----------------------------------------------------------------------
class PaperAdapter(ExchangeInterface):
    """In-memory paper trading adapter — simulates a broker.

    Useful for testing strategies without a real broker connection.
    Maintains open positions, applies SL/TP, tracks equity.
    """

    name = "paper"

    def __init__(self,
                 initial_balance: float = 10000.0,
                 currency: str = "USD",
                 bus: Optional[EventBus] = None):
        self._balance = initial_balance
        self._equity = initial_balance
        self._currency = currency
        self._positions: Dict[int, Position] = {}
        self._ticket_counter = 100000
        self._connected = True
        self._bus = bus or get_bus()
        self._candle_cache: Dict[str, pd.DataFrame] = {}
        self._tick_cache: Dict[str, TickInfo] = {}

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def account_info(self) -> AccountInfo:
        # Recompute equity from open positions
        unrealized = sum(p.profit for p in self._positions.values())
        return AccountInfo(
            balance=self._balance, equity=self._balance + unrealized,
            free_margin=self._balance + unrealized,
            currency=self._currency, connected=self._connected,
            profit=unrealized,
        )

    def set_candle_data(self, symbol: str, df: pd.DataFrame) -> None:
        """Feed historical data into the paper adapter (for backtest/sim)."""
        self._candle_cache[symbol] = df
        if not df.empty:
            last = df.iloc[-1]
            self._tick_cache[symbol] = TickInfo(
                symbol=symbol,
                bid=float(last["close"]),
                ask=float(last["close"]),
                last=float(last["close"]),
                volume=float(last.get("volume", 0)),
            )

    def fetch_candles(self, symbol, timeframe, count=500):
        df = self._candle_cache.get(symbol)
        if df is None:
            return pd.DataFrame()
        return df.tail(count).copy()

    def symbol_tick(self, symbol: str) -> TickInfo:
        return self._tick_cache.get(symbol, TickInfo(symbol=symbol))

    def symbol_info(self, symbol: str) -> SymbolInfo:
        return SymbolInfo(name=symbol)

    def get_symbols_by_pattern(self, patterns: List[str]) -> List[str]:
        return [s for s in self._candle_cache.keys()
                if any(p in s for p in patterns)]

    def positions(self, symbol: Optional[str] = None) -> List[Position]:
        out = list(self._positions.values())
        if symbol:
            out = [p for p in out if p.symbol == symbol]
        return out

    def place_order(self, req: OrderRequest) -> OrderResult:
        if not self._connected:
            return OrderResult(ok=False, comment="not connected")
        tick = self.symbol_tick(req.symbol)
        price = tick.ask if req.side == OrderSide.BUY else tick.bid
        self._ticket_counter += 1
        ticket = self._ticket_counter
        pos = Position(
            ticket=ticket, symbol=req.symbol, type=req.side,
            volume=req.volume, open_price=price, current_price=price,
            sl=req.sl, tp=req.tp, magic=req.magic, comment=req.comment,
            open_time=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._positions[ticket] = pos
        self._bus.emit(EventType.POSITION_OPENED,
                       payload={"ticket": ticket, "symbol": req.symbol,
                                "side": req.side.value, "volume": req.volume,
                                "price": price},
                       source="paper_adapter")
        return OrderResult(ok=True, ticket=ticket, price=price,
                           volume=req.volume, comment="paper fill")

    def modify_order(self, ticket, sl, tp) -> OrderResult:
        p = self._positions.get(ticket)
        if p is None:
            return OrderResult(ok=False, comment="not found")
        p.sl = sl
        p.tp = tp
        return OrderResult(ok=True, ticket=ticket)

    def close_order(self, ticket, symbol, volume, side) -> OrderResult:
        p = self._positions.get(ticket)
        if p is None:
            return OrderResult(ok=False, comment="not found")
        tick = self.symbol_tick(symbol)
        exit_price = tick.bid if p.type == OrderSide.BUY else tick.ask
        pnl = (exit_price - p.open_price) * p.volume * (1 if p.type == OrderSide.BUY else -1)
        self._balance += pnl
        del self._positions[ticket]
        self._bus.emit(EventType.POSITION_CLOSED,
                       payload={"ticket": ticket, "symbol": symbol,
                                "pnl": pnl, "exit_price": exit_price},
                       source="paper_adapter")
        return OrderResult(ok=True, ticket=ticket, price=exit_price,
                           volume=volume, comment=f"closed pnl={pnl:.2f}")

    def pending_orders(self) -> List[Dict[str, Any]]:
        return []

    def order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return []

    # Paper-specific: update P&L on each tick
    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices and check SL/TP for all positions."""
        closed_tickets = []
        for ticket, p in self._positions.items():
            if p.symbol not in prices:
                continue
            p.current_price = prices[p.symbol]
            p.profit = (p.current_price - p.open_price) * p.volume * \
                       (1 if p.type == OrderSide.BUY else -1)
            # Check SL/TP
            if p.type == OrderSide.BUY:
                if p.sl > 0 and p.current_price <= p.sl:
                    closed_tickets.append((ticket, "SL_HIT"))
                elif p.tp > 0 and p.current_price >= p.tp:
                    closed_tickets.append((ticket, "TP_HIT"))
            else:
                if p.sl > 0 and p.current_price >= p.sl:
                    closed_tickets.append((ticket, "SL_HIT"))
                elif p.tp > 0 and p.current_price <= p.tp:
                    closed_tickets.append((ticket, "TP_HIT"))
        for ticket, reason in closed_tickets:
            p = self._positions[ticket]
            event_type = EventType.SL_HIT if reason == "SL_HIT" else EventType.TP_HIT
            self._bus.emit(event_type,
                           payload={"ticket": ticket, "symbol": p.symbol,
                                    "price": p.current_price},
                           source="paper_adapter")
            self.close_order(ticket, p.symbol, p.volume, p.type)


# ----------------------------------------------------------------------
# Exchange Factory
# ----------------------------------------------------------------------
def create_exchange(exchange_type: str, **kwargs) -> ExchangeInterface:
    """Factory: create an exchange adapter by name.

    P0-10 FIX (Phase 3): `mode` is now the canonical selector. The old
    `execution.send_real_orders` config flag is dead and removed — the bot
    must never place a real order because a flag was merely absent. Default
    is `paper` (no broker, in-memory simulation); `demo` and `live` both
    use MT5Adapter but `live` requires an explicit confirmation flag at
    the main.py CLI layer.
    """
    if exchange_type in ("mt5", "demo", "live"):
        return MT5Adapter(
            login=int(kwargs["login"]),
            password=str(kwargs["password"]),
            server=str(kwargs["server"]),
            terminal_path=str(kwargs.get("terminal_path", "")),
            timeout_ms=int(kwargs.get("timeout_ms", 5000)),
            reconnect_attempts=int(kwargs.get("reconnect_attempts", 5)),
            reconnect_delay_s=float(kwargs.get("reconnect_delay_s", 2.0)),
            candle_cache_ttl_s=float(kwargs.get("candle_cache_ttl_s", 900.0)),
        )
    elif exchange_type in ("paper",):
        return PaperAdapter(
            initial_balance=float(kwargs.get("initial_balance", 10000.0)),
            currency=str(kwargs.get("currency", "USD")),
        )
    elif exchange_type == "binance":
        # Future: implement BinanceAdapter
        raise NotImplementedError("BinanceAdapter not yet implemented")
    elif exchange_type == "bybit":
        raise NotImplementedError("BybitAdapter not yet implemented")
    else:
        raise ValueError(f"Unknown exchange type: {exchange_type}")