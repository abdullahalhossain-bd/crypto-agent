"""brokers.mt5_connector
=====================================================================
Thin, defensive wrapper around the MetaTrader5 Python package.

Design goals:
  1. Keep the rest of the codebase MT5-agnostic. Engine code talks to
     *this* module, never to `MetaTrader5` directly.
  2. Survive environments where MetaTrader5 is not installed (Linux / CI)
     by raising a clear `MT5Unavailable` error only when something is
     actually called that needs the real library.
  3. Wrap every MT5 call in retries with exponential back-off — MT5
     drops connections randomly and silent failures are the #1 source
     of ghost trades.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("brokers.mt5")

# MT5 retcodes that represent a transient condition worth retrying
# (network hiccup, requote, stale price) as opposed to a terminal
# rejection (insufficient funds, invalid stops, market closed) that
# should fail fast instead of retrying. See FIX-MT5-01.
_RETRYABLE_RETCODES = frozenset({
    10004,  # TRADE_RETCODE_REQUOTE
    10006,  # TRADE_RETCODE_REJECT (often transient dealer reject)
    10021,  # TRADE_RETCODE_PRICE_OFF
    10024,  # TRADE_RETCODE_TOO_MANY_REQUESTS
    10031,  # TRADE_RETCODE_CONNECTION
    10035,  # TRADE_RETCODE_TIMEOUT
})

# ----------------------------------------------------------------------
# Optional MT5 import — degrade gracefully if not installed
# ----------------------------------------------------------------------
try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    mt5 = None  # type: ignore
    _MT5_AVAILABLE = False


class MT5Unavailable(RuntimeError):
    """Raised when the MetaTrader5 package or terminal isn't reachable."""


# ----------------------------------------------------------------------
# Timeframe resolution
# ----------------------------------------------------------------------
TIMEFRAMES: dict[str, Any] = {}
if _MT5_AVAILABLE:
    TIMEFRAMES = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
    }


# ----------------------------------------------------------------------
# Account / connection info
# ----------------------------------------------------------------------
@dataclass
class AccountInfo:
    login: int
    server: str
    balance: float
    equity: float
    currency: str
    leverage: int

    @classmethod
    def from_raw(cls, raw: Any) -> "AccountInfo":
        return cls(
            login=raw.login,
            server=raw.server,
            balance=float(raw.balance),
            equity=float(raw.equity),
            currency=raw.currency,
            leverage=int(raw.leverage),
        )


# ----------------------------------------------------------------------
# Connector
# ----------------------------------------------------------------------
class MT5Connector:
    """Stateful wrapper around a single MT5 terminal session."""

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        terminal_path: str = "",
        timeout_ms: int = 5000,
        reconnect_attempts: int = 5,
        reconnect_delay_s: float = 2.0,
        bus: Optional[EventBus] = None,
    ) -> None:
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path or None
        self.timeout_ms = timeout_ms
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s
        self._connected = False
        # FIX-MT5-03: last time we actually verified the connection with a
        # cheap round-trip call (not just trusted the flag). Paired with
        # _health_check_interval_s below.
        self._last_health_check = 0.0
        self._health_check_interval_s = 30.0
        # FIX-MT5-02: bus reference so failures can be emitted for
        # self_healing.py's existing MT5_DISCONNECT/IPC_TIMEOUT subscriptions.
        # Optional + defaulted to the process-wide singleton so this remains
        # a backward-compatible constructor signature.
        self._bus = bus or get_bus()

    # ---------------- lifecycle ----------------
    def connect(self) -> bool:
        """Connect to MT5 — v6.2.1 robust connection with terminal_path.

        Strategy:
        1. If terminal_path is set, try mt5.initialize(path=...) FIRST — this
           launches the terminal process directly and is the most reliable.
        2. If no terminal_path (or step 1 fails), try default mt5.initialize()
           (uses cached terminal login).
        3. If that fails, try with explicit credentials from config.
        4. Retry up to 3 times with backoff for IPC timeout (-10005).
        5. Validate by pulling account_info().
        """
        if not _MT5_AVAILABLE:
            raise MT5Unavailable(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )

        # ── v6.2.1: If terminal_path is set, use it FIRST ──────────────
        # This launches the terminal process directly via the path.
        if self.terminal_path and os.path.exists(self.terminal_path):
            log.info("MT5: using terminal path: %s", self.terminal_path)
            # Try with path + credentials
            init_kwargs = {
                "path": self.terminal_path,
                "login": self.login,
                "password": self.password,
                "server": self.server,
                "timeout": self.timeout_ms,
            }
            max_retries = 3
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    log.info("MT5: initialize() with path attempt %d/%d (login=%s server=%s)",
                             attempt, max_retries, self.login, self.server)
                    if mt5.initialize(**init_kwargs):
                        info = mt5.account_info()
                        if info is not None:
                            self._connected = True
                            log.info("MT5 connected (path+login) login=%s server=%s "
                                     "balance=%s equity=%s leverage=1:%s",
                                     info.login, info.server, info.balance,
                                     info.equity, info.leverage)
                            return True
                        else:
                            err = mt5.last_error()
                            log.warning("MT5 initialize OK but account_info() failed: %s", err)
                            last_error = f"account_info() failed: {err}"
                            mt5.shutdown()
                    else:
                        err = mt5.last_error()
                        last_error = str(err)
                        log.warning("MT5 initialize attempt %d/%d failed: %s",
                                    attempt, max_retries, err)
                except Exception as e:
                    last_error = str(e)
                    log.warning("MT5 initialize attempt %d/%d raised: %s",
                                attempt, max_retries, e)
                if attempt < max_retries:
                    import time as _time
                    _time.sleep(2 * attempt)
                try:
                    mt5.shutdown()
                except Exception as e:
                    # Phase 7: log shutdown failure between retries — was
                    # silently swallowed. Acceptable since we're retrying,
                    # but logged at DEBUG for diagnostics.
                    log.debug("MT5: shutdown between retries failed: %r", e)

        # ── v6.2: Try default initialize() (cached login) ────────────────
        try:
            log.info("MT5: attempting default initialize() (cached login)...")
            if mt5.initialize(timeout=self.timeout_ms):
                info = mt5.account_info()
                if info is not None and info.login == self.login:
                    self._connected = True
                    log.info("MT5 connected (cached login) login=%s server=%s "
                             "balance=%s equity=%s leverage=1:%s",
                             info.login, info.server, info.balance,
                             info.equity, info.leverage)
                    return True
                elif info is not None:
                    log.warning("MT5 cached login is %s but config wants %s — "
                                "switching accounts", info.login, self.login)
                    mt5.shutdown()
                else:
                    mt5.shutdown()
            else:
                err = mt5.last_error()
                log.warning("MT5 default initialize() failed: %s — trying explicit", err)
        except Exception as e:
            log.warning("MT5 default initialize() raised: %s — trying explicit", e)
            try:
                mt5.shutdown()
            except Exception as e:
                # Phase 7: log shutdown failure — was silently swallowed.
                log.debug("MT5: shutdown between retries failed: %r", e)

        # ── v6.2: Final try — explicit credentials without path ──────────
        init_kwargs = {
            "login": self.login,
            "password": self.password,
            "server": self.server,
            "timeout": self.timeout_ms,
        }
        max_retries = 3
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                log.info("MT5: explicit initialize() attempt %d/%d (login=%s server=%s)",
                         attempt, max_retries, self.login, self.server)
                if mt5.initialize(**init_kwargs):
                    info = mt5.account_info()
                    if info is None:
                        err = mt5.last_error()
                        log.warning("MT5 initialize OK but account_info() failed: %s", err)
                        mt5.shutdown()
                        last_error = f"account_info() failed: {err}"
                    else:
                        self._connected = True
                        log.info("MT5 connected (explicit login) login=%s server=%s "
                                 "balance=%s equity=%s leverage=1:%s",
                                 info.login, info.server, info.balance,
                                 info.equity, info.leverage)
                        return True
                else:
                    err = mt5.last_error()
                    last_error = str(err)
                    log.warning("MT5 initialize attempt %d/%d failed: %s",
                                attempt, max_retries, err)
            except Exception as e:
                last_error = str(e)
                log.warning("MT5 initialize attempt %d/%d raised: %s",
                            attempt, max_retries, e)
            if attempt < max_retries:
                import time as _time
                _time.sleep(2 * attempt)
            try:
                mt5.shutdown()
            except Exception as e:
                # Phase 7: log shutdown failure — was silently swallowed.
                log.debug("MT5: shutdown between retries failed: %r", e)

        self._connected = False
        self._bus.emit(
            EventType.MT5_DISCONNECT,
            payload={"adapter": "mt5_adapter", "error": str(last_error),
                     "login": self.login, "server": self.server},
            source="mt5_connector",
        )
        raise MT5Unavailable(
            f"MT5 connection failed after all attempts. "
            f"Last error: {last_error}. "
            f"Terminal path: {self.terminal_path or 'not set'}. "
            f"Troubleshooting: "
            f"1) Make sure MT5 terminal is running at: {self.terminal_path or 'default path'}. "
            f"2) Log in to terminal with login={self.login} server={self.server}. "
            f"3) Run: python run_mt5_diagnostic.py"
        )

    def disconnect(self) -> None:
        if _MT5_AVAILABLE and self._connected:
            try:
                mt5.shutdown()
            finally:
                self._connected = False
                log.info("MT5 disconnected")

    def ensure_connected(self) -> None:
        """Re-connect only if truly disconnected. Cheap to call every loop.

        v9.0 fix: Don't call mt5.account_info() as health check on every call.
        That causes IPC timeouts and re-initialization every 5 seconds.
        Just trust the _connected flag — only reconnect if explicitly disconnected.

        FIX-MT5-03: the v9.0 fix over-corrected — trusting the flag FOREVER
        means a silent MT5-side disconnect (terminal closed, VPS reboot,
        broker kick) is invisible until the next real API call happens to
        fail. This adds back a health check, but bounded to once every
        `_health_check_interval_s` (default 30s) rather than every call —
        keeps the fix's original intent (no IPC spam every 5s) while closing
        the "looks connected forever" gap.
        """
        if not _MT5_AVAILABLE:
            raise MT5Unavailable("MT5 backend not installed")
        if self._connected:
            now = time.time()
            if now - self._last_health_check < self._health_check_interval_s:
                return  # Trust the connection — checked recently enough
            self._last_health_check = now
            try:
                if mt5.terminal_info() is not None:
                    return  # still genuinely connected
                log.warning("MT5 periodic health check: terminal_info() "
                           "returned None — treating as disconnected")
                self._connected = False
                self._bus.emit(
                    EventType.MT5_DISCONNECT,
                    payload={"adapter": "mt5_adapter",
                             "error": "periodic health check failed"},
                    source="mt5_connector",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("MT5 periodic health check raised: %r — "
                           "treating as disconnected", e)
                self._connected = False
                self._bus.emit(
                    EventType.MT5_DISCONNECT,
                    payload={"adapter": "mt5_adapter", "error": str(e)},
                    source="mt5_connector",
                )
            # fall through to reconnect logic below

        # Only reach here if _connected is False — attempt reconnect
        last_err: Optional[str] = None
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                self.connect()
                return
            except MT5Unavailable as e:
                # P0-5 fix: don't re-raise immediately — continue retrying.
                # The old code re-raised on the first failure, making the
                # reconnect_attempts config useless.
                last_err = str(e)
                if attempt >= self.reconnect_attempts:
                    raise
                log.warning("MT5 reconnect attempt %d/%d failed: %s — retrying...",
                          attempt, self.reconnect_attempts, last_err)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                log.warning("MT5 reconnect attempt %d/%d failed: %s",
                            attempt, self.reconnect_attempts, e)
                time.sleep(self.reconnect_delay_s * attempt)
        self._bus.emit(
            EventType.IPC_TIMEOUT,
            payload={"adapter": "mt5_adapter", "error": last_err,
                     "attempts": self.reconnect_attempts},
            source="mt5_connector",
        )
        raise MT5Unavailable(f"MT5 reconnect exhausted: {last_err}")

    # ---------------- market data ----------------
    def ensure_symbol(self, symbol: str) -> None:
        self.ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            self._raise_last_error(f"symbol_info({symbol})")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                self._raise_last_error(f"symbol_select({symbol})")
            log.info("Symbol %s selected in MarketWatch", symbol)

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        count: int = 1000,
        as_dataframe: bool = True,
    ):
        """Fetch last `count` OHLCV bars for `symbol`.

        Returns a pandas DataFrame by default (with a normalized 'volume'
        column and UTC 'time'), or a list of raw MT5 namedtuples if
        `as_dataframe=False`.
        """
        self.ensure_connected()
        self.ensure_symbol(symbol)
        tf = TIMEFRAMES.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            self._raise_last_error(f"copy_rates_from_pos({symbol},{timeframe},{count})")
        log.debug("Fetched %d candles for %s %s", len(rates), symbol, timeframe)
        if not as_dataframe:
            return rates
        import pandas as pd
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        # v6.3.1: Normalize volume column — MT5 returns 'tick_volume', not 'volume'
        if "volume" not in df.columns:
            if "tick_volume" in df.columns:
                df["volume"] = df["tick_volume"]
            elif "real_volume" in df.columns:
                df["volume"] = df["real_volume"]
            else:
                df["volume"] = 0.0  # fallback
        return df

    # ---------------- account / positions ----------------
    def account_info(self) -> AccountInfo:
        self.ensure_connected()
        raw = mt5.account_info()
        if raw is None:
            self._raise_last_error("account_info")
        return AccountInfo.from_raw(raw)

    def positions(self, symbol: Optional[str] = None):
        self.ensure_connected()
        return mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()

    # ---------------- order sending (Day 4 entry-points) ----------------
    @staticmethod
    def _result_to_dict(result: Any) -> dict:
        return {
            "retcode": result.retcode,
            "deal":    getattr(result, "deal", 0),
            "order":   getattr(result, "order", 0),
            "volume":  getattr(result, "volume", 0.0),
            "price":   getattr(result, "price", 0.0),
            "bid":     getattr(result, "bid", 0.0),
            "ask":     getattr(result, "ask", 0.0),
            "comment": getattr(result, "comment", ""),
            "request_id": getattr(result, "request_id", 0),
            "rc":      getattr(result, "rc", 0),
        }

    def send_request(self, request: dict, max_retries: int = 3) -> dict:
        """Send a raw MT5 order request dict. Returns MT5 SendResult dict.

        FIX-MT5-01: previously a single unretried call — the highest-value
        operation to make resilient (order placement) had zero resilience
        while connect() had three layers of retry. Now retries transient
        retcodes (requote, timeout, connection drop) with a bounded backoff,
        re-fetching a fresh price for BUY/SELL market requests before each
        retry so a stale requote doesn't just get resubmitted unchanged.
        Terminal rejections (insufficient funds, invalid stops, market
        closed, etc.) are NOT retried — they fail fast on the first attempt.
        """
        self.ensure_connected()
        last_result = None
        for attempt in range(1, max_retries + 1):
            result = mt5.order_send(request)
            if result is None:
                self._raise_last_error("order_send")
            last_result = result

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                if attempt > 1:
                    log.info("order_send succeeded on retry %d/%d (retcode=%s)",
                             attempt, max_retries, result.retcode)
                return self._result_to_dict(result)

            if result.retcode not in _RETRYABLE_RETCODES or attempt == max_retries:
                log.error("order_send FAILED (retcode=%s, comment=%r) — "
                         "%s", result.retcode, result.comment,
                         "not retryable" if result.retcode not in _RETRYABLE_RETCODES
                         else "retries exhausted")
                return self._result_to_dict(result)

            log.warning("order_send retryable failure (retcode=%s, comment=%r), "
                       "attempt %d/%d — retrying with fresh price",
                       result.retcode, result.comment, attempt, max_retries)

            # Refresh price before retrying a market order so we don't
            # resubmit the same stale price that just got requoted/timed out.
            symbol = request.get("symbol")
            if symbol and request.get("type") in (
                getattr(mt5, "ORDER_TYPE_BUY", 0), getattr(mt5, "ORDER_TYPE_SELL", 1)
            ):
                try:
                    tick = self.symbol_tick(symbol)
                    is_buy = request["type"] == getattr(mt5, "ORDER_TYPE_BUY", 0)
                    request["price"] = float(tick.ask if is_buy else tick.bid)
                except Exception as e:  # noqa: BLE001
                    log.warning("could not refresh price for retry: %r", e)

            time.sleep(0.3 * attempt)

        # Should not reach here, but keep a safe fallback.
        return self._result_to_dict(last_result)

    def symbol_tick(self, symbol: str):
        self.ensure_connected()
        self.ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self._raise_last_error(f"symbol_info_tick({symbol})")
        return tick

    def symbol_info(self, symbol: str):
        self.ensure_connected()
        info = mt5.symbol_info(symbol)
        if info is None:
            self._raise_last_error(f"symbol_info({symbol})")
        return info

    def get_symbols_by_pattern(self, patterns: list[str]) -> list[str]:
        """Get symbols matching any of the given patterns (case-insensitive).

        Example: get_symbols_by_pattern(["BTC", "ETH", "XRP", "EUR", "GBP", "Volatility"])
        """
        self.ensure_connected()
        all_syms = mt5.symbols_get() or []
        matched = []
        for s in all_syms:
            name = s.name.upper()
            for p in patterns:
                if p.upper() in name:
                    matched.append(s.name)
                    break
        return sorted(set(matched))

    # ---------------- helpers ----------------
    @staticmethod
    def _raise_last_error(context: str) -> None:
        if not _MT5_AVAILABLE:
            raise MT5Unavailable("MT5 backend not installed")
        err = mt5.last_error()
        msg = f"MT5 error during {context}: {err}"
        log.error(msg)
        raise RuntimeError(msg)

    @property
    def connected(self) -> bool:
        return self._connected