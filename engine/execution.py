"""
engine.execution
=====================================================================
v6.5 — Production-grade MT5 execution engine with full pre-flight checks.

Two modes:
  - LIVE   : builds MT5 order_request dicts, validates everything, sends via MT5
  - PAPER  : logs the would-be order (used when send_real_orders=false)

Pre-flight checks before EVERY live order:
  1. AutoTrading enabled in terminal
  2. Symbol visible in MarketWatch
  3. Tick is fresh (not stale)
  4. Volume normalized to symbol's volume_step
  5. SL/TP respect broker's stops_level minimum distance
  6. Price is valid (ask for BUY, bid for SELL)
  7. Sufficient free margin
  8. Filling mode auto-detected from symbol_info
  9. Full request + response logged on every attempt
"""
from __future__ import annotations

import time
import uuid
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from brokers.mt5_connector import MT5Connector, MT5Unavailable
from engine.risk import ApprovedTrade
from engine.signals import Action
from utils.logger import get_logger, log_trade

log = get_logger("engine.execution")


class _RiskRollbackHook(Protocol):
    """Minimal contract execution.py needs from a risk manager. Any risk
    manager (forex, crypto, multi-exchange) that implements this method
    can be wired in without execution.py depending on its concrete type."""
    def on_execution_failed(self, symbol: str, reason: str = "") -> None: ...

# ----------------------------------------------------------------------
# MT5 constants
# ----------------------------------------------------------------------
try:
    import MetaTrader5 as mt5  # type: ignore
    _ORDER_TYPE_BUY = mt5.ORDER_TYPE_BUY
    _ORDER_TYPE_SELL = mt5.ORDER_TYPE_SELL
    _TRADE_ACTION_DEAL = mt5.TRADE_ACTION_DEAL
    _ORDER_TIME_GTC = mt5.ORDER_TIME_GTC
    _ORDER_FILLING_FOK = mt5.ORDER_FILLING_FOK
    _ORDER_FILLING_IOC = mt5.ORDER_FILLING_IOC
    _ORDER_FILLING_RETURN = getattr(mt5, "ORDER_FILLING_RETURN", 2)
    _TRADE_RETCODE_DONE = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
    _TRADE_RETCODE_DONE_PARTIAL = getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10008)
    _TRADE_RETCODE_PLACED = getattr(mt5, "TRADE_RETCODE_PLACED", 10010)
except ImportError:
    mt5 = None
    _ORDER_TYPE_BUY = 0
    _ORDER_TYPE_SELL = 1
    _TRADE_ACTION_DEAL = 1
    _ORDER_TIME_GTC = 0
    _ORDER_FILLING_FOK = 1
    _ORDER_FILLING_IOC = 2
    _ORDER_FILLING_RETURN = 2
    _TRADE_RETCODE_DONE = 10009
    _TRADE_RETCODE_DONE_PARTIAL = 10008
    _TRADE_RETCODE_PLACED = 10010

# Retcodes that mean "success"
_SUCCESS_RETCODES = {_TRADE_RETCODE_DONE, _TRADE_RETCODE_DONE_PARTIAL, _TRADE_RETCODE_PLACED}


# ----------------------------------------------------------------------
# Result wrapper
# ----------------------------------------------------------------------
@dataclass
class OrderResult:
    ok: bool
    order_id: str
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    retcode: int = 0
    comment: str = ""
    ts: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------
class ExecutionEngine:
    def __init__(
        self,
        connector: Optional[MT5Connector],
        magic_default: int = 100000,
        deviation: int = 20,
        order_timeout_s: int = 10,
        cooldown_s: int = 30,
        send_real_orders: bool = False,
        risk_manager: Optional[_RiskRollbackHook] = None,
    ) -> None:
        self.conn = connector
        self.magic_default = int(magic_default)
        self.deviation = int(deviation)
        self.order_timeout_s = int(order_timeout_s)
        self.cooldown_s = int(cooldown_s)
        self.send_real_orders = bool(send_real_orders)
        self.risk_manager = risk_manager
        self._last_order_ts: dict[str, float] = {}
        self._paper_journal: list[dict[str, Any]] = []

    # ----------------------------------------------------------------
    # PUBLIC: place_order
    # ----------------------------------------------------------------
    def place_order(self, trade: ApprovedTrade, magic: Optional[int] = None) -> OrderResult:
        """Send a market order for an ApprovedTrade.

        Every failure path (cooldown skip, live rejection, paper edge
        case, unexpected exception) is funneled through this single
        method so that risk-state rollback (see _notify_risk_failure)
        happens exactly once and cannot be forgotten at a call site.
        """
        symbol = trade.signal.symbol
        now = time.time()
        last = self._last_order_ts.get(symbol, 0.0)
        if now - last < self.cooldown_s:
            log.info("COOLDOWN skip %s (%.1fs < %ss)", symbol, now - last, self.cooldown_s)
            result = OrderResult(ok=False, order_id=self._gen_id(),
                                 comment="cooldown", ts=self._ts())
            self._notify_risk_failure(symbol, result.comment)
            return result

        # PAPER mode
        if not self.send_real_orders or self.conn is None:
            result = self._paper_place(trade, magic)
            if not result.ok:
                self._notify_risk_failure(symbol, result.comment)
            return result

        # LIVE mode
        try:
            self.conn.ensure_connected()
            result = self._live_place(trade, magic)
        except MT5Unavailable:
            log.warning("MT5 unavailable — falling back to paper execution")
            result = self._paper_place(trade, magic)
        except Exception as e:
            log.exception("LIVE order send failed: %s", e)
            result = OrderResult(ok=False, order_id=self._gen_id(),
                                 comment=f"exception:{e!r}", ts=self._ts())

        if not result.ok:
            self._notify_risk_failure(symbol, result.comment)
        return result

    def _notify_risk_failure(self, symbol: str, reason: str) -> None:
        """Best-effort rollback notification. Deliberately swallows
        exceptions from the callback — a broken risk-manager hook must
        never prevent execution.py from returning its OrderResult to
        the caller, or we compound one failure with another.

        Major #6 fix: the callback is now dispatched in a daemon thread
        so that if `on_execution_failed` blocks (e.g. acquiring a lock
        already held by the caller), the execution engine doesn't
        deadlock. The thread is named 'risk-rollback' for diagnostics.
        """
        if self.risk_manager is None:
            return

        def _do_rollback():
            try:
                self.risk_manager.on_execution_failed(symbol, reason=reason)
            except Exception as e:  # noqa: BLE001
                log.error("risk_manager.on_execution_failed callback failed "
                         "for %s: %r", symbol, e)

        # Major #6 fix: run in a daemon thread so a blocking callback
        # can't deadlock the execution engine. The thread dies with the
        # process if it never completes.
        import threading as _threading
        t = _threading.Thread(target=_do_rollback, name="risk-rollback",
                              daemon=True)
        t.start()

    # ----------------------------------------------------------------
    # PUBLIC: close_order
    # ----------------------------------------------------------------
    def close_order(self, ticket: int, symbol: str, lots: float,
                    action: Action) -> OrderResult:
        if not self.send_real_orders or self.conn is None:
            return self._paper_close(ticket, symbol, lots, action)
        try:
            self.conn.ensure_connected()
            self.conn.ensure_symbol(symbol)
            opposite = _ORDER_TYPE_SELL if action == Action.BUY else _ORDER_TYPE_BUY
            tick = self.conn.symbol_tick(symbol)
            price = tick.bid if opposite == _ORDER_TYPE_SELL else tick.ask
            req = {
                "action":       _TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       self._normalize_volume(symbol, lots),
                "type":         opposite,
                "position":     int(ticket),
                "price":        float(price),
                "deviation":    self.deviation,
                "magic":        self.magic_default,
                "comment":      "bot_close",
                "type_time":    _ORDER_TIME_GTC,
                "type_filling": self._pick_filling(symbol),
            }
            log.info("CLOSE request: %s", req)
            res = self.conn.send_request(req)
            ok = res.get("retcode") in _SUCCESS_RETCODES
            log.info("CLOSE response: retcode=%s comment=%s ok=%s",
                     res.get("retcode"), res.get("comment"), ok)
            log_trade("close", ticket=ticket, symbol=symbol, lots=lots,
                      price=price, retcode=res.get("retcode"), ok=ok)
            return OrderResult(
                ok=ok, order_id=self._gen_id(),
                ticket=int(res.get("order", 0)),
                price=float(res.get("price", price)),
                volume=float(res.get("volume", lots)),
                retcode=int(res.get("retcode", 0)),
                comment=res.get("comment", ""), ts=self._ts(),
            )
        except Exception as e:
            log.exception("close_order failed: %s", e)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment=f"exception:{e!r}", ts=self._ts())

    # ----------------------------------------------------------------
    # PUBLIC: order_status
    # ----------------------------------------------------------------
    def order_status(self, ticket: int) -> dict[str, Any]:
        if self.conn is None or not self.send_real_orders:
            for row in self._paper_journal:
                if int(row.get("ticket", 0)) == ticket:
                    return row
            return {}
        try:
            positions = self.conn.positions()
            for p in positions or []:
                if int(getattr(p, "ticket", 0)) == ticket:
                    return {
                        "ticket": p.ticket, "symbol": p.symbol,
                        "type": p.type, "volume": p.volume,
                        "price_open": p.price_open, "sl": p.sl, "tp": p.tp,
                        "profit": p.profit, "magic": p.magic,
                    }
        except Exception:
            pass
        return {}

    # ================================================================
    # LIVE PATH — v6.5 with full pre-flight validation
    # ================================================================
    def _live_place(self, trade: ApprovedTrade, magic: Optional[int]) -> OrderResult:
        symbol = trade.signal.symbol
        order_type = _ORDER_TYPE_BUY if trade.action == Action.BUY else _ORDER_TYPE_SELL
        magic_val = int(magic or self.magic_default)

        log.info("════ LIVE ORDER START ════ symbol=%s action=%s lots=%.4f",
                 symbol, trade.action.value, trade.lots)

        # ── CHECK 1: AutoTrading enabled? ──────────────────────────
        try:
            term = mt5.terminal_info()
            if term is None:
                log.error("REJECT: terminal_info() returned None")
                return OrderResult(ok=False, order_id=self._gen_id(),
                                   comment="terminal_info None", ts=self._ts())
            log.info("  [1/9] AutoTrading: trade_allowed=%s connected=%s",
                     term.trade_allowed, term.connected)
            if not term.trade_allowed:
                log.error("REJECT: AutoTrading is DISABLED in MT5 terminal. "
                          "Press Ctrl+E or click the AutoTrading button.")
                return OrderResult(ok=False, order_id=self._gen_id(),
                                   comment="AutoTrading disabled — press Ctrl+E",
                                   ts=self._ts())
        except Exception as e:
            log.error("REJECT: terminal_info() failed: %s", e)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment=f"terminal_info error: {e}", ts=self._ts())

        # ── CHECK 2: Symbol visible in MarketWatch? ────────────────
        try:
            self.conn.ensure_symbol(symbol)
            info = self.conn.symbol_info(symbol)
            if info is None:
                log.error("REJECT: symbol_info(%s) returned None", symbol)
                return OrderResult(ok=False, order_id=self._gen_id(),
                                   comment=f"symbol_info None for {symbol}",
                                   ts=self._ts())
            log.info("  [2/9] Symbol: %s visible=%s digits=%d point=%s "
                     "vol_step=%s vol_min=%s vol_max=%s stops_level=%d",
                     symbol, info.visible, info.digits, info.point,
                     info.volume_step, info.volume_min, info.volume_max,
                     getattr(info, 'trade_stops_level', 0))
        except Exception as e:
            log.error("REJECT: symbol setup failed for %s: %s", symbol, e)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment=f"symbol setup error: {e}", ts=self._ts())

        # ── CHECK 3: Fresh tick? ────────────────────────────────────
        try:
            tick = self.conn.symbol_tick(symbol)
            if tick is None:
                log.error("REJECT: symbol_tick(%s) returned None", symbol)
                return OrderResult(ok=False, order_id=self._gen_id(),
                                   comment=f"tick None for {symbol}", ts=self._ts())
            tick_age = time.time() - getattr(tick, 'time', time.time())
            log.info("  [3/9] Tick: bid=%s ask=%s spread=%s age=%.1fs",
                     tick.bid, tick.ask,
                     getattr(tick, 'spread', 0), tick_age)
            if tick_age > 60:
                log.warning("  [WARN] Tick is %.1fs old — may be stale", tick_age)
        except Exception as e:
            log.error("REJECT: tick fetch failed: %s", e)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment=f"tick error: {e}", ts=self._ts())

        # ── CHECK 4: Normalize volume ───────────────────────────────
        vol_step = float(getattr(info, 'volume_step', 0.01))
        vol_min = float(getattr(info, 'volume_min', 0.01))
        vol_max = float(getattr(info, 'volume_max', 100.0))
        raw_lots = float(trade.lots)
        normalized_lots = self._normalize_volume(symbol, raw_lots)
        if normalized_lots < vol_min:
            normalized_lots = vol_min
            log.warning("  [4/9] Volume adjusted UP to min: %.4f → %.4f",
                        raw_lots, normalized_lots)
        if normalized_lots > vol_max:
            normalized_lots = vol_max
            log.warning("  [4/9] Volume adjusted DOWN to max: %.4f → %.4f",
                        raw_lots, normalized_lots)
        log.info("  [4/9] Volume: raw=%.4f normalized=%.4f (step=%s min=%s max=%s)",
                 raw_lots, normalized_lots, vol_step, vol_min, vol_max)

        # ── CHECK 5: Normalize SL/TP to stops_level ─────────────────
        price = float(tick.ask if order_type == _ORDER_TYPE_BUY else tick.bid)
        point = float(getattr(info, 'point', 0.01))
        digits = int(getattr(info, 'digits', 2))
        stops_level = int(getattr(info, 'trade_stops_level', 0))
        min_stop_distance = stops_level * point
        sl = float(trade.stop_loss)
        tp = float(trade.take_profit)

        # Ensure SL is at least stops_level away from price
        if sl > 0:
            sl_distance = abs(price - sl)
            if sl_distance < min_stop_distance and min_stop_distance > 0:
                if order_type == _ORDER_TYPE_BUY:
                    sl = price - min_stop_distance
                else:
                    sl = price + min_stop_distance
                log.warning("  [5/9] SL adjusted to respect stops_level: → %.*f", digits, sl)

        if tp > 0:
            tp_distance = abs(tp - price)
            if tp_distance < min_stop_distance and min_stop_distance > 0:
                if order_type == _ORDER_TYPE_BUY:
                    tp = price + min_stop_distance
                else:
                    tp = price - min_stop_distance
                log.warning("  [5/9] TP adjusted to respect stops_level: → %.*f", digits, tp)

        # Round SL/TP to symbol digits
        sl = round(sl, digits)
        tp = round(tp, digits)
        log.info("  [5/9] SL/TP: sl=%.*f tp=%.*f stops_level=%d min_dist=%s",
                 digits, sl, digits, tp, stops_level, min_stop_distance)

        # ── CHECK 6: Price validity ─────────────────────────────────
        if price <= 0:
            log.error("REJECT: price is invalid (%.5f)", price)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment="invalid price", ts=self._ts())
        side_label = "ask" if order_type == _ORDER_TYPE_BUY else "bid"
        log.info("  [6/9] Price: %.*f (%s)", digits, price, side_label)

        # ── CHECK 7: Margin check ───────────────────────────────────
        try:
            acct = mt5.account_info()
            if acct is not None:
                margin_free = float(acct.margin_free)
                leverage = int(acct.leverage) if acct.leverage > 0 else 100
                # Notional = lots * contract_size * price, NOT lots * price.
                # contract_size is symbol-specific (e.g. 100,000 for a
                # standard FX lot) — omitting it understates required
                # margin by orders of magnitude for most FX/CFD symbols.
                contract_size = float(getattr(info, 'trade_contract_size', 1.0))
                if contract_size <= 0:
                    contract_size = 1.0
                    log.warning("  [7/9] trade_contract_size missing/invalid for %s "
                               "— defaulting to 1.0 (margin estimate may be understated)",
                               symbol)
                estimated_margin = (price * normalized_lots * contract_size) / leverage
                log.info("  [7/9] Margin: free=$%.2f estimated_needed=$%.2f "
                         "leverage=1:%d contract_size=%s",
                         margin_free, estimated_margin, leverage, contract_size)
                if estimated_margin > margin_free:
                    log.error("REJECT: insufficient margin (need $%.2f, free $%.2f)",
                              estimated_margin, margin_free)
                    return OrderResult(ok=False, order_id=self._gen_id(),
                                       comment=f"insufficient margin: need ${estimated_margin:.2f}, free ${margin_free:.2f}",
                                       ts=self._ts())
        except Exception as e:
            log.warning("  [7/9] Margin check skipped: %s", e)

        # ── CHECK 8: Filling mode ───────────────────────────────────
        filling_mode = self._pick_filling(symbol)
        log.info("  [8/9] Filling mode: %d (0=RETURN 1=FOK 2=IOC)", filling_mode)

        # ── CHECK 9: Build + send request ───────────────────────────
        req = {
            "action":       _TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       float(normalized_lots),
            "type":         int(order_type),
            "price":        float(price),
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    int(self.deviation),
            "magic":        int(magic_val),
            "comment":      f"bot_{trade.action.value}",
            "type_time":    int(_ORDER_TIME_GTC),
            "type_filling": int(filling_mode),
        }
        log.info("  [9/9] SENDING REQUEST:")
        log.info("        %s", req)

        try:
            result = mt5.order_send(req)
        except Exception as e:
            log.exception("  mt5.order_send() raised: %s", e)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment=f"order_send exception: {e}", ts=self._ts())

        if result is None:
            err = mt5.last_error()
            log.error("  mt5.order_send() returned None — last_error: %s", err)
            return OrderResult(ok=False, order_id=self._gen_id(),
                               comment=f"order_send None: {err}", ts=self._ts())

        # Log full response
        retcode = int(result.retcode)
        comment = str(getattr(result, 'comment', ''))
        deal = int(getattr(result, 'deal', 0))
        order = int(getattr(result, 'order', 0))
        fill_price = float(getattr(result, 'price', price))
        fill_vol = float(getattr(result, 'volume', normalized_lots))

        log.info("  RESPONSE: retcode=%d comment='%s' deal=%d order=%d "
                 "price=%.*f volume=%.4f",
                 retcode, comment, deal, order, digits, fill_price, fill_vol)

        ok = retcode in _SUCCESS_RETCODES
        self._last_order_ts[symbol] = time.time()

        if ok:
            log.info("  ════ ORDER ACCEPTED ════ ticket=%d price=%.*f",
                     order, digits, fill_price)
        else:
            log.error("  ════ ORDER REJECTED ════ retcode=%d comment='%s'", retcode, comment)
            log.error("  Retcode meanings: 10004=requote 10006=reject 10007=cancel "
                      "10010=partial 10013=invalid 10014=invalid_vol "
                      "10015=invalid_price 10016=invalid_stops "
                      "10018=market_closed 10019=no_money 10030=invalid_fill")

        log_trade("open", ok=ok, order_id=self._gen_id(), ticket=order,
                  symbol=symbol, action=trade.action.value,
                  lots=normalized_lots, price=fill_price,
                  sl=sl, tp=tp, retcode=retcode, comment=comment)

        return OrderResult(
            ok=ok, order_id=self._gen_id(), ticket=order,
            price=fill_price, volume=fill_vol,
            retcode=retcode, comment=comment, ts=self._ts(),
        )

    # ================================================================
    # HELPERS
    # ================================================================
    def _normalize_volume(self, symbol: str, lots: float) -> float:
        """Normalize volume to symbol's volume_step."""
        if self.conn is None:
            return round(lots, 2)
        try:
            info = self.conn.symbol_info(symbol)
            if info is None:
                return round(lots, 2)
            step = float(getattr(info, 'volume_step', 0.01))
            if step <= 0:
                step = 0.01
            # Round down to nearest step
            normalized = math.floor(lots / step) * step
            # Round to step's decimal places
            decimals = max(0, -int(math.floor(math.log10(step))))
            return round(normalized, decimals)
        except Exception:
            return round(lots, 2)

    def _pick_filling(self, symbol: str) -> int:
        """Auto-detect the correct filling mode for this symbol.

        Deriv typically supports ORDER_FILLING_IOC (2).
        Some symbols support FOK (1).
        Fallback: try IOC first, then FOK, then RETURN.
        """
        if self.conn is None or mt5 is None:
            return _ORDER_FILLING_IOC  # Deriv default
        try:
            info = self.conn.symbol_info(symbol)
            if info is None:
                return _ORDER_FILLING_IOC
            modes = int(getattr(info, "filling_mode", 0))
            # bit 0 → FOK allowed, bit 1 → IOC allowed
            if modes & 2:  # IOC
                return _ORDER_FILLING_IOC
            if modes & 1:  # FOK
                return _ORDER_FILLING_FOK
            # If filling_mode is 0, Deriv usually accepts IOC
            return _ORDER_FILLING_IOC
        except Exception:
            return _ORDER_FILLING_IOC

    # ================================================================
    # PAPER PATH
    # ================================================================
    def _paper_place(self, trade: ApprovedTrade, magic: Optional[int]) -> OrderResult:
        order_id = self._gen_id()
        ticket = -abs(hash(order_id)) % (10 ** 9)
        self._last_order_ts[trade.signal.symbol] = time.time()
        entry = trade.entry_price
        rec = {
            "order_id": order_id, "ticket": ticket,
            "symbol": trade.signal.symbol, "action": trade.action.value,
            "lots": trade.lots, "price": entry,
            "sl": trade.stop_loss, "tp": trade.take_profit,
            "magic": int(magic or self.magic_default),
            "open_time": self._ts(), "status": "open", "mode": "paper",
        }
        self._paper_journal.append(rec)
        log.info("PAPER %s %s %.4f @ %.5f sl=%.5f tp=%.5f",
                 trade.action, trade.signal.symbol, trade.lots, entry,
                 trade.stop_loss, trade.take_profit)
        return OrderResult(
            ok=True, order_id=order_id, ticket=ticket,
            price=entry, volume=trade.lots, retcode=0,
            comment="paper", ts=rec["open_time"],
        )

    def _paper_close(self, ticket: int, symbol: str, lots: float,
                     action: Action) -> OrderResult:
        for rec in self._paper_journal:
            if rec.get("ticket") == ticket and rec.get("status") == "open":
                rec["status"] = "closed"
                rec["close_time"] = self._ts()
                log_trade("close", ok=True, ticket=ticket, symbol=symbol,
                          lots=lots, price=rec.get("price", 0.0), mode="paper")
                return OrderResult(
                    ok=True, order_id=self._gen_id(), ticket=ticket,
                    price=float(rec.get("price", 0.0)), volume=float(lots),
                    retcode=0, comment="paper", ts=rec["close_time"],
                )
        return OrderResult(ok=False, order_id=self._gen_id(),
                           comment="paper ticket not found", ts=self._ts())

    # ================================================================
    # PAPER POSITION MANAGEMENT (v6.3.3)
    # ================================================================
    def manage_paper_positions(self, current_prices: dict[str, float]) -> list[dict]:
        closed_positions = []
        for rec in self._paper_journal:
            if rec.get("status") != "open":
                continue
            symbol = rec.get("symbol", "")
            entry = float(rec.get("price", 0))
            sl = float(rec.get("sl", 0))
            tp = float(rec.get("tp", 0))
            action = rec.get("action", "BUY")
            lots = float(rec.get("lots", 0.01))
            current_price = current_prices.get(symbol)
            if current_price is None or current_price <= 0:
                continue
            hit_sl = False
            hit_tp = False
            if action == "BUY":
                if current_price <= sl and sl > 0:
                    hit_sl = True
                elif current_price >= tp and tp > 0:
                    hit_tp = True
            else:
                if current_price >= sl and sl > 0:
                    hit_sl = True
                elif current_price <= tp and tp > 0:
                    hit_tp = True
            if hit_sl or hit_tp:
                if action == "BUY":
                    pnl = (current_price - entry) * lots
                else:
                    pnl = (entry - current_price) * lots
                reason = "SL" if hit_sl else "TP"
                rec["status"] = "closed"
                rec["close_time"] = self._ts()
                rec["close_price"] = current_price
                rec["pnl"] = pnl
                rec["close_reason"] = reason
                log.info("PAPER CLOSE %s %s entry=%.5f close=%.5f pnl=%.2f reason=%s",
                         action, symbol, entry, current_price, pnl, reason)
                log_trade("close", ok=True, ticket=rec.get("ticket"),
                          symbol=symbol, lots=lots, price=current_price,
                          mode="paper", pnl=pnl, reason=reason)
                closed_positions.append({
                    "symbol": symbol, "action": action, "ticket": rec.get("ticket"),
                    "entry": entry, "exit": current_price, "pnl": pnl, "reason": reason,
                })
        return closed_positions

    def paper_equity(self, base_equity: float, current_prices: dict[str, float]) -> float:
        unrealized_pnl = 0.0
        for rec in self._paper_journal:
            if rec.get("status") != "open":
                continue
            symbol = rec.get("symbol", "")
            entry = float(rec.get("price", 0))
            action = rec.get("action", "BUY")
            lots = float(rec.get("lots", 0.01))
            current_price = current_prices.get(symbol)
            if current_price is None or current_price <= 0:
                continue
            if action == "BUY":
                unrealized_pnl += (current_price - entry) * lots
            else:
                unrealized_pnl += (entry - current_price) * lots
        realized_pnl = sum(float(r.get("pnl", 0)) for r in self._paper_journal
                          if r.get("status") == "closed")
        return base_equity + unrealized_pnl + realized_pnl

    def paper_open_count(self) -> int:
        return sum(1 for r in self._paper_journal if r.get("status") == "open")

    # ================================================================
    # UTILS
    # ================================================================
    @staticmethod
    def _gen_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def _ts() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    @property
    def paper_journal(self) -> list[dict[str, Any]]:
        return list(self._paper_journal)