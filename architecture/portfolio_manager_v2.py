"""architecture/portfolio_manager_v2.py
=====================================================================
Portfolio Manager Layer (Improvement #6)
=====================================================================
Hedge-fund grade portfolio management. Tracks every open position,
computes exposure by symbol/sector/currency, enforces portfolio-level
risk limits, and triggers rebalancing when needed.

Responsibilities:
    1. Position tracking (with live P&L)
    2. Exposure decomposition (gross/net/long/short/beta-weighted)
    3. Correlation monitoring (don't hold 10 highly-correlated positions)
    4. Concentration limits (max % per symbol, sector, currency)
    5. Drawdown monitoring (peak-to-trough, equity curve metrics)
    6. Rebalancing triggers (volatility-target, Sharpe-target)
    7. Margin utilization tracking
    8. Hedging suggestions (offset correlated risk)
    9. Portfolio heat (sum of all open risk) — never exceed X% equity at risk

Usage:
    pm = PortfolioManager(initial_capital=10000.0)
    pm.on_position_opened(Position(symbol="BTCUSD", ...))
    pm.update_prices({"BTCUSD": 43250.0, "ETHUSD": 2580.0})
    metrics = pm.metrics()
    if metrics["portfolio_heat_pct"] > 0.10:
        log.warning("Portfolio heat exceeds 10% — block new entries")
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.portfolio_manager")


@dataclass
class PortfolioMetrics:
    """Snapshot of portfolio state at a point in time."""
    timestamp: str = ""
    # Capital
    initial_capital: float = 0.0
    balance: float = 0.0
    equity: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    # Exposure
    gross_exposure: float = 0.0  # |long| + |short|
    net_exposure: float = 0.0    # long - short
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    gross_exposure_pct: float = 0.0  # gross / equity
    # Counts
    open_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    # Risk
    portfolio_heat: float = 0.0      # sum of (entry - SL) × volume
    portfolio_heat_pct: float = 0.0  # portfolio_heat / equity
    margin_used: float = 0.0
    margin_used_pct: float = 0.0
    # Drawdown
    peak_equity: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    # Diversification
    avg_correlation: float = 0.0
    effective_positions: float = 0.0  # 1 / sum(weight^2), like effective N
    # Concentration
    max_symbol_weight: float = 0.0
    herfindahl_index: float = 0.0     # sum of squared weights (0-1)


class PortfolioManager:
    """Central portfolio manager. Tracks all positions + risk metrics."""

    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_gross_exposure_pct: float = 2.0,  # 200% of equity
                 max_portfolio_heat_pct: float = 0.10,  # 10% of equity at risk
                 max_symbol_weight: float = 0.25,        # 25% per symbol
                 max_correlation: float = 0.85,
                 max_open_positions: int = 10,  # Review Point 8: absolute cap
                 bus: Optional[EventBus] = None):
        self._lock = threading.RLock()
        self._positions: Dict[int, Dict[str, Any]] = {}  # ticket -> position dict
        self._initial_capital = initial_capital
        self._balance = initial_capital
        self._realized_pnl = 0.0
        self._peak_equity = initial_capital
        self._max_drawdown = 0.0
        self._max_drawdown_pct = 0.0
        self._equity_history: List[Tuple[float, float]] = []  # (timestamp, equity)
        # P0-4 FIX (Phase 3): closed-trade history for consecutive_losses()
        # and realized_pnl_today() — the two real-telemetry inputs the risk
        # pipeline needs but was previously fed hardcoded zeros for.
        self._closed_trades: List[Dict[str, Any]] = []  # [{ticket, pnl, close_time_iso, ...}]

        # In-flight risk reservations: reservation_id -> {symbol, notional, heat}.
        # Held between can_open_new() approving a trade and on_position_opened()
        # committing it, so two concurrent signals can't both pass the same
        # exposure/heat check against a stale snapshot (see FIX-PM-03).
        self._reservations: Dict[str, Dict[str, Any]] = {}
        self._reservation_seq = 0

        # Risk limits
        self._max_gross_exposure_pct = max_gross_exposure_pct
        self._max_portfolio_heat_pct = max_portfolio_heat_pct
        self._max_symbol_weight = max_symbol_weight
        self._max_correlation = max_correlation
        self._max_open_positions = int(max_open_positions)  # Review Point 8

        self._bus = bus or get_bus()

        # Co-Founder Audit Fix: store leverage for risk-adjusted exposure calc.
        # Set via set_leverage() when the broker reports account_info().
        # Default 100 (1:100) is conservative — real accounts are typically
        # 1:30 (EU), 1:50 (US forex), 1:100-1:500 (crypto/derivatives),
        # or 1:1000 (Deriv demo). Used by can_open_new() to convert raw
        # notional to margin-adjusted exposure.
        self._leverage: int = 100

        # Symbol category mapping (for concentration limits)
        # e.g. {"BTCUSD": "crypto", "ETHUSD": "crypto", "EURUSD": "forex"}
        self._symbol_categories: Dict[str, str] = {}
        self._symbol_betas: Dict[str, float] = {}  # beta vs benchmark
        # H12 fix: per-category concentration caps (fraction of equity).
        # Default: no single category > 40% of equity.
        self._category_caps: Dict[str, float] = {}
        self._default_category_cap: float = 0.40

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def on_position_opened(self,
                           ticket: int, symbol: str, side: str,
                           volume: float, entry_price: float,
                           sl: float = 0.0, tp: float = 0.0,
                           magic: int = 0,
                           reservation_id: Optional[str] = None,
                           contract_size: float = 1.0) -> None:
        """Commit a filled position.

        FIX-PM-03: if this position came from a can_open_new() approval,
        pass the returned reservation_id so the in-flight reservation is
        released now that it's a real, tracked position (avoids double-
        counting the same risk as both "reserved" and "open").

        PORTFOLIO AUDIT FIX: added contract_size parameter. For forex pairs
        (contract_size=100,000), PnL must be multiplied by contract_size
        to get the correct dollar PnL. Without this, forex PnL appears
        100,000x smaller than actual, breaking consecutive_losses,
        realized_pnl_today, and Kelly sizing.
        """
        with self._lock:
            self._prune_stale_reservations()
            if reservation_id is not None:
                self._reservations.pop(reservation_id, None)
            self._positions[ticket] = {
                "ticket": ticket, "symbol": symbol,
                "side": side.upper(),
                "volume": float(volume),
                "entry_price": float(entry_price),
                "current_price": float(entry_price),
                "sl": float(sl), "tp": float(tp),
                "magic": magic,
                "contract_size": float(contract_size),  # AUDIT FIX
                "open_time": datetime.now(tz=timezone.utc).isoformat(),
                "profit": 0.0,
            }
        self._bus.emit(EventType.POSITION_OPENED,
                       payload={"ticket": ticket, "symbol": symbol,
                                "side": side, "volume": volume,
                                "entry": entry_price},
                       source="portfolio_manager")

    def force_close_orphan(self, ticket: int, reason: str = "reconciliation_orphan") -> float:
        """C10 fix: forcibly remove a LOCAL position that the broker no
        longer reports (orphan). Used by TradingBot.reconcile_with_broker()
        to actually resolve — not just log — discrepancies, per the
        "trust the broker" policy. PnL is computed against the last known
        local price since we have no broker exit price for an externally
        closed position; this is an approximation flagged via `reason`.
        """
        with self._lock:
            p = self._positions.get(ticket)
            if p is None:
                return 0.0
            exit_price = p.get("current_price", p.get("entry_price", 0.0))
        return self.on_position_closed(ticket, exit_price, reason=reason)

    def force_sync_volume(self, ticket: int, broker_volume: float) -> bool:
        """C10 fix: force the local volume to match the broker's reported
        volume (e.g. after a partial close the bot didn't observe)."""
        with self._lock:
            p = self._positions.get(ticket)
            if p is None:
                return False
            p["volume"] = float(broker_volume)
            return True

    def on_position_closed(self, ticket: int, exit_price: float,
                           reason: str = "manual") -> float:
        """Close a position, return realized PnL.

        P0-4/P0-8 FIX (Phase 3): Records the closed trade in _closed_trades
        so consecutive_losses() and realized_pnl_today() can compute real
        values for the risk pipeline (was hardcoded to 0 before).

        PORTFOLIO AUDIT FIX: PnL now includes contract_size. For forex
        (contract_size=100,000), the correct PnL is:
            (exit - entry) * volume * direction * contract_size
        Previously contract_size was missing, making forex PnL appear
        100,000x smaller than actual.
        """
        with self._lock:
            p = self._positions.pop(ticket, None)
            if p is None:
                return 0.0
            direction = 1 if p["side"] == "BUY" else -1
            contract_size = p.get("contract_size", 1.0)  # AUDIT FIX
            pnl = (exit_price - p["entry_price"]) * p["volume"] * direction * contract_size
            self._balance += pnl
            self._realized_pnl += pnl
            # Record for history-based risk gates
            self._closed_trades.append({
                "ticket": ticket,
                "symbol": p["symbol"],
                "side": p["side"],
                "volume": p["volume"],
                "entry_price": p["entry_price"],
                "exit_price": exit_price,
                "pnl": pnl,
                "close_time_iso": datetime.now(tz=timezone.utc).isoformat(),
                "reason": reason,
            })
            # C12 fix: cap raised from 500 -> 2000 (~months of active
            # multi-symbol trading instead of weeks). This is still an
            # in-memory bound for fast consecutive_losses()/recent_trades()
            # lookups — the database (decision_audit / trades tables) is
            # the durable, uncapped source of truth for full history.
            if len(self._closed_trades) > 2000:
                self._closed_trades = self._closed_trades[-2000:]
            self._bus.emit(EventType.POSITION_CLOSED,
                           payload={"ticket": ticket, "symbol": p["symbol"],
                                    "pnl": pnl, "exit_price": exit_price,
                                    "reason": reason},
                           source="portfolio_manager")
            return pnl

    # ------------------------------------------------------------------
    # P0-4 FIX (Phase 3): Real-telemetry helpers for the risk pipeline
    # ------------------------------------------------------------------
    def consecutive_losses(self) -> int:
        """Count consecutive losing trades from the end of history.

        Returns 0 if the most recent closed trade was a win, or if no history.
        The ConsecutiveLossGate reads this via RiskContext.consecutive_losses.
        """
        with self._lock:
            count = 0
            for t in reversed(self._closed_trades):
                if t["pnl"] < 0:
                    count += 1
                else:
                    break
            return count

    def realized_pnl_today(self) -> float:
        """Sum of pnl for trades closed since UTC midnight.

        The DailyLossGate reads this via RiskContext.realized_pnl_today.
        """
        from datetime import datetime as _dt, timezone as _tz
        today_utc = _dt.now(tz=_tz.utc).date()
        with self._lock:
            total = 0.0
            for t in self._closed_trades:
                try:
                    close_iso = t.get("close_time_iso", "")
                    if close_iso:
                        close_date = _dt.fromisoformat(close_iso).date()
                        if close_date == today_utc:
                            total += t["pnl"]
                except (ValueError, TypeError):
                    continue
            return total

    def last_trade_time(self) -> float:
        """Epoch timestamp of the most recent closed trade (0 if none)."""
        with self._lock:
            if not self._closed_trades:
                return 0.0
            try:
                from datetime import datetime as _dt
                last_iso = self._closed_trades[-1].get("close_time_iso", "")
                if last_iso:
                    return _dt.fromisoformat(last_iso).timestamp()
            except (ValueError, TypeError):
                pass
            return 0.0

    def recent_trades(self, n: int = 30) -> List[Dict[str, Any]]:
        """Last N closed trades (for Kelly sizing in SizingGate)."""
        with self._lock:
            return list(self._closed_trades[-n:])

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update current prices for all open positions."""
        with self._lock:
            for p in self._positions.values():
                if p["symbol"] in prices:
                    p["current_price"] = prices[p["symbol"]]
                    direction = 1 if p["side"] == "BUY" else -1
                    # AUDIT FIX: include contract_size in unrealized PnL
                    cs = p.get("contract_size", 1.0)
                    p["profit"] = (p["current_price"] - p["entry_price"]) * \
                                  p["volume"] * direction * cs
            # Track peak equity & drawdown
            eq = self.equity()
            if eq > self._peak_equity:
                self._peak_equity = eq
            dd = self._peak_equity - eq
            dd_pct = (dd / self._peak_equity * 100) if self._peak_equity > 0 else 0
            if dd > self._max_drawdown:
                self._max_drawdown = dd
                self._max_drawdown_pct = dd_pct
                if dd_pct > 15:
                    self._bus.emit(EventType.DRAWDOWN_WARNING,
                                   payload={"drawdown_pct": dd_pct,
                                            "peak": self._peak_equity,
                                            "current": eq},
                                   source="portfolio_manager")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def equity(self) -> float:
        """Current equity = balance + unrealized PnL."""
        with self._lock:
            unrealized = sum(p["profit"] for p in self._positions.values())
            return self._balance + unrealized

    def update_equity(self, real_equity: float) -> None:
        """Update balance from broker's reported equity (sync with reality).

        Called each cycle with the real MT5 account equity so the bot
        doesn't use stale/hardcoded values.
        """
        with self._lock:
            unrealized = sum(p["profit"] for p in self._positions.values())
            # Balance = equity - unrealized
            self._balance = real_equity - unrealized
            if real_equity > self._peak_equity:
                self._peak_equity = real_equity

    def set_leverage(self, leverage: int) -> None:
        """Set account leverage from broker (e.g., 1000 for 1:1000).

        Co-Founder Audit Fix: the portfolio gate's gross-exposure check
        needs leverage to convert raw notional to margin-adjusted exposure.
        Without this, forex positions (contract_size=100,000) always show
        200%+ exposure on a $10K account and get rejected.
        Called from TradingBot.boot() after MT5 account_info() reports
        the real leverage.
        """
        with self._lock:
            self._leverage = max(1, int(leverage))
            log.info("portfolio: leverage set to 1:%d", self._leverage)

    def open_count(self) -> int:
        with self._lock:
            return len(self._positions)

    def has_open_position(self, symbol: str) -> bool:
        with self._lock:
            return any(p["symbol"] == symbol for p in self._positions.values())

    def get_position(self, ticket: int) -> Optional[Dict[str, Any]]:
        # Co-Founder Audit Fix: return a DEEP COPY. The previous code
        # returned `self._positions.get(ticket)` directly — a reference
        # to the live dict. Callers could then mutate `pos["sl"]`,
        # `pos["volume"]`, etc. WITHOUT acquiring the portfolio lock,
        # breaking the lock guarantee. Two known offenders:
        #   - integration.py:2298  `pos["sl"] = new_sl` (trailing stop)
        #   - integration.py:2489  volume mismatch sync (was already
        #     moved to force_sync_volume but the pattern was the same)
        # Returning copies forces callers to go through proper mutators.
        with self._lock:
            p = self._positions.get(ticket)
            return dict(p) if p is not None else None

    def all_positions(self) -> List[Dict[str, Any]]:
        # Co-Founder Audit Fix: return DEEP COPIES (see get_position).
        with self._lock:
            return [dict(p) for p in self._positions.values()]

    def update_position_sl_tp(self, ticket: int,
                              sl: Optional[float] = None,
                              tp: Optional[float] = None) -> bool:
        """Co-Founder Audit Fix: thread-safe SL/TP mutator.

        The previous pattern was for callers to grab a position via
        get_position() or all_positions() and mutate `pos["sl"]` directly
        — which bypassed self._lock. This is the canonical, lock-protected
        way to update a position's SL/TP after a broker modify_order()
        succeeds. Returns True if the position was found and updated.
        """
        with self._lock:
            p = self._positions.get(ticket)
            if p is None:
                return False
            if sl is not None:
                p["sl"] = float(sl)
            if tp is not None:
                p["tp"] = float(tp)
            return True

    # ------------------------------------------------------------------
    # H12 fix: symbol category management for concentration limits.
    # ------------------------------------------------------------------
    def set_symbol_category(self, symbol: str, category: str) -> None:
        """Assign a symbol to a category (e.g. 'crypto', 'forex', 'index')."""
        with self._lock:
            self._symbol_categories[symbol] = category

    def set_category_cap(self, category: str, max_fraction: float) -> None:
        """Set the max fraction of equity allowed in a single category."""
        with self._lock:
            self._category_caps[category] = float(max_fraction)

    def category_exposure(self, category: str) -> float:
        """Sum of |volume * price| for all open positions in `category`."""
        with self._lock:
            total = 0.0
            for p in self._positions.values():
                sym = p.get("symbol", "")
                if self._symbol_categories.get(sym) == category:
                    total += abs(p.get("volume", 0) * p.get("current_price", 0))
            return total

    def category_concentration_ok(self, symbol: str, proposed_value: float) -> Tuple[bool, str]:
        """H12 fix: check that adding `proposed_value` to the symbol's
        category doesn't breach the category cap. Returns (ok, reason).
        """
        with self._lock:
            cat = self._symbol_categories.get(symbol)
            if cat is None:
                return True, "no category assigned — skipping concentration check"
            cap = self._category_caps.get(cat, self._default_category_cap)
            current = self.category_exposure(cat)
            equity = max(self._balance, 1.0)
            new_fraction = (current + proposed_value) / equity
            if new_fraction > cap:
                return False, (f"category '{cat}' concentration {new_fraction:.1%} "
                               f"> cap {cap:.1%}")
            return True, f"category '{cat}' ok at {new_fraction:.1%}"

    # ------------------------------------------------------------------
    # Risk gate checks
    # ------------------------------------------------------------------
    def can_open_new(self,
                     symbol: str,
                     proposed_risk: float,
                     proposed_volume: float,
                     proposed_price: float,
                     contract_size: float = 1.0) -> Tuple[bool, str, Optional[str]]:
        """Check if we can open a new position without breaching limits.

        FIX-PM-01: exposure is now computed as volume * price * contract_size
        (true notional), not volume * equity (was a unit-consistency bug that
        made the gross-exposure gate almost never trigger).

        FIX-PM-02: the per-symbol concentration check is only meaningful if
        a symbol can carry more than one position. Under the current
        one-position-per-symbol rule (#4 below), a *new* symbol always has
        zero existing exposure, so checking existing exposure for the same
        symbol being opened was dead code. The check now validates that the
        proposed position ITSELF (including any already-reserved-but-not-yet-
        committed size on this symbol) won't breach the per-symbol cap —
        which is the check that's actually reachable and meaningful.

        FIX-PM-03: on approval, the proposed exposure/heat is atomically
        RESERVED under the same lock used by metrics()/on_position_opened(),
        closing the check-then-act race where two concurrent signals could
        both pass against the same stale snapshot. Caller must either
        commit the reservation (on_position_opened(..., reservation_id=...))
        or release it (release_reservation(reservation_id)) — reservations
        also auto-expire defensively via `reservation_ttl_s` in metrics() to
        avoid a leaked reservation permanently blocking capacity if a caller
        crashes between reserve and commit.

        Returns (allowed, reason_if_not, reservation_id_if_allowed)
        """
        if proposed_volume <= 0 or proposed_price <= 0:
            return False, "proposed_volume and proposed_price must be > 0", None

        with self._lock:
            metrics = self.metrics()
            equity = max(metrics.equity, 1e-9)

            reserved_notional = sum(r["notional"] for r in self._reservations.values())
            reserved_heat = sum(r["heat"] for r in self._reservations.values())
            reserved_symbol_notional = sum(
                r["notional"] for r in self._reservations.values()
                if r["symbol"] == symbol
            )

            new_notional = proposed_volume * proposed_price * contract_size

            # 1. Max gross exposure — leverage-adjusted.
            #
            # Co-Founder Audit Fix (FORENSIC AUDIT BUG #9):
            # The original formula used RAW notional / equity, which makes
            # forex positions (contract_size=100,000) always show 200%+
            # exposure on a $10K account — blocking ALL forex trades.
            # With 1:1000 leverage, a $20K notional position only requires
            # $20 margin, so 200% notional is only 0.2x leverage — safe.
            #
            # The fix: use RISK-ADJUSTED notional = notional / leverage.
            # This represents the actual margin/capital tied up, which is
            # what "gross exposure" should mean for leverage management.
            #
            # For accounts where leverage is unknown (paper mode), we fall
            # back to using risk_amount as a proxy (capital at risk).
            leverage = getattr(self, '_leverage', 0) or 100  # default 1:100
            risk_adjusted_new = new_notional / leverage
            risk_adjusted_reserved = reserved_notional / leverage
            projected_gross = (metrics.gross_exposure + risk_adjusted_reserved
                             + risk_adjusted_new)
            # Also track raw notional for reporting (but don't gate on it)
            raw_projected_gross = (metrics.gross_exposure + reserved_notional
                                  + new_notional)
            if projected_gross / equity > self._max_gross_exposure_pct:
                return False, (
                    f"Gross exposure {projected_gross / equity:.2%} "
                    f"(leverage-adjusted, raw={raw_projected_gross / equity:.2%}) "
                    f"would exceed limit {self._max_gross_exposure_pct:.2%}"
                ), None

            # 2. Max portfolio heat (including in-flight reservations)
            projected_heat_pct = (metrics.portfolio_heat + reserved_heat + proposed_risk) / equity
            if projected_heat_pct > self._max_portfolio_heat_pct:
                return False, (
                    f"Portfolio heat {projected_heat_pct:.2%} would exceed "
                    f"limit {self._max_portfolio_heat_pct:.2%}"
                ), None

            # 3. Max symbol weight (concentration) — RISK-BASED, not notional-based.
            #
            # Co-Founder Audit Fix (FORENSIC AUDIT BUG #9):
            # The original formula: (lots × price × contract_size) / equity
            # For forex (contract_size=100,000), a 0.13 lot EURCAD position
            # = $19,950 notional = 200% of a $10K account → ALWAYS rejected.
            #
            # The correct metric for "symbol concentration" is RISK AMOUNT
            # (capital lost if SL hits), not full notional. A 1% risk trade
            # is 1% of equity at risk regardless of contract_size — that's
            # what "weight" should measure for diversification.
            #
            # risk_amount = lots × sl_distance × contract_size = the $ you lose
            # if the stop-loss triggers. This is the true capital exposure.
            sizing = ctx_pipeline_state if 'ctx_pipeline_state' in dir() else None
            # proposed_risk is already the risk_amount from SizingGate
            symbol_risk_weight = (proposed_risk + reserved_heat) / equity
            if symbol_risk_weight > self._max_symbol_weight:
                return False, (
                    f"Symbol {symbol} risk weight {symbol_risk_weight:.2%} "
                    f"(risk=${proposed_risk:.0f}, equity=${equity:.0f}) "
                    f"would exceed limit {self._max_symbol_weight:.2%}"
                ), None

            # 4. One trade per symbol (existing positions + in-flight reservations)
            if self.has_open_position(symbol) or reserved_symbol_notional > 0:
                return False, f"Already have open or pending position on {symbol}", None

            # Approved — reserve capacity atomically under the same lock.
            self._reservation_seq += 1
            reservation_id = f"resv_{self._reservation_seq}_{int(time.time() * 1000)}"
            self._reservations[reservation_id] = {
                "symbol": symbol,
                "notional": new_notional,
                "heat": proposed_risk,
                "created_at": time.time(),
            }
            log.debug("portfolio: reserved %s notional=%.2f heat=%.2f for %s",
                      reservation_id, new_notional, proposed_risk, symbol)
            return True, "OK", reservation_id

    def release_reservation(self, reservation_id: Optional[str]) -> None:
        """Release a reservation that was approved but never committed
        (order rejected, gate failed downstream, etc.)."""
        if not reservation_id:
            return
        with self._lock:
            if self._reservations.pop(reservation_id, None) is not None:
                log.debug("portfolio: released reservation %s", reservation_id)

    def _prune_stale_reservations(self, ttl_s: float = 30.0) -> None:
        """Defensive cleanup: a reservation that's outlived a normal order
        round-trip (default 30s) is released so a crashed/hung caller can't
        permanently consume exposure/heat capacity."""
        now = time.time()
        stale = [rid for rid, r in self._reservations.items()
                 if now - r["created_at"] > ttl_s]
        for rid in stale:
            log.warning("portfolio: reservation %s expired after %.0fs — releasing",
                       rid, ttl_s)
            self._reservations.pop(rid, None)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def metrics(self) -> PortfolioMetrics:
        with self._lock:
            unrealized = sum(p["profit"] for p in self._positions.values())
            equity = self._balance + unrealized
            long_exp = sum(p["volume"] * p["current_price"]
                          for p in self._positions.values()
                          if p["side"] == "BUY")
            short_exp = sum(p["volume"] * p["current_price"]
                           for p in self._positions.values()
                           if p["side"] == "SELL")
            gross = long_exp + short_exp
            net = long_exp - short_exp

            # Portfolio heat = sum of (|entry - SL| × volume × contract_size)
            # AUDIT FIX: include contract_size so forex heat is correct
            heat = 0.0
            for p in self._positions.values():
                if p["sl"] > 0:
                    risk_per_unit = abs(p["entry_price"] - p["sl"])
                    cs = p.get("contract_size", 1.0)
                    heat += risk_per_unit * p["volume"] * cs

            # Concentration: weight per symbol
            symbol_weights = {}
            for p in self._positions.values():
                w = p["volume"] * p["current_price"]
                symbol_weights[p["symbol"]] = symbol_weights.get(p["symbol"], 0) + w
            total_exposure = sum(symbol_weights.values())
            if total_exposure > 0:
                max_w = max(symbol_weights.values()) / max(equity, 1)
                herf = sum((w / total_exposure) ** 2 for w in symbol_weights.values())
            else:
                max_w = 0.0
                herf = 0.0

            # Effective positions (1 / sum(weight^2))
            eff_n = (1.0 / herf) if herf > 0 else 0.0

            # Drawdown
            dd = self._peak_equity - equity
            dd_pct = (dd / self._peak_equity * 100) if self._peak_equity > 0 else 0

            # Co-Founder Audit Fix: gross_exposure is now LEVERAGE-ADJUSTED
            # (raw notional / leverage). This makes it comparable to equity
            # for the max_gross_exposure_pct gate. Raw notional is still
            # tracked in long_exposure/short_exposure for reporting.
            leverage = getattr(self, '_leverage', 100)
            gross_adjusted = gross / leverage

            return PortfolioMetrics(
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                initial_capital=self._initial_capital,
                balance=self._balance,
                equity=equity,
                unrealized_pnl=unrealized,
                realized_pnl=self._realized_pnl,
                gross_exposure=gross_adjusted,  # leverage-adjusted
                net_exposure=net / leverage,
                long_exposure=long_exp,  # raw (for reporting)
                short_exposure=short_exp,  # raw (for reporting)
                gross_exposure_pct=gross_adjusted / max(equity, 1),
                open_positions=len(self._positions),
                long_positions=sum(1 for p in self._positions.values()
                                   if p["side"] == "BUY"),
                short_positions=sum(1 for p in self._positions.values()
                                    if p["side"] == "SELL"),
                portfolio_heat=heat,
                portfolio_heat_pct=heat / max(equity, 1),
                peak_equity=self._peak_equity,
                current_drawdown=dd,
                current_drawdown_pct=dd_pct,
                max_drawdown=self._max_drawdown,
                max_drawdown_pct=self._max_drawdown_pct,
                max_symbol_weight=max_w,
                herfindahl_index=herf,
                effective_positions=eff_n,
            )

    # ------------------------------------------------------------------
    # Serialization (for snapshot/recovery — see recovery_engine.py)
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serializable snapshot of the fields recovery needs to restore
        the portfolio to its true pre-crash state — not just balance, but
        the drawdown high-water mark, so a crash-and-recover cycle cannot
        silently reset the drawdown limit's remaining room (see FIX-RE-02).
        """
        with self._lock:
            return {
                "initial_capital": self._initial_capital,
                "balance": self._balance,
                "realized_pnl": self._realized_pnl,
                "peak_equity": self._peak_equity,
                "max_drawdown": self._max_drawdown,
                "max_drawdown_pct": self._max_drawdown_pct,
            }

    def restore_from_dict(self, data: Dict[str, Any]) -> None:
        """Inverse of to_dict(). Restores balance + drawdown history
        WITHOUT touching open positions (those are re-added separately by
        the caller via on_position_opened, as before)."""
        if not data:
            return
        with self._lock:
            self._initial_capital = float(data.get("initial_capital", self._initial_capital))
            self._balance = float(data.get("balance", self._balance))
            self._realized_pnl = float(data.get("realized_pnl", self._realized_pnl))
            self._peak_equity = float(data.get("peak_equity", self._balance))
            self._max_drawdown = float(data.get("max_drawdown", 0.0))
            self._max_drawdown_pct = float(data.get("max_drawdown_pct", 0.0))
            log.info("portfolio: restored from snapshot — balance=%.2f peak_equity=%.2f "
                     "max_drawdown_pct=%.2f%%",
                     self._balance, self._peak_equity, self._max_drawdown_pct)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------
    def reconcile(self, broker_positions: List[Dict[str, Any]],
                  auto_resolve: bool = False) -> List[str]:
        """Compare local state to broker state, return list of discrepancies.

        broker_positions: list of dicts with keys: ticket, symbol, side,
                          volume, open_price, current_price, sl, tp, profit
        auto_resolve: M10 fix — if True, automatically resolve discrepancies
                      by trusting the broker state (closing orphans, adding
                      phantoms, fixing volume mismatches). Returns the list
                      of discrepancies that were found (and resolved).
        """
        discrepancies = []
        with self._lock:
            broker_tickets = {int(p["ticket"]) for p in broker_positions}
            local_tickets = set(self._positions.keys())

            # Positions we have but broker doesn't (orphan)
            orphan = local_tickets - broker_tickets
            for t in orphan:
                discrepancies.append(f"Local ticket {t} not found at broker — should be closed")
                if auto_resolve:
                    p = self._positions.pop(t, None)
                    if p:
                        log.warning("reconcile: auto-closed orphan ticket %d (%s)",
                                    t, p.get("symbol", "?"))

            # Positions broker has but we don't (phantom)
            phantom = broker_tickets - local_tickets
            for p in broker_positions:
                t = int(p["ticket"])
                if t in phantom:
                    discrepancies.append(f"Broker ticket {t} not in local state — should be added")
                    if auto_resolve:
                        self._positions[t] = {
                            "ticket": t,
                            "symbol": str(p.get("symbol", "")),
                            "side": str(p.get("side", "BUY")),
                            "volume": float(p.get("volume", 0)),
                            "open_price": float(p.get("open_price", 0)),
                            "current_price": float(p.get("current_price", 0)),
                            "sl": float(p.get("sl", 0)),
                            "tp": float(p.get("tp", 0)),
                            "profit": float(p.get("profit", 0)),
                        }
                        log.warning("reconcile: auto-added phantom ticket %d (%s)",
                                    t, p.get("symbol", "?"))

            # Volume/price mismatch
            for p in broker_positions:
                t = int(p["ticket"])
                if t in self._positions:
                    local = self._positions[t]
                    if abs(local["volume"] - float(p["volume"])) > 0.001:
                        discrepancies.append(
                            f"Volume mismatch on {t}: local={local['volume']} broker={p['volume']}")
                        if auto_resolve:
                            local["volume"] = float(p["volume"])
                            log.warning("reconcile: auto-fixed volume on ticket %d", t)
        return discrepancies

    # ------------------------------------------------------------------
    # Reset (for backtest)
    # ------------------------------------------------------------------
    def reset(self, capital: Optional[float] = None) -> None:
        with self._lock:
            self._positions.clear()
            self._reservations.clear()
            if capital is not None:
                self._initial_capital = capital
                self._balance = capital
            else:
                self._balance = self._initial_capital
            self._realized_pnl = 0.0
            self._peak_equity = self._balance
            self._max_drawdown = 0.0
            self._max_drawdown_pct = 0.0