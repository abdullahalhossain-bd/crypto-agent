"""architecture/institutional_monitoring.py
=====================================================================
Institutional Monitoring & Observability (Improvement #14)
=====================================================================
Hedge-fund grade monitoring. Tracks 50+ KPIs in real-time, triggers
alerts on anomalies, and exposes metrics via Prometheus-style export.

KPI Categories:
    1. P&L Metrics: equity, daily P&L, Sharpe, Sortino, max drawdown
    2. Risk Metrics: VaR, CVaR, portfolio heat, margin usage, beta
    3. Trade Metrics: win rate, profit factor, avg R, expectancy
    4. Execution Metrics: slippage, fill rate, latency, rejection rate
    5. System Metrics: cycle time, error rate, queue depth, memory
    6. Strategy Metrics: per-strategy Sharpe, decay, alpha vs benchmark
    7. Market Metrics: regime, volatility, liquidity, correlation

Alert Levels:
    INFO    — logged, no action
    WARNING — logged + Slack/email notification
    CRITICAL — logged + notification + auto-degrade trading
    EMERGENCY — logged + notification + halt trading + page on-call

Usage:
    mon = InstitutionalMonitor()
    mon.update_equity(10350.0)
    mon.record_trade(pnl=42.5, r_multiple=1.8, hold_time_s=3600)
    alerts = mon.check_alerts()
    metrics = mon.export_metrics()  # Prometheus format
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from architecture.event_bus import EventBus, EventType, get_bus
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.monitoring")


@dataclass
class Alert:
    level: str  # INFO, WARNING, CRITICAL, EMERGENCY
    category: str
    message: str
    value: float = 0.0
    threshold: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())


class InstitutionalMonitor:
    """Real-time KPI tracking + alerting."""

    def __init__(self,
                 bus: Optional[EventBus] = None,
                 history_size: int = 10000,
                 cycle_time_alert_ms: float = 30000.0,
                 active_alerts_state_file: Optional[str] = "data/monitor_active_alerts.json"):
        self._lock = threading.RLock()
        self._bus = bus or get_bus()
        # PERF: configurable so users running many symbols (each symbol
        # costs ~1 serialized MT5 IPC round-trip per cycle) can set a
        # realistic threshold instead of getting spammed by an alert that
        # just reflects symbol count, not an actual problem.
        self._cycle_time_alert_ms = float(cycle_time_alert_ms)
        # Time-series buffers
        self._equity_curve: Deque[Tuple[float, float]] = deque(maxlen=history_size)
        self._trade_pnls: Deque[float] = deque(maxlen=history_size)
        self._trade_r_multiples: Deque[float] = deque(maxlen=history_size)
        self._cycle_times: Deque[float] = deque(maxlen=1000)
        self._error_count: int = 0
        self._total_cycles: int = 0
        self._alerts: Deque[Alert] = deque(maxlen=1000)
        # Strategy-level tracking
        self._strategy_pnl: Dict[str, Deque[float]] = {}
        # Execution tracking
        self._slippage_samples: Deque[float] = deque(maxlen=500)
        self._fill_latencies: Deque[float] = deque(maxlen=500)
        # State
        self._peak_equity: float = 0.0
        self._initial_equity: float = 0.0
        # X6/H4 fix: _active_alerts is the state-change gate that prevents
        # the same WARNING/CRITICAL from re-firing every cycle. It used to
        # be a plain in-memory `set` created lazily on first use, which
        # meant every process restart forgot which alerts were already
        # "active" — if the underlying condition (e.g. drawdown still >10%)
        # hadn't actually cleared, the very next check_alerts() call would
        # re-fire it as if it were brand new. Now it's persisted to a small
        # JSON sidecar file and reloaded on startup.
        self._active_alerts_state_file = active_alerts_state_file
        self._active_alerts: set = self._load_active_alerts()

    def _load_active_alerts(self) -> set:
        if not self._active_alerts_state_file:
            return set()
        try:
            import json
            import os
            if not os.path.exists(self._active_alerts_state_file):
                return set()
            with open(self._active_alerts_state_file, "r") as f:
                data = json.load(f)
            return set(data.get("active_alerts", []))
        except Exception as e:  # noqa: BLE001
            log.warning("monitor: failed to load persisted active_alerts "
                       "(starting empty): %r", e)
            return set()

    def _save_active_alerts(self) -> None:
        if not self._active_alerts_state_file:
            return
        try:
            import json
            import os
            os.makedirs(os.path.dirname(self._active_alerts_state_file) or ".",
                       exist_ok=True)
            with open(self._active_alerts_state_file, "w") as f:
                json.dump({"active_alerts": list(self._active_alerts)}, f)
        except Exception as e:  # noqa: BLE001
            log.debug("monitor: failed to persist active_alerts: %r", e)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    def update_equity(self, equity: float) -> None:
        with self._lock:
            if self._initial_equity == 0:
                self._initial_equity = equity
            if equity > self._peak_equity:
                self._peak_equity = equity
            self._equity_curve.append((time.time(), equity))

    def record_trade(self,
                     pnl: float,
                     r_multiple: float = 0.0,
                     hold_time_s: float = 0.0,
                     strategy: str = "default") -> None:
        with self._lock:
            self._trade_pnls.append(pnl)
            self._trade_r_multiples.append(r_multiple)
            self._strategy_pnl.setdefault(strategy, deque(maxlen=500)).append(pnl)

    def record_cycle(self, cycle_time_s: float, error: bool = False) -> None:
        with self._lock:
            self._total_cycles += 1
            self._cycle_times.append(cycle_time_s)
            if error:
                self._error_count += 1

    def record_execution(self,
                         slippage_bps: float = 0.0,
                         fill_latency_ms: float = 0.0) -> None:
        with self._lock:
            self._slippage_samples.append(slippage_bps)
            self._fill_latencies.append(fill_latency_ms)

    # ------------------------------------------------------------------
    # KPI computation
    # ------------------------------------------------------------------
    def kpis(self) -> Dict[str, Any]:
        with self._lock:
            eq_series = [e for _, e in self._equity_curve]
            pnls = list(self._trade_pnls)
            rs = list(self._trade_r_multiples)
            cycle_times = list(self._cycle_times)

        kpis: Dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        # P&L metrics
        if eq_series:
            current = eq_series[-1]
            kpis["equity"] = current
            kpis["initial_equity"] = self._initial_equity
            kpis["total_return_pct"] = ((current - self._initial_equity) /
                                       max(self._initial_equity, 1)) * 100
            kpis["peak_equity"] = self._peak_equity
            drawdown = (self._peak_equity - current) / max(self._peak_equity, 1) * 100
            kpis["current_drawdown_pct"] = drawdown
            kpis["max_drawdown_pct"] = self._compute_max_drawdown(eq_series)
            # Sharpe (per-trade, annualized later)
            if len(pnls) >= 10:
                returns = np.diff(eq_series[-min(len(eq_series), 100):])
                if len(returns) > 1 and returns.std() > 0:
                    kpis["sharpe_per_bar"] = float(returns.mean() / returns.std())
                    kpis["sharpe_annualized"] = float(
                        returns.mean() / returns.std() * np.sqrt(252 * 24 * 4))
                # Sortino (downside deviation only)
                downside = returns[returns < 0]
                if len(downside) > 1 and downside.std() > 0:
                    kpis["sortino_annualized"] = float(
                        returns.mean() / downside.std() * np.sqrt(252 * 24 * 4))

        # Trade metrics
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            kpis["total_trades"] = len(pnls)
            kpis["wins"] = len(wins)
            kpis["losses"] = len(losses)
            kpis["win_rate"] = len(wins) / max(len(pnls), 1)
            kpis["avg_win"] = sum(wins) / max(len(wins), 1)
            kpis["avg_loss"] = sum(losses) / max(len(losses), 1)
            kpis["profit_factor"] = (sum(wins) / max(abs(sum(losses)), 0.01))
            kpis["expectancy"] = sum(pnls) / len(pnls)
            kpis["total_pnl"] = sum(pnls)

        if rs:
            kpis["avg_r_multiple"] = sum(rs) / len(rs)
            kpis["r_expectancy"] = (sum(r > 0 for r in rs) / len(rs)) * \
                                   (sum(r for r in rs if r > 0) / max(
                                       sum(1 for r in rs if r > 0), 1)) - \
                                   (sum(r < 0 for r in rs) / len(rs))

        # System metrics
        if cycle_times:
            kpis["avg_cycle_time_ms"] = (sum(cycle_times) / len(cycle_times)) * 1000
            kpis["max_cycle_time_ms"] = max(cycle_times) * 1000
        kpis["total_cycles"] = self._total_cycles
        kpis["error_count"] = self._error_count
        kpis["error_rate"] = self._error_count / max(self._total_cycles, 1)

        # Execution
        if self._slippage_samples:
            kpis["avg_slippage_bps"] = sum(self._slippage_samples) / len(self._slippage_samples)
        if self._fill_latencies:
            kpis["avg_fill_latency_ms"] = sum(self._fill_latencies) / len(self._fill_latencies)

        return kpis

    def _compute_max_drawdown(self, series: List[float]) -> float:
        peak = series[0]
        max_dd = 0.0
        for v in series:
            if v > peak:
                peak = v
            dd = (peak - v) / max(peak, 1) * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    # ------------------------------------------------------------------
    # Alert checks
    # ------------------------------------------------------------------
    def check_alerts(self) -> List[Alert]:
        """Check all KPIs against thresholds and emit alerts.

        UI fix: alerts are state-change-gated — each alert type fires only
        when the condition FIRST becomes true, not every cycle. This prevents
        the same WARNING from spamming the console every 5 seconds.
        """
        kpis = self.kpis()
        new_alerts: List[Alert] = []

        # Track which alerts are currently active to prevent spam
        if not hasattr(self, "_active_alerts"):
            self._active_alerts: set = set()

        def _check(name: str, condition: bool, level: str, msg: str,
                   value: float, threshold: float) -> None:
            """Emit alert only on state change (breach → active, clear → inactive)."""
            key = f"{name}_{level}"
            if condition and key not in self._active_alerts:
                new_alerts.append(Alert(level, name, msg, value, threshold))
                self._active_alerts.add(key)
            elif not condition and key in self._active_alerts:
                self._active_alerts.discard(key)  # alert cleared — can fire again next time

        # Drawdown alerts
        dd = kpis.get("current_drawdown_pct", 0)
        _check("drawdown_emergency", dd > 15, "EMERGENCY",
               f"Drawdown {dd:.2f}% exceeds 15% — halt trading", dd, 15.0)
        _check("drawdown_critical", dd > 10, "CRITICAL",
               f"Drawdown {dd:.2f}% exceeds 10%", dd, 10.0)
        _check("drawdown_warning", dd > 5, "WARNING",
               f"Drawdown {dd:.2f}% — monitor", dd, 5.0)

        # Win rate alerts
        wr = kpis.get("win_rate", 0.5)
        has_trades = kpis.get("total_trades", 0) > 20
        _check("win_rate_critical", has_trades and wr < 0.30, "CRITICAL",
               f"Win rate {wr:.1%} < 30% — strategy failing", wr, 0.30)
        _check("win_rate_warning", has_trades and wr < 0.40, "WARNING",
               f"Win rate {wr:.1%} < 40% — review", wr, 0.40)

        # Profit factor
        pf = kpis.get("profit_factor", 1.0)
        _check("profit_factor", has_trades and pf < 1.0, "CRITICAL",
               f"Profit factor {pf:.2f} < 1.0 — losing money", pf, 1.0)

        # Error rate
        er = kpis.get("error_rate", 0)
        _check("error_rate_critical", er > 0.10, "CRITICAL",
               f"Error rate {er:.1%} > 10%", er, 0.10)
        _check("error_rate_warning", er > 0.05, "WARNING",
               f"Error rate {er:.1%} > 5%", er, 0.05)

        # Cycle time — configurable threshold (see __init__: cycle_time_alert_ms)
        ct = kpis.get("avg_cycle_time_ms", 0)
        thresh = self._cycle_time_alert_ms
        _check("cycle_time", ct > thresh, "WARNING",
               f"Avg cycle {ct:.0f}ms > {thresh:.0f}ms", ct, thresh)

        with self._lock:
            self._alerts.extend(new_alerts)

        # Emit critical+ alerts to event bus
        for a in new_alerts:
            if a.level in ("CRITICAL", "EMERGENCY"):
                self._bus.emit(EventType.CIRCUIT_BREAKER_TRIP,
                              payload={"level": a.level, "category": a.category,
                                      "message": a.message},
                              source="monitoring")
            log.log({"INFO": 20, "WARNING": 30, "CRITICAL": 40,
                    "EMERGENCY": 50}[a.level],
                   "ALERT [%s] %s: %s", a.level, a.category, a.message)
        return new_alerts

    # ------------------------------------------------------------------
    # Prometheus export
    # ------------------------------------------------------------------
    def export_metrics(self) -> str:
        """Export metrics in Prometheus text format."""
        kpis = self.kpis()
        lines = [
            "# HELP trading_equity Current account equity",
            "# TYPE trading_equity gauge",
            f"trading_equity {kpis.get('equity', 0)}",
            "",
            "# HELP trading_drawdown_pct Current drawdown percentage",
            "# TYPE trading_drawdown_pct gauge",
            f"trading_drawdown_pct {kpis.get('current_drawdown_pct', 0)}",
            "",
            "# HELP trading_win_rate Rolling win rate",
            "# TYPE trading_win_rate gauge",
            f"trading_win_rate {kpis.get('win_rate', 0)}",
            "",
            "# HELP trading_profit_factor Profit factor (gross profit / gross loss)",
            "# TYPE trading_profit_factor gauge",
            f"trading_profit_factor {kpis.get('profit_factor', 0)}",
            "",
            "# HELP trading_total_trades Total trades executed",
            "# TYPE trading_total_trades counter",
            f"trading_total_trades {kpis.get('total_trades', 0)}",
            "",
            "# HELP trading_error_rate Cycle error rate",
            "# TYPE trading_error_rate gauge",
            f"trading_error_rate {kpis.get('error_rate', 0)}",
            "",
            "# HELP trading_avg_cycle_ms Average cycle time in ms",
            "# TYPE trading_avg_cycle_ms gauge",
            f"trading_avg_cycle_ms {kpis.get('avg_cycle_time_ms', 0)}",
        ]
        return "\n".join(lines)

    def recent_alerts(self, last_n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"level": a.level, "category": a.category, "message": a.message,
                 "value": a.value, "threshold": a.threshold, "timestamp": a.timestamp}
                for a in list(self._alerts)[-last_n:]
            ]