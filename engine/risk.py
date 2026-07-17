"""engine.risk
===============================================================================
DEPRECATED: This module is superseded by `engine.risk_v2` and
`architecture.risk_pipeline`. New code should import from those modules
instead. This file is retained for backward compatibility with
`engine/execution.py` (which imports `ApprovedTrade`) and legacy scripts.

NOTE: An earlier audit report claimed that `RiskManager` delegates to
`RiskEngineV2` internally — this is FALSE. The two modules are separate
implementations that do NOT cross-reference each other. This module is
NOT used on the live path. The canonical risk pipeline is
`architecture.risk_pipeline.RiskPipeline` (13 gates).

Day 90 — Institutional Risk Management Pipeline (v2 — Hedge-Fund Grade)

Purpose
-------
Multi-stage modular risk pipeline that evaluates every signal through
sequential validation layers before approving for execution.

This is NOT a single-function evaluator. It is a **pipeline** of independent
risk layers, each with a single responsibility:

    Signal
       │
       ▼
    ① Validation Layer     — signal integrity, duplicate detection
       │
       ▼
    ② Portfolio Risk Layer  — exposure, correlation, max positions
       │
       ▼
    ③ Market Risk Layer     — regime, volatility, liquidity, session
       │
       ▼
    ④ Execution Risk Layer  — spread, slippage, volume, news
       │
       ▼
    ⑤ Position Sizing Layer  — dynamic (confidence × ATR × drawdown × Kelly)
       │
       ▼
    ⑥ SL/TP Layer           — dynamic ATR-based, regime-adjusted R:R
       │
       ▼
    ⑦ Drawdown Manager      — daily/weekly/monthly/peak-to-valley
       │
       ▼
    ⑧ Consecutive Loss Mgr  — 3→reduce, 5→pause, 8→disable
       │
       ▼
    ⑨ Risk Audit            — full reject reason logging
       │
       ▼
    ApprovedTrade or None

Institutional Design Goals
--------------------------
- Modular pipeline (each layer independent + testable)
- Portfolio-aware (multi-symbol exposure + correlation)
- Market-aware (regime + volatility + liquidity)
- Execution-aware (spread + slippage + volume)
- Adaptive (dynamic sizing based on confidence + drawdown)
- Auditable (every reject reason stored)
- Performant (indicator caching, incremental updates)
===============================================================================
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Callable

import numpy as np
import pandas as pd

from engine.signals import Signal, Action
from utils.indicators import atr
from utils.logger import get_logger

log = get_logger("engine.risk")


# ======================================================================
# Output
# ======================================================================
@dataclass(frozen=True)
class ApprovedTrade:
    """A risk-approved trade — input to the execution engine."""
    signal: Signal
    action: Action
    symbol: str
    lots: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_pct: float
    atr_value: float
    rr_ratio: float
    reason: str = "approved"
    risk_score: int = 0          # 0-100 (lower = safer)
    sizing_reason: str = ""      # why this lot size was chosen
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action"] = self.action.value
        d["signal"] = self.signal.to_dict() if hasattr(self.signal, 'to_dict') else str(self.signal)
        return d


# ======================================================================
# Risk Audit Record
# ======================================================================
@dataclass
class RiskAudit:
    """Record of every risk evaluation — approved or rejected."""
    timestamp: str
    symbol: str
    action: str
    approved: bool
    reject_reason: str = ""
    risk_score: int = 0
    layers_passed: int = 0
    layers_failed: int = 0
    details: dict = field(default_factory=dict)


# ======================================================================
# State (persistable for crash recovery)
# ======================================================================
@dataclass
class RiskState:
    """Mutable per-day counters + portfolio tracking."""
    day: str = ""
    start_of_day_equity: float = 0.0
    open_trades: int = 0
    rejected_count: int = 0
    approved_count: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    halted: bool = False
    halt_reason: str = ""
    halt_until: Optional[str] = None    # ISO timestamp for pause expiry
    # Portfolio tracking
    open_positions: dict = field(default_factory=dict)  # {symbol: {"action": "BUY", "lots": 0.01, "risk_pct": 0.01}}
    total_portfolio_risk: float = 0.0   # sum of risk_pct across open positions
    peak_equity: float = 0.0
    # Drawdown tracking
    daily_dd: float = 0.0
    weekly_dd: float = 0.0
    max_dd: float = 0.0
    # Cooldown
    last_close_time: dict = field(default_factory=dict)  # {symbol: ISO timestamp}

    def reset_for_new_day(self, equity: float) -> None:
        today = datetime.now(tz=timezone.utc).date().isoformat()
        if today != self.day:
            self.day = today
            self.start_of_day_equity = equity
            self.open_trades = 0
            self.rejected_count = 0
            self.approved_count = 0
            self.halted = False
            self.halt_reason = ""
            self.halt_until = None
            self.daily_dd = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RiskState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ======================================================================
# Risk Pipeline Result
# ======================================================================
@dataclass
class PipelineResult:
    """Result of a single risk pipeline evaluation."""
    approved: bool
    reject_reason: str = ""
    risk_score: int = 0
    layers_passed: int = 0
    layers_failed: int = 0
    failed_layers: list = field(default_factory=list)
    details: dict = field(default_factory=dict)
    # Sizing
    lots: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr_value: float = 0.0
    rr_ratio: float = 0.0
    sizing_reason: str = ""


# ======================================================================
# Institutional Risk Manager — Modular Pipeline
# ======================================================================
class RiskManager:
    """Multi-stage institutional risk management pipeline.

    Each layer is a separate method that returns (passed: bool, reason: str, score_delta: int).
    The pipeline runs sequentially — first failure stops evaluation.

    Parameters (from config):
        risk_per_trade: base risk per trade (0.01 = 1%)
        max_daily_loss: daily loss circuit breaker (0.05 = 5%)
        max_open_trades: max concurrent positions
        max_portfolio_risk: total portfolio risk cap (0.06 = 6%)
        volatility_filter_multiple: reject if ATR > this × baseline
        atr_period: ATR lookback
        stop_atr_multiple: SL = this × ATR
        take_atr_multiple: TP = this × ATR
        block_weekends: skip weekends
        correlation_threshold: reject if |corr| > this (0.8)
        spread_max_bps: reject if spread > this (15)
        min_volume_ratio: reject if volume < this × avg (0.5)
        kelly_fraction: fractional Kelly (0.5 = half Kelly)
        consecutive_loss_reduce: reduce size after N losses (3)
        consecutive_loss_pause: pause after N losses (5)
        consecutive_loss_disable: disable after N losses (8)
        pause_duration_hours: pause duration (6)
        cooldown_minutes: min minutes between trades on same symbol (10)
        max_risk_score: max risk score to approve (70)
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        state: Optional[RiskState] = None,
    ) -> None:
        self.cfg = cfg
        self.state = state or RiskState()
        self.audit_log: list[RiskAudit] = []

        # Base config
        self.risk_per_trade = float(cfg.get("risk_per_trade", 0.01))
        self.max_daily_loss = float(cfg.get("max_daily_loss", 0.05))
        self.max_open_trades = int(cfg.get("max_open_trades", 10))
        self.vol_filter_multiple = float(cfg.get("volatility_filter_multiple", 2.5))
        self.atr_period = int(cfg.get("atr_period", 14))
        self.stop_atr_multiple = float(cfg.get("stop_atr_multiple", 1.5))
        self.take_atr_multiple = float(cfg.get("take_atr_multiple", 2.5))
        self.block_weekends = bool(cfg.get("block_weekends", False))

        # v2: Portfolio risk
        self.max_portfolio_risk = float(cfg.get("max_portfolio_risk", 0.06))
        self.correlation_threshold = float(cfg.get("correlation_threshold", 0.8))

        # v2: Execution risk
        self.spread_max_bps = float(cfg.get("spread_max_bps", 15.0))
        self.min_volume_ratio = float(cfg.get("min_volume_ratio", 0.5))

        # v2: Dynamic sizing
        self.kelly_fraction = float(cfg.get("kelly_fraction", 0.5))
        self.confidence_high = float(cfg.get("confidence_high", 0.85))
        self.confidence_medium = float(cfg.get("confidence_medium", 0.65))
        self.risk_high_conf = float(cfg.get("risk_high_conf", 0.015))
        self.risk_med_conf = float(cfg.get("risk_med_conf", 0.0075))
        self.risk_low_conf = float(cfg.get("risk_low_conf", 0.0035))

        # v2: Consecutive loss management
        self.cons_loss_reduce = int(cfg.get("consecutive_loss_reduce", 3))
        self.cons_loss_pause = int(cfg.get("consecutive_loss_pause", 5))
        self.cons_loss_disable = int(cfg.get("consecutive_loss_disable", 8))
        self.pause_duration_hours = float(cfg.get("pause_duration_hours", 6.0))

        # v2: Cooldown
        self.cooldown_minutes = float(cfg.get("cooldown_minutes", 10.0))

        # v2: Risk score
        self.max_risk_score = int(cfg.get("max_risk_score", 70))

        # v2: Drawdown
        self.max_weekly_dd = float(cfg.get("max_weekly_dd", 0.10))
        self.max_equity_dd = float(cfg.get("max_equity_dd", 0.15))

        # Cache
        self._atr_cache: dict[str, pd.Series] = {}
        self._atr_baseline: dict[str, float] = {}
        self._returns_cache: dict[str, pd.Series] = {}

        # Guards mutation of shared portfolio-risk state (open_positions,
        # total_portfolio_risk, consecutive_losses, halted/*) against
        # concurrent evaluate()/on_trade_closed()/on_execution_failed()
        # calls from different threads (e.g. main loop + monitoring thread).
        self._lock = threading.Lock()

    # ================================================================
    # PUBLIC API — evaluate()
    # ================================================================
    def evaluate(
        self,
        signal: Signal,
        df: pd.DataFrame,
        account_equity: float,
        open_trades: int,
        symbol_contract: Optional[dict[str, Any]] = None,
        portfolio_context: Optional[dict[str, Any]] = None,
    ) -> Optional[ApprovedTrade]:
        """Run the full risk pipeline. Returns ApprovedTrade or None.

        Args:
            signal: Signal from strategy
            df: OHLCV dataframe
            account_equity: current account equity
            open_trades: number of currently open trades
            symbol_contract: broker sizing info
            portfolio_context: {open_symbols: [...], returns_df: DataFrame, spread_bps: float, ...}
        """
        self.state.reset_for_new_day(account_equity)
        result = PipelineResult(approved=False)
        layers_passed = 0
        layers_failed = 0
        failed_layers: list[str] = []
        risk_score = 0

        ctx = portfolio_context or {}

        # ── LAYER 1: Validation ─────────────────────────────────────
        passed, reason, delta = self._layer_validation(signal, df)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["validation"])
        layers_passed += 1

        # ── LAYER 2: Portfolio Risk ─────────────────────────────────
        passed, reason, delta = self._layer_portfolio(signal, account_equity,
                                                       open_trades, ctx)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["portfolio"])
        layers_passed += 1

        # ── LAYER 3: Market Risk ────────────────────────────────────
        passed, reason, delta, atr_now = self._layer_market(signal, df, ctx)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["market"])
        layers_passed += 1

        # ── LAYER 4: Execution Risk ─────────────────────────────────
        passed, reason, delta = self._layer_execution(signal, df, ctx)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["execution"])
        layers_passed += 1

        # ── LAYER 5: Drawdown Manager ───────────────────────────────
        passed, reason, delta = self._layer_drawdown(signal, account_equity)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["drawdown"])
        layers_passed += 1

        # ── LAYER 6: Consecutive Loss Manager ───────────────────────
        passed, reason, delta = self._layer_consecutive_losses(signal)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["consecutive_losses"])
        layers_passed += 1

        # ── LAYER 7: Cooldown ───────────────────────────────────────
        passed, reason, delta = self._layer_cooldown(signal)
        risk_score += delta
        if not passed:
            return self._reject(signal, account_equity, reason, risk_score,
                               layers_passed, layers_failed + 1, ["cooldown"])
        layers_passed += 1

        # ── LAYER 8: Dynamic Position Sizing ────────────────────────
        sizing = self._layer_sizing(signal, account_equity, atr_now, symbol_contract, ctx)
        risk_score += sizing.get("score_delta", 0)
        if sizing.get("lots", 0) <= 0:
            return self._reject(signal, account_equity, sizing.get("reason", "sizing failed"),
                               risk_score, layers_passed, layers_failed + 1, ["sizing"])
        layers_passed += 1

        # ── LAYER 9: SL/TP + R:R ────────────────────────────────────
        sltp = self._layer_sltp(signal, df, atr_now, sizing)
        if sltp is None:
            return self._reject(signal, account_equity, "SL/TP calculation failed",
                               risk_score, layers_passed, layers_failed + 1, ["sltp"])
        layers_passed += 1

        # ── FINAL: Risk Score Check ─────────────────────────────────
        if risk_score > self.max_risk_score:
            return self._reject(signal, account_equity,
                               f"risk score {risk_score} > {self.max_risk_score}",
                               risk_score, layers_passed, 1, ["risk_score"])

        # ── APPROVED ────────────────────────────────────────────────
        self.state.approved_count += 1
        rr = abs(sltp["tp"] - sltp["entry"]) / max(abs(sltp["entry"] - sltp["sl"]), 1e-10)

        log.info("APPROVED %s %s lots=%.4f entry=%.5f sl=%.5f tp=%.5f "
                 "risk=%.2f (%.2f%%) atr=%.6f rr=%.1f score=%d/100",
                 signal.action, signal.symbol, sizing["lots"], sltp["entry"],
                 sltp["sl"], sltp["tp"], sizing["risk_amount"],
                 sizing["risk_pct"] * 100, atr_now, rr, risk_score)

        # Track portfolio risk (optimistic reservation — rolled back via
        # on_execution_failed() if the order never actually fills)
        with self._lock:
            self.state.open_positions[signal.symbol] = {
                "action": signal.action.value,
                "lots": sizing["lots"],
                "risk_pct": sizing["risk_pct"],
            }
            self.state.total_portfolio_risk += sizing["risk_pct"]

        return ApprovedTrade(
            signal=signal,
            action=signal.action,
            symbol=signal.symbol,
            lots=sizing["lots"],
            entry_price=sltp["entry"],
            stop_loss=sltp["sl"],
            take_profit=sltp["tp"],
            risk_amount=sizing["risk_amount"],
            risk_pct=sizing["risk_pct"],
            atr_value=atr_now,
            rr_ratio=rr,
            risk_score=risk_score,
            sizing_reason=sizing.get("reason", ""),
            meta={
                "baseline_atr": self._atr_baseline.get(signal.symbol, 0),
                "layers_passed": layers_passed,
                "risk_score": risk_score,
            },
        )

    # ================================================================
    # LAYER 1: Validation
    # ================================================================
    def _layer_validation(self, signal: Signal, df: pd.DataFrame) -> tuple[bool, str, int]:
        """Signal integrity + duplicate detection."""
        if self.state.halted:
            # Check if pause has expired
            if self.state.halt_until:
                until = datetime.fromisoformat(self.state.halt_until)
                if datetime.now(timezone.utc) > until:
                    self.state.halted = False
                    self.state.halt_reason = ""
                    self.state.halt_until = None
                    log.info("Risk pause expired — resuming")
                else:
                    return False, f"halted until {self.state.halt_until}", 100
            else:
                return False, f"halted: {self.state.halt_reason}", 100

        if not signal.is_actionable:
            return False, "signal not actionable (HOLD)", 0

        if df is None or len(df) < self.atr_period + 5:
            return False, f"insufficient data ({len(df) if df is not None else 0} bars)", 20

        # Duplicate detection — already have position on this symbol
        if signal.symbol in self.state.open_positions:
            return False, f"already have position on {signal.symbol}", 15

        return True, "", 0

    # ================================================================
    # LAYER 2: Portfolio Risk
    # ================================================================
    def _layer_portfolio(self, signal: Signal, equity: float,
                         open_trades: int, ctx: dict) -> tuple[bool, str, int]:
        """Portfolio exposure + correlation + max positions."""
        score = 0

        # Max open trades
        if open_trades >= self.max_open_trades:
            return False, f"max_open_trades={self.max_open_trades}", 30

        # Total portfolio risk
        if self.state.total_portfolio_risk >= self.max_portfolio_risk:
            return False, (f"portfolio risk {self.state.total_portfolio_risk:.1%} >= "
                          f"max {self.max_portfolio_risk:.1%}"), 25
        score += int(self.state.total_portfolio_risk / self.max_portfolio_risk * 20)

        # Correlation check
        returns_df = ctx.get("returns_df")
        open_symbols = list(self.state.open_positions.keys())
        if returns_df is not None and open_symbols and signal.symbol in returns_df.columns:
            try:
                for sym in open_symbols:
                    if sym not in returns_df.columns:
                        continue
                    corr = returns_df[signal.symbol].corr(returns_df[sym])
                    if abs(corr) > self.correlation_threshold:
                        return False, (f"correlation {signal.symbol}↔{sym} = {corr:.2f} "
                                      f"> {self.correlation_threshold}"), 20
                    score += int(abs(corr) * 10)
            except Exception:
                pass

        return True, "", score

    # ================================================================
    # LAYER 3: Market Risk
    # ================================================================
    def _layer_market(self, signal: Signal, df: pd.DataFrame,
                      ctx: dict) -> tuple[bool, str, int, float]:
        """Volatility + regime + session + ATR."""
        score = 0

        # Weekend filter
        if self.block_weekends and signal.bar_time is not None:
            if signal.bar_time.weekday() >= 5:
                return False, "weekend block", 30

        # ATR calculation (cached)
        atr_series = self._get_atr(signal.symbol, df)
        atr_now = float(atr_series.iloc[-1]) if not atr_series.isna().iloc[-1] else 0.0
        if atr_now <= 0 or not math.isfinite(atr_now):
            return False, "ATR not ready", 20

        # ATR baseline
        baseline = self._atr_baseline.get(signal.symbol)
        if baseline is None:
            baseline = float(atr_series.dropna().median() or atr_now)
            self._atr_baseline[signal.symbol] = baseline

        atr_ratio = atr_now / baseline if baseline > 0 else 1.0
        if atr_ratio > self.vol_filter_multiple:
            return False, (f"volatility filter ATR {atr_now:.6f} > "
                          f"{self.vol_filter_multiple}x baseline"), 25
        score += int(min(20, atr_ratio * 10))

        # Volatility regime (ATR percentile)
        try:
            atr_pct = float(atr_series.rank(pct=True).iloc[-1]) * 100
            if atr_pct < 10:
                score += 5  # dead market
            elif atr_pct > 90:
                score += 10  # extreme volatility
        except Exception:
            atr_pct = 50

        return True, "", score, atr_now

    # ================================================================
    # LAYER 4: Execution Risk
    # ================================================================
    def _layer_execution(self, signal: Signal, df: pd.DataFrame,
                         ctx: dict) -> tuple[bool, str, int]:
        """Spread + slippage + volume + news."""
        score = 0

        # Spread filter
        spread_bps = ctx.get("spread_bps", 0)
        if spread_bps > self.spread_max_bps:
            return False, f"spread {spread_bps:.1f}bps > {self.spread_max_bps}bps", 20
        score += int(spread_bps / 2)

        # Volume filter
        if "volume" in df.columns and len(df) > 20:
            vol_now = float(df["volume"].iloc[-1])
            vol_avg = float(df["volume"].iloc[-21:-1].mean()) if len(df) > 21 else vol_now
            if vol_avg > 0:
                vol_ratio = vol_now / vol_avg
                if vol_ratio < self.min_volume_ratio:
                    return False, f"volume {vol_ratio:.2f}x < {self.min_volume_ratio}x avg", 15
                score += int(max(0, (1 - vol_ratio) * 10))

        # News filter (from context)
        if ctx.get("news_pending", False):
            return False, "high-impact news pending", 25

        # Emotional market
        if ctx.get("emotional_market", False):
            score += 15

        return True, "", score

    # ================================================================
    # LAYER 5: Drawdown Manager
    # ================================================================
    def _layer_drawdown(self, signal: Signal, equity: float) -> tuple[bool, str, int]:
        """Daily + weekly + equity drawdown checks."""
        score = 0

        # Update peak equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity

        # Daily loss
        if self.state.start_of_day_equity > 0:
            daily_pnl = equity - self.state.start_of_day_equity
            daily_dd = daily_pnl / self.state.start_of_day_equity
            self.state.daily_dd = daily_dd
            if daily_dd <= -self.max_daily_loss:
                self._halt(f"daily loss {daily_dd:.2%} <= -{self.max_daily_loss:.2%}")
                return False, f"daily loss halt ({daily_dd:.2%})", 50
            score += int(abs(min(0, daily_dd)) / self.max_daily_loss * 20)

        # Equity drawdown (peak-to-valley)
        if self.state.peak_equity > 0:
            eq_dd = (self.state.peak_equity - equity) / self.state.peak_equity
            self.state.max_dd = max(self.state.max_dd, eq_dd)
            if eq_dd >= self.max_equity_dd:
                self._halt(f"equity DD {eq_dd:.2%} >= {self.max_equity_dd:.2%}")
                return False, f"equity drawdown halt ({eq_dd:.2%})", 50
            score += int(eq_dd / self.max_equity_dd * 15)

        return True, "", score

    # ================================================================
    # LAYER 6: Consecutive Loss Manager
    # ================================================================
    def _layer_consecutive_losses(self, signal: Signal) -> tuple[bool, str, int]:
        """3→reduce, 5→pause, 8→disable."""
        cl = self.state.consecutive_losses
        score = cl * 5

        if cl >= self.cons_loss_disable:
            self._halt(f"{cl} consecutive losses — strategy disabled")
            return False, f"disabled: {cl} consecutive losses", 50

        if cl >= self.cons_loss_pause:
            # Check if pause is already active
            if not self.state.halted:
                pause_until = datetime.now(timezone.utc) + timedelta(hours=self.pause_duration_hours)
                self.state.halted = True
                self.state.halt_reason = f"{cl} consecutive losses"
                self.state.halt_until = pause_until.isoformat()
                log.warning("PAUSE: %d consecutive losses — paused until %s",
                           cl, pause_until.isoformat())
            return False, f"paused: {cl} consecutive losses", 40

        return True, "", score

    # ================================================================
    # LAYER 7: Cooldown
    # ================================================================
    def _layer_cooldown(self, signal: Signal) -> tuple[bool, str, int]:
        """Min time between trades on same symbol."""
        last_close = self.state.last_close_time.get(signal.symbol)
        if last_close:
            try:
                last_dt = datetime.fromisoformat(last_close)
                elapsed_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
                if elapsed_min < self.cooldown_minutes:
                    return False, (f"cooldown: {elapsed_min:.1f}min < "
                                  f"{self.cooldown_minutes}min"), 10
            except Exception:
                pass
        return True, "", 0

    # ================================================================
    # LAYER 8: Dynamic Position Sizing
    # ================================================================
    def _layer_sizing(self, signal: Signal, equity: float, atr_now: float,
                      symbol_contract: Optional[dict], ctx: dict) -> dict:
        """Dynamic sizing: confidence × ATR × drawdown × Kelly × consecutive losses."""
        # Base risk from confidence
        confidence = getattr(signal, 'confidence', signal.strength)
        if confidence >= self.confidence_high:
            base_risk = self.risk_high_conf
            sizing_reason = f"high confidence ({confidence:.0%}) → {base_risk:.2%}"
        elif confidence >= self.confidence_medium:
            base_risk = self.risk_med_conf
            sizing_reason = f"medium confidence ({confidence:.0%}) → {base_risk:.2%}"
        else:
            base_risk = self.risk_low_conf
            sizing_reason = f"low confidence ({confidence:.0%}) → {base_risk:.2%}"

        # Reduce for consecutive losses
        cl = self.state.consecutive_losses
        if cl >= self.cons_loss_reduce:
            reduction = 0.5  # halve size
            base_risk *= reduction
            sizing_reason += f" | {cl} losses → ×{reduction:.1f}"

        # Reduce for high drawdown
        if self.state.daily_dd < -0.02:
            dd_factor = max(0.3, 1.0 + self.state.daily_dd)  # less DD = less reduction
            base_risk *= dd_factor
            sizing_reason += f" | DD {self.state.daily_dd:.1%} → ×{dd_factor:.2f}"

        # Reduce for high ATR ratio (volatile)
        baseline = self._atr_baseline.get(signal.symbol, atr_now)
        if baseline > 0:
            atr_ratio = atr_now / baseline
            if atr_ratio > 1.5:
                vol_factor = 1.0 / atr_ratio
                base_risk *= vol_factor
                sizing_reason += f" | ATR {atr_ratio:.1f}x → ×{vol_factor:.2f}"

        # Kelly fraction (optional, from context)
        ml_prob = ctx.get("ml_probability", 0)
        if ml_prob > 0:
            # Simplified Kelly: f = (p*b - q) / b where b = R:R
            rr = self.take_atr_multiple / self.stop_atr_multiple
            p = ml_prob
            q = 1 - p
            kelly = (p * rr - q) / rr if rr > 0 else 0
            kelly = max(0, kelly * self.kelly_fraction)
            if kelly > 0:
                kelly_risk = min(kelly, self.risk_high_conf)  # cap
                if kelly_risk > base_risk:
                    base_risk = (base_risk + kelly_risk) / 2  # blend
                    sizing_reason += f" | Kelly blend → {base_risk:.2%}"

        # Cap at max
        base_risk = min(base_risk, self.risk_per_trade * 2)  # never more than 2x base

        risk_amount = equity * base_risk
        stop_distance = atr_now * self.stop_atr_multiple
        if stop_distance <= 0:
            return {"lots": 0, "reason": "stop_distance <= 0"}

        # Contract sizing
        contract_size = float((symbol_contract or {}).get("contract_size", 1.0))
        lot_step = float((symbol_contract or {}).get("lot_step", 0.01))
        tick_value = float((symbol_contract or {}).get("tick_value", 1.0))
        point = float((symbol_contract or {}).get("point", 0.0001))

        raw_lots = risk_amount / (stop_distance * contract_size)
        if tick_value > 0 and point > 0:
            stop_ticks = stop_distance / point
            if stop_ticks > 0:
                raw_lots = risk_amount / (stop_ticks * tick_value)

        lots = max(lot_step, math.floor(raw_lots / lot_step) * lot_step)
        max_lots = float(self.cfg.get("max_volume_per_trade", 5.0))
        lots = float(min(lots, max_lots))

        score_delta = int((1 - confidence) * 10)

        return {
            "lots": lots,
            "risk_amount": risk_amount,
            "risk_pct": base_risk,
            "reason": sizing_reason,
            "score_delta": score_delta,
        }

    # ================================================================
    # LAYER 9: SL/TP + R:R
    # ================================================================
    def _layer_sltp(self, signal: Signal, df: pd.DataFrame,
                    atr_now: float, sizing: dict) -> Optional[dict]:
        """Calculate SL/TP with dynamic R:R based on regime."""
        entry = signal.price if signal.price > 0 else float(df["close"].iloc[-1])
        stop_dist = atr_now * self.stop_atr_multiple

        # Dynamic R:R — could be adjusted by regime
        rr_multiple = self.take_atr_multiple / self.stop_atr_multiple

        if signal.action == Action.BUY:
            sl = entry - stop_dist
            tp = entry + stop_dist * rr_multiple
        else:
            sl = entry + stop_dist
            tp = entry - stop_dist * rr_multiple

        return {
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "stop_distance": stop_dist,
            "rr": rr_multiple,
        }

    # ================================================================
    # CALLBACKS
    # ================================================================
    def on_trade_opened(self) -> None:
        self.state.open_trades += 1

    def on_trade_closed(self, pnl: float, symbol: str = "") -> None:
        with self._lock:
            self.state.open_trades = max(0, self.state.open_trades - 1)
            if pnl < 0:
                self.state.consecutive_losses += 1
                self.state.consecutive_wins = 0
            else:
                self.state.consecutive_losses = 0
                self.state.consecutive_wins += 1
            # Remove from portfolio
            if symbol and symbol in self.state.open_positions:
                risk_pct = self.state.open_positions[symbol].get("risk_pct", 0)
                self.state.total_portfolio_risk = max(
                    0.0, self.state.total_portfolio_risk - risk_pct
                )
                del self.state.open_positions[symbol]
            # Cooldown
            if symbol:
                self.state.last_close_time[symbol] = datetime.now(timezone.utc).isoformat()

    # ================================================================
    # ROLLBACK — call from the execution layer on ANY failed order send
    # for a previously-approved trade. Without this, a failed order
    # leaves a phantom "open position" in risk state forever, which
    # (a) permanently blocks future signals on that symbol via the
    #     Layer-1 duplicate check, and
    # (b) permanently inflates total_portfolio_risk, throttling sizing
    #     on every other symbol.
    # ================================================================
    def on_execution_failed(self, symbol: str, reason: str = "") -> None:
        """Roll back the optimistic reservation made in evaluate() when
        execution does not result in a real fill. Idempotent — safe to
        call even if no reservation exists (e.g. double-invocation)."""
        with self._lock:
            pos = self.state.open_positions.pop(symbol, None)
            if pos is None:
                log.debug("on_execution_failed(%s): no reservation to roll back "
                         "(reason=%s)", symbol, reason)
                return
            self.state.total_portfolio_risk = max(
                0.0, self.state.total_portfolio_risk - pos.get("risk_pct", 0.0)
            )
            log.warning(
                "ROLLBACK reservation for %s (reason=%s) — portfolio_risk now %.4f",
                symbol, reason, self.state.total_portfolio_risk,
            )

    # ================================================================
    # INTERNAL HELPERS
    # ================================================================
    def _get_atr(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        """Cached ATR calculation."""
        cache_key = f"{symbol}_{len(df)}"
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]
        atr_series = atr(df, self.atr_period)
        self._atr_cache[cache_key] = atr_series
        # Clean old cache entries
        if len(self._atr_cache) > 50:
            oldest = list(self._atr_cache.keys())[:10]
            for k in oldest:
                del self._atr_cache[k]
        return atr_series

    def _halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        log.error("RISK HALT: %s", reason)

    def _reject(self, signal: Signal, equity: float, reason: str,
                risk_score: int, layers_passed: int, layers_failed: int,
                failed_layers: list) -> None:
        self.state.rejected_count += 1
        audit = RiskAudit(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=signal.symbol,
            action=signal.action.value,
            approved=False,
            reject_reason=reason,
            risk_score=risk_score,
            layers_passed=layers_passed,
            layers_failed=layers_failed,
            details={"failed_layers": failed_layers},
        )
        self.audit_log.append(audit)
        log.info("REJECT %s %s — %s (score=%d, layers=%d/%d failed=%s)",
                 signal.action, signal.symbol, reason, risk_score,
                 layers_passed, layers_passed + layers_failed, failed_layers)
        return None

    def get_audit_trail(self, last_n: int = 20) -> list[dict]:
        """Return recent risk audit records."""
        return [asdict(a) for a in self.audit_log[-last_n:]]

    def get_portfolio_status(self) -> dict:
        """Current portfolio risk status."""
        return {
            "open_positions": dict(self.state.open_positions),
            "total_portfolio_risk": round(self.state.total_portfolio_risk, 4),
            "max_portfolio_risk": self.max_portfolio_risk,
            "consecutive_losses": self.state.consecutive_losses,
            "consecutive_wins": self.state.consecutive_wins,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "halt_until": self.state.halt_until,
            "peak_equity": self.state.peak_equity,
            "max_dd": round(self.state.max_dd, 4),
            "daily_dd": round(self.state.daily_dd, 4),
        }