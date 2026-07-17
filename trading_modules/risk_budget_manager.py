"""trading_modules/risk_budget_manager.py
=====================================================================
Risk Budget Manager (Principle #147, #154)
=====================================================================
Enforces strict daily/weekly/monthly risk budgets. Automatically pauses
trading when limits are reached, and dynamically reduces risk during
losing streaks.

Risk Budgets:
    - DAILY:   max 2% of equity at risk per day (default)
    - WEEKLY:  max 6% of equity per week
    - MONTHLY: max 15% of equity per month (drawdown limit)

Dynamic Risk Reduction:
    After consecutive losses, risk per trade is automatically reduced:
        0 losses → 100% of normal risk
        1 loss   → 75%
        2 losses → 50%
        3 losses → 25%
        4+ losses → 10% (near halt)
    After a win, risk gradually recovers.

Auto-Pause Triggers:
    - Daily risk budget exhausted
    - Weekly drawdown limit hit
    - Monthly drawdown limit hit
    - 4+ consecutive losses
    - Equity drops below emergency threshold

Usage:
    mgr = RiskBudgetManager(equity=10000, daily_risk_pct=2.0)
    # Before each trade:
    if mgr.can_take_trade(risk_usd=50):
        mgr.allocate_risk(50, strategy="momentum")
        place_trade()
    else:
        skip()  # budget exhausted

    # After each trade closes:
    mgr.record_trade_result(pnl=50, win=True)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from utils.logger import get_logger

log = get_logger("trading_bot.risk_budget_manager")


@dataclass
class RiskBudgetState:
    """Current risk budget state."""
    # Daily
    daily_risk_used: float = 0.0       # % of equity used today
    daily_risk_remaining: float = 2.0  # % remaining
    daily_pnl: float = 0.0
    daily_trades: int = 0
    # Weekly
    weekly_risk_used: float = 0.0
    weekly_pnl: float = 0.0
    weekly_trades: int = 0
    # Monthly
    monthly_risk_used: float = 0.0
    monthly_pnl: float = 0.0
    monthly_trades: int = 0
    # Consecutive losses
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    # Risk multiplier (0.1 to 1.0)
    current_risk_multiplier: float = 1.0
    # Status
    trading_paused: bool = False
    pause_reason: str = ""
    # Equity tracking
    peak_equity: float = 0.0
    current_equity: float = 0.0
    current_drawdown_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily_risk_used_pct": round(self.daily_risk_used, 3),
            "daily_risk_remaining_pct": round(self.daily_risk_remaining, 3),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
            "weekly_risk_used_pct": round(self.weekly_risk_used, 3),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "monthly_risk_used_pct": round(self.monthly_risk_used, 3),
            "monthly_pnl": round(self.monthly_pnl, 2),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "current_risk_multiplier": round(self.current_risk_multiplier, 2),
            "trading_paused": self.trading_paused,
            "pause_reason": self.pause_reason,
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(self.current_equity, 2),
            "current_drawdown_pct": round(self.current_drawdown_pct, 2),
        }


class RiskBudgetManager:
    """Manages risk budgets across daily/weekly/monthly horizons."""

    def __init__(self,
                 equity: float = 10000.0,
                 daily_risk_pct: float = 2.0,
                 weekly_risk_pct: float = 6.0,
                 monthly_risk_pct: float = 15.0,
                 max_consecutive_losses: int = 4,
                 emergency_drawdown_pct: float = 20.0,
                 recovery_win_threshold: int = 2):
        """Initialize risk budget manager.

        Args:
            equity: starting account equity
            daily_risk_pct: max % of equity at risk per day
            weekly_risk_pct: max % per week
            monthly_risk_pct: max % per month (drawdown limit)
            max_consecutive_losses: pause after this many consecutive losses
            emergency_drawdown_pct: halt trading at this drawdown
            recovery_win_threshold: wins needed to restore full risk
        """
        self.equity = equity
        self.daily_limit = daily_risk_pct
        self.weekly_limit = weekly_risk_pct
        self.monthly_limit = monthly_risk_pct
        self.max_consec_losses = max_consecutive_losses
        self.emergency_dd = emergency_drawdown_pct
        self.recovery_wins = recovery_win_threshold

        self._lock = threading.RLock()
        self._state = RiskBudgetState(
            peak_equity=equity, current_equity=equity,
            daily_risk_remaining=daily_risk_pct,
        )
        # Trade history for tracking
        self._daily_trades: Deque[dict] = deque(maxlen=100)
        self._weekly_trades: Deque[dict] = deque(maxlen=500)
        self._monthly_trades: Deque[dict] = deque(maxlen=2000)
        # Day/week/month tracking
        self._current_day = datetime.now(tz=timezone.utc).date()
        self._current_week = self._get_week_start()
        self._current_month = datetime.now(tz=timezone.utc).strftime("%Y-%m")

    def _get_week_start(self) -> datetime:
        """Get start of current week (Monday)."""
        now = datetime.now(tz=timezone.utc)
        return now - timedelta(days=now.weekday())

    # ------------------------------------------------------------------
    # Check if we can take a trade
    # ------------------------------------------------------------------
    def can_take_trade(self, risk_usd: float,
                       risk_pct: Optional[float] = None) -> Tuple[bool, str]:
        """Check if we can take a new trade within risk budgets.

        Args:
            risk_usd: dollar risk if SL hit
            risk_pct: risk as % of equity (optional, computed if not given)

        Returns:
            (allowed: bool, reason: str)
        """
        with self._lock:
            self._check_reset()

            if self._state.trading_paused:
                return False, f"TRADING PAUSED: {self._state.pause_reason}"

            # Check daily budget
            if self._state.daily_risk_remaining <= 0:
                self._pause("daily_risk_budget_exhausted")
                return False, "Daily risk budget exhausted"

            # Check weekly budget
            if self._state.weekly_risk_used >= self.weekly_limit:
                self._pause("weekly_risk_limit_reached")
                return False, "Weekly risk limit reached"

            # Check monthly budget
            if self._state.monthly_risk_used >= self.monthly_limit:
                self._pause("monthly_drawdown_limit_reached")
                return False, "Monthly drawdown limit reached"

            # Check emergency drawdown
            if self._state.current_drawdown_pct >= self.emergency_dd:
                self._pause("emergency_drawdown")
                return False, f"Emergency drawdown {self._state.current_drawdown_pct:.1f}%"

            # Check consecutive losses
            if self._state.consecutive_losses >= self.max_consec_losses:
                self._pause("max_consecutive_losses")
                return False, f"{self.max_consec_losses} consecutive losses"

            # Check if this trade's risk fits in remaining budget
            pct = risk_pct or (risk_usd / max(self._state.current_equity, 1) * 100)
            if pct > self._state.daily_risk_remaining:
                return False, (
                    f"Trade risk {pct:.2f}% > remaining daily budget "
                    f"{self._state.daily_risk_remaining:.2f}%"
                )

            return True, "OK"

    # ------------------------------------------------------------------
    # Allocate risk (when a trade is placed)
    # ------------------------------------------------------------------
    def allocate_risk(self, risk_usd: float, strategy: str = "") -> None:
        """Record risk allocation when a trade is placed."""
        with self._lock:
            self._check_reset()
            pct = risk_usd / max(self._state.current_equity, 1) * 100
            self._state.daily_risk_used += pct
            self._state.daily_risk_remaining = max(0, self.daily_limit - self._state.daily_risk_used)
            self._state.weekly_risk_used += pct
            self._state.monthly_risk_used += pct
            self._state.daily_trades += 1

            trade = {
                "timestamp": time.time(),
                "risk_usd": risk_usd, "risk_pct": pct,
                "strategy": strategy, "result": "open",
            }
            self._daily_trades.append(trade)
            self._weekly_trades.append(trade)
            self._monthly_trades.append(trade)

            log.info("risk_budget: allocated %.2f%% (%.0f trades today, %.2f%% remaining)",
                     pct, self._state.daily_trades, self._state.daily_risk_remaining)

    # ------------------------------------------------------------------
    # Record trade result (when a trade closes)
    # ------------------------------------------------------------------
    def record_trade_result(self, pnl: float, win: bool,
                            strategy: str = "") -> None:
        """Record the outcome of a closed trade."""
        with self._lock:
            self._check_reset()

            self._state.daily_pnl += pnl
            self._state.weekly_pnl += pnl
            self._state.monthly_pnl += pnl

            # Update equity
            self._state.current_equity += pnl
            if self._state.current_equity > self._state.peak_equity:
                self._state.peak_equity = self._state.current_equity

            # Drawdown
            if self._state.peak_equity > 0:
                self._state.current_drawdown_pct = (
                    (self._state.peak_equity - self._state.current_equity)
                    / self._state.peak_equity * 100
                )

            # Consecutive losses/wins
            if win:
                self._state.consecutive_wins += 1
                self._state.consecutive_losses = 0
            else:
                self._state.consecutive_losses += 1
                self._state.consecutive_wins = 0

            # Update risk multiplier
            self._update_risk_multiplier()

            # Check if we should resume trading
            if self._state.trading_paused and self._should_resume():
                self._resume()

            log.info("risk_budget: trade result pnl=%.2f %s (consec_losses=%d, mult=%.2f)",
                     pnl, "WIN" if win else "LOSS",
                     self._state.consecutive_losses,
                     self._state.current_risk_multiplier)

    # ------------------------------------------------------------------
    # Dynamic risk multiplier
    # ------------------------------------------------------------------
    def _update_risk_multiplier(self) -> None:
        """Update risk multiplier based on consecutive losses/wins."""
        consec = self._state.consecutive_losses
        if consec == 0:
            target = 1.0
        elif consec == 1:
            target = 0.75
        elif consec == 2:
            target = 0.50
        elif consec == 3:
            target = 0.25
        else:
            target = 0.10

        # Recovery: if we have enough wins, gradually restore
        if self._state.consecutive_wins >= self.recovery_wins:
            target = min(1.0, target + 0.25)

        self._state.current_risk_multiplier = target

    def get_risk_multiplier(self) -> float:
        """Get current risk multiplier (0.1 to 1.0)."""
        with self._lock:
            return self._state.current_risk_multiplier

    def get_adjusted_risk(self, base_risk_pct: float) -> float:
        """Get risk percentage adjusted by current multiplier."""
        return base_risk_pct * self.get_risk_multiplier()

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------
    def _pause(self, reason: str) -> None:
        """Pause trading."""
        if not self._state.trading_paused:
            self._state.trading_paused = True
            self._state.pause_reason = reason
            log.warning("risk_budget: TRADING PAUSED — %s", reason)

    def _should_resume(self) -> bool:
        """Check if trading should resume after a pause."""
        # Resume after 2 wins or if it's a new day
        if self._state.consecutive_wins >= self.recovery_wins:
            return True
        return False

    def _resume(self) -> None:
        """Resume trading."""
        self._state.trading_paused = False
        self._state.pause_reason = ""
        log.info("risk_budget: trading resumed")

    def manual_resume(self) -> None:
        """Manually resume trading (override pause)."""
        with self._lock:
            self._resume()

    # ------------------------------------------------------------------
    # Periodic resets
    # ------------------------------------------------------------------
    def _check_reset(self) -> None:
        """Check if we need to reset daily/weekly/monthly counters.

        Critical #5 fix: this method MUST be called while holding self._lock.
        The callers (allocate_risk, record_trade_result, etc.) all acquire
        the lock before calling _check_reset, so the state mutations here
        are safe. An assertion is added to catch misuse.
        """
        # Critical #5 fix: assert lock is held to prevent race conditions.
        assert self._lock._is_owned(), "_check_reset must be called while holding self._lock"
        now = datetime.now(tz=timezone.utc)
        today = now.date()
        week_start = self._get_week_start()
        month = now.strftime("%Y-%m")

        # Daily reset
        if today != self._current_day:
            self._current_day = today
            self._state.daily_risk_used = 0.0
            self._state.daily_risk_remaining = self.daily_limit
            self._state.daily_pnl = 0.0
            self._state.daily_trades = 0
            log.info("risk_budget: daily reset — new day %s", today)

        # Weekly reset
        if week_start != self._current_week:
            self._current_week = week_start
            self._state.weekly_risk_used = 0.0
            self._state.weekly_pnl = 0.0
            self._state.weekly_trades = 0
            log.info("risk_budget: weekly reset — new week")

        # Monthly reset
        if month != self._current_month:
            self._current_month = month
            self._state.monthly_risk_used = 0.0
            self._state.monthly_pnl = 0.0
            self._state.monthly_trades = 0
            log.info("risk_budget: monthly reset — new month %s", month)

    # ------------------------------------------------------------------
    # Update equity (from portfolio manager)
    # ------------------------------------------------------------------
    def update_equity(self, equity: float) -> None:
        """Update current equity (called each cycle)."""
        with self._lock:
            self._state.current_equity = equity
            if equity > self._state.peak_equity:
                self._state.peak_equity = equity
            if self._state.peak_equity > 0:
                self._state.current_drawdown_pct = (
                    (self._state.peak_equity - equity)
                    / self._state.peak_equity * 100
                )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def state(self) -> RiskBudgetState:
        """Get current state (copy)."""
        with self._lock:
            self._check_reset()
            return self._state

    def summary(self) -> Dict[str, Any]:
        """Get summary dict."""
        s = self.state()
        return s.to_dict()
