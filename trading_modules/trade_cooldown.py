"""
Trade Cooldown Manager — "Don't revenge-trade losses"
=======================================================

After losses, human traders revenge-trade. AI bots do the same thing —
they detect a "great" signal immediately after a loss and pile in to
recover. This almost always leads to a second loss.

This module enforces discipline:

    1. LOSS_STREAK_PAUSE — after N consecutive losses, pause for X minutes
    2. DAILY_LOSS_HALT   — once daily loss limit hit, halt for the rest of day
    3. SYMBOL_COOLDOWN   — after a loss on a symbol, skip that symbol for X min
    4. RECENT_TRADE_GAP  — minimum minutes between trades on the same symbol

The state is persisted to a JSON file so it survives restarts.

Usage:
    from trading_modules.trade_cooldown import TradeCooldownManager

    cd = TradeCooldownManager(state_path="data/cooldown.json")

    if not cd.can_trade("BTCUSD"):
        reason = cd.reason("BTCUSD")
        log.info(f"Cooldown active: {reason}")

    cd.record_trade("BTCUSD", result_pnl=-50.0)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class CooldownState:
    # Recent trade results per symbol (most recent last)
    # Each entry: {"time": ISO, "pnl": float, "symbol": str}
    recent_trades: list[dict] = field(default_factory=list)
    # Consecutive loss counter (resets on win)
    consecutive_losses: int = 0
    # Today's realized PnL
    daily_pnl: float = 0.0
    daily_pnl_date: str = ""   # YYYY-MM-DD
    # Per-symbol last trade timestamp
    last_trade_time: dict[str, str] = field(default_factory=dict)
    # Per-symbol cooldown-until timestamp (set after loss)
    symbol_cooldown_until: dict[str, str] = field(default_factory=dict)
    # Global cooldown-until (after loss streak or daily limit)
    global_cooldown_until: str = ""


class TradeCooldownManager:
    """
    Enforce post-loss and post-trade cooldowns.

    Parameters:
        state_path: where to persist state JSON
        loss_streak_threshold: # consecutive losses to trigger pause (default 3)
        loss_streak_pause_minutes: pause duration after loss streak (default 60)
        daily_loss_limit_usd: halt for the day once this is hit (default -500)
        daily_loss_limit_pct: alternative — halt once daily PnL <= -x% of initial capital
                              (mutually exclusive with usd; if both set, usd wins)
        symbol_loss_cooldown_minutes: per-symbol pause after a loss (default 30)
        min_minutes_between_trades: minimum gap between trades on same symbol (default 5)
        initial_capital: used for pct-based daily limit
    """

    def __init__(
        self,
        state_path: str = "data/cooldown.json",
        loss_streak_threshold: int = 3,
        loss_streak_pause_minutes: int = 60,
        daily_loss_limit_usd: float = 500.0,
        daily_loss_limit_pct: Optional[float] = None,
        symbol_loss_cooldown_minutes: int = 30,
        min_minutes_between_trades: int = 5,
        initial_capital: float = 10000.0,
    ) -> None:
        self.state_path = state_path
        self.loss_streak_threshold = loss_streak_threshold
        self.loss_streak_pause_minutes = loss_streak_pause_minutes
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.symbol_loss_cooldown_minutes = symbol_loss_cooldown_minutes
        self.min_minutes_between_trades = min_minutes_between_trades
        self.initial_capital = initial_capital

        self.state = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def can_trade(self, symbol: str, now: Optional[datetime] = None) -> bool:
        """Return True if a new trade is allowed on `symbol`."""
        return self._check(symbol, now)[0]

    def reason(self, symbol: str, now: Optional[datetime] = None) -> str:
        """Return human-readable reason if can_trade is False."""
        return self._check(symbol, now)[1]

    def record_trade(self, symbol: str, pnl: float, now: Optional[datetime] = None) -> None:
        """Record a closed trade. Positive pnl = win, negative = loss."""
        now = now or datetime.now(timezone.utc)
        self._rollover_day_if_needed(now)

        self.state.recent_trades.append({
            "time": now.isoformat(),
            "symbol": symbol,
            "pnl": float(pnl),
        })
        # Trim history to last 50 trades
        if len(self.state.recent_trades) > 50:
            self.state.recent_trades = self.state.recent_trades[-50:]

        # Consecutive loss counter
        if pnl < 0:
            self.state.consecutive_losses += 1
            # Set per-symbol cooldown
            cooldown_until = now + timedelta(minutes=self.symbol_loss_cooldown_minutes)
            self.state.symbol_cooldown_until[symbol] = cooldown_until.isoformat()
            # Set global cooldown if loss streak hit
            if self.state.consecutive_losses >= self.loss_streak_threshold:
                global_until = now + timedelta(minutes=self.loss_streak_pause_minutes)
                self.state.global_cooldown_until = global_until.isoformat()
                log.warning(
                    "Loss streak = %d → global cooldown until %s",
                    self.state.consecutive_losses, global_until.isoformat(),
                )
        else:
            self.state.consecutive_losses = 0

        # Daily PnL
        self.state.daily_pnl += float(pnl)

        # Last trade time
        self.state.last_trade_time[symbol] = now.isoformat()

        self._save()

    def reset(self) -> None:
        """Clear all state (use with caution)."""
        self.state = CooldownState()
        self._save()

    def status(self) -> dict:
        """Return a snapshot of current cooldown state."""
        return asdict(self.state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check(self, symbol: str, now: Optional[datetime] = None) -> tuple[bool, str]:
        now = now or datetime.now(timezone.utc)
        self._rollover_day_if_needed(now)

        # 1. Global cooldown (loss streak)
        if self.state.global_cooldown_until:
            until = self._parse_iso(self.state.global_cooldown_until)
            if until and now < until:
                return False, (f"Global cooldown (loss streak {self.state.consecutive_losses}) "
                               f"until {until.isoformat()}")

        # 2. Daily loss limit
        if self._daily_limit_hit():
            return False, (f"Daily loss limit hit: PnL=${self.state.daily_pnl:.2f}, "
                           f"limit=-${self.daily_loss_limit_usd}")

        # 3. Per-symbol cooldown
        sc = self.state.symbol_cooldown_until.get(symbol)
        if sc:
            until = self._parse_iso(sc)
            if until and now < until:
                return False, (f"Symbol {symbol} cooldown after loss "
                               f"until {until.isoformat()}")

        # 4. Minimum gap between trades
        last_iso = self.state.last_trade_time.get(symbol)
        if last_iso:
            last = self._parse_iso(last_iso)
            if last:
                gap_min = (now - last).total_seconds() / 60.0
                if gap_min < self.min_minutes_between_trades:
                    return False, (f"Min gap not met: {gap_min:.1f}min < "
                                   f"{self.min_minutes_between_trades}min")

        return True, "OK"

    def _daily_limit_hit(self) -> bool:
        if self.daily_loss_limit_usd and self.state.daily_pnl <= -abs(self.daily_loss_limit_usd):
            return True
        if (self.daily_loss_limit_pct and self.initial_capital > 0):
            limit_usd = abs(self.daily_loss_limit_pct) * self.initial_capital
            if self.state.daily_pnl <= -limit_usd:
                return True
        return False

    def _rollover_day_if_needed(self, now: datetime) -> None:
        today = now.strftime("%Y-%m-%d")
        if self.state.daily_pnl_date != today:
            self.state.daily_pnl_date = today
            self.state.daily_pnl = 0.0
            self._save()

    @staticmethod
    def _parse_iso(s: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> CooldownState:
        if not os.path.exists(self.state_path):
            return CooldownState()
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            return CooldownState(**data)
        except Exception as e:
            log.warning("Failed to load cooldown state: %s — starting fresh", e)
            return CooldownState()

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
            with open(self.state_path, "w") as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            log.warning("Failed to save cooldown state: %s", e)
