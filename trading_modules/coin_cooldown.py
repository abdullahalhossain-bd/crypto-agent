"""
Coin Cooldown Manager — Per-Symbol Loss Tracking
==================================================

Novel behavioral risk control: prevents revenge trading on a specific symbol.
If a symbol has burned you recently, the system enforces a cooldown before
allowing new entries.

Features:
  - Per-symbol loss tracking (consecutive losses, total loss)
  - Cooldown enforcement after consecutive losses
  - Historical loss penalty (downweight signals on symbols with bad history)
  - Configurable thresholds per risk profile

Source: NexusQuant (review #29) — coinCooldownManager.ts concept

Usage:
    from coin_cooldown import CoinCooldownManager

    ccm = CoinCooldownManager()

    # Record a loss
    ccm.record_trade("BTCUSDT", pnl_usd=-150.0, result="loss")

    # Check if symbol is in cooldown
    if ccm.is_in_cooldown("BTCUSDT"):
        info = ccm.get_cooldown_info("BTCUSDT")
        print(f"BTCUSDT in cooldown until {info['cooldown_until']}")
        print(f"Reason: {info['reason']}")

    # Get loss penalty for signal scoring
    penalty = ccm.get_loss_penalty("BTCUSDT")
    # penalty = 0.5 means signal strength should be halved
    adjusted_signal = raw_signal * (1.0 - penalty)
"""

from __future__ import annotations

import threading

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SymbolStats:
    """Per-symbol trading statistics."""
    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    total_pnl_usd: float = 0.0
    last_trade_time: Optional[str] = None
    last_loss_time: Optional[str] = None
    cooldown_until: Optional[str] = None
    cooldown_reason: Optional[str] = None

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl_usd / self.total_trades if self.total_trades > 0 else 0.0


class CoinCooldownManager:
    """
    Per-symbol cooldown manager.

    Prevents revenge trading by enforcing cooldowns after consecutive losses.
    Also applies a loss penalty to signal scoring for symbols with bad history.
    """

    # Default cooldown thresholds
    COOLDOWN_AFTER_CONSECUTIVE_LOSSES = 3
    COOLDOWN_DURATION_HOURS = 24  # 24-hour cooldown

    # Loss penalty scaling
    MAX_LOSS_PENALTY = 0.5  # Max 50% signal reduction
    PENALTY_PER_CONSECUTIVE_LOSS = 0.15  # 15% per consecutive loss

    def __init__(
        self,
        storage_path: str | Path = "memory_data/coin_cooldown.json",
        cooldown_after: int = COOLDOWN_AFTER_CONSECUTIVE_LOSSES,
        cooldown_hours: int = COOLDOWN_DURATION_HOURS,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown_after = cooldown_after
        self.cooldown_hours = cooldown_hours
        self._lock = threading.Lock()  # Critical #2 fix
        self._stats: dict[str, SymbolStats] = self._load()

    def record_trade(self, symbol: str, pnl_usd: float, result: str = None) -> None:
        """
        Record a trade outcome for a symbol.

        Args:
            symbol: Trading symbol
            pnl_usd: Profit/Loss in USD (negative = loss)
            result: "win" / "loss" / "breakeven" (auto-detected if None)
        """
        symbol = symbol.upper()

        if result is None:
            if pnl_usd > 0:
                result = "win"
            elif pnl_usd < 0:
                result = "loss"
            else:
                result = "breakeven"

        stats = self._stats.get(symbol, SymbolStats(symbol=symbol))
        stats.total_trades += 1
        stats.total_pnl_usd += pnl_usd
        stats.last_trade_time = datetime.now(timezone.utc).isoformat()

        if result == "win":
            stats.wins += 1
            stats.consecutive_losses = 0
        elif result == "loss":
            stats.losses += 1
            stats.consecutive_losses += 1
            stats.last_loss_time = datetime.now(timezone.utc).isoformat()

            # Check if cooldown should be triggered
            if stats.consecutive_losses >= self.cooldown_after:
                cooldown_until = datetime.now(timezone.utc) + timedelta(hours=self.cooldown_hours)
                stats.cooldown_until = cooldown_until.isoformat()
                stats.cooldown_reason = (
                    f"{stats.consecutive_losses} consecutive losses on {symbol}"
                )
                logger.warning(
                    f"⛔ {symbol} entered {self.cooldown_hours}h cooldown after "
                    f"{stats.consecutive_losses} consecutive losses"
                )

        self._stats[symbol] = stats
        self._save()

    def is_in_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is currently in cooldown."""
        symbol = symbol.upper()
        stats = self._stats.get(symbol)
        if not stats or not stats.cooldown_until:
            return False

        try:
            cooldown_end = datetime.fromisoformat(stats.cooldown_until.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) < cooldown_end:
                return True
            else:
                # Cooldown expired — clear it
                stats.cooldown_until = None
                stats.cooldown_reason = None
                stats.consecutive_losses = 0  # Reset after cooldown
                self._save()
                return False
        except (ValueError, TypeError):
            return False

    def get_cooldown_info(self, symbol: str) -> Optional[dict]:
        """Get cooldown details for a symbol."""
        symbol = symbol.upper()
        stats = self._stats.get(symbol)
        if not stats:
            return None

        return {
            "symbol": symbol,
            "in_cooldown": self.is_in_cooldown(symbol),
            "cooldown_until": stats.cooldown_until,
            "cooldown_reason": stats.cooldown_reason,
            "consecutive_losses": stats.consecutive_losses,
            "total_trades": stats.total_trades,
            "win_rate": stats.win_rate,
            "total_pnl": stats.total_pnl_usd,
        }

    def get_loss_penalty(self, symbol: str) -> float:
        """
        Get signal penalty for a symbol based on loss history.

        Returns 0.0 (no penalty) to MAX_LOSS_PENALTY (0.5 = halve signal).
        """
        symbol = symbol.upper()
        stats = self._stats.get(symbol)
        if not stats:
            return 0.0

        penalty = stats.consecutive_losses * self.PENALTY_PER_CONSECUTIVE_LOSS
        return min(penalty, self.MAX_LOSS_PENALTY)

    def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all tracked symbols."""
        return {
            symbol: asdict(stats) for symbol, stats in self._stats.items()
        }

    def clear_cooldown(self, symbol: str) -> None:
        """Manually clear cooldown for a symbol (operator override)."""
        symbol = symbol.upper()
        stats = self._stats.get(symbol)
        if stats:
            stats.cooldown_until = None
            stats.cooldown_reason = None
            stats.consecutive_losses = 0
            self._save()
            logger.info(f"Cooldown cleared for {symbol}")

    def _load(self) -> dict[str, SymbolStats]:
        """Load stats from JSON file."""
        if not self.storage_path.exists():
            return {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                k: SymbolStats(**v) for k, v in data.items()
            }
        except (json.JSONDecodeError, OSError, TypeError):
            return {}

    def _save(self) -> None:
        """Critical #2 fix: thread-safe save with file lock."""
        with self._lock:
            try:
                tmp = self.storage_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(
                        {k: asdict(v) for k, v in self._stats.items()},
                        f, indent=2, ensure_ascii=False, default=str,
                    )
                import os
                os.replace(tmp, self.storage_path)
            except OSError as e:
                logger.warning(f"Failed to save coin cooldown: {e}")
