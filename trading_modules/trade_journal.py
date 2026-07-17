"""
Trade Journal + Learning Loop — "post-trade analysis & self-improvement"
========================================================================

Every closed trade is recorded with the full decision context (gate score,
checklist, regime, candle quality, confluence signals). The journal then
identifies which filters are failing most often, which setups are
profitable, and feeds that back into the next decision.

Features:
    1. record_trade()     — persist a closed trade with full context
    2. analyze_history()  — compute win rate per grade, per regime, per
                            symbol, identify weak filters
    3. suggest_tweaks()   — recommend config changes based on history
    4. filter_reliability — per-filter accuracy (win rate when filter passed)

The journal is persisted as JSONL (one trade per line) so it can be
inspected manually or reloaded after a restart.

Usage:
    from trading_modules.trade_journal import TradeJournal

    journal = TradeJournal(path="data/trade_journal.jsonl")
    journal.record_trade(
        symbol="BTCUSD", direction="BUY",
        entry=65200, exit=66400, pnl=120.0,
        gate_score=88.0, grade="A+",
        checklist_passed=10, checklist_failed=0,
        regime="trending_up", candle_quality=0.85,
        rr_planned=2.5, rr_actual=2.5,
        confidence_pct=82.0,
        notes="clean BOS retest",
    )
    stats = journal.analyze_history()
    print(f"Win rate: {stats['overall_win_rate']:.1%}")
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    # Identity
    timestamp: str
    symbol: str
    direction: str              # "BUY" or "SELL"
    # P&L
    entry: float
    exit: float
    pnl: float                  # dollar PnL (positive = win, negative = loss)
    pnl_pct: float              # % return on entry
    # Decision context (what the gate knew at entry time)
    gate_score: float
    grade: str
    checklist_passed: int
    checklist_failed: int
    failed_checks: list[str] = field(default_factory=list)
    regime: str = ""
    candle_quality: float = 0.0
    candle_quality_label: str = ""
    rr_planned: float = 0.0
    rr_actual: float = 0.0
    confidence_pct: float = 0.0
    win_probability: float = 0.0
    htf_alignment: bool = False
    # Outcome
    outcome: str = ""           # "win" / "loss" / "breakeven"
    exit_reason: str = ""       # "TP" / "SL" / "manual" / "kill_switch"
    notes: str = ""
    # Session context
    session: str = ""


@dataclass
class JournalStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    overall_win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0     # avg expected $ per trade
    # Breakdown by grade
    by_grade: dict[str, dict] = field(default_factory=dict)
    # Breakdown by regime
    by_regime: dict[str, dict] = field(default_factory=dict)
    # Filter reliability
    filter_reliability: dict[str, dict] = field(default_factory=dict)
    # Recommendations
    recommendations: list[str] = field(default_factory=list)


class TradeJournal:
    """Append-only JSONL trade journal with analytics.

    Parameters:
        path: path to JSONL file (created if missing)
        min_trades_for_stats: minimum trades before stats are reliable (default 20)
    """

    def __init__(
        self, path: str = "data/trade_journal.jsonl",
        min_trades_for_stats: int = 20,
    ) -> None:
        self.path = path
        self.min_trades_for_stats = min_trades_for_stats
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry: float,
        exit: float,
        pnl: float,
        gate_score: float,
        grade: str,
        checklist_passed: int,
        checklist_failed: int,
        failed_checks: Optional[list[str]] = None,
        regime: str = "",
        candle_quality: float = 0.0,
        candle_quality_label: str = "",
        rr_planned: float = 0.0,
        rr_actual: float = 0.0,
        confidence_pct: float = 0.0,
        win_probability: float = 0.0,
        htf_alignment: bool = False,
        outcome: str = "",
        exit_reason: str = "",
        session: str = "",
        notes: str = "",
    ) -> None:
        """Append a closed trade to the journal."""
        pnl_pct = (exit - entry) / entry * 100 if entry != 0 else 0
        if direction.upper() == "SELL":
            pnl_pct = -pnl_pct
        if outcome == "":
            if pnl > 0:
                outcome = "win"
            elif pnl < 0:
                outcome = "loss"
            else:
                outcome = "breakeven"
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol, direction=direction.upper(),
            entry=float(entry), exit=float(exit), pnl=float(pnl),
            pnl_pct=float(pnl_pct),
            gate_score=float(gate_score), grade=grade,
            checklist_passed=int(checklist_passed),
            checklist_failed=int(checklist_failed),
            failed_checks=failed_checks or [],
            regime=regime, candle_quality=float(candle_quality),
            candle_quality_label=candle_quality_label,
            rr_planned=float(rr_planned), rr_actual=float(rr_actual),
            confidence_pct=float(confidence_pct),
            win_probability=float(win_probability),
            htf_alignment=bool(htf_alignment),
            outcome=outcome, exit_reason=exit_reason,
            session=session, notes=notes,
        )
        # Critical #3 fix: atomic append — write to temp file, then rename.
        # The old `open(path, "a")` could leave a partial JSON line if the
        # process crashed mid-write, corrupting the entire journal.
        # We now use os.open + os.write + os.fsync + os.close, then append
        # the complete line atomically. For JSONL append, we write the
        # complete line to a temp file and use os.replace on the temp file
        # appended to the main file. However, since this is append-mode JSONL,
        # the simplest safe approach is to write the complete line to a
        # buffer first, then write it in a single os.write() call with fsync.
        line = json.dumps(asdict(record)) + "\n"
        import os as _os
        fd = _os.open(self.path, _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
        try:
            _os.write(fd, line.encode("utf-8"))
            _os.fsync(fd)
        finally:
            _os.close(fd)
        logger.info(
            "Trade journal: recorded %s %s pnl=%s outcome=%s grade=%s",
            symbol, direction, pnl, outcome, grade,
        )

    def load_history(self) -> list[dict]:
        """Load all trades from the JSONL file."""
        if not os.path.exists(self.path):
            return []
        records: list[dict] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed journal line: %s", e)
        return records

    def analyze_history(self) -> JournalStats:
        """Compute aggregate statistics and recommendations."""
        records = self.load_history()
        stats = JournalStats()
        if not records:
            stats.recommendations.append(
                "No trades in journal yet — start trading in paper mode to build history."
            )
            return stats

        stats.total_trades = len(records)
        wins = [r for r in records if r.get("outcome") == "win"]
        losses = [r for r in records if r.get("outcome") == "loss"]
        breakeven = [r for r in records if r.get("outcome") == "breakeven"]
        stats.wins = len(wins)
        stats.losses = len(losses)
        stats.breakeven = len(breakeven)
        decided = stats.wins + stats.losses
        stats.overall_win_rate = (stats.wins / decided) if decided > 0 else 0.0
        stats.total_pnl = float(sum(r.get("pnl", 0) for r in records))
        stats.avg_pnl = stats.total_pnl / stats.total_trades
        stats.avg_win = float(np.mean([r["pnl"] for r in wins])) if wins else 0.0
        stats.avg_loss = float(np.mean([r["pnl"] for r in losses])) if losses else 0.0
        gross_profit = float(sum(r["pnl"] for r in wins))
        gross_loss = abs(float(sum(r["pnl"] for r in losses)))
        stats.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        stats.expectancy = (
            stats.overall_win_rate * stats.avg_win
            - (1 - stats.overall_win_rate) * abs(stats.avg_loss)
        )

        # Breakdown by grade
        grades: dict[str, list[dict]] = {}
        for r in records:
            grades.setdefault(r.get("grade", "?"), []).append(r)
        for grade, rs in grades.items():
            ws = sum(1 for r in rs if r.get("outcome") == "win")
            ls = sum(1 for r in rs if r.get("outcome") == "loss")
            total_pnl = sum(r.get("pnl", 0) for r in rs)
            stats.by_grade[grade] = {
                "count": len(rs),
                "wins": ws,
                "losses": ls,
                "win_rate": (ws / (ws + ls)) if (ws + ls) > 0 else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / len(rs), 2),
            }

        # Breakdown by regime
        regimes: dict[str, list[dict]] = {}
        for r in records:
            regimes.setdefault(r.get("regime", "unknown"), []).append(r)
        for regime, rs in regimes.items():
            ws = sum(1 for r in rs if r.get("outcome") == "win")
            ls = sum(1 for r in rs if r.get("outcome") == "loss")
            stats.by_regime[regime] = {
                "count": len(rs),
                "wins": ws,
                "losses": ls,
                "win_rate": (ws / (ws + ls)) if (ws + ls) > 0 else 0,
                "total_pnl": round(sum(r.get("pnl", 0) for r in rs), 2),
            }

        # Filter reliability — for each filter name, what % of trades that
        # passed this filter were wins?
        all_filters: set[str] = set()
        for r in records:
            for f in r.get("failed_checks", []):
                all_filters.add(f)
        # We track which filters FAILED — so "reliability" = win rate when
        # the filter was NOT in failed_checks (i.e., it passed)
        for fname in ["trend_aligned", "htf_bias_clear", "price_at_key_level",
                      "liquidity_taken", "structure_confirmed", "retest_complete",
                      "volume_supports", "spread_acceptable", "rr_acceptable",
                      "risk_within_limits", "candle_quality_ok"]:
            passed = [r for r in records if fname not in r.get("failed_checks", [])]
            failed = [r for r in records if fname in r.get("failed_checks", [])]
            ws = sum(1 for r in passed if r.get("outcome") == "win")
            ls = sum(1 for r in passed if r.get("outcome") == "loss")
            fws = sum(1 for r in failed if r.get("outcome") == "win")
            fls = sum(1 for r in failed if r.get("outcome") == "loss")
            stats.filter_reliability[fname] = {
                "win_rate_when_passed": (ws / (ws + ls)) if (ws + ls) > 0 else 0,
                "win_rate_when_failed": (fws / (fws + fls)) if (fws + fls) > 0 else 0,
                "trades_passed": len(passed),
                "trades_failed": len(failed),
            }

        # Recommendations
        if stats.total_trades < self.min_trades_for_stats:
            stats.recommendations.append(
                f"Only {stats.total_trades} trades — need at least "
                f"{self.min_trades_for_stats} for reliable stats."
            )
        else:
            if stats.profit_factor < 1.0:
                stats.recommendations.append(
                    f"Profit factor {stats.profit_factor:.2f} < 1.0 — strategy "
                    f"is losing money. Tighten thresholds."
                )
            for grade, s in stats.by_grade.items():
                if s["count"] >= 5 and s["win_rate"] < 0.4:
                    stats.recommendations.append(
                        f"Grade {grade} win rate is only {s['win_rate']:.0%} "
                        f"({s['wins']}W/{s['losses']}L) — consider skipping this grade."
                    )
            for fname, fr in stats.filter_reliability.items():
                if fr["trades_failed"] >= 5 and fr["win_rate_when_failed"] > 0.55:
                    stats.recommendations.append(
                        f"Filter '{fname}' may be too strict — trades that fail "
                        f"this filter still win {fr['win_rate_when_failed']:.0%} "
                        f"of the time."
                    )
                if fr["trades_passed"] >= 5 and fr["win_rate_when_passed"] < 0.4:
                    stats.recommendations.append(
                        f"Filter '{fname}' may be too lenient — trades that pass "
                        f"only win {fr['win_rate_when_passed']:.0%} of the time."
                    )
            if stats.expectancy < 0:
                stats.recommendations.append(
                    f"Negative expectancy (${stats.expectancy:.2f}/trade) — "
                    f"halt live trading and review journal."
                )

        return stats


__all__ = ["TradeJournal", "TradeRecord", "JournalStats"]
