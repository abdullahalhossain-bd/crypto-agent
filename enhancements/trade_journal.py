"""enhancements.trade_journal
=====================================================================
Day 149-151 — Trade journal with post-trade analysis.

Different from metrics/decision_traces (machine-generated). This is
a HUMAN-reviewable journal where each closed trade gets:
  - Full context (entry/exit/setup)
  - What worked / what didn't
  - Lessons learned (operator can annotate)
  - Tags for categorisation
  - Periodic analysis (best/worst setups, common mistakes)

The journal persists to JSON and can be exported for review.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("enhancements.journal")


@dataclass
class JournalEntry:
    entry_id: str
    symbol: str
    timeframe: str
    side: str                    # long / short
    strategy: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    lots: float
    pnl: float
    pnl_pct: float
    r_multiple: float
    hold_bars: int
    # Setup context
    market_state: str = ""
    regime: str = ""
    confluence_grade: str = ""
    pattern: str = ""
    # Quality
    quality_grade: str = ""
    # Post-trade analysis (operator-filled)
    what_worked: str = ""
    what_didnt: str = ""
    lessons: str = ""
    tags: list[str] = field(default_factory=list)
    # Outcome classification
    outcome: str = ""            # "win" / "loss" / "breakeven" / "scratch"
    mistake: str = ""            # "early_entry" / "late_exit" / "wrong_regime" etc.
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(tz=timezone.utc).isoformat()
        if not self.outcome:
            if self.pnl > 0:
                self.outcome = "win"
            elif self.pnl < 0:
                self.outcome = "loss"
            else:
                self.outcome = "breakeven"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class TradeJournal:
    def __init__(self, path: str = "data/trade_journal.jsonl") -> None:
        self.path = path
        self._entries: list[JournalEntry] = []
        self._load()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ----------------------------------------------------------------
    def record(self, entry: JournalEntry) -> None:
        self._entries.append(entry)
        self._persist(entry)
        log.info("Journal entry recorded: %s %s %s pnl=%+.2f",
                 entry.symbol, entry.side, entry.outcome, entry.pnl)

    def annotate(self, entry_id: str, what_worked: str = "",
                  what_didnt: str = "", lessons: str = "",
                  tags: Optional[list[str]] = None,
                  mistake: str = "") -> bool:
        """Add operator annotations to an existing entry.

        Major #5 fix: annotations are now stored in a separate append-only
        file (`<path>.annotations.jsonl`) instead of rewriting the entire
        journal. This avoids the O(n) I/O cost of rewriting hundreds of
        entries when annotating a single trade. The annotations are merged
        into entries on load.
        """
        for e in self._entries:
            if e.entry_id == entry_id:
                if what_worked:
                    e.what_worked = what_worked
                if what_didnt:
                    e.what_didnt = what_didnt
                if lessons:
                    e.lessons = lessons
                if tags:
                    e.tags = list(tags)
                if mistake:
                    e.mistake = mistake
                # Major #5 fix: append annotation to a separate file
                # instead of rewriting the entire journal.
                self._append_annotation(entry_id, what_worked, what_didnt,
                                        lessons, tags, mistake)
                return True
        return False

    def _append_annotation(self, entry_id: str, what_worked: str,
                            what_didnt: str, lessons: str,
                            tags: Optional[list[str]], mistake: str) -> None:
        """Major #5 fix: append a single annotation line to a separate file.
        This is O(1) instead of O(n) — no full rewrite needed."""
        ann_path = self.path + ".annotations.jsonl"
        try:
            import os as _os
            _os.makedirs(_os.path.dirname(ann_path) or ".", exist_ok=True)
            ann = {
                "entry_id": entry_id,
                "what_worked": what_worked,
                "what_didnt": what_didnt,
                "lessons": lessons,
                "tags": list(tags) if tags else [],
                "mistake": mistake,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            with open(ann_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ann, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("journal annotation append failed: %r", e)

    # ----------------------------------------------------------------
    def all_entries(self) -> list[JournalEntry]:
        return list(self._entries)

    def filter(self, symbol: Optional[str] = None,
                strategy: Optional[str] = None,
                outcome: Optional[str] = None,
                since: Optional[str] = None) -> list[JournalEntry]:
        out = []
        for e in self._entries:
            if symbol and e.symbol != symbol:
                continue
            if strategy and e.strategy != strategy:
                continue
            if outcome and e.outcome != outcome:
                continue
            if since and e.entry_time < since:
                continue
            out.append(e)
        return out

    # ----------------------------------------------------------------
    def analyse(self) -> "JournalAnalysis":
        """Produce a summary analysis of all entries."""
        if not self._entries:
            return JournalAnalysis()
        n = len(self._entries)
        wins = [e for e in self._entries if e.outcome == "win"]
        losses = [e for e in self._entries if e.outcome == "loss"]
        pnls = [e.pnl for e in self._entries]
        total_pnl = sum(pnls)
        win_rate = len(wins) / n
        avg_win = (sum(e.pnl for e in wins) / len(wins)) if wins else 0
        avg_loss = (sum(e.pnl for e in losses) / len(losses)) if losses else 0
        # Best/worst
        best = max(self._entries, key=lambda e: e.pnl)
        worst = min(self._entries, key=lambda e: e.pnl)
        # Per-strategy
        per_strategy: dict[str, dict[str, Any]] = {}
        for e in self._entries:
            s = per_strategy.setdefault(e.strategy, {"n": 0, "wins": 0, "pnl": 0.0})
            s["n"] += 1
            if e.outcome == "win":
                s["wins"] += 1
            s["pnl"] += e.pnl
        for s in per_strategy.values():
            s["win_rate"] = s["wins"] / s["n"] if s["n"] else 0
        # Common mistakes
        mistakes: dict[str, int] = {}
        for e in self._entries:
            if e.mistake:
                mistakes[e.mistake] = mistakes.get(e.mistake, 0) + 1
        # Common tags
        tags: dict[str, int] = {}
        for e in self._entries:
            for t in e.tags:
                tags[t] = tags.get(t, 0) + 1
        return JournalAnalysis(
            n_trades=n, n_wins=len(wins), n_losses=len(losses),
            win_rate=win_rate, total_pnl=total_pnl,
            avg_win=avg_win, avg_loss=avg_loss,
            best_trade=best.to_dict(), worst_trade=worst.to_dict(),
            per_strategy=per_strategy,
            common_mistakes=dict(mistakes),
            common_tags=dict(tags),
        )

    # ----------------------------------------------------------------
    def export_markdown(self) -> str:
        """Export the journal as markdown for human review."""
        analysis = self.analyse()
        lines = [
            "# Trade Journal\n",
            f"**Total trades:** {analysis.n_trades}  ",
            f"**Win rate:** {analysis.win_rate:.1%}  ",
            f"**Total PnL:** {analysis.total_pnl:+.2f}\n",
            "## Per-Strategy Breakdown\n",
            "| Strategy | Trades | Win Rate | Total PnL |",
            "|----------|--------|----------|-----------|",
        ]
        for strat, stats in analysis.per_strategy.items():
            lines.append(f"| {strat} | {stats['n']} | {stats['win_rate']:.1%} | {stats['pnl']:+.2f} |")
        lines.append("\n## Common Mistakes\n")
        for m, count in sorted(analysis.common_mistakes.items(),
                                key=lambda x: -x[1]):
            lines.append(f"- **{m}**: {count} times")
        lines.append("\n## Recent Trades\n")
        lines.append("| Time | Symbol | Side | Strategy | PnL | R | Outcome |")
        lines.append("|------|--------|------|----------|-----|---|---------|")
        for e in self._entries[-20:]:  # last 20
            lines.append(f"| {e.entry_time[:10]} | {e.symbol} | {e.side} | "
                         f"{e.strategy} | {e.pnl:+.2f} | {e.r_multiple:.1f}R | {e.outcome} |")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    def _persist(self, entry: JournalEntry) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("journal persist failed: %r", e)

    def _rewrite_all(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                for e in self._entries:
                    f.write(json.dumps(e.to_dict(), default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            log.warning("journal rewrite failed: %r", e)

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self._entries.append(JournalEntry(**d))
                    except Exception:  # noqa: BLE001
                        continue
            # Major #5 fix: merge annotations from the separate file.
            self._load_annotations()
            log.info("Journal loaded: %d entries", len(self._entries))
        except Exception as e:  # noqa: BLE001
            log.warning("journal load failed: %r", e)

    def _load_annotations(self) -> None:
        """Major #5 fix: load annotations from the separate append-only file
        and merge them into the in-memory entries."""
        ann_path = self.path + ".annotations.jsonl"
        if not os.path.isfile(ann_path):
            return
        try:
            # Build a lookup of entry_id -> entry for fast merge.
            entry_map = {e.entry_id: e for e in self._entries}
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ann = json.loads(line)
                        entry_id = ann.get("entry_id", "")
                        if entry_id in entry_map:
                            e = entry_map[entry_id]
                            # Apply the latest annotation (last write wins).
                            if ann.get("what_worked"):
                                e.what_worked = ann["what_worked"]
                            if ann.get("what_didnt"):
                                e.what_didnt = ann["what_didnt"]
                            if ann.get("lessons"):
                                e.lessons = ann["lessons"]
                            if ann.get("tags"):
                                e.tags = list(ann["tags"])
                            if ann.get("mistake"):
                                e.mistake = ann["mistake"]
                    except Exception:  # noqa: BLE001
                        continue
        except Exception as e:  # noqa: BLE001
            log.warning("journal annotation load failed: %r", e)


# ----------------------------------------------------------------------
@dataclass
class JournalAnalysis:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: dict[str, Any] = field(default_factory=dict)
    worst_trade: dict[str, Any] = field(default_factory=dict)
    per_strategy: dict[str, Any] = field(default_factory=dict)
    common_mistakes: dict[str, int] = field(default_factory=dict)
    common_tags: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
