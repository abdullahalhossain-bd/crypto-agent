"""trading_modules/weekly_self_audit.py
=====================================================================
Weekly Self-Audit Engine (Principle #135 — Continuous Self-Audit)
=====================================================================
Every week, the bot audits its own performance across all dimensions
and generates a report card with improvement recommendations.

What It Audits:
    1. PERFORMANCE — win rate, profit factor, Sharpe, total P&L
    2. RISK — max drawdown, risk-adjusted return, VaR accuracy
    3. EXECUTION — slippage, fill rate, latency
    4. STRATEGY — per-strategy ranking, decay detection
    5. MARKET FIT — which regimes/sessions/setups worked best
    6. PSYCHOLOGY — discipline score (did we follow rules?)
    7. INFRASTRUCTURE — system uptime, error rate
    8. LEARNING — what did we learn this week?

Output:
    - Report card (A+ to F grade per dimension)
    - Overall GPA
    - Top 3 strengths
    - Top 3 weaknesses
    - Action items for next week

Usage:
    auditor = WeeklySelfAuditor()

    # Record data throughout the week
    auditor.record_trade(...)
    auditor.record_error(...)
    auditor.record_system_metric(...)

    # Generate weekly report
    report = auditor.audit_week()
    # report = {
    #     "week_ending": "2024-01-07",
    #     "overall_gpa": 3.2,
    #     "grades": {"performance": "B", "risk": "A", ...},
    #     "strengths": [...],
    #     "weaknesses": [...],
    #     "action_items": [...]
    # }
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.weekly_self_audit")


@dataclass
class GradeCard:
    """Grade card for a single dimension."""
    dimension: str
    grade: str = "B"     # A+, A, B, C, D, F
    gpa: float = 3.0     # 0-4
    score: float = 0.0   # 0-100
    metrics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "grade": self.grade,
            "gpa": self.gpa,
            "score": round(self.score, 1),
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class WeeklyAuditReport:
    """Complete weekly audit report."""
    week_ending: str = ""
    overall_gpa: float = 0.0
    overall_grade: str = "B"
    grades: Dict[str, GradeCard] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "week_ending": self.week_ending,
            "overall_gpa": round(self.overall_gpa, 2),
            "overall_grade": self.overall_grade,
            "grades": {k: v.to_dict() for k, v in self.grades.items()},
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "action_items": self.action_items,
            "summary": self.summary,
        }


def score_to_grade(score: float) -> Tuple[str, float]:
    """Convert 0-100 score to letter grade + GPA."""
    if score >= 95:
        return "A+", 4.0
    elif score >= 90:
        return "A", 3.7
    elif score >= 85:
        return "A-", 3.3
    elif score >= 80:
        return "B+", 3.0
    elif score >= 75:
        return "B", 2.7
    elif score >= 70:
        return "B-", 2.3
    elif score >= 65:
        return "C+", 2.0
    elif score >= 60:
        return "C", 1.7
    elif score >= 50:
        return "D", 1.0
    else:
        return "F", 0.0


class WeeklySelfAuditor:
    """Weekly self-audit of trading performance.

    Records data throughout the week, then audits on schedule
    (default: Sunday 00:00 UTC).
    """

    def __init__(self,
                 audit_day: int = 6,        # 0=Mon, 6=Sun
                 initial_equity: float = 10000.0):
        """Initialize auditor.

        Args:
            audit_day: day of week for audit (0=Mon, 6=Sun)
            initial_equity: starting equity for the week
        """
        self.audit_day = audit_day
        self.initial_equity = initial_equity
        self._lock = threading.RLock()
        self._trades: Deque[dict] = deque(maxlen=500)
        self._errors: Deque[dict] = deque(maxlen=100)
        self._system_metrics: Deque[dict] = deque(maxlen=1000)
        self._last_audit: Optional[datetime] = None
        self._weekly_reports: List[WeeklyAuditReport] = []

    # ------------------------------------------------------------------
    # Record data
    # ------------------------------------------------------------------
    def record_trade(self, strategy: str, symbol: str, pnl: float,
                     r_multiple: float, hold_time_s: float,
                     slippage_bps: float = 0, confidence: float = 0.5,
                     regime: str = "unknown", session: str = "unknown",
                     setup: str = "unknown") -> None:
        """Record a completed trade."""
        with self._lock:
            self._trades.append({
                "timestamp": time.time(),
                "strategy": strategy, "symbol": symbol,
                "pnl": pnl, "r_multiple": r_multiple,
                "hold_time_s": hold_time_s, "slippage_bps": slippage_bps,
                "confidence": confidence, "regime": regime,
                "session": session, "setup": setup,
                "win": pnl > 0,
            })

    def record_error(self, component: str, error: str) -> None:
        """Record a system error."""
        with self._lock:
            self._errors.append({
                "timestamp": time.time(),
                "component": component,
                "error": error,
            })

    def record_system_metric(self, metric: str, value: float) -> None:
        """Record a system metric (CPU, memory, latency, etc.)."""
        with self._lock:
            self._system_metrics.append({
                "timestamp": time.time(),
                "metric": metric,
                "value": value,
            })

    # ------------------------------------------------------------------
    # Main audit
    # ------------------------------------------------------------------
    def audit_week(self) -> WeeklyAuditReport:
        """Run the weekly audit.

        Returns:
            WeeklyAuditReport with grades + recommendations
        """
        report = WeeklyAuditReport()
        report.week_ending = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            trades = list(self._trades)
            errors = list(self._errors)
            metrics = list(self._system_metrics)

        if not trades:
            report.summary = "No trades this week — nothing to audit"
            return report

        # === Grade each dimension ===
        report.grades["performance"] = self._audit_performance(trades)
        report.grades["risk"] = self._audit_risk(trades)
        report.grades["execution"] = self._audit_execution(trades)
        report.grades["strategy"] = self._audit_strategy(trades)
        report.grades["market_fit"] = self._audit_market_fit(trades)
        report.grades["discipline"] = self._audit_discipline(trades)
        report.grades["infrastructure"] = self._audit_infrastructure(errors, metrics)
        report.grades["learning"] = self._audit_learning(trades)

        # === Overall GPA ===
        gpas = [g.gpa for g in report.grades.values()]
        report.overall_gpa = sum(gpas) / max(len(gpas), 1)
        report.overall_grade, _ = score_to_grade(report.overall_gpa / 4 * 100)

        # === Strengths + weaknesses ===
        sorted_grades = sorted(report.grades.items(), key=lambda x: x[1].gpa, reverse=True)
        report.strengths = [f"{k}: {v.grade} ({v.notes})"
                           for k, v in sorted_grades[:3] if v.gpa >= 3.0]
        report.weaknesses = [f"{k}: {v.grade} ({v.notes})"
                            for k, v in sorted_grades[-3:] if v.gpa < 3.0]

        # === Action items ===
        report.action_items = self._generate_action_items(report)

        # === Summary ===
        report.summary = self._generate_summary(report, trades)

        # Save report
        with self._lock:
            self._weekly_reports.append(report)
            self._last_audit = datetime.now(tz=timezone.utc)
            # Clear week's data
            self._trades.clear()
            self._errors.clear()

        return report

    # ------------------------------------------------------------------
    # Dimension auditors
    # ------------------------------------------------------------------
    def _audit_performance(self, trades: list) -> GradeCard:
        """Audit raw performance: P&L, win rate, Sharpe."""
        card = GradeCard(dimension="performance")
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / max(len(pnls), 1)
        total_pnl = sum(pnls)
        profit_factor = sum(wins) / max(abs(sum(losses)), 0.01)
        sharpe = (np.mean(pnls) / max(np.std(pnls), 1e-10) * np.sqrt(252)
                 if len(pnls) >= 5 else 0)

        # Score: weighted combination
        score = (
            min(40, win_rate * 60) +           # win rate (0-40)
            min(30, profit_factor * 15) +      # profit factor (0-30)
            min(30, max(0, sharpe) * 15)       # Sharpe (0-30)
        )
        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "trades": len(pnls), "win_rate": round(win_rate, 3),
            "total_pnl": round(total_pnl, 2),
            "profit_factor": round(profit_factor, 2),
            "sharpe": round(sharpe, 3),
            "avg_pnl": round(np.mean(pnls), 2),
        }
        card.notes = (f"{len(pnls)} trades, WR={win_rate:.0%}, "
                     f"PF={profit_factor:.2f}, Sharpe={sharpe:.2f}")
        return card

    def _audit_risk(self, trades: list) -> GradeCard:
        """Audit risk management: drawdown, R-multiples."""
        card = GradeCard(dimension="risk")
        rs = [t["r_multiple"] for t in trades]
        cum_r = np.cumsum(rs)
        peak = np.maximum.accumulate(cum_r)
        dd = peak - cum_r
        max_dd_r = float(np.max(dd)) if len(dd) > 0 else 0
        avg_r = float(np.mean(rs))
        ev_r = avg_r  # expected value in R

        # Score
        score = (
            min(40, max(0, ev_r * 30 + 20)) +   # EV in R (0-40)
            min(30, max(0, 30 - max_dd_r * 10)) +  # Max DD (0-30, lower DD = higher)
            min(30, max(0, avg_r * 25 + 15))     # Avg R (0-30)
        )
        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "ev_r": round(ev_r, 3), "avg_r": round(avg_r, 3),
            "max_dd_r": round(max_dd_r, 2),
            "risk_reward_avg": round(np.mean([abs(r) for r in rs]), 2),
        }
        card.notes = f"EV={ev_r:.2f}R, MaxDD={max_dd_r:.1f}R, AvgR={avg_r:.2f}"
        return card

    def _audit_execution(self, trades: list) -> GradeCard:
        """Audit execution quality: slippage, fill time."""
        card = GradeCard(dimension="execution")
        slippages = [t["slippage_bps"] for t in trades if t["slippage_bps"] > 0]
        avg_slippage = float(np.mean(slippages)) if slippages else 0

        # Score: lower slippage = better
        if avg_slippage < 1:
            score = 95
        elif avg_slippage < 2:
            score = 85
        elif avg_slippage < 5:
            score = 70
        elif avg_slippage < 10:
            score = 50
        else:
            score = 30

        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "avg_slippage_bps": round(avg_slippage, 2),
            "max_slippage_bps": round(max(slippages), 2) if slippages else 0,
        }
        card.notes = f"Avg slippage: {avg_slippage:.1f}bps"
        return card

    def _audit_strategy(self, trades: list) -> GradeCard:
        """Audit per-strategy performance."""
        card = GradeCard(dimension="strategy")
        by_strategy: Dict[str, list] = defaultdict(list)
        for t in trades:
            by_strategy[t["strategy"]].append(t["pnl"])

        strategy_pfs = {}
        for strat, pnls in by_strategy.items():
            wins = sum(p for p in pnls if p > 0)
            losses = abs(sum(p for p in pnls if p < 0))
            pf = wins / max(losses, 0.01)
            strategy_pfs[strat] = round(pf, 2)

        # Score: average profit factor across strategies
        avg_pf = float(np.mean(list(strategy_pfs.values()))) if strategy_pfs else 0
        score = min(100, avg_pf * 40)
        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "strategies_traded": len(by_strategy),
            "per_strategy_pf": strategy_pfs,
            "best_strategy": max(strategy_pfs, key=strategy_pfs.get) if strategy_pfs else None,
            "worst_strategy": min(strategy_pfs, key=strategy_pfs.get) if strategy_pfs else None,
        }
        card.notes = f"{len(by_strategy)} strategies, avg PF={avg_pf:.2f}"
        return card

    def _audit_market_fit(self, trades: list) -> GradeCard:
        """Audit which regimes/sessions/setups worked best."""
        card = GradeCard(dimension="market_fit")
        by_regime: Dict[str, list] = defaultdict(list)
        by_session: Dict[str, list] = defaultdict(list)
        for t in trades:
            by_regime[t["regime"]].append(t["pnl"])
            by_session[t["session"]].append(t["pnl"])

        # Find best/worst regimes
        regime_pnls = {r: sum(ps) for r, ps in by_regime.items()}
        session_pnls = {s: sum(ps) for s, ps in by_session.items()}

        best_regime = max(regime_pnls, key=regime_pnls.get) if regime_pnls else None
        worst_regime = min(regime_pnls, key=regime_pnls.get) if regime_pnls else None

        # Score: did we trade the right regimes?
        profitable_regimes = sum(1 for v in regime_pnls.values() if v > 0)
        total_regimes = len(regime_pnls)
        regime_fit = profitable_regimes / max(total_regimes, 1)

        score = regime_fit * 100
        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "regimes_traded": total_regimes,
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "best_session": max(session_pnls, key=session_pnls.get) if session_pnls else None,
        }
        card.notes = f"Best regime: {best_regime}, Worst: {worst_regime}"
        return card

    def _audit_discipline(self, trades: list) -> GradeCard:
        """Audit discipline: did we follow rules?"""
        card = GradeCard(dimension="discipline")
        # Count trades that violated rules (low confidence, oversized, etc.)
        low_confidence = sum(1 for t in trades if t["confidence"] < 0.5)
        discipline_pct = 1 - (low_confidence / max(len(trades), 1))

        score = discipline_pct * 100
        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "total_trades": len(trades),
            "low_confidence_trades": low_confidence,
            "discipline_pct": round(discipline_pct, 3),
        }
        card.notes = f"Discipline: {discipline_pct:.0%} ({low_confidence} rule violations)"
        return card

    def _audit_infrastructure(self, errors: list, metrics: list) -> GradeCard:
        """Audit system health."""
        card = GradeCard(dimension="infrastructure")
        error_rate = len(errors) / 100  # errors per 100 cycles (approx)

        if error_rate < 0.5:
            score = 95
        elif error_rate < 1.0:
            score = 85
        elif error_rate < 2.0:
            score = 70
        elif error_rate < 5.0:
            score = 50
        else:
            score = 30

        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "errors": len(errors),
            "error_rate": round(error_rate, 2),
        }
        card.notes = f"{len(errors)} errors this week"
        return card

    def _audit_learning(self, trades: list) -> GradeCard:
        """Audit learning: did we improve over the week?"""
        card = GradeCard(dimension="learning")
        if len(trades) < 20:
            card.score = 50
            card.grade, card.gpa = score_to_grade(50)
            card.notes = "Insufficient trades for learning analysis"
            return card

        # Compare first half vs second half
        half = len(trades) // 2
        first_half_ev = np.mean([t["r_multiple"] for t in trades[:half]])
        second_half_ev = np.mean([t["r_multiple"] for t in trades[half:]])
        improvement = second_half_ev - first_half_ev

        if improvement > 0.3:
            score = 90
        elif improvement > 0:
            score = 75
        elif improvement > -0.2:
            score = 60
        else:
            score = 40

        card.score = score
        card.grade, card.gpa = score_to_grade(score)
        card.metrics = {
            "first_half_ev_r": round(float(first_half_ev), 3),
            "second_half_ev_r": round(float(second_half_ev), 3),
            "improvement": round(float(improvement), 3),
        }
        card.notes = f"Improvement: {improvement:+.2f}R over the week"
        return card

    # ------------------------------------------------------------------
    # Action items + summary
    # ------------------------------------------------------------------
    def _generate_action_items(self, report: WeeklyAuditReport) -> List[str]:
        """Generate actionable items for next week."""
        items = []
        for dim, card in report.grades.items():
            if card.gpa < 2.0:
                if dim == "performance":
                    items.append("Review strategy parameters — win rate or profit factor too low")
                elif dim == "risk":
                    items.append("Reduce position sizing — max drawdown too high")
                elif dim == "execution":
                    items.append("Investigate slippage — consider limit orders or different broker")
                elif dim == "strategy":
                    items.append("Disable underperforming strategies (see Strategy Health Monitor)")
                elif dim == "market_fit":
                    items.append("Skip worst-performing regime — focus on best regime")
                elif dim == "discipline":
                    items.append("Tighten entry filters — too many low-confidence trades")
                elif dim == "infrastructure":
                    items.append("Fix system errors — check logs for recurring failures")
                elif dim == "learning":
                    items.append("Performance declining — review what changed mid-week")
        return items

    def _generate_summary(self, report: WeeklyAuditReport, trades: list) -> str:
        """Generate human-readable summary."""
        total_pnl = sum(t["pnl"] for t in trades)
        return (f"Week of {report.week_ending}: "
                f"{len(trades)} trades, P&L=${total_pnl:.2f}, "
                f"GPA={report.overall_gpa:.1f} ({report.overall_grade})")

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def history(self) -> List[WeeklyAuditReport]:
        """Get all past weekly reports."""
        with self._lock:
            return list(self._weekly_reports)

    def should_audit(self) -> bool:
        """Check if it's time for the weekly audit."""
        now = datetime.now(tz=timezone.utc)
        if self._last_audit is None:
            return True
        days_since = (now - self._last_audit).days
        return days_since >= 7
