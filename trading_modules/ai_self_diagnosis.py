"""trading_modules/ai_self_diagnosis.py
=====================================================================
AI Self-Diagnosis System (Principle #177)
=====================================================================
Weekly self-diagnosis: the AI examines its own performance, identifies
weaknesses, and generates improvement recommendations.

Diagnosis Dimensions:
    1. STRATEGY WEAKNESS — which strategies are losing edge?
    2. SESSION WEAKNESS — which sessions perform worst?
    3. REGIME WEAKNESS — which regimes are we bad at trading?
    4. EXECUTION WEAKNESS — where are we leaking alpha?
    5. RISK WEAKNESS — are we taking too much / too little risk?
    6. DISCIPLINE WEAKNESS — are we following our rules?
    7. LEARNING WEAKNESS — are we improving over time?

For each weakness, the system performs ROOT CAUSE ANALYSIS:
    - What is the symptom? (e.g., "low win rate in Asia session")
    - What is the likely cause? (e.g., "low liquidity in Asia")
    - What is the recommended fix? (e.g., "reduce Asia session size 50%")

Output:
    - Diagnosis report with severity scores (0-10 per dimension)
    - Top 3 weaknesses with root cause analysis
    - Actionable improvement recommendations
    - 30-day improvement trajectory

Usage:
    diag = AISelfDiagnosis()

    # Feed it trade data
    for trade in trades:
        diag.record_trade(...)

    # Run weekly diagnosis
    report = diag.diagnose()
    # report = {
    #     "overall_health": 6.5,  # out of 10
    #     "weaknesses": [
    #         {"dimension": "session", "severity": 7,
    #          "symptom": "Asia WR=30%", "cause": "low liquidity",
    #          "fix": "reduce Asia size 50%"},
    #         ...
    #     ],
    #     "recommendations": [...],
    #     "improvement_trajectory": +0.3,
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

log = get_logger("trading_bot.ai_self_diagnosis")


@dataclass
class Weakness:
    """Identified weakness with root cause analysis."""
    dimension: str          # strategy/session/regime/execution/risk/discipline/learning
    severity: float = 0.0   # 0-10 (10 = critical)
    symptom: str = ""       # what we observed
    cause: str = ""         # likely root cause
    fix: str = ""           # recommended action
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosisReport:
    """Complete self-diagnosis report."""
    diagnosis_date: str = ""
    overall_health: float = 0.0      # 0-10
    improvement_trajectory: float = 0.0  # -10 to +10 (vs last week)

    # Per-dimension scores (0-10, higher = better)
    strategy_health: float = 5.0
    session_health: float = 5.0
    regime_health: float = 5.0
    execution_health: float = 5.0
    risk_health: float = 5.0
    discipline_health: float = 5.0
    learning_health: float = 5.0

    # Weaknesses
    weaknesses: List[Weakness] = field(default_factory=list)
    top_weaknesses: List[Weakness] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # Self-reflection
    self_awareness_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_date": self.diagnosis_date,
            "overall_health": round(self.overall_health, 2),
            "improvement_trajectory": round(self.improvement_trajectory, 2),
            "dimensions": {
                "strategy": round(self.strategy_health, 1),
                "session": round(self.session_health, 1),
                "regime": round(self.regime_health, 1),
                "execution": round(self.execution_health, 1),
                "risk": round(self.risk_health, 1),
                "discipline": round(self.discipline_health, 1),
                "learning": round(self.learning_health, 1),
            },
            "weaknesses": [
                {"dimension": w.dimension, "severity": round(w.severity, 1),
                 "symptom": w.symptom, "cause": w.cause, "fix": w.fix}
                for w in self.weaknesses
            ],
            "top_weaknesses": [
                {"dimension": w.dimension, "severity": round(w.severity, 1),
                 "symptom": w.symptom, "cause": w.cause, "fix": w.fix}
                for w in self.top_weaknesses
            ],
            "recommendations": self.recommendations,
            "action_items": self.action_items,
            "self_awareness_notes": self.self_awareness_notes,
        }


class AISelfDiagnosis:
    """Weekly AI self-diagnosis system."""

    def __init__(self,
                 min_trades_for_diagnosis: int = 20,
                 comparison_window_days: int = 7):
        """Initialize self-diagnosis.

        Args:
            min_trades_for_diagnosis: minimum trades to run diagnosis
            comparison_window_days: compare this week vs last N days
        """
        self.min_trades = min_trades_for_diagnosis
        self.window = comparison_window_days
        self._lock = threading.RLock()
        self._trades: Deque[dict] = deque(maxlen=2000)
        self._last_diagnosis: Optional[DiagnosisReport] = None
        self._diagnosis_history: List[DiagnosisReport] = []

    # ------------------------------------------------------------------
    # Record trades
    # ------------------------------------------------------------------
    def record_trade(self, strategy: str, session: str, regime: str,
                     pnl: float, r_multiple: float, confidence: float,
                     slippage_bps: float = 0, hold_time_s: float = 0,
                     setup: str = "", symbol: str = "",
                     followed_rules: bool = True) -> None:
        """Record a trade for diagnosis."""
        with self._lock:
            self._trades.append({
                "timestamp": time.time(),
                "strategy": strategy, "session": session, "regime": regime,
                "setup": setup, "symbol": symbol,
                "pnl": pnl, "r_multiple": r_multiple,
                "confidence": confidence, "slippage_bps": slippage_bps,
                "hold_time_s": hold_time_s,
                "win": pnl > 0,
                "followed_rules": followed_rules,
            })

    # ------------------------------------------------------------------
    # Run diagnosis
    # ------------------------------------------------------------------
    def diagnose(self) -> DiagnosisReport:
        """Run full self-diagnosis."""
        report = DiagnosisReport(
            diagnosis_date=datetime.now(tz=timezone.utc).isoformat(),
        )

        with self._lock:
            trades = list(self._trades)

        if len(trades) < self.min_trades:
            report.self_awareness_notes = (
                f"Insufficient data for diagnosis ({len(trades)}/{self.min_trades} trades). "
                f"Continue collecting data."
            )
            return report

        # Split into this period vs previous
        now = time.time()
        period_ago = now - self.window * 86400
        prev_ago = now - 2 * self.window * 86400
        this_period = [t for t in trades if t["timestamp"] >= period_ago]
        prev_period = [t for t in trades if prev_ago <= t["timestamp"] < period_ago]

        if not this_period:
            report.self_awareness_notes = "No trades in current period."
            return report

        # === Diagnose each dimension ===
        report.strategy_health, strat_weak = self._diagnose_strategies(this_period)
        report.session_health, sess_weak = self._diagnose_sessions(this_period)
        report.regime_health, regime_weak = self._diagnose_regimes(this_period)
        report.execution_health, exec_weak = self._diagnose_execution(this_period)
        report.risk_health, risk_weak = self._diagnose_risk(this_period)
        report.discipline_health, disc_weak = self._diagnose_discipline(this_period)
        report.learning_health, learn_weak = self._diagnose_learning(this_period, prev_period)

        # Collect all weaknesses
        all_weaknesses = strat_weak + sess_weak + regime_weak + exec_weak + \
                        risk_weak + disc_weak + learn_weak
        report.weaknesses = all_weaknesses
        # Top 3 by severity
        report.top_weaknesses = sorted(all_weaknesses, key=lambda w: w.severity, reverse=True)[:3]

        # Overall health
        dimensions = [
            report.strategy_health, report.session_health, report.regime_health,
            report.execution_health, report.risk_health, report.discipline_health,
            report.learning_health,
        ]
        report.overall_health = float(np.mean(dimensions))

        # Improvement trajectory
        if self._last_diagnosis:
            report.improvement_trajectory = report.overall_health - self._last_diagnosis.overall_health

        # Recommendations + action items
        report.recommendations = self._recommend(report)
        report.action_items = self._action_items(report)

        # Self-awareness notes
        report.self_awareness_notes = self._self_reflect(report)

        # Save
        with self._lock:
            self._last_diagnosis = report
            self._diagnosis_history.append(report)
            if len(self._diagnosis_history) > 52:  # keep 1 year
                self._diagnosis_history = self._diagnosis_history[-52:]

        return report

    # ------------------------------------------------------------------
    # Dimension diagnosticians
    # ------------------------------------------------------------------
    def _diagnose_strategies(self, trades: list) -> Tuple[float, List[Weakness]]:
        """Diagnose per-strategy performance."""
        by_strategy: Dict[str, list] = defaultdict(list)
        for t in trades:
            by_strategy[t["strategy"]].append(t["r_multiple"])

        weaknesses = []
        scores = []
        for strat, rs in by_strategy.items():
            avg_r = float(np.mean(rs))
            win_rate = sum(1 for r in rs if r > 0) / max(len(rs), 1)
            # Score: 10 = excellent, 0 = terrible
            if avg_r > 0.5:
                score = 9
            elif avg_r > 0.2:
                score = 7
            elif avg_r > 0:
                score = 6
            elif avg_r > -0.2:
                score = 4
            else:
                score = 2
                # Weakness
                weaknesses.append(Weakness(
                    dimension="strategy", severity=8,
                    symptom=f"Strategy '{strat}' EV={avg_r:.2f}R, WR={win_rate:.0%}",
                    cause="Strategy edge has decayed or market has changed",
                    fix=f"Pause '{strat}' and retrain with recent data",
                    evidence={"avg_r": avg_r, "win_rate": win_rate, "trades": len(rs)},
                ))
            scores.append(score)
        return float(np.mean(scores)) if scores else 5.0, weaknesses

    def _diagnose_sessions(self, trades: list) -> Tuple[float, List[Weakness]]:
        """Diagnose per-session performance."""
        by_session: Dict[str, list] = defaultdict(list)
        for t in trades:
            by_session[t["session"]].append(t["r_multiple"])

        weaknesses = []
        scores = []
        for sess, rs in by_session.items():
            avg_r = float(np.mean(rs))
            if avg_r > 0.3:
                score = 8
            elif avg_r > 0:
                score = 6
            elif avg_r > -0.2:
                score = 4
                weaknesses.append(Weakness(
                    dimension="session", severity=6,
                    symptom=f"Session '{sess}' EV={avg_r:.2f}R",
                    cause="Poor liquidity or wrong strategy for this session",
                    fix=f"Reduce size 50% in {sess} or skip entirely",
                ))
            else:
                score = 2
                weaknesses.append(Weakness(
                    dimension="session", severity=8,
                    symptom=f"Session '{sess}' EV={avg_r:.2f}R (losing)",
                    cause="This session doesn't suit our strategies",
                    fix=f"Stop trading in {sess} session",
                ))
            scores.append(score)
        return float(np.mean(scores)) if scores else 5.0, weaknesses

    def _diagnose_regimes(self, trades: list) -> Tuple[float, List[Weakness]]:
        """Diagnose per-regime performance."""
        by_regime: Dict[str, list] = defaultdict(list)
        for t in trades:
            by_regime[t["regime"]].append(t["r_multiple"])

        weaknesses = []
        scores = []
        for regime, rs in by_regime.items():
            avg_r = float(np.mean(rs))
            if avg_r > 0.3:
                score = 8
            elif avg_r > 0:
                score = 6
            elif avg_r > -0.2:
                score = 4
            else:
                score = 2
                weaknesses.append(Weakness(
                    dimension="regime", severity=7,
                    symptom=f"Regime '{regime}' EV={avg_r:.2f}R",
                    cause="Strategy not suited to this market regime",
                    fix=f"Skip trades in {regime} regime or switch strategy",
                ))
            scores.append(score)
        return float(np.mean(scores)) if scores else 5.0, weaknesses

    def _diagnose_execution(self, trades: list) -> Tuple[float, List[Weakness]]:
        """Diagnose execution quality."""
        slippages = [t["slippage_bps"] for t in trades if t["slippage_bps"] > 0]
        avg_slip = float(np.mean(slippages)) if slippages else 0.0

        if avg_slip < 1:
            score = 9
        elif avg_slip < 2:
            score = 8
        elif avg_slip < 5:
            score = 6
        elif avg_slip < 10:
            score = 4
            weaknesses = [Weakness(
                dimension="execution", severity=5,
                symptom=f"High slippage: {avg_slip:.1f}bps average",
                cause="Poor liquidity or wrong order type",
                fix="Use limit orders instead of market orders",
            )]
        else:
            score = 2
            weaknesses = [Weakness(
                dimension="execution", severity=8,
                symptom=f"Excessive slippage: {avg_slip:.1f}bps",
                cause="Very poor execution environment",
                fix="Switch broker or stop trading in low-liquidity hours",
            )]

        return score, weaknesses if avg_slip >= 5 else []

    def _diagnose_risk(self, trades: list) -> Tuple[float, List[Weakness]]:
        """Diagnose risk management."""
        rs = [t["r_multiple"] for t in trades]
        max_loss = min(rs) if rs else 0
        cumulative = np.cumsum(rs)
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        if max_dd < 2:
            score = 9
        elif max_dd < 4:
            score = 7
        elif max_dd < 6:
            score = 5
            weaknesses = [Weakness(
                dimension="risk", severity=6,
                symptom=f"Max drawdown {max_dd:.1f}R",
                cause="Risk per trade too high or consecutive losses",
                fix="Reduce position size by 30%",
            )]
        else:
            score = 2
            weaknesses = [Weakness(
                dimension="risk", severity=9,
                symptom=f"Excessive drawdown {max_dd:.1f}R",
                cause="Poor risk management or strategy failure",
                fix="Halve position size immediately, review all strategies",
            )]
        return score, weaknesses if max_dd >= 4 else []

    def _diagnose_discipline(self, trades: list) -> Tuple[float, List[Weakness]]:
        """Diagnose rule-following discipline."""
        violations = sum(1 for t in trades if not t["followed_rules"])
        violation_rate = violations / max(len(trades), 1)

        if violation_rate < 0.05:
            score = 9
        elif violation_rate < 0.10:
            score = 7
        elif violation_rate < 0.20:
            score = 5
            weaknesses = [Weakness(
                dimension="discipline", severity=6,
                symptom=f"Rule violation rate {violation_rate:.0%}",
                cause="Strategy not enforcing rules strictly",
                fix="Tighten rule enforcement, add hard blocks",
            )]
        else:
            score = 2
            weaknesses = [Weakness(
                dimension="discipline", severity=9,
                symptom=f"High rule violation rate {violation_rate:.0%}",
                cause="Discipline breakdown",
                fix="Stop trading until discipline restored",
            )]
        return score, weaknesses if violation_rate >= 0.10 else []

    def _diagnose_learning(self, this_period: list,
                           prev_period: list) -> Tuple[float, List[Weakness]]:
        """Diagnose if we're improving over time."""
        if not prev_period:
            return 5.0, []

        this_ev = float(np.mean([t["r_multiple"] for t in this_period]))
        prev_ev = float(np.mean([t["r_multiple"] for t in prev_period]))
        improvement = this_ev - prev_ev

        if improvement > 0.2:
            score = 9
        elif improvement > 0:
            score = 7
        elif improvement > -0.1:
            score = 5
        else:
            score = 3
            weaknesses = [Weakness(
                dimension="learning", severity=7,
                symptom=f"Performance declining ({improvement:+.2f}R)",
                cause="Strategy not adapting to market changes",
                fix="Review what changed, retrain models",
            )]
        return score, weaknesses if improvement < -0.1 else []

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------
    def _recommend(self, report: DiagnosisReport) -> List[str]:
        """Generate recommendations from weaknesses."""
        recs = []
        for w in report.top_weaknesses:
            recs.append(f"[{w.dimension}] {w.symptom} → {w.fix}")
        if report.improvement_trajectory > 0:
            recs.append(f"Improving (+{report.improvement_trajectory:.1f}) — maintain approach")
        elif report.improvement_trajectory < 0:
            recs.append(f"Declining ({report.improvement_trajectory:.1f}) — review urgently")
        return recs

    def _action_items(self, report: DiagnosisReport) -> List[str]:
        """Generate concrete action items."""
        items = []
        for w in report.top_weaknesses:
            if w.severity >= 7:
                items.append(f"URGENT: {w.fix}")
            elif w.severity >= 5:
                items.append(f"This week: {w.fix}")
        return items

    def _self_reflect(self, report: DiagnosisReport) -> str:
        """Generate self-awareness reflection."""
        if report.overall_health >= 8:
            return (f"System is healthy ({report.overall_health:.1f}/10). "
                   f"Maintain current approach, keep learning.")
        elif report.overall_health >= 6:
            return (f"System is acceptable ({report.overall_health:.1f}/10) but has "
                   f"{len(report.top_weaknesses)} weaknesses to address.")
        elif report.overall_health >= 4:
            return (f"System is struggling ({report.overall_health:.1f}/10). "
                   f"Reduce risk, focus on top weaknesses.")
        else:
            return (f"System is in critical condition ({report.overall_health:.1f}/10). "
                   f"Halt new strategies, emergency review needed.")

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def history(self) -> List[DiagnosisReport]:
        """Get diagnosis history."""
        with self._lock:
            return list(self._diagnosis_history)
