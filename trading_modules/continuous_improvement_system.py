"""trading_modules/continuous_improvement_system.py
=====================================================================
Continuous Improvement System (Principle #156, #158, #160)
=====================================================================
The master learning loop that ties everything together:
    Trade → Feedback → Database → Analysis → Retraining → Better Policy

Three Improvement Cycles:
    1. DAILY    — quick stats review, parameter nudges
    2. WEEKLY   — full audit (uses WeeklySelfAuditor), strategy review
    3. MONTHLY  — model retraining, parameter optimization, benchmarking

What It Tracks:
    - Performance trajectory (are we getting better over time?)
    - Parameter drift (have optimal parameters changed?)
    - Strategy lifecycle (which strategies are gaining/losing edge?)
    - Feature importance (which features predict wins now?)
    - Benchmark comparison (are we beating buy-and-hold?)

What It Outputs:
    - Daily improvement report
    - Weekly audit summary
    - Monthly retraining recommendations
    - Parameter adjustment suggestions
    - Feature engineering recommendations

Usage:
    cis = ContinuousImprovementSystem()

    # Record each trade
    cis.record_trade(strategy="momentum", pnl=42, r=1.8, ...)

    # Daily review
    daily = cis.daily_review()
    # daily = {"improvement_score": 0.05, "parameter_nudges": {...}}

    # Weekly audit
    weekly = cis.weekly_audit()

    # Monthly review
    monthly = cis.monthly_review()
    # monthly = {"retrain_recommended": True, "strategies_to_retrain": [...]}
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

log = get_logger("trading_bot.continuous_improvement_system")


@dataclass
class ImprovementReport:
    """Improvement report (daily/weekly/monthly)."""
    period: str = "daily"  # daily, weekly, monthly
    period_start: str = ""
    period_end: str = ""

    # Performance trajectory
    performance_trend: str = "stable"  # improving, stable, declining
    improvement_score: float = 0.0     # -1 to +1
    ev_trend: float = 0.0              # EV change over period
    sharpe_trend: float = 0.0          # Sharpe change

    # Parameter drift
    parameter_changes: Dict[str, float] = field(default_factory=dict)
    parameter_nudges: Dict[str, float] = field(default_factory=dict)  # suggested changes

    # Strategy review
    strategies_gaining: List[str] = field(default_factory=list)
    strategies_losing: List[str] = field(default_factory=list)
    strategies_to_retrain: List[str] = field(default_factory=list)

    # Feature importance
    top_features: List[Tuple[str, float]] = field(default_factory=list)

    # Benchmark
    vs_benchmark: float = 0.0  # alpha vs buy-and-hold

    # Recommendations
    retrain_recommended: bool = False
    recommendations: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "performance_trend": self.performance_trend,
            "improvement_score": round(self.improvement_score, 3),
            "ev_trend": round(self.ev_trend, 3),
            "sharpe_trend": round(self.sharpe_trend, 3),
            "parameter_changes": {k: round(v, 4) for k, v in self.parameter_changes.items()},
            "parameter_nudges": {k: round(v, 4) for k, v in self.parameter_nudges.items()},
            "strategies_gaining": self.strategies_gaining,
            "strategies_losing": self.strategies_losing,
            "strategies_to_retrain": self.strategies_to_retrain,
            "top_features": [(f, round(v, 3)) for f, v in self.top_features],
            "vs_benchmark": round(self.vs_benchmark, 3),
            "retrain_recommended": self.retrain_recommended,
            "recommendations": self.recommendations,
            "action_items": self.action_items,
        }


class ContinuousImprovementSystem:
    """Master continuous improvement loop."""

    def __init__(self,
                 benchmark_return_pct: float = 0.02,  # monthly benchmark
                 min_trades_for_review: int = 20,
                 improvement_threshold: float = 0.05):
        """Initialize CIS.

        Args:
            benchmark_return_pct: monthly benchmark return (e.g., 2% = buy & hold)
            min_trades_for_review: min trades before generating report
            improvement_threshold: improvement > this = "improving"
        """
        self.benchmark = benchmark_return_pct
        self.min_trades = min_trades_for_review
        self.threshold = improvement_threshold

        self._lock = threading.RLock()
        self._trades: Deque[dict] = deque(maxlen=2000)
        self._parameters: Dict[str, float] = {}
        self._parameter_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._feature_importance: Dict[str, float] = {}
        self._last_daily: Optional[datetime] = None
        self._last_weekly: Optional[datetime] = None
        self._last_monthly: Optional[datetime] = None
        self._reports: List[ImprovementReport] = []

    # ------------------------------------------------------------------
    # Record trade
    # ------------------------------------------------------------------
    def record_trade(self, strategy: str, symbol: str, pnl: float,
                     r_multiple: float, confidence: float,
                     features: Optional[Dict[str, float]] = None,
                     win: Optional[bool] = None) -> None:
        """Record a completed trade for improvement analysis."""
        if win is None:
            win = pnl > 0
        trade = {
            "timestamp": time.time(),
            "strategy": strategy, "symbol": symbol,
            "pnl": pnl, "r_multiple": r_multiple,
            "confidence": confidence,
            "features": features or {},
            "win": win,
        }
        with self._lock:
            self._trades.append(trade)
            # Update feature importance (correlation with win)
            if features:
                self._update_feature_importance(features, win)

    def _update_feature_importance(self, features: Dict[str, float], win: bool) -> None:
        """Update rolling feature importance (correlation with winning)."""
        # Simple: track average feature value for wins vs losses
        for feat, val in features.items():
            if feat not in self._feature_importance:
                self._feature_importance[feat] = 0.0
            # Nudge: + if win & high val, - if loss & high val
            nudge = 0.001 * (1 if win else -1) * (1 if val > 0.5 else -1)
            self._feature_importance[feat] = 0.95 * self._feature_importance[feat] + 0.05 * nudge

    # ------------------------------------------------------------------
    # Set / track parameters
    # ------------------------------------------------------------------
    def set_parameter(self, name: str, value: float) -> None:
        """Set a tunable parameter."""
        with self._lock:
            old = self._parameters.get(name, value)
            self._parameters[name] = value
            self._parameter_history[name].append((time.time(), value))
            if abs(value - old) > 0.001:
                log.info("cis: parameter %s: %.4f → %.4f", name, old, value)

    def get_parameter(self, name: str, default: float = 0.0) -> float:
        return self._parameters.get(name, default)

    # ------------------------------------------------------------------
    # Daily review
    # ------------------------------------------------------------------
    def daily_review(self) -> ImprovementReport:
        """Generate daily improvement report.

        - Quick stats review
        - Parameter nudges (small adjustments)
        - Performance trajectory
        """
        report = ImprovementReport(period="daily")
        now = datetime.now(tz=timezone.utc)
        report.period_end = now.isoformat()
        report.period_start = (now - timedelta(days=1)).isoformat()

        with self._lock:
            # Get today's trades
            day_ago = time.time() - 86400
            today_trades = [t for t in self._trades if t["timestamp"] >= day_ago]
            # Previous day for comparison
            two_days_ago = time.time() - 2 * 86400
            yesterday_trades = [t for t in self._trades
                               if two_days_ago <= t["timestamp"] < day_ago]

        if len(today_trades) < 5:
            report.recommendations.append("Too few trades today for analysis")
            return report

        # === Performance trajectory ===
        today_ev = np.mean([t["r_multiple"] for t in today_trades])
        yest_ev = np.mean([t["r_multiple"] for t in yesterday_trades]) if yesterday_trades else 0
        report.ev_trend = float(today_ev - yest_ev)

        if report.ev_trend > self.threshold:
            report.performance_trend = "improving"
            report.improvement_score = min(1.0, report.ev_trend * 2)
        elif report.ev_trend < -self.threshold:
            report.performance_trend = "declining"
            report.improvement_score = max(-1.0, report.ev_trend * 2)
        else:
            report.performance_trend = "stable"

        # === Parameter nudges ===
        report.parameter_nudges = self._suggest_parameter_nudges(today_trades)

        # === Top features ===
        sorted_feats = sorted(self._feature_importance.items(),
                             key=lambda x: abs(x[1]), reverse=True)
        report.top_features = sorted_feats[:5]

        # === Recommendations ===
        report.recommendations = self._daily_recommendations(report)
        report.action_items = self._daily_action_items(report)

        with self._lock:
            self._last_daily = now
            self._reports.append(report)

        return report

    def _suggest_parameter_nudges(self, trades: list) -> Dict[str, float]:
        """Suggest small parameter adjustments based on today's trades."""
        nudges: Dict[str, float] = {}
        # If win rate is low, suggest raising min_confidence
        win_rate = sum(1 for t in trades if t["win"]) / max(len(trades), 1)
        if win_rate < 0.40:
            current = self.get_parameter("min_confidence", 0.60)
            nudges["min_confidence"] = current + 0.02
        elif win_rate > 0.70:
            current = self.get_parameter("min_confidence", 0.60)
            nudges["min_confidence"] = max(0.40, current - 0.01)

        # If avg R is negative, suggest tightening stops
        avg_r = np.mean([t["r_multiple"] for t in trades])
        if avg_r < 0:
            current = self.get_parameter("sl_atr_multiple", 1.5)
            nudges["sl_atr_multiple"] = max(1.0, current - 0.1)

        return nudges

    def _daily_recommendations(self, report: ImprovementReport) -> List[str]:
        recs = []
        if report.performance_trend == "declining":
            recs.append("Performance declining — review today's losing trades")
        if report.performance_trend == "improving":
            recs.append("Performance improving — maintain current approach")
        for param, val in report.parameter_nudges.items():
            recs.append(f"Consider adjusting {param} → {val:.3f}")
        return recs

    def _daily_action_items(self, report: ImprovementReport) -> List[str]:
        items = []
        if report.performance_trend == "declining":
            items.append("Reduce position size by 25% tomorrow")
        for param, val in report.parameter_nudges.items():
            items.append(f"Apply parameter: {param}={val:.3f}")
        return items

    # ------------------------------------------------------------------
    # Weekly audit
    # ------------------------------------------------------------------
    def weekly_audit(self) -> ImprovementReport:
        """Generate weekly improvement report.

        - Full performance review
        - Strategy gaining/losing identification
        - Feature importance review
        - Benchmark comparison
        """
        report = ImprovementReport(period="weekly")
        now = datetime.now(tz=timezone.utc)
        report.period_end = now.isoformat()
        report.period_start = (now - timedelta(days=7)).isoformat()

        with self._lock:
            week_ago = time.time() - 7 * 86400
            week_trades = [t for t in self._trades if t["timestamp"] >= week_ago]
            prev_week_trades = [t for t in self._trades
                               if time.time() - 14 * 86400 <= t["timestamp"] < week_ago]

        if len(week_trades) < self.min_trades:
            report.recommendations.append("Too few trades this week for audit")
            return report

        # === Performance trajectory ===
        week_ev = np.mean([t["r_multiple"] for t in week_trades])
        prev_ev = np.mean([t["r_multiple"] for t in prev_week_trades]) if prev_week_trades else 0
        report.ev_trend = float(week_ev - prev_ev)

        # Sharpe trend
        week_pnls = [t["pnl"] for t in week_trades]
        prev_pnls = [t["pnl"] for t in prev_week_trades] if prev_week_trades else [0]
        week_sharpe = np.mean(week_pnls) / max(np.std(week_pnls), 1e-10) if week_pnls else 0
        prev_sharpe = np.mean(prev_pnls) / max(np.std(prev_pnls), 1e-10) if prev_pnls else 0
        report.sharpe_trend = float(week_sharpe - prev_sharpe)

        if report.ev_trend > self.threshold:
            report.performance_trend = "improving"
            report.improvement_score = min(1.0, report.ev_trend * 2)
        elif report.ev_trend < -self.threshold:
            report.performance_trend = "declining"
            report.improvement_score = max(-1.0, report.ev_trend * 2)

        # === Strategy review ===
        by_strategy: Dict[str, list] = defaultdict(list)
        for t in week_trades:
            by_strategy[t["strategy"]].append(t["r_multiple"])
        for strat, rs in by_strategy.items():
            avg_r = np.mean(rs)
            if avg_r > 0.3:
                report.strategies_gaining.append(strat)
            elif avg_r < -0.2:
                report.strategies_losing.append(strat)
                # Suggest retrain if losing for 2+ weeks
                if len(prev_week_trades) > 0:
                    prev_strat_rs = [t["r_multiple"] for t in prev_week_trades
                                    if t["strategy"] == strat]
                    if prev_strat_rs and np.mean(prev_strat_rs) < 0:
                        report.strategies_to_retrain.append(strat)

        # === Top features ===
        sorted_feats = sorted(self._feature_importance.items(),
                             key=lambda x: abs(x[1]), reverse=True)
        report.top_features = sorted_feats[:10]

        # === Benchmark ===
        total_pnl = sum(t["pnl"] for t in week_trades)
        benchmark_pnl = self.benchmark / 4 * self._get_equity()  # weekly = monthly/4
        report.vs_benchmark = float(total_pnl - benchmark_pnl)

        # === Recommendations ===
        report.recommendations = self._weekly_recommendations(report)
        report.action_items = self._weekly_action_items(report)
        report.retrain_recommended = len(report.strategies_to_retrain) > 0

        with self._lock:
            self._last_weekly = now
            self._reports.append(report)

        return report

    def _get_equity(self) -> float:
        """Get current equity (placeholder — should be set externally)."""
        return getattr(self, "_equity", 10000.0)

    def set_equity(self, equity: float) -> None:
        self._equity = equity

    def _weekly_recommendations(self, report: ImprovementReport) -> List[str]:
        recs = []
        if report.performance_trend == "improving":
            recs.append(f"Week improved by {report.ev_trend:+.2f}R — maintain approach")
        elif report.performance_trend == "declining":
            recs.append(f"Week declined by {report.ev_trend:+.2f}R — review strategies")
        for strat in report.strategies_gaining:
            recs.append(f"Strategy '{strat}' gaining edge — consider increasing allocation")
        for strat in report.strategies_losing:
            recs.append(f"Strategy '{strat}' losing edge — reduce allocation or pause")
        if report.vs_benchmark < 0:
            recs.append(f"Underperforming benchmark by ${abs(report.vs_benchmark):.0f} — review approach")
        return recs

    def _weekly_action_items(self, report: ImprovementReport) -> List[str]:
        items = []
        for strat in report.strategies_to_retrain:
            items.append(f"RETRAIN strategy '{strat}' — 2 weeks of negative EV")
        if report.vs_benchmark < 0:
            items.append("Review why we're underperforming benchmark")
        return items

    # ------------------------------------------------------------------
    # Monthly review
    # ------------------------------------------------------------------
    def monthly_review(self) -> ImprovementReport:
        """Generate monthly improvement report.

        - Model retraining recommendations
        - Parameter optimization
        - Benchmarking
        """
        report = ImprovementReport(period="monthly")
        now = datetime.now(tz=timezone.utc)
        report.period_end = now.isoformat()
        report.period_start = (now - timedelta(days=30)).isoformat()

        with self._lock:
            month_ago = time.time() - 30 * 86400
            month_trades = [t for t in self._trades if t["timestamp"] >= month_ago]

        if len(month_trades) < 50:
            report.recommendations.append("Too few trades this month for review")
            return report

        # === Performance ===
        month_ev = np.mean([t["r_multiple"] for t in month_trades])
        month_pnls = [t["pnl"] for t in month_trades]
        month_sharpe = np.mean(month_pnls) / max(np.std(month_pnls), 1e-10)

        report.ev_trend = float(month_ev)
        report.sharpe_trend = float(month_sharpe)

        if month_ev > self.threshold:
            report.performance_trend = "improving"
            report.improvement_score = min(1.0, month_ev)
        elif month_ev < -self.threshold:
            report.performance_trend = "declining"
            report.improvement_score = max(-1.0, month_ev)

        # === Strategy review ===
        by_strategy: Dict[str, list] = defaultdict(list)
        for t in month_trades:
            by_strategy[t["strategy"]].append(t["r_multiple"])
        for strat, rs in by_strategy.items():
            avg_r = np.mean(rs)
            if avg_r > 0.3:
                report.strategies_gaining.append(strat)
            elif avg_r < -0.1:
                report.strategies_losing.append(strat)
                report.strategies_to_retrain.append(strat)

        # === Benchmark ===
        total_pnl = sum(t["pnl"] for t in month_trades)
        benchmark_pnl = self.benchmark * self._get_equity() / 100
        report.vs_benchmark = float(total_pnl - benchmark_pnl)

        # === Retrain recommendation ===
        report.retrain_recommended = (
            len(report.strategies_to_retrain) > 0 or
            report.performance_trend == "declining" or
            report.vs_benchmark < 0
        )

        # === Recommendations ===
        report.recommendations = self._monthly_recommendations(report)
        report.action_items = self._monthly_action_items(report)

        with self._lock:
            self._last_monthly = now
            self._reports.append(report)

        return report

    def _monthly_recommendations(self, report: ImprovementReport) -> List[str]:
        recs = []
        if report.retrain_recommended:
            recs.append("RETRAINING RECOMMENDED — schedule model retraining")
        for strat in report.strategies_to_retrain:
            recs.append(f"Retrain strategy '{strat}' — edge decayed over month")
        if report.vs_benchmark > 0:
            recs.append(f"Outperforming benchmark by ${report.vs_benchmark:.0f} — maintain approach")
        else:
            recs.append(f"Underperforming benchmark — major review needed")
        return recs

    def _monthly_action_items(self, report: ImprovementReport) -> List[str]:
        items = []
        if report.retrain_recommended:
            items.append("Schedule model retraining session")
        for strat in report.strategies_to_retrain:
            items.append(f"Retrain '{strat}' with latest data")
        items.append("Update parameter optimization")
        items.append("Benchmark vs buy-and-hold")
        return items

    # ------------------------------------------------------------------
    # History + status
    # ------------------------------------------------------------------
    def history(self) -> List[ImprovementReport]:
        """Get all past reports."""
        with self._lock:
            return list(self._reports)

    def status(self) -> Dict[str, Any]:
        """Get current status."""
        with self._lock:
            return {
                "total_trades_recorded": len(self._trades),
                "parameters_tracked": len(self._parameters),
                "features_tracked": len(self._feature_importance),
                "reports_generated": len(self._reports),
                "last_daily": self._last_daily.isoformat() if self._last_daily else None,
                "last_weekly": self._last_weekly.isoformat() if self._last_weekly else None,
                "last_monthly": self._last_monthly.isoformat() if self._last_monthly else None,
            }
