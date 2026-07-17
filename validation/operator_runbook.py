"""validation.operator_runbook
=====================================================================
Day 121-125 — Operator incident response runbook.

A documented set of procedures for what to do when things go wrong.
This isn't code — it's structured procedural knowledge that an
operator can follow under stress without thinking.

Each runbook entry has:
  - Trigger           : what condition activates this procedure
  - Severity          : INFO / WARN / CRITICAL
  - Immediate actions : what to do in the first 60 seconds
  - Investigation     : how to diagnose root cause
  - Resolution        : how to fix
  - Escalation        : when and who to escalate to
  - Prevention        : how to prevent recurrence

The runbook is generated as a structured dict so it can be rendered
to HTML, PDF, or markdown by the dashboard layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("validation.runbook")


@dataclass
class RunbookEntry:
    """One incident response procedure."""
    trigger: str
    severity: str             # INFO / WARN / CRITICAL
    immediate_actions: list[str] = field(default_factory=list)
    investigation_steps: list[str] = field(default_factory=list)
    resolution_steps: list[str] = field(default_factory=list)
    escalation: str = ""
    prevention: str = ""
    category: str = ""        # execution / risk / data / system / market

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
class OperatorRunbook:
    """Institutional incident response runbook."""

    def __init__(self) -> None:
        self.entries: dict[str, RunbookEntry] = {}
        self._build_default_runbook()

    # ----------------------------------------------------------------
    def get(self, trigger: str) -> Optional[RunbookEntry]:
        return self.entries.get(trigger)

    def add(self, entry: RunbookEntry) -> None:
        self.entries[entry.trigger] = entry

    def all_entries(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries.values()]

    def for_severity(self, severity: str) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries.values()
                if e.severity == severity]

    # ----------------------------------------------------------------
    @property
    def ready(self) -> bool:
        """Runbook is 'ready' when it has at least the 8 mandatory entries."""
        mandatory = [
            "kill_switch_activated",
            "max_daily_loss_breach",
            "max_drawdown_breach",
            "mt5_disconnect",
            "anomaly_detected",
            "execution_slippage_spike",
            "strategy_decay_detected",
            "correlation_collapse",
        ]
        return all(m in self.entries for m in mandatory)

    # ----------------------------------------------------------------
    def _build_default_runbook(self) -> None:
        self.add(RunbookEntry(
            trigger="kill_switch_activated",
            severity="CRITICAL",
            category="system",
            immediate_actions=[
                "DO NOT disarm the kill switch until root cause is understood",
                "Verify all open positions are closed (check MT5 terminal)",
                "Check data/decision_traces.jsonl for the last 10 decisions",
                "Record the timestamp and trigger reason in the incident log",
            ],
            investigation_steps=[
                "Check logs/system.log for the triggering event",
                "Identify which guardrail fired (daily_loss / drawdown / anomaly)",
                "Check if the trigger was a false positive (data feed issue?)",
                "Verify broker state matches bot state (no orphan positions)",
            ],
            resolution_steps=[
                "If false positive: disarm with `python main_v3.py --disarm`",
                "If real: keep kill switch armed, review positions manually",
                "If real and positions are open: close them via MT5 terminal",
                "Document the incident before resuming",
            ],
            escalation="If position state is unclear or orphaned, escalate to broker support immediately",
            prevention="Add the trigger condition to guardrail config with tighter thresholds",
        ))

        self.add(RunbookEntry(
            trigger="max_daily_loss_breach",
            severity="CRITICAL",
            category="risk",
            immediate_actions=[
                "Verify the loss is real (check MT5 account balance)",
                "Check if any positions are still open",
                "DO NOT resume trading for the rest of the day",
                "Record the actual loss vs configured limit",
            ],
            investigation_steps=[
                "Identify which trades caused the loss (check trades.log)",
                "Check if risk engine approved the losing trades correctly",
                "Verify position sizing was within risk_per_trade limits",
                "Check market conditions — was there a news event?",
            ],
            resolution_steps=[
                "Close all open positions manually if bot hasn't",
                "Wait until next trading day before resuming",
                "Review risk parameters with risk manager",
                "Consider reducing risk_per_trade if losses are recurring",
            ],
            escalation="If loss exceeds 2x the daily limit, halt trading for the week",
            prevention="Tighten max_daily_loss or reduce max_open_trades",
        ))

        self.add(RunbookEntry(
            trigger="max_drawdown_breach",
            severity="CRITICAL",
            category="risk",
            immediate_actions=[
                "Halt all trading immediately",
                "Verify drawdown calculation (is it real or a data issue?)",
                "Check equity curve in data/metrics.jsonl",
                "Document the drawdown magnitude and duration",
            ],
            investigation_steps=[
                "Identify which strategies contributed most to the drawdown",
                "Check if drawdown is concentrated in one symbol or spread",
                "Verify correlation assumptions held during the drawdown",
                "Check if regime shifted (data/dashboard.html)",
            ],
            resolution_steps=[
                "Reduce capital allocation via `python main_v3.py --promote MICRO`",
                "Auto-retire strategies with decay_score < 0.4",
                "Consider pausing trading for 24-48 hours",
                "Re-review strategy pool before resuming",
            ],
            escalation="If drawdown exceeds 20%, halt trading indefinitely pending review",
            prevention="Tighten max_drawdown_pct in risk.v2 config",
        ))

        self.add(RunbookEntry(
            trigger="mt5_disconnect",
            severity="WARN",
            category="system",
            immediate_actions=[
                "Check MT5 terminal is running on the host machine",
                "Verify network connectivity to broker",
                "Check logs/system.log for the disconnect timestamp",
                "DO NOT place new orders until connection is restored",
            ],
            investigation_steps=[
                "Verify MT5 terminal login is still valid",
                "Check broker server status (broker website / status page)",
                "Review reconnect attempts in the log",
                "Identify any positions that were open when disconnect happened",
            ],
            resolution_steps=[
                "Restart MT5 terminal if needed",
                "Restart the bot — it will reconcile state via recovery.py",
                "Verify all open positions are accounted for",
                "If positions are missing, contact broker immediately",
            ],
            escalation="If disconnect lasts > 5 minutes with open positions, call broker",
            prevention="Improve reconnect_attempts and reconnect_delay_s in mt5 config",
        ))

        self.add(RunbookEntry(
            trigger="anomaly_detected",
            severity="WARN",
            category="system",
            immediate_actions=[
                "Check what anomaly was detected (data feed / latency / price gap)",
                "Verify market data is sane (compare to external source)",
                "If data is corrupted: halt trading until feed is restored",
                "Document the anomaly type and timestamp",
            ],
            investigation_steps=[
                "Check system_mon.health() for latency / uptime issues",
                "Check risk_mon.health() for VaR / correlation spikes",
                "Check alpha_mon.health() for feature drift",
                "Identify whether anomaly is data-side or strategy-side",
            ],
            resolution_steps=[
                "If data-side: pause trading, wait for clean data",
                "If strategy-side: auto-retire the offending strategy",
                "If system-side: restart the bot after fix",
                "Run kill switch validator before resuming",
            ],
            escalation="If anomaly persists > 15 minutes, treat as kill switch scenario",
            prevention="Tighten anomaly_kill_switch_score in guardrails config",
        ))

        self.add(RunbookEntry(
            trigger="execution_slippage_spike",
            severity="WARN",
            category="execution",
            immediate_actions=[
                "Check execution_metrics.jsonl for recent fill prices",
                "Compare actual slippage to slippage model prediction",
                "If slippage > 2x prediction: halt new orders",
                "Document the slippage distribution",
            ],
            investigation_steps=[
                "Identify which orders had high slippage",
                "Check market conditions at fill time (volatility / spread)",
                "Verify order size relative to ADV (was it too large?)",
                "Check if multiple strategies traded the same symbol simultaneously",
            ],
            resolution_steps=[
                "Reduce order size via capital scaler",
                "Enable more aggressive order slicing",
                "If persistent: disable the offending strategy",
                "Update slippage model with new data",
            ],
            escalation="If slippage > 25 bps consistently, halt the strategy",
            prevention="Reduce max_lot_per_trade and increase slicing",
        ))

        self.add(RunbookEntry(
            trigger="strategy_decay_detected",
            severity="WARN",
            category="alpha",
            immediate_actions=[
                "Identify which strategy has decayed (check decay_detector)",
                "Check decay_score — if < 0.4, auto-retire should fire",
                "Reduce that strategy's allocation weight to 0",
                "Document the decay pattern (sudden vs gradual)",
            ],
            investigation_steps=[
                "Compare recent Sharpe to baseline Sharpe",
                "Check if regime shifted (alpha_monitor.regime_mismatch)",
                "Check feature drift (alpha_monitor.drifted_features)",
                "Identify when decay started (decision traces)",
            ],
            resolution_steps=[
                "If gradual: auto-retire the strategy",
                "If sudden: investigate regime change, may be temporary",
                "Consider retraining ML filter with recent data",
                "Run survival_test on the strategy before re-enabling",
            ],
            escalation="If 2+ strategies decay simultaneously, halt trading — likely regime shift",
            prevention="Tighten retirement_threshold in decay_detector config",
        ))

        self.add(RunbookEntry(
            trigger="correlation_collapse",
            severity="CRITICAL",
            category="risk",
            immediate_actions=[
                "Halt all new orders immediately",
                "Check risk_mon.health() for correlation spike",
                "Verify open positions — they may all be long or all short",
                "Document which symbols suddenly correlated",
            ],
            investigation_steps=[
                "Check correlation_matrix in portfolio manager",
                "Identify the market event (flash crash? news?)",
                "Verify exposure_model — is net exposure concentrated?",
                "Check if guardrails.max_correlated_exposure fired",
            ],
            resolution_steps=[
                "Close the most correlated positions first",
                "Reduce net exposure to < 0.3 until correlations normalise",
                "Consider manual override of capital tier to MICRO",
                "Re-evaluate portfolio allocation once correlations subside",
            ],
            escalation="If correlation > 0.9 across 3+ symbols, treat as black swan",
            prevention="Tighten max_correlated_exposure in guardrails config",
        ))

    # ----------------------------------------------------------------
    def to_markdown(self) -> str:
        """Render the runbook as markdown for documentation."""
        lines = ["# Operator Incident Response Runbook\n"]
        for entry in self.entries.values():
            lines.append(f"## {entry.trigger}")
            lines.append(f"\n**Severity:** {entry.severity}  ")
            lines.append(f"**Category:** {entry.category}\n")
            lines.append("### Immediate Actions (first 60 seconds)")
            for i, a in enumerate(entry.immediate_actions, 1):
                lines.append(f"{i}. {a}")
            lines.append("\n### Investigation")
            for i, a in enumerate(entry.investigation_steps, 1):
                lines.append(f"{i}. {a}")
            lines.append("\n### Resolution")
            for i, a in enumerate(entry.resolution_steps, 1):
                lines.append(f"{i}. {a}")
            lines.append(f"\n### Escalation\n{entry.escalation}\n")
            lines.append(f"### Prevention\n{entry.prevention}\n")
            lines.append("---\n")
        return "\n".join(lines)
