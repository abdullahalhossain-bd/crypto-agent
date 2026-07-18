"""agents.agent_graph
=====================================================================
Multi-Agent Trading Graph — orchestrates the full workflow.

Inspired by TradingAgents' LangGraph setup, but implemented as a
lightweight pure-Python graph (no LangGraph/LangChain dependency).

Flow:
    1. Analyst Team runs in parallel:
       fundamentals + news + sentiment + technical → 4 reports
    2. Bull/Bear Debate:
       Researchers read reports → N rounds of debate → debate history
    3. Research Manager:
       Synthesizes debate → ResearchPlan
    4. Trader:
       Turns plan into TraderProposal (action, lots, SL, TP)
    5. Risk Debate:
       Aggressive/Conservative/Neutral debate the proposal
    6. Portfolio Manager:
       Final decision: approve/reject + adjusted lots
    7. Memory Log:
       Store decision for future reflection

The graph is the single entry point. Call:
    graph = MultiAgentTradingGraph()
    result = graph.propagate(symbol="BTCUSD", df=ohlcv_df)

FIXES (Batch 2 audit):
  - C1/X1: exposed `resolve_trade_outcome()` on the graph so the main
    trading loop can call it after a trade closes, completing the
    memory-reflection learning loop.
  - C6/X4: `propagate()` now accepts an optional `portfolio_state`
    (equity, open positions, exposure) which is forwarded to the
    Portfolio Manager so decisions aren't made in a vacuum.
  - C9: each agent call is wrapped in a hard timeout (default 60s,
    configurable via `agent_timeout_s`). A hung LLM provider can no
    longer freeze the graph indefinitely.
  - C14: top-level try/except around every agent stage. A failure in
    one stage logs the error and produces a safe fallback instead of
    crashing the whole graph.
  - C18: memory log rotation is handled by `TradingMemoryLog.max_entries`
    (enforced inside memory_log.py).
  - C20: `TradingGraphResult` now carries `token_usage` and
    `agent_latencies` dicts so callers can observe per-agent cost.
  - H1: analysts now run in parallel via `concurrent.futures.ThreadPoolExecutor`.
  - H8: `propagate()` validates `df` up-front (not None, not empty,
    has a 'close' column) and returns a safe HOLD result if invalid.
  - H15: `max_lot` and `agent_timeout_s` can be read from `config.agents`
    if a config dict is supplied to the constructor.
  - H20: if a `Database` instance is supplied, the final decision is
    logged to the `decisions` table as a structured audit trail.
  - M8: the `context` parameter is now explicitly forwarded to
    `analyst_team.run()` (it was already passed, but this is now
    documented and also enriched with external data if available).
  - X2: `propagate()` accepts an optional `context_fetchers` list of
    callables that populate news/macro/sentiment data into the context
    dict before analysts run, so analysts aren't operating on empty
    context.
"""
from __future__ import annotations

import concurrent.futures
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from agents.analysts import AnalystTeam, AnalystReport
from agents.researchers import BullBearDebate, DebateResult
from agents.research_manager import ResearchManager
from agents.risk_debators import RiskDebate, RiskDebateResult
from agents.trader import Trader
from agents.portfolio_manager import PortfolioManager
from agents.memory_log import TradingMemoryLog
from agents.schemas import ResearchPlan, TraderProposal, PortfolioDecision, PortfolioRating, TraderAction
from external.llm_provider import LLMProvider
from utils.logger import get_logger

log = get_logger("agents.graph")

_DEFAULT_AGENT_TIMEOUT_S = 60.0
_DEFAULT_MAX_WORKERS = 4


@dataclass
class TradingGraphResult:
    """Full output of the multi-agent graph."""
    symbol: str
    trade_date: str
    analyst_reports: dict[str, Any] = field(default_factory=dict)
    debate: Optional[DebateResult] = None
    research_plan: Optional[ResearchPlan] = None
    trader_proposal: Optional[TraderProposal] = None
    risk_debate: Optional[RiskDebateResult] = None
    final_decision: Optional[PortfolioDecision] = None
    timestamp: str = ""
    # C20 fix: token + latency observability.
    token_usage: dict[str, int] = field(default_factory=dict)
    agent_latencies: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trade_date": self.trade_date,
            "analyst_reports": {k: v.to_dict() if hasattr(v, "to_dict") else v
                                 for k, v in self.analyst_reports.items()},
            "debate": self.debate.to_dict() if self.debate else None,
            "research_plan": self.research_plan.to_dict() if self.research_plan else None,
            "trader_proposal": self.trader_proposal.to_dict() if self.trader_proposal else None,
            "risk_debate": self.risk_debate.to_dict() if self.risk_debate else None,
            "final_decision": self.final_decision.to_dict() if self.final_decision else None,
            "timestamp": self.timestamp,
            "token_usage": dict(self.token_usage),
            "agent_latencies": dict(self.agent_latencies),
            "errors": list(self.errors),
        }


# ----------------------------------------------------------------------
def _run_with_timeout(func: Callable, timeout_s: float,
                       label: str, *args, **kwargs) -> Any:
    """Run `func(*args, **kwargs)` in a worker thread with a hard timeout.

    C9 fix: a hung LLM provider can no longer freeze the graph. If the
    worker is still alive after `timeout_s`, we log, return None, and
    let the caller produce a safe fallback.

    M7 FIX (Chief AI Architect Audit): properly shut down the executor
    on all paths (success, timeout, exception) to prevent thread pool
    leaks. The previous code created a new ThreadPoolExecutor per call
    and could leak worker threads on timeout.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(func, *args, **kwargs)
    try:
        result = fut.result(timeout=timeout_s)
        executor.shutdown(wait=False, cancel_futures=True)
        return result
    except concurrent.futures.TimeoutError:
        log.error("Graph stage %s timed out after %.1fs — abandoning", label, timeout_s)
        executor.shutdown(wait=False, cancel_futures=True)
        return None
    except Exception as e:
        log.error("Graph stage %s raised: %r", label, e)
        executor.shutdown(wait=False, cancel_futures=True)
        return None


def _safe_call(stage_name: str, func: Callable, timeout_s: float,
                result: TradingGraphResult, *args, **kwargs) -> Any:
    """C14 fix: wrap an agent call in try/except + timeout. On failure,
    log the error into `result.errors` and return None so the caller
    can produce a safe fallback.
    """
    t0 = time.monotonic()
    try:
        out = _run_with_timeout(func, timeout_s, stage_name, *args, **kwargs)
        elapsed = time.monotonic() - t0
        result.agent_latencies[stage_name] = round(elapsed, 3)
        return out
    except Exception as e:  # noqa: BLE001 — top-level graph guard
        elapsed = time.monotonic() - t0
        result.agent_latencies[stage_name] = round(elapsed, 3)
        err_msg = f"{stage_name}: {e!r}"
        result.errors.append(err_msg)
        log.error("Graph stage %s failed: %r\n%s", stage_name, e, traceback.format_exc())
        return None


# ----------------------------------------------------------------------
class MultiAgentTradingGraph:
    """Main orchestrator — runs the full multi-agent workflow."""

    def __init__(
        self,
        llm: Optional[LLMProvider] = None,
        selected_analysts: Optional[list[str]] = None,
        max_debate_rounds: int = 1,
        max_risk_rounds: int = 1,
        memory_log_path: str = "data/agent_memory_log.jsonl",
        max_lot: float = 0.1,
        agent_timeout_s: float = _DEFAULT_AGENT_TIMEOUT_S,
        max_workers: int = _DEFAULT_MAX_WORKERS,
        config: Optional[dict] = None,
        database: Any = None,  # Database instance for audit logging (H20)
    ) -> None:
        # H15 fix: read max_lot and agent_timeout_s from config if provided.
        if config:
            agents_cfg = config.get("agents", {})
            if "max_lot" in agents_cfg:
                max_lot = float(agents_cfg["max_lot"])
            if "agent_timeout_s" in agents_cfg:
                agent_timeout_s = float(agents_cfg["agent_timeout_s"])
            if "max_workers" in agents_cfg:
                max_workers = int(agents_cfg["max_workers"])
            if "max_debate_rounds" in agents_cfg:
                max_debate_rounds = int(agents_cfg["max_debate_rounds"])
            if "max_risk_rounds" in agents_cfg:
                max_risk_rounds = int(agents_cfg["max_risk_rounds"])

        self.llm = llm or LLMProvider()
        self.selected_analysts = selected_analysts or [
            "fundamentals", "news", "sentiment", "technical",
        ]
        self.max_debate_rounds = int(max_debate_rounds)
        self.max_risk_rounds = int(max_risk_rounds)
        self.max_lot = float(max_lot)
        self.agent_timeout_s = float(agent_timeout_s)
        self.max_workers = max(1, int(max_workers))
        self.database = database  # optional Database for audit logging (H20)
        # Components
        self.analyst_team = AnalystTeam(self.llm, self.selected_analysts)
        self.bull_bear = BullBearDebate(self.llm, self.max_debate_rounds)
        self.research_manager = ResearchManager(self.llm)
        self.trader = Trader(self.llm)
        self.risk_debate = RiskDebate(self.llm, self.max_risk_rounds)
        self.portfolio_manager = PortfolioManager(self.llm)
        self.memory_log = TradingMemoryLog(memory_log_path, llm=self.llm)

    # ----------------------------------------------------------------
    def propagate(
        self,
        symbol: str,
        df: pd.DataFrame,
        context: Optional[dict] = None,
        trade_date: Optional[str] = None,
        portfolio_state: Optional[dict] = None,
        context_fetchers: Optional[list[Callable[[str], dict]]] = None,
    ) -> TradingGraphResult:
        """Run the full multi-agent workflow.

        Args:
            symbol: ticker symbol, e.g. "BTCUSD".
            df: OHLCV DataFrame. Must have a 'close' column.
            context: optional dict of news/macro/sentiment data for analysts.
            trade_date: override the trade date string (defaults to today UTC).
            portfolio_state: optional dict with equity/open_positions/exposure
                for the Portfolio Manager (C6/X4 fix).
            context_fetchers: optional list of callables that each take the
                symbol and return a dict of external data to merge into
                context before analysts run (X2 fix — lets the caller wire
                in real news/sentiment providers without the graph needing
                to know about them).
        """
        context = dict(context or {})
        if trade_date is None:
            trade_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        # Reset LLM cycle counter
        self.llm.reset_cycle()
        result = TradingGraphResult(symbol=symbol, trade_date=trade_date)
        log.info("=== Multi-Agent Graph started for %s ===", symbol)

        # H8 fix: validate df up-front.
        if df is None or df.empty or "close" not in df.columns:
            err = "DataFrame is empty or missing 'close' column — cannot run analysts."
            result.errors.append(err)
            log.error("Graph: %s", err)
            result.final_decision = PortfolioDecision(
                rating=PortfolioRating.HOLD,
                approved=False,
                final_lots=0.0,
                rationale=err,
            )
            result.timestamp = datetime.now(tz=timezone.utc).isoformat()
            self._record_token_usage(result)
            return result

        # X2 fix: if the caller supplied context_fetchers, invoke each
        # one and merge its output into the context dict so analysts
        # have real external data to work with.
        if context_fetchers:
            for fetcher in context_fetchers:
                try:
                    fetched = fetcher(symbol) or {}
                    if isinstance(fetched, dict):
                        context.update(fetched)
                except Exception as e:  # noqa: BLE001
                    log.warning("Context fetcher %s raised: %r", fetcher.__name__, e)
                    result.errors.append(f"context_fetcher: {e!r}")

        # 1. Analyst Team — H1 fix: run in parallel.
        log.info("[1/6] Running analyst team (parallel, max_workers=%d)...", self.max_workers)
        reports = _safe_call(
            "analyst_team", self._run_analysts_parallel, self.agent_timeout_s,
            result, symbol, df, context,
        )
        if reports is None:
            reports = {}
            result.errors.append("analyst_team: returned None — using empty reports")
        result.analyst_reports = reports
        fund_report = reports.get("fundamentals", AnalystReport("fundamentals", symbol, "")).report
        news_report = reports.get("news", AnalystReport("news", symbol, "")).report
        sent_report = reports.get("sentiment", AnalystReport("sentiment", symbol, "")).report
        tech_report = reports.get("technical", AnalystReport("technical", symbol, "")).report

        # 2. Bull/Bear Debate
        log.info("[2/6] Running bull/bear debate...")
        debate = _safe_call(
            "bull_bear_debate", self.bull_bear.run, self.agent_timeout_s,
            result, symbol, fund_report, news_report, sent_report, tech_report,
        )
        if debate is not None:
            result.debate = debate
            debate_history = debate.history
        else:
            debate_history = ""

        # 3. Research Manager
        log.info("[3/6] Research manager creating plan...")
        plan = _safe_call(
            "research_manager", self.research_manager.create_plan, self.agent_timeout_s,
            result, symbol, debate_history,
            fund_report, news_report, sent_report, tech_report,
        )
        if plan is None:
            # Safe fallback: HOLD plan.
            plan = ResearchPlan(
                recommendation=PortfolioRating.HOLD,
                rationale="[Research manager unavailable] Defaulting to Hold.",
                strategic_actions="No action — await next cycle.",
                confidence=0.3,
            )
            result.errors.append("research_manager: used fallback HOLD plan")
        result.research_plan = plan

        # 4. Trader
        log.info("[4/6] Trader creating proposal...")
        proposal = _safe_call(
            "trader", self.trader.propose, self.agent_timeout_s,
            result, symbol, df, plan,
            fund_report, news_report, sent_report, tech_report,
            max_lot=self.max_lot,
        )
        if proposal is None:
            proposal = TraderProposal(
                action=TraderAction.HOLD, lots=0.0,
                reasoning="[Trader unavailable] Defaulting to Hold.",
            )
            result.errors.append("trader: used fallback HOLD proposal")
        result.trader_proposal = proposal

        # 5. Risk Debate (only if trader proposed a trade)
        if proposal.action.value != "Hold" and proposal.lots > 0:
            log.info("[5/6] Running risk debate...")
            # H6/X6 fix: pass portfolio_state as risk_metrics if available.
            risk_metrics = portfolio_state or {}
            risk_result = _safe_call(
                "risk_debate", self.risk_debate.run, self.agent_timeout_s,
                result, symbol, proposal.to_markdown(),
                fund_report, news_report, sent_report, tech_report,
                risk_metrics,
            )
            if risk_result is not None:
                result.risk_debate = risk_result
                risk_history = risk_result.history
            else:
                risk_history = ""

            # 6. Portfolio Manager — C6 fix: pass portfolio_state.
            log.info("[6/6] Portfolio manager deciding...")
            past_context = _safe_call(
                "memory_log.get_past_context", self.memory_log.get_past_context,
                self.agent_timeout_s, result, symbol,
            ) or ""
            decision = _safe_call(
                "portfolio_manager", self.portfolio_manager.decide, self.agent_timeout_s,
                result, symbol, plan, proposal, risk_history, past_context,
                portfolio_state,
            )
            if decision is None:
                decision = PortfolioDecision(
                    rating=plan.recommendation,
                    approved=False,
                    final_lots=0.0,
                    rationale="[PM unavailable] Defaulting to reject.",
                )
                result.errors.append("portfolio_manager: used fallback reject")
            result.final_decision = decision
            # Store in memory log
            if decision.approved:
                try:
                    self.memory_log.store_decision(
                        ticker=symbol, trade_date=trade_date,
                        final_decision=decision.to_markdown(),
                    )
                except Exception as e:  # noqa: BLE001
                    result.errors.append(f"memory_log.store_decision: {e!r}")
                    log.warning("Memory log store failed: %r", e)
        else:
            log.info("[5/6] Skipping risk debate — trader proposed HOLD")
            # Auto-hold decision
            result.final_decision = PortfolioDecision(
                rating=plan.recommendation,
                approved=False,
                final_lots=0.0,
                rationale="Trader proposed HOLD — no action needed.",
            )

        # C20 fix: record token usage.
        self._record_token_usage(result)

        # H20 fix: structured audit logging to the Database decisions table.
        if self.database is not None:
            try:
                self._audit_log_decision(symbol, trade_date, result)
            except Exception as e:  # noqa: BLE001
                log.warning("Audit log to DB failed: %r", e)

        result.timestamp = datetime.now(tz=timezone.utc).isoformat()
        log.info("=== Multi-Agent Graph complete for %s ===", symbol)
        log.info("Final: rating=%s approved=%s lots=%.4f tokens=%d errors=%d",
                 result.final_decision.rating.value,
                 result.final_decision.approved,
                 result.final_decision.final_lots,
                 sum(result.token_usage.values()),
                 len(result.errors))
        return result

    # ----------------------------------------------------------------
    def _run_analysts_parallel(self, symbol: str, df: pd.DataFrame,
                                 context: dict) -> dict[str, AnalystReport]:
        """H1 fix: run all selected analysts in parallel using a thread pool.

        Critical #3 fix: check the AnalystTeam cache FIRST — if we already
        have reports for this (symbol, bar_time, context_hash), return them
        immediately instead of re-calling the LLM. The cache is context-aware
        (Critical #2 fix), so different context produces fresh reports.
        """
        # Critical #3 fix: check cache before spawning threads.
        # M6 FIX: use lock for cache access to prevent race condition.
        cache_key = self.analyst_team._cache_key(symbol, df, context)
        with self.analyst_team._cache_lock:
            if cache_key is not None and cache_key in self.analyst_team._cache:
                cached = self.analyst_team._cache[cache_key]
                log.info("AnalystTeam: cache hit for %s @ %s (ctx hash=%s) — skipping parallel run",
                         symbol, cache_key[1], cache_key[2])
                return cached

        results: dict[str, AnalystReport] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_name = {}
            for name in self.selected_analysts:
                analyst = self.analyst_team.analysts.get(name)
                if analyst is None:
                    log.warning("Unknown analyst %r — skipping (M1)", name)
                    continue
                fut = pool.submit(self._run_one_analyst, name, analyst, symbol, df, context)
                future_to_name[fut] = name
            for fut in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    results[name] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[name] = AnalystReport(
                        analyst=name, symbol=symbol, report="",
                        success=False, error=str(e),
                    )
                    log.warning("Analyst %s failed: %r", name, e)

        # Critical #3 fix: store results in the AnalystTeam cache so
        # subsequent calls (e.g. re-evaluation within the same cycle) hit
        # the cache instead of re-calling the LLM.
        # M6 FIX: use lock for cache store to prevent race condition.
        if cache_key is not None:
            with self.analyst_team._cache_lock:
                self.analyst_team._cache[cache_key] = results
                # Limit cache to last 100 entries.
                if len(self.analyst_team._cache) > 100:
                    oldest = next(iter(self.analyst_team._cache))
                    del self.analyst_team._cache[oldest]

        return results

    @staticmethod
    def _run_one_analyst(name: str, analyst, symbol: str,
                          df: pd.DataFrame, context: dict) -> AnalystReport:
        """Run a single analyst with its own try/except."""
        try:
            return analyst.analyze(symbol, df, context)
        except Exception as e:  # noqa: BLE001
            return AnalystReport(
                analyst=name, symbol=symbol, report="",
                success=False, error=str(e),
            )

    # ----------------------------------------------------------------
    def _record_token_usage(self, result: TradingGraphResult) -> None:
        """C20 fix: pull token-usage stats from the LLM provider."""
        try:
            stats = self.llm.stats
            result.token_usage = {
                "calls_this_cycle": stats.get("calls_this_cycle", 0),
                "calls_this_min": stats.get("calls_this_min", 0),
                "max_per_cycle": stats.get("max_per_cycle", 0),
            }
        except Exception:  # noqa: BLE001
            pass

    # ----------------------------------------------------------------
    def _audit_log_decision(self, symbol: str, trade_date: str,
                              result: TradingGraphResult) -> None:
        """H20 fix: write the final decision to the Database decisions table."""
        import uuid as _uuid
        decision = result.final_decision
        if decision is None:
            return
        audit_id = str(_uuid.uuid4())
        # The decisions table schema (from database.py) expects these columns.
        # We store the full graph result as the feature_vector for forensic review.
        feature_vector = {
            "analyst_reports": list(result.analyst_reports.keys()),
            "debate_rounds": result.debate.rounds if result.debate else 0,
            "risk_debate_rounds": result.risk_debate.rounds if result.risk_debate else 0,
            "agent_latencies": result.agent_latencies,
            "token_usage": result.token_usage,
            "errors": result.errors,
        }
        try:
            self.database.save_decision(
                audit_id=audit_id,
                correlation_id=audit_id,
                symbol=symbol,
                cycle=0,
                bar_close=float(result.trader_proposal.entry_price) if result.trader_proposal else 0.0,
                feature_vector=feature_vector,
                account_equity=0.0,
                open_positions=0,
                current_drawdown_pct=0.0,
                strategy_action=decision.rating.value,
                strategy_strength=float(result.research_plan.confidence) if result.research_plan else 0.0,
                strategy_meta={
                    "trade_date": trade_date,
                    "rationale": decision.rationale[:500],
                },
            )
            self.database.finalize_decision(
                audit_id=audit_id,
                approved=decision.approved,
                final_lots=decision.final_lots,
                final_sl=result.trader_proposal.stop_loss if result.trader_proposal else 0.0,
                final_tp=result.trader_proposal.take_profit if result.trader_proposal else 0.0,
                entry_price=result.trader_proposal.entry_price if result.trader_proposal else 0.0,
                ticket=0,
                reject_reason="" if decision.approved else decision.rationale[:200],
            )
            log.info("Audit-logged decision %s for %s", audit_id[:8], symbol)
        except Exception as e:  # noqa: BLE001
            log.warning("Audit log DB write failed: %r", e)

    # ----------------------------------------------------------------
    # C1/X1 fix: expose memory reflection on the graph so the main
    # trading loop can call it after a trade closes.
    # ----------------------------------------------------------------
    def resolve_trade_outcome(self, symbol: str, trade_date: str,
                                raw_return: float, alpha_return: float = 0.0,
                                benchmark: str = "BTC") -> str:
        """Call this from the main trading loop once a trade's outcome is known.

        Generates a reflection on the decision and marks the memory-log
        entry as resolved, completing the learning loop (C1/X1 fix).
        """
        try:
            return self.memory_log.resolve_trade_outcome(
                ticker=symbol, trade_date=trade_date,
                raw_return=raw_return, alpha_return=alpha_return,
                benchmark=benchmark,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("resolve_trade_outcome failed: %r", e)
            return ""
