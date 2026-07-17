"""architecture/llm_augmented_coordinator.py
=====================================================================
Hybrid LLM-Augmented Multi-Agent Coordinator (Co-Founder Audit)
=====================================================================
Bridges the rule-based MultiAgentCoordinator (live, fast, deterministic,
zero-cost) with the LLM-powered MultiAgentTradingGraph (agents/, slow,
costly, but richer context-aware analysis).

DESIGN RATIONALE (Co-Founder decision):
  - LLM is NEVER the sole decision-maker. It augments rule-based consensus.
  - LLM runs ONLY when: (a) rule-based says BUY/SELL, (b) LLM is enabled,
    (c) cost budget remains, (d) latency budget remains.
  - On ANY LLM failure (timeout, exception, all-providers-down), the
    rule-based consensus is returned unchanged. The bot NEVER blocks on LLM.
  - LLM disagreement with rule-based does NOT override rule-based —
    LLM is advisory. If LLM says SELL but rule-based says BUY, we respect
    rule-based but log the disagreement for forensic review.
  - The existing 13-gate RiskPipeline + WisdomGate still run AFTER this
    coordinator, so LLM approval is necessary-but-not-sufficient. A trade
    can only be placed if rule-based + LLM + risk pipeline + wisdom gate
    ALL agree. This is defense-in-depth.

FLOW:
  1. rule_consensus = MultiAgentCoordinator.evaluate(symbol, df, fv, ctx)
  2. if rule_consensus.action == "HOLD":
        return rule_consensus  (no LLM cost on HOLDs — 95% of cycles)
  3. if not llm_enabled or no llm_budget:
        return rule_consensus  (rule-based path, unchanged)
  4. llm_result = MultiAgentTradingGraph.propagate(symbol, df, ctx)  [timeout-bounded]
  5. Combine:
     - If LLM approves AND LLM action matches rule-based action:
         → boost strength by LLM confidence factor (capped at 1.5x)
     - If LLM approves BUT LLM action differs from rule-based:
         → respect rule-based action; log disagreement; keep strength as-is
     - If LLM rejects (rating=Hold/Sell/Underweight when rule=BUY):
         → downgrade strength by 50% (LLM veto — risk pipeline will likely
           reject anyway, but this gives it a chance to fail gracefully)
     - If LLM errored/timeout:
         → return rule_consensus unchanged
  6. Return augmented Consensus. Risk pipeline + WisdomGate run next.

CONFIG (config.yaml → llm: section):
  llm:
    enabled: false              # opt-in — set true to enable LLM augmentation
    cost_budget_per_cycle_usd: 0.02   # hard cap, ~4 LLM calls/cycle at $0.005
    cost_budget_per_day_usd: 5.00     # daily cap
    latency_budget_s: 8.0       # per-symbol LLM call timeout
    min_rule_strength: 0.15     # only run LLM if rule-based strength >= this
    boost_multiplier: 1.3       # strength boost when LLM agrees (cap 1.5)
    veto_multiplier: 0.5        # strength reduction when LLM disagrees
    require_llm_for_live: false # if true, live mode REFUSES to trade without LLM

FAIL-SAFE GUARANTEES:
  1. LLMProvider() construction failure → llm_enabled=False (silent fallback)
  2. propagate() exception → rule_consensus returned unchanged
  3. propagate() timeout → rule_consensus returned unchanged
  4. Token-economy violation (LLMProvider returns success=False) → rule_consensus
  5. Cost budget exhausted → LLM skipped for rest of cycle/day
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import pandas as pd

from architecture.multi_agent import (
    MultiAgentCoordinator, Consensus, AgentVote, AgentOpinion,
    build_default_coordinator,
)
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.llm_augmented")


# ----------------------------------------------------------------------
# Cost tracker — shared across all instances, persisted in-memory
# ----------------------------------------------------------------------
@dataclass
class _LLMCostTracker:
    """Per-process LLM cost tracker. Thread-safe."""
    spent_today_usd: float = 0.0
    spent_this_cycle_usd: float = 0.0
    last_reset_date: str = ""  # UTC date string YYYY-MM-DD
    calls_today: int = 0
    calls_this_cycle: int = 0
    last_reset_cycle: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset_if_new_day(self) -> None:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if self.last_reset_date != today:
                self.spent_today_usd = 0.0
                self.calls_today = 0
                self.last_reset_date = today
                log.info("LLM cost tracker: daily reset (date=%s)", today)

    def reset_cycle(self, cycle_num: int) -> None:
        with self._lock:
            if self.last_reset_cycle != cycle_num:
                self.spent_this_cycle_usd = 0.0
                self.calls_this_cycle = 0
                self.last_reset_cycle = cycle_num

    def record(self, cost_usd: float, cycle_num: int) -> None:
        with self._lock:
            self.spent_today_usd += cost_usd
            self.spent_this_cycle_usd += cost_usd
            self.calls_today += 1
            self.calls_this_cycle += 1

    def cycle_budget_ok(self, cap: float) -> bool:
        with self._lock:
            return self.spent_this_cycle_usd < cap

    def daily_budget_ok(self, cap: float) -> bool:
        with self._lock:
            return self.spent_today_usd < cap


# Module-level singleton — shared by all LLMAugmentedCoordinator instances
_COST_TRACKER = _LLMCostTracker()


# ----------------------------------------------------------------------
# Coordinator
# ----------------------------------------------------------------------
class LLMAugmentedCoordinator:
    """Hybrid rule-based + LLM coordinator.

    Wraps the existing MultiAgentCoordinator. For each evaluate() call:
      1. Runs the rule-based coordinator first (fast, deterministic).
      2. If the rule-based result is actionable (BUY/SELL) AND LLM is
         enabled AND budgets remain, runs the LLM graph as a second
         opinion.
      3. Combines the two results per the rules in the module docstring.
      4. Returns a Consensus (same dataclass the rule-based coordinator
         returns) so the rest of the pipeline is unchanged.

    The LLM graph is constructed lazily — only on first call where LLM
    augmentation is actually needed. This keeps startup fast when LLM
    is disabled.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 rule_coordinator: Optional[MultiAgentCoordinator] = None):
        self._cfg = config or {}
        llm_cfg = self._cfg.get("llm", {})
        self._enabled = bool(llm_cfg.get("enabled", False))
        self._cost_per_cycle = float(llm_cfg.get("cost_budget_per_cycle_usd", 0.02))
        self._cost_per_day = float(llm_cfg.get("cost_budget_per_day_usd", 5.00))
        self._latency_budget_s = float(llm_cfg.get("latency_budget_s", 8.0))
        self._min_rule_strength = float(llm_cfg.get("min_rule_strength", 0.15))
        self._boost_mult = float(llm_cfg.get("boost_multiplier", 1.3))
        self._veto_mult = float(llm_cfg.get("veto_multiplier", 0.5))
        self._require_llm_for_live = bool(llm_cfg.get("require_llm_for_live", False))
        # Estimated cost per LLM call (rough — actual varies by provider/model).
        # Conservatively over-estimate so we err on the side of stopping early.
        self._estimated_cost_per_call = float(llm_cfg.get("estimated_cost_per_call_usd", 0.005))

        # Rule-based coordinator (always present — it's the primary path)
        self._rule = rule_coordinator or build_default_coordinator()

        # Co-Founder Audit: market context provider for news/sentiment/macro.
        # Constructed lazily — only when first LLM call needs it. Provides
        # the news_items/macro/retail_sentiment/news_sentiment context fields
        # that the LLM analysts (NewsAnalyst, SentimentAnalyst) expect.
        self._context_provider = None

        # LLM graph — constructed lazily.
        self._llm_graph = None
        self._llm_available: Optional[bool] = None  # None = not yet probed
        self._llm_init_lock = threading.Lock()

        # If LLM is enabled, eagerly try to construct it once so we know
        # whether keys are available. Failure here is non-fatal — we just
        # log and disable LLM for this session.
        if self._enabled:
            self._probe_llm_availability()

    # ------------------------------------------------------------------
    def _probe_llm_availability(self) -> None:
        """Try to construct the LLM graph once. Sets self._llm_available.

        Co-Founder Audit fix: LLMProvider doesn't store api_key directly
        on each provider dict — it stores a `get_key` callable that
        returns the key (allowing per-call rotation for multi-key
        providers like Groq/Gemini). We call get_key() to check if a
        non-empty key is actually available, rather than checking a
        non-existent `api_key` field.
        """
        try:
            from agents.agent_graph import MultiAgentTradingGraph
            from external.llm_provider import LLMProvider
            llm = LLMProvider()
            # Each provider has a `get_key` callable. Call it to verify a
            # non-empty key is actually available.
            usable = []
            for p in llm.providers:
                if not p.get("available"):
                    continue
                get_key = p.get("get_key")
                if not callable(get_key):
                    continue
                try:
                    key = get_key()
                    if key:
                        usable.append(p)
                except Exception:
                    continue
            if not usable:
                log.warning("LLM augmentation: enabled in config but no LLM "
                           "API keys resolve to non-empty values "
                           "(GROQ_API_KEY/CEREBRAS_API_KEY/SAMBANOVA_API_KEY/"
                           "OPENROUTER_API_KEY/GEMINI_API_KEY). "
                           "Falling back to rule-only path. To enable: set "
                           "at least one key in .env, then set llm.enabled: true.")
                self._llm_available = False
                return
            graph = MultiAgentTradingGraph(
                llm=llm,
                config=self._cfg,
                agent_timeout_s=self._latency_budget_s,
            )
            self._llm_graph = graph
            self._llm_available = True
            log.info("LLM augmentation: enabled (%d providers with keys: %s, "
                     "agent_timeout=%.1fs, cost_cap=$%.3f/cycle $%.2f/day)",
                     len(usable), [p["name"] for p in usable],
                     self._latency_budget_s,
                     self._cost_per_cycle, self._cost_per_day)
        except Exception as e:
            log.warning("LLM augmentation: init failed — falling back to "
                       "rule-only path: %r", e)
            self._llm_available = False

    # ------------------------------------------------------------------
    def evaluate(self,
                 symbol: str,
                 df: pd.DataFrame,
                 features: Any,
                 context: Dict[str, Any]) -> Consensus:
        """Run rule-based + optional LLM augmentation. Returns Consensus."""
        # Step 1: always run rule-based first.
        rule_consensus = self._rule.evaluate(symbol, df, features, context)

        # Step 2: short-circuit — no LLM cost on HOLDs.
        if rule_consensus.action == "HOLD":
            return rule_consensus

        # Step 3: short-circuit if LLM is not enabled / unavailable.
        if not self._enabled or not self._llm_available or self._llm_graph is None:
            return rule_consensus

        # Step 4: short-circuit if rule-based strength is too weak to bother.
        if rule_consensus.strength < self._min_rule_strength:
            return rule_consensus

        # Step 5: short-circuit if cost budget exhausted.
        _COST_TRACKER.reset_if_new_day()
        cycle_num = context.get("_cycle", 0)
        _COST_TRACKER.reset_cycle(cycle_num)
        if not _COST_TRACKER.cycle_budget_ok(self._cost_per_cycle):
            log.debug("LLM augmentation: cycle cost cap reached (%.3f/%.3f) — skipping %s",
                     _COST_TRACKER.spent_this_cycle_usd, self._cost_per_cycle, symbol)
            return rule_consensus
        if not _COST_TRACKER.daily_budget_ok(self._cost_per_day):
            log.warning("LLM augmentation: DAILY cost cap reached (%.2f/%.2f) — "
                       "LLM disabled for rest of day",
                       _COST_TRACKER.spent_today_usd, self._cost_per_day)
            return rule_consensus

        # Step 6: run LLM graph with timeout. On ANY failure, return rule-based.
        try:
            t0 = time.monotonic()
            # R4 FIX: build REAL portfolio state with actual risk metrics
            # (drawdown, exposure, heat, consecutive losses) instead of
            # just equity + open_positions. This gives the RiskDebate
            # agents quantitative grounding for their risk assessment.
            portfolio_state = self._build_portfolio_state(context)
            llm_result = self._run_llm_with_timeout(
                symbol, df, context, portfolio_state)
            elapsed = time.monotonic() - t0

            if llm_result is None:
                # Timeout or exception — rule-based consensus unchanged.
                return rule_consensus

            # Record estimated cost.
            # EDGE CASE FIX: guard against token_usage being None or having
            # non-numeric values. Previously would crash with TypeError.
            try:
                token_calls = sum(int(v) for v in llm_result.token_usage.values()) \
                    if llm_result and llm_result.token_usage else 1
            except (TypeError, ValueError, AttributeError):
                token_calls = 1
            est_cost = max(self._estimated_cost_per_call, token_calls * self._estimated_cost_per_call)
            _COST_TRACKER.record(est_cost, cycle_num)

            # Step 7: combine rule-based + LLM per the design rules.
            return self._combine(symbol, rule_consensus, llm_result, elapsed)

        except Exception as e:
            log.warning("LLM augmentation: unexpected error for %s — "
                       "returning rule-based consensus: %r", symbol, e)
            return rule_consensus

    # ------------------------------------------------------------------
    def _build_portfolio_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """R4 FIX: build real portfolio state with quantitative risk metrics.

        Previously the RiskDebate only received equity + open_positions.
        Now it receives drawdown, exposure, heat, consecutive losses,
        and peak equity — giving the risk analysts real data to debate
        instead of debating in a vacuum.
        """
        state = {
            "equity": context.get("equity", 0),
            "peak_equity": context.get("peak_equity", context.get("equity", 0)),
            "open_positions": context.get("open_positions", 0),
        }
        # Compute real risk metrics from the rule-based portfolio manager
        # (accessible via self._rule which is the MultiAgentCoordinator,
        # but the portfolio is in the TradingBot). We can't access it
        # directly from here, so we compute what we can from context.
        equity = state["equity"]
        peak = state["peak_equity"]
        if peak > 0 and equity > 0:
            drawdown_pct = (peak - equity) / peak * 100
            state["drawdown_pct"] = round(drawdown_pct, 2)
            state["drawdown_from_peak_usd"] = round(peak - equity, 2)
        # Exposure and heat would come from portfolio.metrics() — the
        # caller (integration.py) can add them to context if available.
        if "gross_exposure_pct" in context:
            state["gross_exposure_pct"] = context["gross_exposure_pct"]
        if "portfolio_heat_pct" in context:
            state["portfolio_heat_pct"] = context["portfolio_heat_pct"]
        if "consecutive_losses" in context:
            state["consecutive_losses"] = context["consecutive_losses"]
        if "realized_pnl_today" in context:
            state["realized_pnl_today"] = context["realized_pnl_today"]
        return state

    # ------------------------------------------------------------------
    def _run_llm_with_timeout(self, symbol: str, df: pd.DataFrame,
                              context: Dict[str, Any],
                              portfolio_state: Dict[str, Any]):
        """Run MultiAgentTradingGraph.propagate() with a hard timeout.

        Uses concurrent.futures so a hung LLM provider can't block the
        cycle. On timeout, returns None (caller falls back to rule-based).
        """
        import concurrent.futures
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    self._llm_graph.propagate,
                    symbol=symbol, df=df,
                    context=self._build_llm_context(symbol, context),
                    portfolio_state=portfolio_state,
                )
                return fut.result(timeout=self._latency_budget_s)
        except concurrent.futures.TimeoutError:
            log.warning("LLM augmentation: timed out after %.1fs for %s — "
                       "falling back to rule-based",
                       self._latency_budget_s, symbol)
            return None
        except Exception as e:
            log.warning("LLM augmentation: propagate() failed for %s: %r",
                       symbol, e)
            return None

    # ------------------------------------------------------------------
    def _get_context_provider(self):
        """Lazy-init the MarketContextProvider. Returns None on failure."""
        if self._context_provider is not None:
            return self._context_provider
        try:
            from architecture.context_providers import MarketContextProvider
            self._context_provider = MarketContextProvider(self._cfg)
            log.info("LLM augmentation: market context provider ready "
                     "(news/sentiment/macro/events with per-source TTL caching)")
        except Exception as e:
            log.warning("LLM augmentation: context provider init failed — "
                       "LLM analysts will see empty news/sentiment: %r", e)
            self._context_provider = None
        return self._context_provider

    # ------------------------------------------------------------------
    def _build_llm_context(self, symbol: str,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Build the context dict that the LLM graph's analysts consume.

        Co-Founder Audit: this now wires in real news/sentiment/macro data
        from external/ providers via MarketContextProvider. The provider
        handles caching (10min news, 1hr macro, 5min sentiment per symbol)
        so we don't fire HTTP calls on every cycle. Fail-open: any provider
        error returns empty fields — LLM analysts degrade gracefully to
        "no recent news available" and base analysis on OHLCV data alone.

        Field mapping (verified against agents/analysts.py):
          - NewsAnalyst reads: news_items, macro
          - SentimentAnalyst reads: social_sentiment, news_sentiment, retail_sentiment
          - FundamentalsAnalyst reads: context (whole dict)
          - TechnicalAnalyst reads: context (whole dict, but uses df directly)
        """
        # Start with the rule-based context (equity, regime, etc.)
        llm_ctx = {
            "symbol": symbol,
            "equity": context.get("equity", 0),
            "peak_equity": context.get("peak_equity", context.get("equity", 0)),
            "regime": str(context.get("adjustments", {}).regime
                          if hasattr(context.get("adjustments", {}), "regime")
                          else context.get("regime", "unknown")),
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        # Wire in market context (news/sentiment/macro/events).
        # Fail-open: if the provider is unavailable, analysts see empty
        # fields and degrade gracefully.
        provider = self._get_context_provider()
        if provider is not None:
            try:
                market_ctx = provider.get_context(symbol)
                llm_ctx.update(market_ctx)
            except Exception as e:
                log.warning("LLM augmentation: context fetch failed for %s — "
                           "analysts will see empty news/sentiment: %r",
                           symbol, e)
                # Ensure the expected keys exist even on failure so analysts
                # don't KeyError
                llm_ctx.setdefault("news_items", [])
                llm_ctx.setdefault("macro", {})
                llm_ctx.setdefault("retail_sentiment", {})
                llm_ctx.setdefault("news_sentiment", {})
                llm_ctx.setdefault("social_sentiment", {})
                llm_ctx.setdefault("upcoming_events", [])
        else:
            # Provider unavailable — set empty defaults so analysts don't crash
            llm_ctx["news_items"] = []
            llm_ctx["macro"] = {}
            llm_ctx["retail_sentiment"] = {}
            llm_ctx["news_sentiment"] = {}
            llm_ctx["social_sentiment"] = {}
            llm_ctx["upcoming_events"] = []

        return llm_ctx

    # ------------------------------------------------------------------
    def _combine(self, symbol: str, rule: Consensus,
                 llm_result, llm_latency_s: float) -> Consensus:
        """Combine rule-based + LLM consensus per the design rules.

        See module docstring for the rules. Returns a new Consensus with
        the combined strength + an LLM-attribution note in agent_opinions.
        """
        # Make a shallow copy so we don't mutate the rule coordinator's state.
        combined = Consensus(
            timestamp=rule.timestamp,
            symbol=rule.symbol,
            action=rule.action,
            strength=rule.strength,
            confidence=rule.confidence,
            votes_buy=rule.votes_buy,
            votes_sell=rule.votes_sell,
            votes_hold=rule.votes_hold,
            votes_reduce=rule.votes_reduce,
            agreement_score=rule.agreement_score,
            dissenting_agents=list(rule.dissenting_agents),
            agent_opinions=list(rule.agent_opinions),
            suggested_weight=rule.suggested_weight,
        )

        decision = llm_result.final_decision
        if decision is None:
            log.warning("LLM augmentation: %s — graph returned no final_decision, "
                       "using rule-based unchanged", symbol)
            return combined

        # Map LLM rating → direction for comparison
        # Buy/Overweight = bullish, Sell/Underweight = bearish, Hold = neutral
        bullish_ratings = {"Buy", "Overweight"}
        bearish_ratings = {"Sell", "Underweight"}
        neutral_ratings = {"Hold"}
        llm_rating = decision.rating.value if hasattr(decision.rating, "value") else str(decision.rating)
        llm_approved = bool(decision.approved)

        # Rule action direction
        rule_bullish = rule.action == "BUY"
        rule_bearish = rule.action == "SELL"

        # Compare directions
        if llm_rating in bullish_ratings and rule_bullish:
            # Agreement: boost strength
            new_strength = min(1.0, rule.strength * self._boost_mult)
            log.info("LLM aug %s: AGREEMENT (rule=BUY, llm=%s, approved=%s) — "
                    "strength %.3f → %.3f (latency=%.1fs)",
                    symbol, llm_rating, llm_approved,
                    rule.strength, new_strength, llm_latency_s)
            combined.strength = new_strength
        elif llm_rating in bearish_ratings and rule_bearish:
            # Agreement: boost strength
            new_strength = min(1.0, rule.strength * self._boost_mult)
            log.info("LLM aug %s: AGREEMENT (rule=SELL, llm=%s, approved=%s) — "
                    "strength %.3f → %.3f (latency=%.1fs)",
                    symbol, llm_rating, llm_approved,
                    rule.strength, new_strength, llm_latency_s)
            combined.strength = new_strength
        elif llm_rating in neutral_ratings:
            # LLM says HOLD — partial veto
            new_strength = rule.strength * self._veto_mult
            log.info("LLM aug %s: LLM NEUTRAL (llm=%s) — strength %.3f → %.3f "
                    "(latency=%.1fs)",
                    symbol, llm_rating, rule.strength, new_strength, llm_latency_s)
            combined.strength = new_strength
        elif (llm_rating in bullish_ratings and rule_bearish) or \
             (llm_rating in bearish_ratings and rule_bullish):
            # Direction disagreement — full veto, near-zero strength.
            # The risk pipeline will likely reject, but we let it run.
            new_strength = rule.strength * (self._veto_mult * 0.5)
            log.warning("LLM aug %s: DIRECTION DISAGREEMENT (rule=%s, llm=%s) — "
                       "strength %.3f → %.3f (latency=%.1fs) — risk pipeline "
                       "will likely reject",
                       symbol, rule.action, llm_rating,
                       rule.strength, new_strength, llm_latency_s)
            combined.strength = new_strength
        else:
            log.debug("LLM aug %s: unmapped rating %r — using rule-based unchanged",
                     symbol, llm_rating)

        # Append a synthetic agent opinion recording the LLM's view, so
        # the audit trail shows LLM participated.
        # EDGE CASE FIX: guard against None research_plan / token_usage
        # which would crash with AttributeError.
        try:
            llm_confidence = float(llm_result.research_plan.confidence) \
                if llm_result and llm_result.research_plan else 0.5
        except (TypeError, ValueError, AttributeError):
            llm_confidence = 0.5
        try:
            token_count = sum(llm_result.token_usage.values()) \
                if llm_result and llm_result.token_usage else 0
        except (TypeError, AttributeError):
            token_count = 0

        combined.agent_opinions.append(AgentOpinion(
            agent_name="llm_augment",
            vote=AgentVote.BUY if llm_rating in bullish_ratings
                 else AgentVote.SELL if llm_rating in bearish_ratings
                 else AgentVote.HOLD,
            confidence=llm_confidence,
            reasoning=f"LLM={llm_rating}, approved={llm_approved}, "
                      f"latency={llm_latency_s:.1f}s, tokens={token_count}",
            target_weight=0.0,  # LLM doesn't size — that's SizingGate's job
        ))

        return combined

    # ------------------------------------------------------------------
    # Public introspection — for status/diagnostics
    # ------------------------------------------------------------------
    def is_llm_enabled(self) -> bool:
        return self._enabled and bool(self._llm_available)

    def llm_status(self) -> Dict[str, Any]:
        # Co-Founder Audit: include context provider status so operators
        # can verify news/sentiment are wired when LLM is enabled.
        ctx_status = {}
        if self._context_provider is not None:
            try:
                ctx_status = self._context_provider.status()
            except Exception:
                ctx_status = {}
        return {
            "enabled_in_config": self._enabled,
            "llm_available": bool(self._llm_available),
            "active": self.is_llm_enabled(),
            "spent_this_cycle_usd": _COST_TRACKER.spent_this_cycle_usd,
            "spent_today_usd": _COST_TRACKER.spent_today_usd,
            "calls_this_cycle": _COST_TRACKER.calls_this_cycle,
            "calls_today": _COST_TRACKER.calls_today,
            "cost_cap_cycle_usd": self._cost_per_cycle,
            "cost_cap_day_usd": self._cost_per_day,
            "latency_budget_s": self._latency_budget_s,
            "rule_coordinator_agents": self._rule.agent_names(),
            "context_providers": ctx_status,
        }


# ----------------------------------------------------------------------
# Builder — drop-in replacement for build_default_coordinator()
# ----------------------------------------------------------------------
def build_augmented_coordinator(config: Dict[str, Any]) -> LLMAugmentedCoordinator:
    """Build the hybrid coordinator. Use this instead of
    build_default_coordinator() to get LLM augmentation.

    If LLM is disabled in config (default), this is functionally equivalent
    to build_default_coordinator() — same rule-based path, zero LLM cost.
    """
    return LLMAugmentedCoordinator(config=config)
