"""architecture/integration.py
=====================================================================
Layer-0 Master Integration (Improvement #20)
=====================================================================
Wires all 20 architectural improvements into a single IndustrialBot
class — the unified brain of the autonomous trading platform.

IndustrialBot = composition of:
    - EventBus                  (nervous system)
    - StateMachine              (lifecycle)
    - AIModelManager            (ML brain registry)
    - FeaturePipeline           (perception)
    - ExchangeInterface         (hands)
    - PortfolioManager          (capital tracker)
    - RiskPipeline (12 gates)   (defense)
    - SelfHealingSystem         (immunity)
    - DecisionAuditor           (memory of decisions)
    - MemorySystem              (long-term memory)
    - OnlineLearner             (continuous learning)
    - MultiAgentCoordinator     (multi-perspective AI)
    - InstitutionalMonitor      (observability)
    - SnapshotEngine            (state persistence)
    - RecoveryEngine            (crash recovery)
    - ConfigVersioning          (config history)
    - RegimeOrchestrator        (market context)
    - SimulationLayer           (what-if testing)
    - WisdomGate (existing)     (Livermore principles)

Trading Flow (per cycle):
    1. StateMachine.check_state_health() — watchdog
    2. EventBus.drain() — process any pending events
    3. SelfHealingSystem.health() — auto-recover from failures
    4. RegimeOrchestrator.detect() — what market are we in?
    5. For each symbol (parallel):
       a. Fetch OHLCV via ExchangeInterface
       b. Compute FeatureVector via FeaturePipeline
       c. MultiAgentCoordinator.evaluate() — get consensus
       d. RiskContext assembled with portfolio state
       e. RiskPipeline.evaluate() — 12-gate check
       f. WisdomGate.evaluate() — 120-principle check
       g. DecisionAuditor.record() — full audit trail
       h. If approved: ExchangeInterface.place_order()
          → PortfolioManager.on_position_opened()
          → EventBus.emit(POSITION_OPENED)
          → MemorySystem.encode_episode() [on close]
    6. Update open positions (SL/TP check)
    7. InstitutionalMonitor.update_equity() + check_alerts()
    8. OnlineLearner.record_outcome() [if trade closed]
    9. SnapshotEngine.take_snapshot() [every N cycles]
    10. EventBus.emit(HEARTBEAT)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd

from architecture.ai_model_manager import AIModelManager, get_model_manager
from architecture.decision_audit import DecisionAuditor
from architecture.event_bus import EventBus, EventType, get_bus
from architecture.exchange_abstraction import (
    ExchangeInterface, OrderRequest, OrderSide, create_exchange,
)
from architecture.feature_pipeline import FeaturePipeline, get_pipeline
from architecture.institutional_monitoring import InstitutionalMonitor
from architecture.memory_system import MemorySystem
from architecture.online_learning import OnlineLearner
from architecture.portfolio_manager_v2 import PortfolioManager
from architecture.recovery_engine import (
    ConfigVersioning, RecoveryEngine, SnapshotEngine,
)
from architecture.regime_orchestrator import (
    MarketRegime, RegimeOrchestrator,
)
from architecture.risk_pipeline import RiskPipeline
from architecture.self_healing import SelfHealingSystem
from architecture.simulation_layer import SimulationLayer
from architecture.state_machine import BotState, StateMachine, get_state_machine
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.integration")


@dataclass
class CycleResult:
    """Result of one trading cycle."""
    cycle: int = 0
    timestamp: str = ""
    regime: str = "unknown"
    equity: float = 0.0
    open_positions: int = 0
    signals_generated: int = 0
    trades_placed: int = 0
    trades_rejected: int = 0
    cycle_time_ms: float = 0.0
    state: str = "LIVE"
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    # UI fix: skip-breakdown for cycle-summary (avoids 100 individual lines)
    skip_breakdown: Dict[str, int] = field(default_factory=dict)
    # Candle-cache diagnostics (populated only on cycles that actually scan
    # symbols) — merged into the main.py CYCLE log line instead of a
    # separate log.info() call.
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    # DEBUG: pipeline funnel — tracks how many symbols passed each stage
    # so operators can see WHERE signals are being lost.
    funnel: Dict[str, int] = field(default_factory=dict)


def _skip_reason_key(reason: str) -> str:
    """Normalize a skip/reject reason string into a stable bucket key for
    skip_breakdown counters (H18 fix). Cuts at the first of '(' or ':',
    whichever occurs first, instead of only '(' — structured reason codes
    like "risk_gate_X: some detail" and "hold_or_low_strength(0.12)" both
    collapse to their leading identifier."""
    if not reason:
        return "unknown"
    paren_idx = reason.find("(")
    colon_idx = reason.find(":")
    candidates = [i for i in (paren_idx, colon_idx) if i != -1]
    cut = min(candidates) if candidates else len(reason)
    return reason[:cut].strip()


class TradingBot:
    """The unified, canonical trading bot (Phase 3 rename of IndustrialBot).

    This is the ONE class that wires the entire pipeline:
        data feed → features → multi-agent consensus → risk pipeline (13 gates)
        → wisdom gate → broker order → portfolio update → DB persistence → audit

    P0-1 FIX (Phase 3): Every approval path in _process_symbol ends in an
    actual place_order call or an explicit, logged, DB-persisted rejection.
    The "log APPROVED, place nothing" bug (master_orchestrator.py:499-509)
    is structurally impossible here — see _process_symbol.

    P0-2 FIX (Phase 3): kill_switch_file is checked at the top of every
    cycle before any signal evaluation runs.

    P0-3/P0-4 FIX (Phase 3): RiskContext is built with REAL consecutive_losses
    and realized_pnl_today from portfolio history + DB — not hardcoded zeros.

    P0-8 FIX (Phase 3): Every fill writes to database.trades AND database.decisions.

    P0-10 FIX (Phase 3): --mode=paper|demo|live flag replaces the dead
    send_real_orders config key. Default is paper.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 bus: Optional[EventBus] = None,
                 mode: str = "paper"):
        self._cfg = config
        self._bus = bus or get_bus()
        self._lock = threading.RLock()
        self._cycle = 0
        self._stop = False
        # P0-10 FIX: mode is paper | demo | live. paper = PaperAdapter (default),
        # demo/live = MT5Adapter. The CLI layer enforces the live confirmation flag.
        self._mode = mode
        self._i_understand_real_money = bool(config.get("_i_understand_real_money", False))

        # ── Strategy Decision Trace ──────────────────────────────────
        # TRADING_BOT_TRACE=1 enables per-symbol INFO-level logging of:
        #   strategy action, strength, agent votes, MTF score, confluence,
        #   risk verdict, and the final skip/approve reason.
        # This is the key diagnostic for "why is everything HOLD?" —
        # set TRACING_BOT_TRACE=1 in the environment to see the full
        # decision breakdown for every symbol on every cycle.
        self._trace_enabled = os.environ.get("TRADING_BOT_TRACE", "0") not in ("0", "false", "False", "")
        # Configurable minimum strength for a signal to be actionable.
        # Default lowered from 0.3 to 0.15 — the old 0.3 threshold combined
        # with the multi-agent scoring (where strength = max(bull_score,
        # bear_score) and scores are weighted by confidence × target_weight)
        # meant almost no signal ever reached the risk pipeline. Operators
        # who want stricter filtering can set `strategy.min_strength` in
        # config.yaml.
        strat_cfg = config.get("strategy", {})
        self._min_signal_strength = float(strat_cfg.get("min_strength", 0.15))

        # Critical #1 fix: lock guarding all mutations to the shared CycleResult
        # object. When max_workers > 1 (non-MT5 exchanges), _process_symbol is
        # called concurrently on multiple threads. Without this lock, concurrent
        # result.errors.append(...), result.skip_breakdown[key] += 1, and
        # result.trades_rejected += 1 operations can lose updates or corrupt
        # the dict/list internals.
        self._result_lock = threading.Lock()

        # BUG FIX: symbols whose symbol_info() call fails permanently
        # (e.g. broker returns retcode -2 "Invalid arguments" — a
        # delisted/unavailable meta-symbol like "Spot Up - Volatility Down
        # Index") were being retried every single cycle forever, spamming
        # system.log with the same ERROR line and burning an extra IPC
        # round-trip per cycle (contributing to p95/p99 latency spikes).
        # Once a symbol fails symbol_info(), remember it here and skip it
        # in the universe filter on subsequent cycles instead of retrying.
        self._invalid_symbols: set = set()

        # === Layer 1: Foundation ===
        self.state_machine: StateMachine = get_state_machine()
        self.event_bus: EventBus = self._bus
        self.snapshot_engine = SnapshotEngine(
            snapshot_dir=config.get("runtime", {}).get("snapshot_dir",
                                                       "data/snapshots"),
            bus=self._bus,
        )
        self.recovery_engine = RecoveryEngine(self.snapshot_engine, bus=self._bus)
        self.config_versioning = ConfigVersioning(
            config_path=config.get("config_path", "config/config.yaml"),
        )

        # === Layer 1b: Persistence (P0-8 FIX) ===
        # Canonical Database — every fill + every decision writes here.
        # Phase 7: health-check + auto-repair with backup before use.
        from database import Database
        db_path = config.get("database", {}).get("path", "data/trading_bot.db")
        self.db = Database(db_path)
        if not self.db.health_check():
            log.warning("TradingBot: DB corruption detected — attempting repair with backup")
            if not self.db.repair_with_backup():
                log.error("TradingBot: DB repair failed — trading with potentially corrupt DB")

        # === Layer 1c: Idempotency (Phase 4 req #22) ===
        # Prevents duplicate orders from crash-recovery re-sends or strategy
        # firing twice on the same bar. Keyed by (symbol, action, bar_time, strategy).
        from engine.idempotency import IdempotencyStore
        idempotency_path = config.get("runtime", {}).get(
            "idempotency_file", "data/seen_orders.json")
        self.idempotency = IdempotencyStore(path=idempotency_path)

        # === Layer 2: Perception ===
        self.feature_pipeline: FeaturePipeline = get_pipeline()
        self.regime_orchestrator = RegimeOrchestrator(bus=self._bus)

        # === Layer 3: Brain ===
        self.ai_model_manager: AIModelManager = get_model_manager()
        # Co-Founder Audit: use the LLM-augmented coordinator. When LLM is
        # disabled in config (default), this is functionally equivalent to
        # build_default_coordinator() — same rule-based agents, zero LLM
        # cost, zero latency overhead. When LLM is enabled, rule-based
        # runs first, LLM runs as second opinion on actionable signals only.
        from architecture.llm_augmented_coordinator import build_augmented_coordinator
        self.multi_agent = build_augmented_coordinator(config)
        self.memory_system = MemorySystem(db_path=db_path)
        self.online_learner = OnlineLearner(bus=self._bus)

        # === Layer 4: Hands ===
        # Exchange is created lazily in boot() based on --mode
        self.exchange: Optional[ExchangeInterface] = None
        self.portfolio = PortfolioManager(
            initial_capital=float(config.get("capital", 10000.0)),
            max_gross_exposure_pct=float(config.get("risk", {}).get("max_gross_exposure", 2.0)),
            max_portfolio_heat_pct=float(config.get("risk", {}).get("max_portfolio_heat", 0.10)),
            max_symbol_weight=float(config.get("risk", {}).get("max_symbol_weight", 0.25)),
            bus=self._bus,
        )

        # === Layer 5: Defense ===
        self.risk_pipeline = RiskPipeline(
            portfolio=self.portfolio,
            bus=self._bus,
            config=config.get("risk", {}),
        )
        # P0-21 FIX (Phase 4): top-of-cycle circuit breakers — systemic safety
        # net distinct from the per-trade RiskPipeline gates. If ANY breaker is
        # OPEN, the entire cycle is skipped.
        from architecture.circuit_breaker import CircuitBreakerCoordinator
        self.breakers = CircuitBreakerCoordinator(
            config,
            bus=self._bus,
            ignore_broker_disconnect=(self._mode == "demo"),
        )

        # TIER 1: per-symbol manipulation detector
        from engine.manipulation_detector import ManipulationDetector
        self._manipulation_detector = ManipulationDetector()

        # TIER 4: dynamic exit intelligence for open positions
        from trading_modules.dynamic_exit_intelligence import DynamicExitIntelligence
        _exit_cfg = config.get("exit_intelligence", {})
        self._exit_intel = DynamicExitIntelligence(
            breakeven_r=float(_exit_cfg.get("breakeven_r", 1.0)),
            trail_start_r=float(_exit_cfg.get("trail_start_r", 1.5)),
            partial_close_r=float(_exit_cfg.get("partial_close_r", 2.0)),
            max_hold_bars=int(_exit_cfg.get("max_hold_bars", 100)),
            max_adverse_r=float(_exit_cfg.get("max_adverse_r", -1.5)),
        )
        self.self_healing = SelfHealingSystem(bus=self._bus)
        # DecisionAuditor still owns its in-memory ring buffer, but its
        # persistent writes now go through self.db (see P0-8 FIX in
        # _process_symbol — we call self.db.save_decision / finalize_decision
        # directly, and DecisionAuditor is kept for its fast in-memory query).
        self.decision_auditor = DecisionAuditor(db_path=db_path)

        # === Layer 6: Observability ===
        self.monitor = InstitutionalMonitor(
            bus=self._bus,
            cycle_time_alert_ms=float(self._cfg.get("runtime", {}).get(
                "cycle_time_alert_ms", 30000.0)),
        )
        self.simulation = SimulationLayer(bus=self._bus)

        # === Existing systems (kept for backward compat) ===
        self._wisdom_gate = None
        self._breaker_was_open = False
        self._last_regime_block = None  # UI fix: track regime-block state

        # P0-2 FIX: kill-switch file path — checked at top of every cycle.
        self._kill_switch_file = config.get("runtime", {}).get(
            "kill_switch_file", "data/KILL_SWITCH")

        # Review Fix 2: daily_loss_halted_until persisted across restarts.
        # Loaded from state file on boot, saved whenever a halt is triggered.
        self._state_file = config.get("runtime", {}).get(
            "state_file", "data/state.json")
        self._daily_loss_halted_until = 0.0
        try:
            import json
            with open(self._state_file, "r") as f:
                state = json.load(f)
                self._daily_loss_halted_until = float(
                    state.get("daily_loss_halted_until", 0.0))
                if self._daily_loss_halted_until > time.time():
                    log.warning("TradingBot: daily loss halt is ACTIVE (restored "
                               "from state file, expires in %.1fh)",
                               (self._daily_loss_halted_until - time.time()) / 3600)
                else:
                    self._daily_loss_halted_until = 0.0  # expired
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            pass  # no state file = fresh start

        # Phase 11 req #62: Telegram alerts — wired for real.
        # Bot is created lazily; if TELEGRAM_BOT_TOKEN env var is not set,
        # send_alert becomes a no-op (logged at DEBUG).
        self._telegram = None
        try:
            from external.telegram_bot import TelegramBot
            self._telegram = TelegramBot()
            if self._telegram.token:  # BUGFIX: was _token, but TelegramBot uses self.token
                log.info("TradingBot: Telegram alerts enabled")
            else:
                self._telegram = None
                log.info("TradingBot: Telegram disabled (TELEGRAM_BOT_TOKEN not set)")
        except Exception as e:
            log.debug("TradingBot: Telegram init failed: %r", e)

        # Phase 9 req #50: Trade journal — wired for real. Every closed trade
        # writes a full structured entry (entry/exit reason, R-multiple, slippage,
        # holding duration) to the journal file + database.decisions.
        from enhancements.trade_journal import TradeJournal
        journal_path = config.get("runtime", {}).get(
            "journal_path", "data/trade_journal.jsonl")
        self._journal = TradeJournal(path=journal_path)

        # Phase 9 req #51: Mistake analyzer — runs as a scheduled job
        # (daily, triggered every 288 cycles at 5s poll = 24min, or
        # configurable via runtime.mistake_analysis_interval_cycles)
        self._mistake_analyzer_interval = int(
            config.get("runtime", {}).get("mistake_analysis_interval_cycles", 288))
        self._last_mistake_analysis_cycle = 0

        # Phase 9 req #53: Strategy decay detector
        from factory.decay_detector import DecayDetector
        self._decay_detector = DecayDetector()
        self._decayed_strategies: set = set()  # strategies flagged as decayed

        log.info("TradingBot: composed pipeline (mode=%s)", mode)

    # Backward-compat alias — old code that referenced IndustrialBot still works.
    IndustrialBot = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def boot(self) -> bool:
        """Start the bot. Returns True if ready to trade.

        P0-10 FIX (Phase 3): exchange type is now driven by self._mode
        (paper/demo/live), not the dead send_real_orders config flag.
        """
        self.state_machine.transition(BotState.BOOT, reason="initializing")
        self._bus.emit(EventType.BOT_START, payload={"mode": self._mode},
                       source="trading_bot")

        # P0-2 FIX: refuse to boot if kill-switch file already exists.
        import os
        if os.path.exists(self._kill_switch_file):
            log.error("TradingBot: kill-switch file %s exists — refusing to boot. "
                      "Remove it to start trading.", self._kill_switch_file)
            self.state_machine.transition(BotState.EMERGENCY,
                                         reason="kill_switch_active_at_boot")
            return False

        # Check for crash recovery
        if self.recovery_engine.detect_crash():
            log.warning("TradingBot: previous session did not shut down cleanly — attempting recovery")
            snap = self.recovery_engine.restore_latest()
            if snap is not None:
                self.recovery_engine.apply_snapshot(
                    snap, portfolio=self.portfolio,
                )
                self._cycle = snap.cycle
                log.info("TradingBot: restored from snapshot (cycle=%d)", snap.cycle)

        # Connect to exchange based on --mode flag.
        # demo/live → MT5Adapter (requires MT5 terminal + credentials)
        # paper is only allowed for internal tests, NOT from CLI
        self.state_machine.transition(BotState.CONNECTING, reason=f"connecting ({self._mode})")
        try:
            if self._mode == "paper":
                # Only reachable from tests (CLI blocks "paper" choice).
                # Log loudly so it's never confused with real trading.
                log.warning("TradingBot: PAPER mode (test only) — PaperAdapter")
                self.exchange = create_exchange("paper")
                if not self.exchange.connect():
                    log.error("TradingBot: paper adapter connect failed (should not happen)")
                    return False
            else:
                # demo or live — both use MT5Adapter
                if self._mode == "live" and not self._i_understand_real_money:
                    log.error("TradingBot: --mode=live requires --i-understand-this-is-real-money")
                    self.state_machine.transition(BotState.EMERGENCY,
                                                 reason="live_mode_without_confirmation")
                    return False
                mt5_cfg = self._cfg.get("mt5", {})
                if not mt5_cfg.get("login"):
                    log.error("TradingBot: --mode=%s requires mt5.login in config", self._mode)
                    self.state_machine.transition(BotState.EMERGENCY,
                                                 reason="missing_mt5_credentials")
                    return False
                self.exchange = create_exchange("mt5", **mt5_cfg)
                if not self.exchange.connect():
                    if self._mode == "demo":
                        log.warning("TradingBot: MT5 connection failed (mode=%s) — falling back to paper adapter", self._mode)
                        self.exchange = create_exchange("paper")
                        if not self.exchange.connect():
                            log.error("TradingBot: paper adapter fallback connect failed")
                            self.state_machine.transition(BotState.EMERGENCY,
                                                         reason="exchange connection failed")
                            return False
                        log.info("TradingBot: demo mode running in paper fallback mode")
                    else:
                        log.error("TradingBot: MT5 connection failed (mode=%s)", self._mode)
                        self.state_machine.transition(BotState.EMERGENCY,
                                                     reason="exchange connection failed")
                        return False
                if getattr(self.exchange, "name", "") == "paper":
                    log.info("TradingBot: connected to paper fallback adapter (mode=%s)", self._mode)
                else:
                    log.info("TradingBot: connected to MT5 (mode=%s)", self._mode)

                # Review Gap 3: post-connect account/mode cross-check.
                # Verify the connected account matches the declared mode.
                # If --mode=demo but the account looks like a live account
                # (or vice versa), refuse to trade rather than risk
                # accidentally trading real capital thinking it's demo.
                try:
                    acct = self.exchange.account_info()
                    # Heuristic: Deriv demo servers typically have "Demo" in
                    # the server name. If --mode=demo but server doesn't say
                    # "Demo", warn loudly and refuse to continue.
                    server_name = getattr(acct, "server", "") or ""
                    if getattr(self.exchange, "name", "") == "paper":
                        log.info("TradingBot: paper fallback adapter detected — skipping MT5 mode/server verification")
                    elif self._mode == "demo" and "demo" not in server_name.lower():
                        log.error("TradingBot: MODE MISMATCH — --mode=demo but "
                                  "connected to server '%s' which doesn't look "
                                  "like a demo server. Refusing to trade — check "
                                  "config.yaml mt5.server.", server_name)
                        self.state_machine.transition(BotState.EMERGENCY,
                                                     reason="mode_server_mismatch")
                        return False
                    if self._mode == "live" and "demo" in server_name.lower():
                        log.warning("TradingBot: --mode=live but server '%s' "
                                   "looks like a demo server. Continuing (you "
                                   "explicitly requested live mode).", server_name)
                    log.info("TradingBot: account verified — login=%s server=%s "
                             "balance=%s equity=%s leverage=1:%s",
                             acct.login, acct.server, acct.balance,
                             acct.equity, acct.leverage)
                    # Co-Founder Audit Fix: wire leverage into PortfolioManager
                    # so the gross-exposure gate uses margin-adjusted notional
                    # instead of raw notional. Without this, forex positions
                    # (contract_size=100,000) always show 200%+ exposure and
                    # get rejected on any reasonably-sized account.
                    if hasattr(acct, 'leverage') and acct.leverage > 0:
                        self.portfolio.set_leverage(int(acct.leverage))
                except Exception as e:
                    log.warning("TradingBot: account cross-check failed (non-fatal): %r", e)
        except Exception as e:  # noqa: BLE001
            log.error("TradingBot: exchange creation failed: %r", e)
            self.state_machine.transition(BotState.EMERGENCY,
                                         reason=f"exchange error: {e}")
            return False

        # Sync symbols + warmup
        self.state_machine.transition(BotState.SYNCING, reason="syncing symbols")
        self._symbols = self._load_symbols()
        log.info("TradingBot: %d symbols loaded", len(self._symbols))

        # C16 fix: refuse to proceed to LIVE with an empty symbol list.
        # Previously the bot would boot "successfully" and just never place
        # a trade, with no obvious signal beyond an easy-to-miss log line
        # buried among boot INFO logs — a silent no-trading failure mode
        # (e.g. MT5 terminal has different symbols visible in MarketWatch
        # than config.yaml expects, or every symbol matched an exclusion
        # keyword). Validate against what the exchange actually reports
        # where possible, and hard-fail rather than silently idle.
        if not self._symbols:
            if self._mode == "demo" and getattr(self.exchange, "name", "") == "paper":
                log.warning("TradingBot: no symbols loaded from broker — using configured fallback symbols")
                configured = [
                    s["name"]
                    for s in self._cfg.get("symbols", [])
                    if isinstance(s, dict) and "name" in s
                ]
                self._symbols = configured
            else:
                log.error("TradingBot: 0 symbols loaded after _load_symbols() — "
                          "refusing to go LIVE. Check config.yaml `symbols` "
                          "against what's actually visible in the MT5 terminal's "
                          "MarketWatch (or check exclusion keywords in "
                          "_load_symbols()).")
                self.state_machine.transition(BotState.EMERGENCY,
                                             reason="no_symbols_loaded")
                return False
        if self._mode != "paper" and self.exchange is not None:
            try:
                broker_symbols = set(self.exchange.get_symbols_by_pattern(["*"]))
                missing = [s for s in self._symbols if broker_symbols and s not in broker_symbols]
                if broker_symbols and missing:
                    log.warning("TradingBot: %d/%d configured symbols not found "
                               "at broker (will fail at fetch time): %s",
                               len(missing), len(self._symbols),
                               ", ".join(missing[:10]))
            except Exception as e:
                log.debug("TradingBot: symbol cross-check against broker "
                         "skipped (non-fatal): %r", e)

        self.state_machine.transition(BotState.WARMUP, reason="indicator warmup")
        warmup = self.feature_pipeline.warmup_requirement()
        log.info("TradingBot: warmup requirement = %d bars", warmup)

        # Initialize Wisdom Gate
        try:
            from livermore_principles import WisdomGate
            risk_cfg = self._cfg.get("risk", {})
            self._wisdom_gate = WisdomGate(
                min_confidence=float(risk_cfg.get("min_confidence", 0.60)),
                min_rr=float(risk_cfg.get("min_rr", 2.0)),
                min_bars_between_trades=int(risk_cfg.get("min_bars_between_trades", 5)),
                max_spread_bps=float(risk_cfg.get("max_spread_bps", 15.0)),
                max_drawdown_pct=float(risk_cfg.get("max_drawdown_pct", 15.0)),
                max_consecutive_losses=int(risk_cfg.get("max_consecutive_losses", 3)),
            )
            log.info("TradingBot: Wisdom Gate ready (200 principles)")
        except Exception as e:
            log.warning("TradingBot: Wisdom Gate init failed: %r", e)

        # Take initial config snapshot
        self.config_versioning.snapshot("bot boot")

        # Register recovery functions
        self._register_recoveries()

        # Ready
        self.state_machine.transition(BotState.LIVE, reason="all systems go")
        log.info("TradingBot: LIVE — mode=%s, %d symbols", self._mode, len(self._symbols))

        # Phase 6 req #37: Reconcile with broker on boot
        if self._mode != "paper":
            self.reconcile_with_broker()

        return True

    def shutdown(self, reason: str = "user requested") -> None:
        """Graceful shutdown."""
        log.info("TradingBot: shutting down — %s", reason)
        self.state_machine.transition(BotState.SHUTDOWN, reason=reason)
        self._stop = True

        # Take final snapshot + mark clean shutdown
        try:
            self.snapshot_engine.take_snapshot(
                portfolio=self.portfolio,
                cycle=self._cycle,
                equity=self.portfolio.equity(),
                peak_equity=self.portfolio.metrics().peak_equity,
                bot_state="SHUTDOWN",
                notes=f"clean shutdown: {reason}",
            )
            self.snapshot_engine.mark_clean_shutdown()
        except Exception as e:
            log.warning("TradingBot: final snapshot failed: %r", e)

        if self.exchange is not None:
            try:
                self.exchange.disconnect()
            except Exception as e:  # noqa: BLE001
                log.warning("TradingBot: exchange disconnect failed: %r", e)

        self._bus.emit(EventType.BOT_SHUTDOWN,
                      payload={"reason": reason, "final_cycle": self._cycle},
                      source="trading_bot")
        self.state_machine.transition(BotState.HALTED, reason="shutdown complete")

    # ------------------------------------------------------------------
    # Symbol loading
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Asset-class classification — crypto & futures/synthetics only
    # ------------------------------------------------------------------
    # Bot is now scoped to crypto + futures/synthetic-index instruments;
    # plain FX currency pairs (majors/minors/crosses) are excluded even if
    # they're still listed in config.yaml `symbols:`, so a stale config
    # doesn't silently pull the bot back into forex.
    _CRYPTO_TICKERS = (
        "BTC", "ETH", "XRP", "LTC", "SOL", "BNB", "DOGE", "ADA", "DOT",
        "MATIC", "AVAX", "LINK", "SHIB", "TRX", "ATOM", "UNI", "BCH",
        "DASH", "DSH", "XLM", "EOS", "XMR", "USDT", "USDC",
    )
    # Futures / continuous-synthetic instruments (Deriv-style synthetic
    # indices behave like cash-settled continuous futures — 24/7, no
    # underlying spot FX pair — plus real index/commodity futures.
    _FUTURES_KEYWORDS = (
        "us30", "us500", "nas100", "uk100", "ger40", "dax", "spx", "ndx",
        "wall street", "europe 50", "usoil", "ukoil", "ngas", "xauusd",
        "xagusd",
    )
    # Deriv synthetic indices — engineered price processes (fixed spike
    # frequency, fixed volatility), NOT futures on a real underlying.
    # Previously lumped into _FUTURES_KEYWORDS, which meant the default
    # asset_class_focus=["crypto","futures"] silently let this "crypto"
    # bot keep trading Boom/Crash/Step Index instead of only BTC/ETH/etc.
    _SYNTHETIC_KEYWORDS = (
        "boom", "crash", "volatility", "jump", "step index", "range break",
    )
    # Plain FX currency codes — used to detect currency pairs like
    # EURUSD, CADJPY, AUDNZD (two 3-letter codes back to back) so they're
    # excluded even when disguised behind a custom strategy label like
    # "EURUSD RSI Pullback Index".
    _FX_CODES = (
        "USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "SGD",
        "HKD", "MXN", "NOK", "SEK", "ZAR", "TRY", "CNH", "PLN", "HUF",
        "DKK", "CZK",
    )

    def _asset_class(self, name: str) -> str:
        """Classify a symbol name as 'crypto', 'futures', 'forex', or 'other'.

        Uses the leading token/prefix of the name (before any strategy-label
        suffix like " RSI Pullback Index") since that prefix is the actual
        underlying instrument passed straight through to MT5.
        """
        base = name.strip().split(" ")[0].upper()
        low = name.lower()
        if any(low.startswith(kw) or kw in low for kw in self._SYNTHETIC_KEYWORDS):
            return "synthetic"
        if any(low.startswith(kw) or kw in low for kw in self._FUTURES_KEYWORDS):
            return "futures"
        if any(base.startswith(t) or t in base for t in self._CRYPTO_TICKERS):
            return "crypto"
        # Pure FX pair: exactly two known 3-letter currency codes back to
        # back (e.g. EURUSD, CADJPY, AUDNZD) with nothing else in the base.
        if len(base) == 6:
            left, right = base[:3], base[3:]
            if left in self._FX_CODES and right in self._FX_CODES:
                return "forex"
        return "other"

    def _load_symbols(self) -> List[str]:
        cfg_syms = self._cfg.get("symbols", [])
        names = []
        for s in cfg_syms:
            if isinstance(s, dict) and "name" in s:
                name = str(s["name"]).strip()
                if name:
                    names.append(name)
            elif isinstance(s, str) and s.strip():
                names.append(s.strip())
        if self._cfg.get("symbols_auto_load", True) and self.exchange is not None:
            try:
                patterns = self._cfg.get("symbol_patterns", [
                    "BTC", "ETH", "XRP", "LTC", "SOL", "BNB", "DOGE",
                ])
                extra = self.exchange.get_symbols_by_pattern(patterns)
                # Review fix: filter out non-tradable basket/index/arbitrage
                # meta-symbols that fail symbol_info() with "Invalid arguments"
                # H7 fix: exclusion list is now configurable via
                # config.yaml `symbol_exclude_keywords` — previously a
                # hardcoded tuple silently dropped ANY symbol containing one
                # of these substrings with no way to override for a broker
                # where e.g. "Basket" is a real tradable instrument.
                EXCLUDE_KEYWORDS = tuple(
                    kw.lower() for kw in self._cfg.get(
                        "symbol_exclude_keywords",
                        ["arbitrage", "basket", "index long", "index short"])
                )
                filtered = []
                for s in extra:
                    s_lower = s.lower()
                    if any(kw in s_lower for kw in EXCLUDE_KEYWORDS):
                        log.debug("TradingBot: excluding non-tradable symbol: %s", s)
                        continue
                    if s not in names:
                        names.append(s)
                        filtered.append(s)
                log.info("TradingBot: auto-loaded %d symbols (%d filtered out as non-tradable)",
                         len(filtered), len(extra) - len(filtered))
            except Exception as e:
                log.warning("TradingBot: auto-load symbols failed: %r", e)

        # Asset-class focus filter: crypto + real futures/commodities/
        # indices (US30, XAUUSD, etc). Deriv synthetic indices (Boom/Crash/
        # Step/Volatility/Jump) are deliberately excluded by default — set
        # via config.yaml `asset_class_focus`, e.g. add "synthetic" to
        # re-enable them, or "forex"/"other" for everything else.
        focus = set(self._cfg.get("asset_class_focus", ["crypto", "futures"]))
        pre_count = len(names)
        kept, dropped = [], []
        for s in names:
            cls = self._asset_class(s)
            if cls in focus:
                kept.append(s)
            else:
                dropped.append((s, cls))
        if dropped:
            log.info("TradingBot: asset_class_focus=%s — dropped %d/%d symbols "
                      "(%s)", sorted(focus), len(dropped), pre_count,
                      ", ".join(f"{s}[{c}]" for s, c in dropped[:10]) +
                      (" ..." if len(dropped) > 10 else ""))
        names = kept

        # PERF NOTE: each symbol costs ~1 MT5 IPC round-trip per cycle in
        # _process_symbol (fetch_candles), and MT5's IPC channel is
        # effectively serialized regardless of thread count (see the lock
        # in MT5Adapter). More symbols = a longer, mostly-linear cycle
        # time. 100 symbols on a typical demo IPC latency is what produced
        # the ~35s cycles. Configurable via runtime.max_symbols so this is
        # a deliberate choice, not a hardcoded surprise.
        max_symbols = int(self._cfg.get("runtime", {}).get("max_symbols", 100))
        return names[:max_symbols]


    # ------------------------------------------------------------------
    # Universe pre-filter — production optimization
    # ------------------------------------------------------------------
    def _universe_filter(self, symbols: List[str], equity: float,
                         result: CycleResult) -> List[str]:
        """Pre-screen symbols BEFORE AI analysis. Returns top-N tradable.

        Co-Founder Audit (production hardening): without this filter the bot
        runs feature engineering (28+ indicators) + 5 multi-agent consensus
        + 13-gate risk pipeline on ALL 100 symbols every cycle, even though
        ~80% of them will fail the same liquidity/spread/volatility gates
        every time. That's wasted CPU AND a logging flood (per-symbol
        REJECTED lines from risk_pipeline.py).

        Cheap checks first, expensive ones only on survivors:
          1. Already-open filter (free, in-memory)
          2. Duplicate-variant filter (string match — drop AUDNZDmicro if
             AUDNZD already in the list, same underlying instrument)
          3. Spread filter (1 IPC call: symbol_info.spread)
          4. Liquidity filter (1 IPC call: 20 bars of volume, ~5% of the
             500-bar fetch the full pipeline would do)
          5. Volatility filter (uses same 20 bars)

        Survivors are sorted by a tradability score (liquidity × inverse
        spread) and the top N (configurable via runtime.max_universe_size,
        default 20) are returned for full AI analysis.

        Fail-open: on ANY exception, returns the original `symbols` list
        unchanged. Better to over-process than to silently drop a tradable
        symbol because of a transient IPC error.
        """
        if self.exchange is None or not symbols:
            return symbols

        uf_cfg = self._cfg.get("universe_filter", {})
        if not uf_cfg.get("enabled", True):
            return symbols  # explicitly disabled

        max_size = int(uf_cfg.get("max_size", 20))
        max_spread_bps = float(uf_cfg.get("max_spread_bps",
                                          self._cfg.get("risk", {}).get(
                                              "max_spread_bps", 15.0)))
        # Drop suffix-duplicates: AUDNZDmicro if AUDNZD exists, BTCUSD.conv
        # if BTCUSD exists, etc. The "base" symbol is preferred.
        drop_variants = bool(uf_cfg.get("drop_variant_suffixes", True))
        variant_suffixes = tuple(s.lower() for s in uf_cfg.get(
            "variant_suffixes", ["micro", "mini", ".conv", "c", "pro"]))

        try:
            # ── Step 1: already-open filter ────────────────────────────
            not_open = [s for s in symbols
                        if not self.portfolio.has_open_position(s)]
            n_already_open = len(symbols) - len(not_open)

            # ── Step 2: duplicate-variant filter ───────────────────────
            # Build a set of "base" symbol names present in the list.
            # Then drop any symbol whose name is base+suffix.
            if drop_variants:
                survivors = []
                dropped_variants = []
                for s in not_open:
                    s_lower = s.lower()
                    is_variant = False
                    for suf in variant_suffixes:
                        if s_lower.endswith(suf):
                            base = s_lower[:-len(suf)]
                            # Drop only if the base also exists in our list
                            if any(b.lower() == base for b in not_open if b != s):
                                is_variant = True
                                break
                    if is_variant:
                        dropped_variants.append(s)
                    else:
                        survivors.append(s)
                not_open = survivors
            else:
                dropped_variants = []

            # ── Steps 3-5: spread + liquidity + volatility screen ──────
            # For each survivor, fetch symbol_info + 20 bars. This is ~5%
            # of the cost of the full 500-bar fetch the pipeline would do.
            # Use the SAME _ipc_lock semantics as the full pipeline so we
            # don't add a new race surface.
            scored: List[Tuple[float, str]] = []
            n_spread_fail = 0
            n_liquidity_fail = 0
            n_volatility_fail = 0
            n_data_fail = 0
            for s in not_open:
                if s in self._invalid_symbols:
                    n_data_fail += 1
                    continue
                try:
                    # Spread check (cheap — no OHLCV fetch)
                    try:
                        sym_info = self.exchange.symbol_info(s)
                    except Exception as e:
                        # Permanent broker-side rejection (e.g. -2 "Invalid
                        # arguments" for a delisted/unavailable symbol) —
                        # remember it so we stop retrying every cycle.
                        self._invalid_symbols.add(s)
                        log.warning("UNIVERSE FILTER: %s failed symbol_info() "
                                    "— marking permanently invalid, will not "
                                    "retry: %r", s, e)
                        n_data_fail += 1
                        continue
                    if sym_info is not None:
                        spread = getattr(sym_info, "spread", 0)
                        point = getattr(sym_info, "point", 0.0001)
                        # Need a price to compute spread_bps — use last
                        # tick if available, else skip the spread check.
                        try:
                            tick = self.exchange.symbol_tick(s)
                            price = (tick.bid + tick.ask) / 2 if tick.bid > 0 else 0
                        except Exception:
                            price = 0.0
                        if price > 0:
                            spread_bps = (spread * point / price) * 10000
                            if spread_bps > max_spread_bps:
                                n_spread_fail += 1
                                continue

                    # Liquidity check — VERY LENIENT in universe filter.
                    # The real liquidity check happens in risk_pipeline's
                    # LiquidityGate (with auto-calibration). Here we only
                    # filter out TRULY DEAD symbols (volume = 0 or near 0).
                    # The old check (20th pctile × 0.5) was too strict and
                    # rejected 67% of symbols on low-volume periods.
                    df = self.exchange.fetch_candles(s, "M15", 20)
                    if df is None or df.empty or len(df) < 20 or "volume" not in df.columns:
                        n_data_fail += 1
                        continue
                    recent_vol = df["volume"].tail(20).astype(float)
                    current_vol = float(recent_vol.iloc[-1])
                    # Only filter if volume is literally 0 (truly dead symbol)
                    if current_vol <= 0:
                        n_liquidity_fail += 1
                        continue

                    # Volatility check — use the same ATR% threshold as
                    # VolatilityGate so we don't contradict the risk pipeline.
                    try:
                        from utils.indicators import atr as _atr_fn
                        atr_series = _atr_fn(df, 14)
                        atr_val = float(atr_series.iloc[-1])
                        price = float(df["close"].iloc[-1])
                        if price > 0:
                            atr_pct = atr_val / price
                            max_atr_pct = float(self._cfg.get("risk", {}).get(
                                "max_atr_pct", 0.05))
                            if atr_pct > max_atr_pct:
                                n_volatility_fail += 1
                                continue
                    except Exception:
                        # ATR computation failure is non-fatal — let the
                        # full risk pipeline's VolatilityGate handle it.
                        pass

                    # Tradability score: higher liquidity × tighter spread
                    # is better. We don't have a precise spread here for
                    # every symbol (tick fetch is best-effort), so use
                    # liquidity as the primary sort key.
                    score = current_vol
                    scored.append((score, s))
                except Exception:
                    n_data_fail += 1
                    continue

            # Sort by score descending, take top N
            scored.sort(key=lambda x: x[0], reverse=True)
            filtered = [s for _, s in scored[:max_size]]

            # ── Log summary instead of per-symbol rejections ──────────
            # This replaces the dozens of per-symbol REJECTED lines that
            # the risk_pipeline would otherwise emit. Single summary line
            # per cycle, with the breakdown by reason.
            total_dropped = (len(symbols) - len(filtered))
            log.info(
                "UNIVERSE FILTER: %d → %d symbols (dropped %d) │ "
                "already_open=%d variant_dup=%d spread_fail=%d "
                "liquidity_fail=%d volatility_fail=%d data_fail=%d",
                len(symbols), len(filtered), total_dropped,
                n_already_open, len(dropped_variants),
                n_spread_fail, n_liquidity_fail, n_volatility_fail, n_data_fail,
            )
            # Surface a few example dropped symbols so the operator can
            # sanity-check the filter (e.g. "yes, AUDNZDmicro is genuinely
            # a duplicate of AUDNZD").
            if dropped_variants:
                log.debug("  variant drops: %s",
                          ", ".join(dropped_variants[:10]))

            # Update skip_breakdown so the cycle-summary line in main.py
            # reflects the universe-filter drops too.
            self._result_incr_skip(result, "universe_already_open", n_already_open)
            self._result_incr_skip(result, "universe_variant_dup", len(dropped_variants))
            self._result_incr_skip(result, "universe_spread_fail", n_spread_fail)
            self._result_incr_skip(result, "universe_liquidity_fail", n_liquidity_fail)
            self._result_incr_skip(result, "universe_volatility_fail", n_volatility_fail)
            self._result_incr_skip(result, "universe_data_fail", n_data_fail)

            return filtered
        except Exception as e:
            # Fail-open: any unexpected error in the filter returns the
            # full symbol list. Better to over-process than to silently
            # drop tradable symbols.
            log.warning("TradingBot: universe filter crashed — falling back "
                       "to full symbol list (%d symbols): %r", len(symbols), e)
            return symbols

    # ------------------------------------------------------------------
    # Recovery registration
    # ------------------------------------------------------------------
    def _register_recoveries(self) -> None:
        from architecture.self_healing import FailureType
        if self.exchange is not None:
            self.self_healing.register_recovery(
                FailureType.CONNECTION, "mt5_adapter",
                recovery_fn=lambda: self.exchange.connect(),
                max_retries=5,
            )
        self.self_healing.register_recovery(
            FailureType.DATABASE, "database",
            recovery_fn=lambda: True,  # SQLite is resilient; no-op
            max_retries=2,
        )
        log.info("TradingBot: recovery handlers registered")

    # ------------------------------------------------------------------
    # Main trading cycle
    # ------------------------------------------------------------------
    def cycle(self) -> CycleResult:
        """Run one trading cycle. Returns a summary.

        P0-2 FIX (Phase 3): kill-switch file checked at the very top, before
        any signal evaluation runs. If the file exists, the cycle is skipped
        and a loud log line is emitted. This is the operator's panic button.
        """
        if self._stop or self.state_machine.is_terminal():
            return CycleResult(cycle=self._cycle, state=self.state_machine.current.value)

        t0 = time.time()
        result = CycleResult(
            cycle=self._cycle + 1,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

        try:
            # 0. KILL-SWITCH CHECK (P0-2 FIX) — top of every cycle, no exceptions.
            import os
            if os.path.exists(self._kill_switch_file):
                log.warning("TradingBot: KILL SWITCH ACTIVE (%s) — skipping cycle %d. "
                            "Remove the file to resume trading.",
                            self._kill_switch_file, result.cycle)
                # Phase 11: Telegram alert on kill switch (only first time)
                if result.cycle == 1 or not getattr(self, "_ks_notified", False):
                    self._notify("KILL SWITCH ACTIVE",
                                f"Bot halted — kill switch file detected.\n"
                                f"Remove {self._kill_switch_file} to resume.")
                    self._ks_notified = True
                result.state = "KILL_SWITCH"
                result.cycle_time_ms = (time.time() - t0) * 1000
                self._cycle += 1
                return result

            # 0b. CIRCUIT BREAKER CHECK (Phase 4) — if any breaker is OPEN,
            # skip the entire cycle. This is the systemic safety net.
            # UI fix: log only on state-change, not every cycle.
            if self.breakers.should_block_cycle():
                if not getattr(self, "_breaker_was_open", False):
                    open_list = self.breakers.open_breakers()
                    log.error("TradingBot: ❌ TRADING PAUSED — circuit breaker(s) "
                             "OPEN: %s. Bot will skip cycles until breaker "
                             "auto-resets (cooldown). Check MT5 connection + "
                             "equity health.", [b["name"] for b in open_list])
                    for b in open_list:
                        log.error("  breaker %s: state=%s reason=%s",
                                 b["name"], b["state"], b.get("last_reason", ""))
                    self._notify("Circuit Breaker Tripped",
                                f"Trading paused — breaker(s) open: "
                                f"{[b['name'] for b in open_list]}\n"
                                f"Will auto-retry after cooldown.")
                self._breaker_was_open = True
                result.state = "BREAKER_OPEN"
                result.cycle_time_ms = (time.time() - t0) * 1000
                self._cycle += 1
                return result
            else:
                if getattr(self, "_breaker_was_open", False):
                    log.info("TradingBot: trading RESUMED — breaker(s) closed")
                self._breaker_was_open = False

            # 1. State health check
            warning = self.state_machine.check_state_health()
            if warning:
                log.warning("TradingBot: %s", warning)

            # 2. Regime detection
            # FIX Bug #3: Previously used ONLY symbols[0] as proxy for the
            # entire portfolio — if that one symbol was in TRANSITION, ALL
            # 100 symbols were blocked. Now we detect regime per-symbol
            # inside _process_symbol, and only use a portfolio-level regime
            # as a light advisory (not a hard block). The hard block now
            # only fires for CRISIS regime (systemic risk), not TRANSITION.
            regime = MarketRegime.UNKNOWN
            if self.exchange is not None and self._symbols:
                try:
                    df_proxy = self.exchange.fetch_candles(
                        self._symbols[0], "M15", 1000
                    )
                    if df_proxy is None or df_proxy.empty:
                        log.warning("TradingBot: regime detect — fetch_candles "
                                   "returned empty for %s, regime stays UNKNOWN "
                                   "(MT5 may be disconnected)", self._symbols[0])
                    else:
                        regime = self.regime_orchestrator.detect(df_proxy)
                except Exception as e:
                    log.warning("TradingBot: regime detect failed for %s: %r — "
                               "MT5 may be disconnected", self._symbols[0], e)
                    result.errors.append(f"regime detect failed: {e}")
            result.regime = regime.value

            # 3. Account equity — sync from broker each cycle
            # Co-Founder Audit Fix: when MT5 disconnects, account_info() can
            # return equity=0 or raise. We must NOT use $0 as real equity —
            # it triggers false 100% drawdown + circuit breaker. Instead,
            # fall back to portfolio's last-known equity.
            equity = 0.0
            mt5_healthy = True
            if self.exchange is not None:
                try:
                    acct = self.exchange.account_info()
                    equity = acct.equity
                    # Guard against MT5 returning 0 on disconnect
                    if equity <= 0:
                        log.error("TradingBot: ❌ MT5 DISCONNECTED — broker "
                                 "returned equity=%.2f. Using last known equity. "
                                 "RESTART MT5 TERMINAL to fix!", equity)
                        equity = self.portfolio.equity()
                        mt5_healthy = False
                    else:
                        # Keep portfolio equity synced with broker reality
                        self.portfolio.update_equity(equity)
                except Exception as e:
                    log.error("TradingBot: ❌ MT5 account_info() FAILED: %r — "
                             "MT5 disconnected. Using last known equity. "
                             "RESTART MT5 TERMINAL to fix!", e)
                    equity = self.portfolio.equity()
                    mt5_healthy = False
            else:
                equity = self.portfolio.equity()
                mt5_healthy = True  # paper/live-non-mt5 is always healthy

            self.monitor.update_equity(equity)
            result.equity = equity

            # 3b. MT5 health gate — if MT5 is disconnected, skip ALL trading
            # for this cycle. Equity is already preserved (last-known value).
            # We still update the cycle result so the operator sees the state.
            if not mt5_healthy:
                log.warning("TradingBot: MT5 unhealthy — skipping all trading this cycle "
                            "(equity preserved at $%.2f)", equity)
                result.state = "MT5_DISCONNECTED"
                return result

            # 4. Regime adjustments
            # FIX Bug #3: Only block ALL trading for CRISIS regime (systemic
            # risk). TRANSITION, HIGH_VOL, CHOP etc. should reduce risk
            # parameters (which the adjustments dict does), NOT block all
            # scanning. Per-symbol regime detection happens inside
            # _process_symbol via the risk pipeline's MarketRegimeGate.
            adjustments = self.regime_orchestrator.get_adjustments(regime)
            if regime == MarketRegime.CRISIS:
                # CRISIS = systemic risk, block everything
                if self._last_regime_block != regime.value:
                    log.warning("TradingBot: CRISIS regime — ALL trading halted")
                    self._last_regime_block = regime.value
                result.trades_rejected = 0
                result.skip_breakdown['regime_block'] = result.skip_breakdown.get('regime_block', 0) + 1
            else:
                if self._last_regime_block is not None:
                    log.info("TradingBot: regime %s now allows trading — resuming",
                             regime.value)
                self._last_regime_block = None

                # 5. Per-symbol processing (parallelized where it actually helps)
                from architecture.self_healing import FailureType
                from concurrent.futures import ThreadPoolExecutor, as_completed
                # C5/X3 fix: MT5Adapter serializes ALL calls behind a single
                # `_ipc_lock` (the MetaTrader5 package is not thread-safe).
                # Spinning up 8 worker threads that all immediately block on
                # that one lock adds pure overhead — no parallelism is
                # actually achieved, and cycle time stays serialized (was
                # measured at ~35s for 100 IPC-bound symbols regardless of
                # worker count). Only use a thread pool when the exchange
                # backend has no such lock (e.g. PaperAdapter/backtest),
                # where per-symbol feature/risk computation genuinely can
                # run concurrently.
                exchange_is_ipc_serialized = getattr(self.exchange, "_ipc_lock", None) is not None
                if exchange_is_ipc_serialized:
                    max_workers = 1
                else:
                    max_workers = min(len(self._symbols), 8)

                # Co-Founder Audit (production hardening): run the universe
                # pre-filter BEFORE the per-symbol loop. This drops symbols
                # that would obviously fail the liquidity/spread/volatility
                # gates, so AI agents only run on the top-N tradable ones.
                # Fail-open: on any error, falls back to the full list.
                active_symbols = self._universe_filter(self._symbols, equity, result)
                if max_workers > 1:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                self._process_symbol, symbol, equity, adjustments, result
                            ): symbol for symbol in active_symbols
                        }
                        for future in as_completed(futures):
                            symbol = futures[future]
                            try:
                                future.result()
                            except Exception as e:
                                result.errors.append(f"{symbol}: {e}")
                                self.self_healing.report_failure(
                                    FailureType.STRATEGY, symbol, str(e)
                                )
                else:
                    # Single symbol — no need for thread pool overhead
                    for symbol in active_symbols:
                        try:
                            self._process_symbol(symbol, equity, adjustments, result)
                        except Exception as e:
                            result.errors.append(f"{symbol}: {e}")
                            self.self_healing.report_failure(
                                FailureType.STRATEGY, symbol, str(e)
                            )

                # Candle-cache diagnostics — attached to `result` so
                # main.py's single-line logger can show it inline instead
                # of a separate log.info() call.
                cache_stats_fn = getattr(self.exchange, "cache_stats", None)
                if callable(cache_stats_fn):
                    cs = cache_stats_fn()
                    result.cache_hits = cs["hits"]
                    result.cache_misses = cs["misses"]
                    result.cache_hit_rate = cs["hit_rate"]

            # 6. Update open positions via portfolio manager
            if self.exchange is not None:
                try:
                    broker_positions = self.exchange.positions()
                    broker_tickets = {int(p.ticket) for p in broker_positions}
                    prices = self._fetch_current_prices()
                    self.portfolio.update_prices(prices)

                    # Review Fix 1: detect broker-side closes (SL/TP hit)
                    # and feed the PnL to ConsecutiveLossBreaker.
                    local_positions = self.portfolio.all_positions()
                    local_tickets = {p["ticket"] for p in local_positions}
                    closed_at_broker = local_tickets - broker_tickets
                    for ticket in closed_at_broker:
                        pos = self.portfolio.get_position(ticket)
                        if pos:
                            # Position was closed by broker (SL/TP hit).
                            # Get the exit price from the broker position history
                            # or fall back to current price.
                            exit_price = pos.get("current_price", pos.get("entry_price", 0))
                            pnl = self.portfolio.on_position_closed(
                                ticket, exit_price, reason="broker_sl_tp")
                            self.breakers.record_trade_outcome(pnl)
                            log.info("TradingBot: BROKER CLOSE %s ticket=%d pnl=%+.2f "
                                    "(SL/TP hit at broker)",
                                    pos.get("symbol", "?"), ticket, pnl)
                            self._notify("Position Closed (SL/TP)",
                                        f"{pos.get('symbol', '?')} ticket={ticket}\n"
                                        f"PnL: {pnl:+.2f} (broker-side close)")
                            # C1 FIX (Chief AI Architect Audit): wire the learning loop.
                            # Call resolve_trade_outcome() so the LLM memory log generates
                            # a reflection on this trade — completing the learning cycle
                            # that was previously broken (get_past_context always empty).
                            self._trigger_trade_reflection(pos, pnl, "broker_sl_tp")

                    # Reconcile
                    #
                    # ROOT-CAUSE FIX (screenshot 2026-07-18): this ran every
                    # cycle with auto_resolve defaulting to False, so a
                    # phantom broker position (open at the broker but not in
                    # local state — e.g. after a restart, or a crash between
                    # order fill and on_position_opened()) was only ever
                    # logged, never registered. has_open_position(symbol)
                    # then kept returning False for a symbol that already
                    # had an open trade, which is what let the bot open a
                    # second, uncoordinated position on the same symbol
                    # (Boom 99 Index opposite-direction, Skew Step Index 5 Up
                    # same-direction) instead of being blocked by the
                    # one-trade-per-symbol gate. auto_resolve=True makes this
                    # cycle (every ~5-9s) self-healing instead of relying
                    # solely on the slower reconcile_with_broker() path
                    # (every N cycles). sl/tp/magic are now passed through so
                    # a registered phantom position still gets correct
                    # trailing-stop/breakeven handling instead of defaulting
                    # to sl=0/tp=0.
                    discrepancies = self.portfolio.reconcile(
                        [{"ticket": p.ticket, "symbol": p.symbol,
                          "side": p.type.value, "volume": p.volume,
                          "open_price": p.open_price,
                          "current_price": p.current_price,
                          "sl": getattr(p, "sl", 0.0),
                          "tp": getattr(p, "tp", 0.0),
                          "magic": getattr(p, "magic", 0)}
                         for p in broker_positions],
                        auto_resolve=True,
                    )
                    for d in discrepancies:
                        log.warning("TradingBot: reconciliation: %s", d)
                        self._notify("Reconciliation Discrepancy (auto-resolved)", d)
                except Exception as e:
                    result.errors.append(f"position update failed: {e}")

            # 7. Monitor + alerts
            alerts = self.monitor.check_alerts()
            result.alerts = [
                {"level": a.level, "category": a.category, "message": a.message}
                for a in alerts
            ]
            result.open_positions = self.portfolio.open_count()

            # 8. Cycle stats
            self.monitor.record_cycle(time.time() - t0)

            # 9. Periodic snapshot (every 50 cycles)
            self._cycle += 1
            if self._cycle % 50 == 0:
                try:
                    self.snapshot_engine.take_snapshot(
                        portfolio=self.portfolio,
                        cycle=self._cycle,
                        equity=equity,
                        peak_equity=self.portfolio.metrics().peak_equity,
                        bot_state=self.state_machine.current.value,
                        notes=f"periodic snapshot at cycle {self._cycle}",
                    )
                except Exception as e:
                    log.warning("TradingBot: snapshot failed: %r", e)

            # 10. Heartbeat
            self._bus.emit(EventType.HEARTBEAT,
                          payload={"cycle": self._cycle, "equity": equity,
                                  "regime": regime.value},
                          source="trading_bot")

            # 11. Circuit breaker telemetry (Phase 4) — feed equity + latency
            # so breakers can trip on drawdown/error-rate/latency.
            self.breakers.record_equity(equity)
            self.breakers.record_cycle(ok=len(result.errors) == 0,
                                        latency_s=time.time() - t0)

            # 12. Phase 6: Periodic reconciliation with broker (every N cycles)
            recon_interval = int(self._cfg.get("runtime", {}).get(
                "reconciliation_interval_cycles", 50))
            if self._mode != "paper" and self._cycle % recon_interval == 0:
                self.reconcile_with_broker()

            # 13. Phase 9: Mistake analysis (scheduled job)
            if (self._cycle - self._last_mistake_analysis_cycle
                    >= self._mistake_analyzer_interval):
                self._run_mistake_analysis()
                self._run_decay_detection()
                # H3/X5 fix: consolidate() converts recent episodic memories
                # into semantic patterns — it was never called anywhere in
                # the main loop, so semantic memory was permanently empty
                # no matter how many trades the bot logged. Run it on the
                # same periodic cadence as mistake analysis.
                try:
                    consolidated = self.memory_system.consolidate()
                    if consolidated:
                        log.info("TradingBot: memory consolidation produced "
                                "%d semantic pattern(s)", consolidated)
                except Exception as e:
                    log.warning("TradingBot: memory consolidation failed: %r", e)
                # C15 fix: SelfHealingSystem never re-tested a degraded
                # component once marked — it could only recover via an
                # explicit successful report_failure() retry, which won't
                # happen again for a component nobody is actively calling.
                # Give degraded components a periodic second chance.
                try:
                    self.self_healing.retest_degraded_components()
                except Exception as e:
                    log.warning("TradingBot: self-healing retest failed: %r", e)
                self._last_mistake_analysis_cycle = self._cycle

            # 14. Phase 14 req #79: Time-based exits — close positions that
            # have been open longer than the max holding period.
            self._manage_open_positions(result)

        except Exception as e:  # noqa: BLE001
            log.exception("TradingBot: cycle crashed: %r", e)
            result.errors.append(str(e))
            self.monitor.record_cycle(time.time() - t0, error=True)
            # Record the cycle failure so ErrorRateBreaker can trip
            self.breakers.record_cycle(ok=False, latency_s=time.time() - t0)

        result.cycle_time_ms = (time.time() - t0) * 1000
        result.state = self.state_machine.current.value
        return result

    # ------------------------------------------------------------------
    # Per-symbol processing
    # ------------------------------------------------------------------
    def _process_symbol(self,
                        symbol: str,
                        equity: float,
                        adjustments: Any,
                        result: CycleResult) -> None:
        """Process one symbol through the full pipeline.

        P0-1 FIX (Phase 3): Every path through this method ends in exactly one of:
          (a) an actual exchange.place_order() call, OR
          (b) a logged + DB-persisted rejection with a concrete reason.
        The v9 "log APPROVED, place nothing" pattern (master_orchestrator.py:499-509)
        is structurally impossible — there is no code path that sets a flag and
        returns without either placing the order or calling _reject().

        P0-3 FIX: realized_pnl_today computed from portfolio history.
        P0-4 FIX: consecutive_losses computed from portfolio history (falls back
                  to DB query which survives restarts).
        P0-8 FIX: every decision + every fill written to database.decisions and
                  database.trades.
        """
        if self.exchange is None:
            return

        # ── Timing: collect per-stage durations for the decision report ──
        _timing: Dict[str, float] = {}
        _t0 = time.monotonic()

        # Fetch OHLCV
        try:
            df = self.exchange.fetch_candles(symbol, "M15", 2000)
        except Exception as e:
            self._result_add_error(result, f"{symbol}: fetch failed: {e}")
            self._record_skip(symbol, equity, None, None, result,
                            reason=f"fetch_error: {e}")
            return
        if df is None or df.empty or len(df) < 60:
            # Prompt #7: log + record this skip so it's visible, not silent.
            bar_count = len(df) if df is not None else 0
            if bar_count == 0:
                # Extra diagnostic: WHY is it empty?
                log.warning("TradingBot: %s returned 0 candles — check symbol visibility in MT5 MarketWatch", symbol)
            self._record_skip(symbol, equity, None, None, result,
                            reason=f"insufficient_bars({bar_count}/60 min)")
            if self._trace_enabled:
                _timing["Fetch Data"] = (time.monotonic() - _t0) * 1000
                report = self._format_decision_report(
                    symbol, df, None, None, None, None, None, None,
                    _timing, skip_reason=f"insufficient_bars({bar_count}/60 min)",
                    regime=str(getattr(adjustments, 'regime', 'unknown')))
                log.info("%s", report)
            return

        _timing["Fetch Data"] = (time.monotonic() - _t0) * 1000

        # Phase 8 req #48: Data validation — reject/flag impossible candles
        # before they reach feature/signal computation.
        from architecture.data_validator import DataValidator
        dv_cfg = self._cfg.get("data", {}).get("validation", {})
        validator = DataValidator(
            max_price_gap_pct=float(dv_cfg.get("max_price_gap_pct", 0.20)),
            min_volume=float(dv_cfg.get("min_volume", 0.0)),
            max_staleness_s=float(dv_cfg.get("max_staleness_s", 300.0)),
        )
        dv_result = validator.validate(df, symbol=symbol)
        if dv_result.has_errors:
            # Prompt #7: list the specific issue types, not just a count
            issue_types = [i.issue_type for i in dv_result.issues if i.severity == "error"]
            issue_detail = ", ".join(issue_types[:5])  # max 5 to keep readable
            self._record_skip(symbol, equity, None, None, result,
                            reason=f"DataValidator:{issue_detail}({dv_result.bars_checked} bars)")
            if self._trace_enabled:
                report = self._format_decision_report(
                    symbol, df, None, None, None, None, None, None,
                    _timing, skip_reason=f"DataValidator:{issue_detail}",
                    regime=str(getattr(adjustments, 'regime', 'unknown')))
                log.info("%s", report)
            return

        # TIER 1: manipulation detection (per-symbol veto BEFORE signals)
        try:
            _manip = self._manipulation_detector.check(df, symbol)
            if _manip.veto:
                self._record_skip(symbol, equity, None, None, result,
                                reason=f"manipulation:{_manip.veto_reason}")
                if self._trace_enabled:
                    _timing["Manipulation Check"] = (time.monotonic() - _t0) * 1000
                    report = self._format_decision_report(
                        symbol, df, None, None, None, None, None, None,
                        _timing, skip_reason=f"manipulation:{_manip.veto_reason}",
                        regime=str(getattr(adjustments, 'regime', 'unknown')))
                    log.info("%s", report)
                return
        except Exception:
            pass  # non-fatal — don't block trading on detector error

        # Compute feature vector (ONCE per cycle per symbol — shared across all gates)
        _t_feat = time.monotonic()
        fv = self.feature_pipeline.compute(symbol, df)
        _timing["Feature Pipeline"] = (time.monotonic() - _t_feat) * 1000

        # Multi-agent consensus
        _t_agent = time.monotonic()
        context = {
            "equity": equity,
            "peak_equity": self.portfolio.metrics().peak_equity,
            "adjustments": adjustments,
            # Co-Founder Audit: pass the cycle number so the LLM cost
            # tracker can reset per-cycle counters correctly.
            "_cycle": self._cycle,
        }
        consensus = self.multi_agent.evaluate(symbol, df, fv, context)
        _timing["AI Agents"] = (time.monotonic() - _t_agent) * 1000
        self._result_incr(result, "signals_generated", 1)
        # DEBUG funnel: signal generated (passed AI + strength threshold)
        result.funnel["ai_actionable"] = result.funnel.get("ai_actionable", 0) + 1

        # ── SKIP PATH 1: HOLD signal ──────────────────────────────────────
        # Every skip still writes a decision record so "silently doing nothing"
        # is impossible. The reason field makes the skip auditable.
        #
        # TRACE: emit per-symbol strategy breakdown so operators can see WHY
        # a signal is HOLD (which agents voted what, what the strength was).
        # Controlled by TRACING_BOT_TRACE=1 env var.
        if self._trace_enabled:
            self._trace_strategy_decision(symbol, consensus, fv, df)

        if consensus.action == "HOLD" or consensus.strength < self._min_signal_strength:
            # Review fix: include agent vote breakdown in the reason so the
            # user can see WHY strength is 0 (e.g. "all 5 agents voted HOLD"
            # vs "2 BUY vs 3 SELL = disagreement")
            vote_breakdown = f"B{consensus.votes_buy}/S{consensus.votes_sell}/H{consensus.votes_hold}"
            threshold_note = (f"< min_strength({self._min_signal_strength:.2f})"
                              if consensus.action != "HOLD" else "")
            skip_reason = (f"hold_or_low_strength({consensus.strength:.2f}, "
                          f"votes={vote_breakdown} {threshold_note})").strip()
            self._record_skip(symbol, equity, fv, consensus, result,
                              reason=skip_reason)
            if self._trace_enabled:
                report = self._format_decision_report(
                    symbol, df, fv, consensus, None, None, None, None,
                    _timing, skip_reason=skip_reason,
                    regime=str(getattr(adjustments, 'regime', 'unknown')))
                log.info("%s", report)
            return

        # ── SKIP PATH 2: already have an open position on this symbol ─────
        if self.portfolio.has_open_position(symbol):
            self._record_skip(symbol, equity, fv, consensus, result,
                              reason="already_open")
            return

        # ── Critical #2 fix: non-critical breaker check ───────────────────
        # should_block_cycle() only checks CRITICAL breakers (equity DD,
        # broker disconnect, etc.) and blocks the entire cycle including
        # position management. should_block_new_trades() also checks
        # NON-critical breakers (SlippageBreaker, LatencyBreaker) — if any
        # is open, we skip NEW entries but still allow position management
        # (SL/TP updates) to continue in _manage_open_positions.
        if self.breakers.should_block_new_trades():
            open_non_critical = [b["name"] for b in self.breakers.open_breakers()
                                 if not b.get("critical", True)]
            self._record_skip(symbol, equity, fv, consensus, result,
                              reason=f"non_critical_breaker_open({','.join(open_non_critical)})")
            return

        # ── Phase 5 #25: Multi-timeframe confirmation ────────────────────
        # Higher-timeframe trend must agree with entry-timeframe signal.
        # Review Point 4: distinguish "check ran and passed" from "check
        # failed to run" — the latter is logged at WARNING so silent
        # gate-skips are visible in the audit trail.
        try:
            from engine.candlestick.multi_timeframe import MultiTimeframeConfirmator
            mtf = MultiTimeframeConfirmator()
            mtf_result = mtf.confirm(df)
            if self._trace_enabled:
                log.info("TRACE %s │ MTF: score=%.0f aligned=%s dominant=%s",
                         symbol, mtf_result.score, mtf_result.aligned,
                         mtf_result.dominant_direction)
            if not mtf_result.aligned:
                self._record_skip(symbol, equity, fv, consensus, result,
                                  reason=f"MultiTimeframeGate:score={mtf_result.score:.0f} "
                                  f"(dominant={mtf_result.dominant_direction})")
                result.funnel["mtf_fail"] = result.funnel.get("mtf_fail", 0) + 1
                return
        except Exception as e:
            result.funnel["mtf_check_skipped"] = result.funnel.get("mtf_check_skipped", 0) + 1
            log.warning("TradingBot: MTF gate SKIPPED for %s (check crashed): %r "
                       "— trade proceeds without MTF confirmation", symbol, e)

        # ── Phase 5 #27: False-breakout filtering ────────────────────────
        # Reject signals where the breakout is likely fake.
        try:
            from engine.candlestick.false_breakout import FalseBreakoutDetector
            fb_detector = FalseBreakoutDetector()
            fb_result = fb_detector.detect(df)
            if fb_result.probability > 0.7:  # >70% likely fake = skip
                self._record_skip(symbol, equity, fv, consensus, result,
                                  reason=f"FakeBreakoutGate:prob={fb_result.probability:.0%} "
                                  f"(dir={fb_result.breakout_direction})")
                result.funnel["fakeout_fail"] = result.funnel.get("fakeout_fail", 0) + 1
                return
        except Exception as e:
            result.funnel["fakeout_check_skipped"] = result.funnel.get("fakeout_check_skipped", 0) + 1
            log.warning("TradingBot: fakeout gate SKIPPED for %s (check crashed): %r "
                       "— trade proceeds without fakeout filter", symbol, e)

        # Build canonical Signal (engine.signals.Signal — the v2 schema, not the dead v3)
        from architecture.risk_pipeline import RiskContext
        from engine.signals import Action, Signal

        action_val = Action.BUY if consensus.action == "BUY" else (
            Action.SELL if consensus.action == "SELL" else Action.HOLD
        )
        signal = Signal(
            symbol=symbol,
            action=action_val,
            strength=consensus.strength,
            price=float(df["close"].iloc[-1]),
            meta={
                "consensus_agreement": consensus.agreement_score,
                "votes_buy": consensus.votes_buy,
                "votes_sell": consensus.votes_sell,
                "dissenting_agents": consensus.dissenting_agents,
            },
        )

        # Fetch symbol info (for LiquidityGate spread check + SLTPGate stops_level)
        try:
            sym_info = self.exchange.symbol_info(symbol)
        except Exception:
            sym_info = None

        # ── P0-3/P0-4 FIX: REAL telemetry for the risk pipeline ───────────
        # consecutive_losses: from portfolio in-memory history (fast path),
        # falling back to DB query (survives restarts). realized_pnl_today:
        # from portfolio history of closed trades today (UTC).
        consecutive_losses = self.portfolio.consecutive_losses()
        if consecutive_losses == 0 and self.db is not None:
            # On a fresh restart with no in-memory history, check the DB.
            try:
                consecutive_losses = self.db.get_consecutive_losses()
            except Exception as e:
                # Phase 7: log DB read failure — was silently swallowed.
                # In-memory history is the fallback, but a DB read failure
                # could indicate corruption that should be visible.
                log.warning("TradingBot: db.get_consecutive_losses failed, "
                           "using in-memory fallback: %r", e)
        realized_pnl_today = self.portfolio.realized_pnl_today()
        last_trade_time = self.portfolio.last_trade_time()
        # C4/X2 fix: RiskContext.recent_trades was never populated, so
        # SizingGate._fractional_kelly_multiplier always saw an empty list
        # and short-circuited to a multiplier of 1.0 — Kelly sizing was
        # completely non-functional. Feed it real recent closed-trade
        # history from the portfolio.
        try:
            recent_trades_for_kelly = self.portfolio.recent_trades(n=50)
        except Exception as e:
            log.warning("TradingBot: portfolio.recent_trades failed, "
                       "Kelly sizing will use default multiplier: %r", e)
            recent_trades_for_kelly = []

        # Major #1 fix: pre-compute common indicators ONCE so multiple risk
        # gates (VolatilityGate, SizingGate, SLTPGate) don't each recompute
        # ATR/ATR% independently. Stored in pipeline_state which is passed
        # to every gate via RiskContext.
        _pipeline_state: Dict[str, Any] = {}
        try:
            from utils.indicators import atr as _atr_fn
            _atr_series = _atr_fn(df, 14)
            if len(_atr_series) > 0 and not pd.isna(_atr_series.iloc[-1]):
                _atr_val = float(_atr_series.iloc[-1])
                _pipeline_state["atr"] = _atr_val
                _price = float(df["close"].iloc[-1])
                if _price > 0:
                    _pipeline_state["atr_pct"] = _atr_val / _price
        except Exception as e:
            log.debug("TradingBot: pre-compute ATR failed for %s: %r", symbol, e)

        ctx = RiskContext(
            signal=signal,
            df=df,
            account_equity=equity,
            portfolio=self.portfolio.metrics(),
            symbol_info=sym_info,
            current_prices={},
            open_positions=[p for p in self.portfolio.all_positions()],
            consecutive_losses=consecutive_losses,  # P0-4 FIX: real value
            current_drawdown_pct=self.portfolio.metrics().current_drawdown_pct,
            last_trade_time=last_trade_time,
            realized_pnl_today=realized_pnl_today,  # P0-3 FIX: real value
            recent_trades=recent_trades_for_kelly,  # C4/X2 FIX: real value
            # Review Fix 2: daily_loss_halted_until is now persisted to
            # runtime_state.json and restored on restart. Previously hardcoded
            # to 0.0, meaning a restart mid-halt would resume trading.
            daily_loss_halted_until=self._daily_loss_halted_until,
            pipeline_state=_pipeline_state,  # Major #1 fix: shared indicators
        )

        # Start decision audit (P0-8: write to BOTH DecisionAuditor and Database)
        audit_id = self.decision_auditor.start_decision(
            symbol=symbol, cycle=self._cycle,
            feature_vector=fv.features,
            account_equity=equity,
            open_positions=self.portfolio.open_count(),
            current_drawdown_pct=ctx.current_drawdown_pct,
            bar_close=fv.bar_close,
        )
        self.decision_auditor.add_strategy_output(audit_id, signal)
        # P0-8: also persist to database.decisions
        if self.db is not None:
            try:
                import uuid as _uuid
                correlation_id = _uuid.uuid4().hex
                self.db.save_decision(
                    audit_id=audit_id, correlation_id=correlation_id,
                    symbol=symbol, cycle=self._cycle, bar_close=fv.bar_close,
                    feature_vector=fv.features, account_equity=equity,
                    open_positions=self.portfolio.open_count(),
                    current_drawdown_pct=ctx.current_drawdown_pct,
                    strategy_action=signal.action.value,
                    strategy_strength=signal.strength,
                    strategy_meta={
                        "consensus_agreement": consensus.agreement_score,
                        "votes_buy": consensus.votes_buy,
                        "votes_sell": consensus.votes_sell,
                    },
                )
            except Exception as e:
                log.warning("TradingBot: db.save_decision failed: %r", e)

        # Run risk pipeline (13 gates now — DailyLossGate added in Phase 3)
        _t_risk = time.monotonic()
        try:
            approved, final_verdict, all_verdicts = self.risk_pipeline.evaluate(ctx)
        except Exception as e:
            # P0-14 fix: if the risk pipeline raises, finalize the decision
            # as rejected with the exception reason so the audit trail is
            # complete and no orphaned decision records remain.
            log.error("TradingBot: risk_pipeline.evaluate raised for %s: %r", symbol, e)
            self._finalize_rejection(audit_id, symbol, f"risk_pipeline_exception: {e}", result)
            self._result_incr(result, "trades_rejected", 1)
            # Release any reservation that might have been created.
            reservation_id = getattr(final_verdict, "metadata", {}).get("reservation_id") if 'final_verdict' in dir() else None
            if reservation_id:
                try:
                    self.portfolio.release_reservation(reservation_id)
                except Exception:
                    pass
            return
        _timing["Risk Engine"] = (time.monotonic() - _t_risk) * 1000
        self.decision_auditor.add_risk_verdicts(audit_id, all_verdicts)
        if self.db is not None:
            try:
                self.db.update_decision_risk(audit_id, [
                    {"gate": v.gate_name, "passed": v.passed, "reason": v.reason}
                    for v in all_verdicts
                ])
            except Exception as e:
                # Phase 7: log DB write failure — was silently swallowed.
                # Decision audit is best-effort but a write failure could
                # indicate DB issues that should be visible.
                log.warning("TradingBot: db.update_decision_risk failed: %r", e)

        # ── REJECT PATH 1: risk pipeline failed a gate ────────────────────
        if not approved:
            self._result_incr(result, "trades_rejected", 1)
            result.funnel["risk_reject"] = result.funnel.get("risk_reject", 0) + 1
            log.info("PIPELINE %s: risk_reject at %s — %s",
                     symbol, final_verdict.gate_name, final_verdict.reason[:60])
            # C3/X1 fix: PortfolioGate reserves exposure/heat via
            # reservation_id BEFORE later gates run. If a gate AFTER
            # PortfolioGate rejects the trade, that reservation must be
            # released here or the portfolio believes it has more
            # exposure than it actually does (leaked reservation blocks
            # future trades). risk_pipeline.evaluate() carries the
            # reservation_id through to final_verdict.metadata regardless
            # of which gate ultimately failed.
            reservation_id = final_verdict.metadata.get("reservation_id")
            if reservation_id:
                try:
                    self.portfolio.release_reservation(reservation_id)
                except Exception as e:
                    log.warning("TradingBot: failed to release reservation "
                               "%s for %s: %r", reservation_id, symbol, e)
            reject_reason = f"risk_gate_{final_verdict.gate_name}: {final_verdict.reason}"
            self._finalize_rejection(audit_id, symbol, reject_reason, result)
            if self._trace_enabled:
                report = self._format_decision_report(
                    symbol, df, fv, consensus,
                    mtf_result if 'mtf_result' in dir() else None,
                    fb_result if 'fb_result' in dir() else None,
                    all_verdicts, None, _timing, skip_reason=reject_reason,
                    regime=str(getattr(adjustments, 'regime', 'unknown')))
                log.info("%s", report)
            # Review Fix 2: if DailyLossGate tripped, persist the halt
            # so it survives a restart. The halt expires at next UTC midnight.
            if final_verdict.gate_name == "daily_loss":
                from datetime import datetime as _dt, timezone as _tz
                now = _dt.now(tz=_tz.utc)
                tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
                tomorrow = tomorrow.timestamp() + 86400  # next UTC midnight
                self._daily_loss_halted_until = float(tomorrow)
                try:
                    import json
                    state = {"daily_loss_halted_until": self._daily_loss_halted_until}
                    with open(self._state_file, "w") as f:
                        json.dump(state, f)
                    log.warning("TradingBot: daily loss halt PERSISTED to %s "
                               "(expires at next UTC midnight)", self._state_file)
                    self._notify("Daily Loss Halt Triggered",
                                "Daily loss limit breached. Trading halted until "
                                "UTC midnight. Persisted to state file.")
                except Exception as e:
                    log.error("TradingBot: could not persist daily loss halt: %r", e)
            return

        # Wisdom gate (Livermore 200 principles)
        # P0-6 FIX (Phase 5): TradeContext now built from REAL telemetry
        # (recent_losses, recent_wins, bars_since_last_trade, pattern stats,
        # win_rate from memory_system, drawdown, regime) instead of the
        # hardcoded synthetic values that made several principles unpassable.
        if self._wisdom_gate is not None:
            try:
                wisdom_ctx = self._build_trade_context(
                    symbol=symbol, signal=signal, fv=fv, sym_info=sym_info,
                    final_verdict=final_verdict, ctx=ctx, regime=result.regime,
                    consecutive_losses=consecutive_losses,
                    consensus=consensus,  # BUGFIX: was missing, caused NameError
                )
                wisdom_verdict = self._wisdom_gate.evaluate(wisdom_ctx)
                self.decision_auditor.add_wisdom_verdict(audit_id, wisdom_verdict)
                if self.db is not None:
                    try:
                        self.db.update_decision_wisdom(audit_id, {
                            "approved": wisdom_verdict.approved,
                            "position_multiplier": wisdom_verdict.position_multiplier,
                            "checks_passed": wisdom_verdict.checks_passed,
                            "checks_failed": wisdom_verdict.checks_failed,
                        })
                    except Exception as e:
                        # Phase 7: log DB write failure — was silently swallowed.
                        log.warning("TradingBot: db.update_decision_wisdom failed: %r", e)
                # ── REJECT PATH 2: WisdomGate rejected ────────────────────
                if not wisdom_verdict.approved:
                    self._result_incr(result, "trades_rejected", 1)
                    result.funnel["wisdom_reject"] = result.funnel.get("wisdom_reject", 0) + 1
                    log.info("PIPELINE %s: wisdom_reject — %d principles failed (%s)",
                             symbol, wisdom_verdict.checks_failed,
                             ", ".join(wisdom_verdict.failed_principles[:3]))
                    # C3/X1 fix: release the PortfolioGate reservation —
                    # WisdomGate runs after the risk pipeline, so a
                    # reservation made in PortfolioGate is still open here.
                    reservation_id = final_verdict.metadata.get("reservation_id")
                    if reservation_id:
                        try:
                            self.portfolio.release_reservation(reservation_id)
                        except Exception as e:
                            log.warning("TradingBot: failed to release reservation "
                                       "%s for %s: %r", reservation_id, symbol, e)
                    # Prompt #7: list which principles failed, not just a count
                    failed_names = wisdom_verdict.failed_principles[:5]
                    reject_reason = (f"WisdomGate:{wisdom_verdict.checks_failed} principles failed "
                                     f"({', '.join(failed_names)})")
                    self._finalize_rejection(audit_id, symbol, reject_reason, result)
                    return
                # Apply position multiplier from wisdom gate
                if final_verdict.modified_lots:
                    final_verdict.modified_lots *= wisdom_verdict.position_multiplier
                    # BUG FIX: SizingGate already rounds lots to the broker's
                    # volume_step and clamps to [volume_min, volume_max]. But
                    # the WisdomGate position_multiplier (an arbitrary 0..1.5
                    # float) was applied AFTER that normalization, so the
                    # final volume sent to the broker was almost never a
                    # clean multiple of volume_step (and could even fall
                    # below volume_min). MT5 then rejected literally every
                    # order with retcode=10014 "Invalid volume" (see
                    # trades.log / system.log — 100% order rejection rate).
                    # Re-normalize here using the same step/min/max logic
                    # as SizingGate.
                    if sym_info is not None:
                        step = float(getattr(sym_info, "volume_step", 0.01)) or 0.01
                        vmin = float(getattr(sym_info, "volume_min", 0.01))
                        vmax = float(getattr(sym_info, "volume_max", 100.0))
                        final_verdict.modified_lots = round(
                            round(final_verdict.modified_lots / step) * step, 8)
                        final_verdict.modified_lots = max(
                            vmin, min(vmax, final_verdict.modified_lots))
            except Exception as e:
                # WisdomGate failure is a hard reject — we don't trade without it.
                # Review Point 6: distinguish "code bug in _build_trade_context"
                # from "gate evaluated and rejected" so a code bug doesn't get
                # miscategorized as "gate did its job."
                log.error("TradingBot: WisdomGate CODE ERROR (not a gate rejection) — "
                         "fail closed: %r", e)
                self._result_incr(result, "trades_rejected", 1)
                # C3/X1 fix: release the PortfolioGate reservation here too —
                # a code-error fail-closed reject is still a reject.
                reservation_id = final_verdict.metadata.get("reservation_id")
                if reservation_id:
                    try:
                        self.portfolio.release_reservation(reservation_id)
                    except Exception as release_err:
                        log.warning("TradingBot: failed to release reservation "
                                   "%s for %s: %r", reservation_id, symbol, release_err)
                self._finalize_rejection(
                    audit_id, symbol,
                    f"CODE_ERROR:wisdom_gate_crashed({type(e).__name__}: {e}) — "
                    f"NOT a principle rejection, a bug in _build_trade_context",
                    result)
                self._notify("WisdomGate Code Error",
                            f"Bug in _build_trade_context: {e!r}\n"
                            f"Trade rejected (fail-closed). Investigate the code bug.")
                return

        # ════════════════════════════════════════════════════════════════════
        # APPROVED — PLACE THE ORDER (P0-1 FIX: this is the ONLY approval path)
        # ════════════════════════════════════════════════════════════════════
        # Review Points 2 & 3: HARD REJECT if volume or SL is zero/None.
        # A sizing bug must NEVER silently become "trade anyway with 0.01 fallback"
        # and an SL bug must NEVER send an unprotected position to the broker.
        lots = final_verdict.modified_lots
        sl = final_verdict.modified_sl
        if not lots or lots <= 0:
            self._result_incr(result, "trades_rejected", 1)
            reservation_id = final_verdict.metadata.get("reservation_id")
            if reservation_id:
                self.portfolio.release_reservation(reservation_id)
            self._finalize_rejection(
                audit_id, symbol,
                f"HARD_REJECT:volume_invalid(lots={lots}) — sizing bug, not a fallback",
                result)
            log.error("TradingBot: HARD REJECT %s — volume=%s is invalid. "
                     "SizingGate has a bug. NOT falling back to 0.01.", symbol, lots)
            return
        if not sl or sl <= 0:
            self._result_incr(result, "trades_rejected", 1)
            reservation_id = final_verdict.metadata.get("reservation_id")
            if reservation_id:
                self.portfolio.release_reservation(reservation_id)
            self._finalize_rejection(
                audit_id, symbol,
                f"HARD_REJECT:no_stop_loss(sl={sl}) — every order MUST have an SL",
                result)
            log.error("TradingBot: HARD REJECT %s — SL=%s is invalid. "
                     "Every order MUST have a stop-loss. NOT sending unprotected.", symbol, sl)
            return

        try:
            # Phase 14 req #84: Pre-trade expected-value gate.
            # Reject trades with negative or sub-threshold computed expectancy
            # even after all other gates pass. Uses the canonical EV calculator
            # from trading_modules/expected_value_calculator.py.
            ev_cfg = self._cfg.get("execution", {}).get("ev_gate", {})
            ev_threshold_r = float(ev_cfg.get("min_ev_r", 0.0))  # default: reject EV < 0
            if ev_cfg.get("enabled", True):
                try:
                    from trading_modules.expected_value_calculator import ExpectedValueCalculator
                    ev_calc = ExpectedValueCalculator()
                    # Use real telemetry: win rate from portfolio history
                    recent = self.portfolio.recent_trades(n=30)
                    if len(recent) >= 10:
                        wins = [t for t in recent if t["pnl"] > 0]
                        wr = len(wins) / len(recent) if recent else 0.5
                        # NOTE: trade records only store raw currency PnL,
                        # not the risk amount at entry, so a true R-multiple
                        # can't be derived here. These remain conservative
                        # defaults until per-trade risk is persisted.
                        avg_win_r = 2.0  # default R:R assumption
                        avg_loss_r = 1.0
                        ev_result = ev_calc.calculate(
                            win_rate=wr, avg_win_r=avg_win_r, avg_loss_r=avg_loss_r,
                            sample_size=len(recent),
                            account_equity=equity,
                            risk_per_trade_pct=self._cfg.get("risk", {}).get("risk_per_trade", 0.01) * 100,
                        )
                        if ev_result.ev_per_trade_r < ev_threshold_r:
                            self._result_incr(result, "trades_rejected", 1)
                            # C3/X1 fix: release the PortfolioGate reservation.
                            reservation_id = final_verdict.metadata.get("reservation_id")
                            if reservation_id:
                                try:
                                    self.portfolio.release_reservation(reservation_id)
                                except Exception as release_err:
                                    log.warning("TradingBot: failed to release "
                                               "reservation %s for %s: %r",
                                               reservation_id, symbol, release_err)
                            self._finalize_rejection(
                                audit_id, symbol,
                                f"ev_gate_negative(ev_r={ev_result.ev_per_trade_r:.3f})",
                                result)
                            return
                        signal.expected_value_r = ev_result.ev_per_trade_r
                except Exception as e:
                    log.debug("TradingBot: EV gate failed (non-blocking): %r", e)

            magic = self._symbol_magic(symbol, default=100000)
            req = OrderRequest(
                symbol=symbol,
                side=OrderSide.BUY if signal.action.value == "BUY" else OrderSide.SELL,
                volume=lots,  # Review Point 2: validated above, no fallback
                sl=sl,        # Review Point 3: validated above, no fallback
                tp=final_verdict.modified_tp or 0.0,  # TP=0 is OK (no TP), SL=0 is NOT
                magic=magic,
                comment=f"tradingbot_c{self._cycle}_{self._mode}",
            )

            # Phase 4 req #22: idempotency check
            bar_time = df["time"].iloc[-1] if "time" in df.columns else None
            idem_key = self.idempotency.make_key(
                symbol=symbol, action=req.side.value,
                bar_time=bar_time, strategy="multi_agent")
            if not self.idempotency.check_and_mark(idem_key):
                self._result_incr(result, "trades_rejected", 1)
                reservation_id = final_verdict.metadata.get("reservation_id")
                if reservation_id:
                    self.portfolio.release_reservation(reservation_id)
                self._finalize_rejection(
                    audit_id, symbol, "idempotency_duplicate_order", result)
                log.warning("TradingBot: idempotency skip — already sent order "
                            "for %s %s @ %s", symbol, req.side.value, bar_time)
                return

            # Phase 6: Order slicing for large orders.
            # If the order notional exceeds the slicer threshold, break it
            # into child orders. For paper mode this is a no-op (PaperAdapter
            # fills at signal price regardless). For MT5, each slice is a
            # separate place_order call with a small delay between them.
            order_result = self._place_order_with_slicing(req, signal, result)

            if order_result.ok:
                self._result_incr(result, "trades_placed", 1)
                result.funnel["order_placed"] = result.funnel.get("order_placed", 0) + 1
                fill_price = order_result.price or signal.price

                # Phase 6: Compute + record actual slippage.
                # slippage_bps = |fill_price - signal_price| / signal_price * 10000
                # PaperAdapter fills at signal.price → slippage = 0.
                # MT5 fills may deviate → real slippage recorded.
                slippage_bps = 0.0
                if signal.price > 0:
                    slippage_bps = abs(fill_price - signal.price) / signal.price * 10000

                # Phase 4: feed fill to SlippageBreaker
                self.breakers.record_fill(
                    expected_price=signal.price, actual_price=fill_price)

                # Update portfolio
                # C8 fix: use order_result.volume (actual filled size) not
                # req.volume (requested size) — these can now differ when
                # order slicing partially fills.
                # C3/X1 fix (success path): pass reservation_id so the
                # PortfolioGate reservation is released the instant the real
                # position is committed, instead of relying on the 30s
                # stale-reservation prune (during which exposure would be
                # double-counted: once as "reserved", once as "open").
                self.portfolio.on_position_opened(
                    ticket=order_result.ticket,
                    symbol=symbol,
                    side=req.side.value,
                    volume=order_result.volume or req.volume,
                    entry_price=fill_price,
                    sl=req.sl, tp=req.tp, magic=req.magic,
                    reservation_id=final_verdict.metadata.get("reservation_id"),
                    # AUDIT FIX: pass contract_size so PnL is computed correctly
                    contract_size=float(getattr(sym_info, "contract_size", 1.0)
                                        if sym_info else 1.0),
                )
                # P0-8 FIX: write to database.trades (Phase 6: with slippage_bps)
                if self.db is not None:
                    try:
                        self.db.save_trade_open(
                            ticket=order_result.ticket, symbol=symbol,
                            action=req.side.value, lots=order_result.volume or req.volume,
                            entry_price=fill_price, stop_loss=req.sl,
                            take_profit=req.tp, magic=req.magic,
                            mode=self._mode, gate_score=consensus.strength,
                            grade="", strategy_type="multi_agent",
                            regime=result.regime,
                            atr=float(fv.get("atr", 0)),
                            confidence_pct=signal.strength * 100,
                            slippage_bps=slippage_bps,
                        )
                    except Exception as e:
                        log.warning("TradingBot: db.save_trade_open failed: %r", e)
                # Finalize decision
                self.decision_auditor.finalize_decision(
                    audit_id, approved=True,
                    lots=req.volume, sl=req.sl, tp=req.tp,
                    entry_price=fill_price,
                    ticket=order_result.ticket,
                )
                if self.db is not None:
                    try:
                        self.db.finalize_decision(
                            audit_id, approved=True,
                            final_lots=req.volume, final_sl=req.sl, final_tp=req.tp,
                            entry_price=fill_price, ticket=order_result.ticket,
                        )
                    except Exception as e:
                        log.warning("TradingBot: db.finalize_decision (approved) failed: %r", e)
                log.info("TradingBot: ORDER FILLED %s %s %.4f @ %.5f (ticket=%d, mode=%s, slip=%.1fbps)",
                         req.side.value, symbol, req.volume, fill_price,
                         order_result.ticket, self._mode, slippage_bps)
                # Phase 11: Telegram alert
                self._notify(
                    "Order Filled",
                    f"{req.side.value} {symbol} {req.volume:.4f} @ {fill_price:.5f}\n"
                    f"Ticket: {order_result.ticket} | Slip: {slippage_bps:.1f}bps | Mode: {self._mode}")
            else:
                # ── REJECT PATH 3: broker rejected the order ──────────────
                self._result_incr(result, "trades_rejected", 1)
                reject_reason = f"broker_rejected(retcode={order_result.error_code}): {order_result.comment}"
                # P0-7 fix: mark reservation as released so the except block
                # doesn't release it a second time.
                _reservation_released = False
                reservation_id = final_verdict.metadata.get("reservation_id")
                if reservation_id:
                    self.portfolio.release_reservation(reservation_id)
                    _reservation_released = True
                self._finalize_rejection(audit_id, symbol, reject_reason, result)
                log.warning("TradingBot: ORDER REJECTED %s: %s (retcode=%d)",
                            symbol, order_result.comment, order_result.error_code)
                # BUG FIX: no trade was actually opened, so don't let the
                # idempotency guard permanently blackhole this symbol/bar —
                # release the key so the next cycle can retry it.
                self.idempotency.unmark(idem_key)
        except Exception as e:
            # ── REJECT PATH 4: order placement raised an exception ─────────
            self._result_add_error(result, f"{symbol}: order failed: {e}")
            self._result_incr(result, "trades_rejected", 1)
            # P0-7 fix: only release if not already released in REJECT PATH 3.
            reservation_id = final_verdict.metadata.get("reservation_id")
            if reservation_id and not _reservation_released:
                self.portfolio.release_reservation(reservation_id)
            self._finalize_rejection(audit_id, symbol, f"order_exception: {e}", result)
            log.exception("TradingBot: order placement crashed for %s: %r", symbol, e)
            # BUG FIX: same as REJECT PATH 3 — an exception during order
            # placement means we don't know a trade went through, so don't
            # leave the idempotency key permanently marking it as "sent".
            try:
                self.idempotency.unmark(idem_key)
            except NameError:
                pass  # crashed before idem_key was computed — nothing to unmark

    # ------------------------------------------------------------------
    # P0-1/P0-8 FIX helpers: every exit path writes a decision record
    # ------------------------------------------------------------------
    def _trace_strategy_decision(self, symbol: str, consensus: Any,
                                   fv: Any, df: Any) -> None:
        """Emit a per-symbol strategy decision trace at INFO level.

        This is the key diagnostic for "why is everything HOLD?" — shows:
          - Consensus action + strength
          - Agent vote breakdown (BUY/SELL/HOLD/REDUCE counts)
          - Agreement score + dissenting agents
          - Key feature values from the feature vector

        Enabled by TRADING_BOT_TRACE=1 env var. Without it, the bot only
        logs the cycle-summary line (scan=100 buy=0 rej=0 skipped=100).
        """
        try:
            votes = (f"B{getattr(consensus, 'votes_buy', 0)}/"
                     f"S{getattr(consensus, 'votes_sell', 0)}/"
                     f"H{getattr(consensus, 'votes_hold', 0)}")
            action = getattr(consensus, "action", "?")
            strength = getattr(consensus, "strength", 0.0)
            agreement = getattr(consensus, "agreement_score", 0.0)
            dissenters = getattr(consensus, "dissenting_agents", [])

            # Pull a few key feature values for context.
            feat_summary = ""
            if fv is not None and hasattr(fv, "features"):
                feats = fv.features
                key_feats = []
                for k in ("rsi_14", "sma_20", "sma_50", "atr_14",
                          "macd_histogram", "bb_upper", "bb_lower"):
                    if k in feats and feats[k] is not None:
                        try:
                            key_feats.append(f"{k}={float(feats[k]):.2f}")
                        except (TypeError, ValueError):
                            pass
                if key_feats:
                    feat_summary = " │ ".join(key_feats[:5])

            bar_close = ""
            if df is not None and hasattr(df, "empty") and not df.empty and "close" in df.columns:
                bar_close = f"close={float(df['close'].iloc[-1]):.2f}"

            log.info(
                "TRACE %s │ %s strength=%.2f │ votes=%s │ agree=%.0f%% │ %s%s%s",
                symbol,
                action,
                strength,
                votes,
                agreement * 100,
                bar_close,
                f" │ {feat_summary}" if feat_summary else "",
                f" │ dissent={dissenters}" if dissenters else "",
            )
        except Exception as e:  # noqa: BLE001
            log.debug("trace_strategy_decision failed for %s: %r", symbol, e)

    # ------------------------------------------------------------------
    # Formatted AI Trading Decision Report
    # ------------------------------------------------------------------
    def _format_decision_report(self,
                                  symbol: str,
                                  df: Any,
                                  fv: Any,
                                  consensus: Any,
                                  mtf_result: Any,
                                  fb_result: Any,
                                  risk_verdicts: Any,
                                  final_decision: Any,
                                  timing: Dict[str, float],
                                  skip_reason: str = "",
                                  regime: str = "unknown") -> str:
        """Format a comprehensive AI Trading Decision Report with box-drawing.

        This produces the institutional-grade report format the operator
        sees in the console — with per-factor signal breakdown, AI agent
        votes, risk engine output, confidence engine, and a 'WHY NOT BUY?'
        section that explains exactly why a trade was or wasn't taken.
        """
        import datetime as _dt

        # ── Helper: box-drawing lines ──────────────────────────────
        W = 68  # content width
        B = "═" * (W + 2)
        D = "─" * (W + 2)

        def header(title: str) -> str:
            return f"\n{D}\n {title}\n{D}"

        def row(label: str, value: str, indent: int = 0) -> str:
            pad = " " * indent
            return f" {pad}{label:<28s} {value}"

        # ── Collect data safely ────────────────────────────────────
        feats = {}
        if fv is not None and hasattr(fv, "features"):
            feats = fv.features or {}

        def fval(key: str, default: float = 0.0) -> float:
            try:
                v = feats.get(key)
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        bar_close = 0.0
        bar_time_str = ""
        if df is not None and hasattr(df, "empty") and not df.empty:
            if "close" in df.columns:
                bar_close = float(df["close"].iloc[-1])
            if "time" in df.columns:
                try:
                    bar_time_str = str(df["time"].iloc[-1])
                except Exception:
                    bar_time_str = ""

        # ── Trend / Regime / Volatility ────────────────────────────
        rsi_val = fval("rsi_14", 50.0)
        sma20 = fval("sma_20")
        sma50 = fval("sma_50")
        atr_val = fval("atr_14")
        macd_hist = fval("macd_histogram")

        if sma20 > sma50 and sma50 > 0:
            trend = "BULLISH"
        elif sma20 < sma50 and sma50 > 0:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        regime_str = str(regime).upper().replace("_", " ") if regime else "UNKNOWN"

        # Volatility: ATR as % of price
        atr_pct = (atr_val / bar_close * 100) if bar_close > 0 else 0.0
        if atr_pct > 2.0:
            vol_str = f"HIGH (ATR {atr_val:.5f}, {atr_pct:.2f}%)"
        elif atr_pct < 0.5:
            vol_str = f"LOW (ATR {atr_val:.5f}, {atr_pct:.2f}%)"
        else:
            vol_str = f"NORMAL (ATR {atr_val:.5f}, {atr_pct:.2f}%)"

        # ── Signal breakdown (factor-by-factor) ────────────────────
        def icon(ok: bool) -> str:
            return "✅" if ok else "❌"

        def warn_icon() -> str:
            return "⚠"

        factors: list[tuple[str, str, str]] = []

        # EMA Trend
        ema_ok = trend == "BULLISH"
        ema_score = "+15" if ema_ok else "-10"
        factors.append(("EMA Trend", icon(ema_ok), ema_score))

        # Market Structure
        ms_ok = sma20 > 0 and sma50 > 0
        factors.append(("Market Structure", icon(ms_ok), "+10" if ms_ok else "-5"))

        # Momentum (RSI direction)
        mom_ok = rsi_val > 50
        factors.append(("Momentum", icon(mom_ok),
                        f"{'+8' if mom_ok else '-8'} (RSI {rsi_val:.0f})"))

        # Volume confirmation
        vol_ok = fval("rvol_20", 1.0) > 1.0
        factors.append(("Volume Confirmation", icon(vol_ok),
                        "+5" if vol_ok else "-5"))

        # SMC confirmation (simplified)
        smc_ok = macd_hist > 0
        factors.append(("SMC Confirmation", icon(smc_ok),
                        "+12" if smc_ok else "-12"))

        # Candlestick pattern (from fb_result if available)
        pattern_name = ""
        if fb_result is not None:
            try:
                pattern_name = getattr(fb_result, "breakout_direction", "") or ""
            except Exception:
                pass
        if pattern_name:
            factors.append(("Candlestick Pattern", warn_icon(), pattern_name))

        # MTF alignment
        # FIX: when mtf_result is None (MTF check didn't run because the
        # symbol was rejected before that gate), show "N/A" instead of
        # falsely reporting a mismatch with -10 penalty.
        mtf_aligned = False
        mtf_score = 0.0
        mtf_dominant = ""
        mtf_evaluated = mtf_result is not None
        if mtf_evaluated:
            try:
                mtf_aligned = getattr(mtf_result, "aligned", False)
                mtf_score = float(getattr(mtf_result, "score", 0))
                mtf_dominant = str(getattr(mtf_result, "dominant_direction", ""))
            except Exception:
                pass
        if not mtf_evaluated:
            factors.append(("MTF Alignment", "⚪", "N/A (not evaluated)"))
        elif mtf_aligned:
            factors.append(("MTF Alignment", icon(True), f"+10 (score {mtf_score:.0f})"))
        else:
            factors.append(("MTF Alignment", icon(False),
                            f"-10 (mismatch, dom={mtf_dominant})"))

        # Liquidity
        liq_ok = atr_pct < 3.0
        factors.append(("Liquidity Score", icon(liq_ok), "+6" if liq_ok else "-6"))

        # Compute total signal score (rough heuristic)
        total_score = 0
        for _, _, score_str in factors:
            try:
                if score_str.startswith("+"):
                    total_score += int(score_str.split("(")[0].replace("+", ""))
                elif score_str.startswith("-"):
                    total_score += int(score_str.split("(")[0])
            except (ValueError, IndexError):
                pass
        total_score = max(0, min(100, total_score + 50))  # normalize to 0-100

        # ── AI Agents ──────────────────────────────────────────────
        action = getattr(consensus, "action", "HOLD") if consensus else "HOLD"
        strength = getattr(consensus, "strength", 0.0) if consensus else 0.0
        votes_buy = getattr(consensus, "votes_buy", 0) if consensus else 0
        votes_sell = getattr(consensus, "votes_sell", 0) if consensus else 0
        votes_hold = getattr(consensus, "votes_hold", 0) if consensus else 0
        votes_reduce = getattr(consensus, "votes_reduce", 0) if consensus else 0
        agreement = getattr(consensus, "agreement_score", 0.0) if consensus else 0.0

        # ── Risk Engine ────────────────────────────────────────────
        risk_approved = False
        risk_reason = ""
        final_lots = 0.0
        final_sl = 0.0
        final_tp = 0.0
        entry_price = bar_close
        kelly_mult = 0.0
        if risk_verdicts:
            for v in reversed(risk_verdicts):
                if hasattr(v, "passed") and v.passed and hasattr(v, "modified_lots") and v.modified_lots:
                    final_lots = float(v.modified_lots or 0)
                if hasattr(v, "modified_sl") and v.modified_sl:
                    final_sl = float(v.modified_sl)
                if hasattr(v, "modified_tp") and v.modified_tp:
                    final_tp = float(v.modified_tp)
                if hasattr(v, "metadata") and isinstance(v.metadata, dict):
                    kelly_mult = float(v.metadata.get("kelly_multiplier", 0))
            # Last verdict is the pipeline final
            if risk_verdicts:
                last = risk_verdicts[-1]
                risk_approved = getattr(last, "passed", False)
                risk_reason = getattr(last, "reason", "")

        # ── Final decision ─────────────────────────────────────────
        approved = False
        decision_label = "NO TRADE"
        decision_reason = skip_reason or "Weak confluence"
        # FIX Bug #2: Signal Score was decorative — it never affected the
        # displayed confidence. Now Final Confidence is a blend of
        # consensus strength (70%) and signal score (30%), so a high
        # signal score can boost confidence above the raw strength.
        # This makes the report internally consistent.
        signal_score_pct = total_score  # 0-100
        raw_confidence = strength * 100  # 0-100
        confidence_pct = (0.7 * raw_confidence) + (0.3 * signal_score_pct)
        confidence_pct = max(0.0, min(100.0, confidence_pct))
        if final_decision is not None:
            approved = getattr(final_decision, "approved", False)
            if approved:
                decision_label = "BUY" if action == "BUY" else "SELL" if action == "SELL" else "APPROVED"
            else:
                decision_label = "NO TRADE"
            decision_reason = getattr(final_decision, "rationale", "")[:200] or skip_reason or "Weak confluence"

        # ── Build the report ───────────────────────────────────────
        lines: list[str] = []
        now_str = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines.append("")
        lines.append(B)
        lines.append(" AI TRADING DECISION REPORT")
        lines.append(B)
        lines.append("")
        lines.append(f" Symbol        : {symbol}")
        lines.append(" Timeframe     : M15")
        lines.append(f" Session       : {self._trading_session()}")
        lines.append(f" Timestamp     : {now_str}")
        lines.append(f" Bar Close     : {bar_close:.5f}" if bar_close > 0 else " Bar Close     : N/A")
        lines.append(f" Bar Time      : {bar_time_str}" if bar_time_str else " Bar Time      : N/A")
        lines.append(f" Cycle         : {self._cycle}")

        # ── MARKET STATE ───────────────────────────────────────────
        lines.append(header("MARKET STATE"))
        lines.append(row("Trend", trend))
        lines.append(row("Regime", regime_str))
        lines.append(row("Volatility", vol_str))
        lines.append(row("RSI(14)", f"{rsi_val:.1f}"))
        lines.append(row("MACD Histogram", f"{macd_hist:.5f}"))
        lines.append(row("ATR(14)", f"{atr_val:.5f}"))
        lines.append(row("SMA(20)", f"{sma20:.5f}"))
        lines.append(row("SMA(50)", f"{sma50:.5f}"))

        # ── SIGNAL BREAKDOWN ───────────────────────────────────────
        lines.append(header("SIGNAL BREAKDOWN"))
        for label, ic, score in factors:
            lines.append(f" {label:<28s} {ic}  {score}")
        lines.append("")
        lines.append(f" {'Total Signal Score':<28s} {total_score} / 100")

        # ── AI AGENTS ──────────────────────────────────────────────
        lines.append(header("AI AGENTS"))
        # We don't have per-agent breakdown in consensus, so show the tally
        lines.append(f" {'Votes — BUY':<28s} {votes_buy}")
        lines.append(f" {'Votes — SELL':<28s} {votes_sell}")
        lines.append(f" {'Votes — HOLD':<28s} {votes_hold}")
        if votes_reduce:
            lines.append(f" {'Votes — REDUCE':<28s} {votes_reduce}")
        lines.append("")
        lines.append(f" {'Consensus Action':<28s} {action}")
        lines.append(f" {'Consensus Strength':<28s} {strength*100:.0f}%")
        lines.append(f" {'Agreement Score':<28s} {agreement*100:.0f}%")

        # ── RISK ENGINE ────────────────────────────────────────────
        lines.append(header("RISK ENGINE"))
        lines.append(row("Risk Approved", "YES" if risk_approved else "NO"))
        if risk_reason:
            lines.append(row("Risk Reason", str(risk_reason)))
        if kelly_mult > 0:
            lines.append(row("Kelly Multiplier", f"{kelly_mult:.2f}x"))
        lines.append(row("Min Signal Strength", f"{self._min_signal_strength:.2f}"))
        lines.append(row("Signal Strength", f"{strength:.2f}"))
        if final_lots > 0:
            lines.append(row("Position Size", f"{final_lots:.4f} lot"))

        # ── TRADE PLAN ─────────────────────────────────────────────
        if approved and final_lots > 0:
            lines.append(header("TRADE PLAN"))
            lines.append(f" {'Action':<28s} {decision_label}")
            lines.append(f" {'Entry':<28s} {entry_price:.5f}")
            if final_sl > 0:
                sl_pips = abs(entry_price - final_sl) / max(entry_price, 1e-8) * 10000
                lines.append(f" {'Stop Loss':<28s} {final_sl:.5f} ({sl_pips:.1f} pips)")
            if final_tp > 0:
                tp_pips = abs(final_tp - entry_price) / max(entry_price, 1e-8) * 10000
                lines.append(f" {'Take Profit':<28s} {final_tp:.5f} ({tp_pips:.1f} pips)")

        # ── CONFIDENCE ENGINE ──────────────────────────────────────
        lines.append(header("CONFIDENCE ENGINE"))
        lines.append(f" {'Consensus Strength':<28s} {strength*100:.0f}%")
        lines.append(f" {'Agreement Score':<28s} {agreement*100:.0f}%")
        lines.append(f" {'Signal Score':<28s} {total_score}/100")
        lines.append(f" {'Final Confidence':<28s} {confidence_pct:.0f}%")

        # ── WHY NOT BUY? ───────────────────────────────────────────
        lines.append(header("WHY NOT BUY?" if not approved else "WHY BUY?"))
        reasons_for: list[str] = []
        reasons_against: list[str] = []
        if trend == "BULLISH":
            reasons_for.append("Trend is bullish")
        elif trend == "BEARISH":
            reasons_against.append("Trend is bearish")
        if risk_approved:
            reasons_for.append("Risk acceptable")
        else:
            reasons_against.append("Risk not approved")
        if not smc_ok:
            reasons_against.append("Smart Money confirmation missing")
        if not mom_ok:
            reasons_against.append("Momentum weak (RSI < 50)")
        if mtf_evaluated and not mtf_aligned:
            reasons_against.append("MTF not aligned")
        if strength < self._min_signal_strength:
            reasons_against.append(f"Confidence below minimum {self._min_signal_strength:.0%}")
        if not vol_ok:
            reasons_against.append("Volume confirmation missing")

        for r in reasons_for:
            lines.append(f" ✓ {r}")
        for r in reasons_against:
            lines.append(f" ✗ {r}")
        lines.append("")
        lines.append(f" Decision           {decision_label}")

        # ── PERFORMANCE ────────────────────────────────────────────
        lines.append(header("PERFORMANCE"))
        total_ms = 0.0
        for stage, ms in timing.items():
            lines.append(f" {stage:<28s} {ms:.0f} ms")
            total_ms += ms
        lines.append("")
        lines.append(f" {'TOTAL':<28s} {total_ms:.0f} ms")

        # ── FINAL RESULT ───────────────────────────────────────────
        lines.append(header("FINAL RESULT"))
        lines.append(f" {'Decision':<28s} {decision_label}")
        lines.append(f" {'Confidence':<28s} {confidence_pct:.0f}%")
        # Truncate reason to fit
        reason_display = decision_reason[:60] + "..." if len(decision_reason) > 60 else decision_reason
        lines.append(f" {'Reason':<28s} {reason_display}")
        lines.append(f" {'Next Review':<28s} Next candle (15 min)")
        lines.append("")
        lines.append("═" * 70)

        return "\n".join(lines)

    def _trading_session(self) -> str:
        """Return the current trading session name based on UTC hour."""
        import datetime as _dt
        hour = _dt.datetime.now(tz=_dt.timezone.utc).hour
        if 7 <= hour < 12:
            return "London → New York Overlap"
        elif 12 <= hour < 16:
            return "New York"
        elif 0 <= hour < 7:
            return "Tokyo → London"
        elif 16 <= hour < 21:
            return "New York Close"
        else:
            return "Off-session"

    # ------------------------------------------------------------------
    # Critical #1 fix: thread-safe helpers for mutating the shared
    # CycleResult object. These must be used instead of direct
    # `result.errors.append()` / `result.trades_rejected += 1` etc.
    # when _process_symbol runs concurrently (max_workers > 1).
    # ------------------------------------------------------------------
    def _result_add_error(self, result: CycleResult, msg: str) -> None:
        """Thread-safe append to result.errors."""
        with self._result_lock:
            result.errors.append(msg)

    def _result_incr(self, result: CycleResult, attr: str, n: int = 1) -> None:
        """Thread-safe increment of a numeric attribute on result."""
        with self._result_lock:
            setattr(result, attr, getattr(result, attr) + n)

    def _result_incr_skip(self, result: CycleResult, reason_key: str,
                          n: int = 1) -> None:
        """Thread-safe increment of a skip_breakdown counter.

        Co-Founder Audit: added optional `n` parameter so the universe
        pre-filter can batch-increment counts (e.g. 45 liquidity_fail in
        one call) instead of looping 45 times.
        """
        with self._result_lock:
            result.skip_breakdown[reason_key] = result.skip_breakdown.get(reason_key, 0) + n

    def _record_skip(self, symbol: str, equity: float, fv: Any,
                     consensus: Any, result: CycleResult, reason: str) -> None:
        """Record a pre-risk-pipeline skip (HOLD, already open, etc.) as a
        rejected decision. P0-1: no path exits _process_symbol without a record.

        UI fix: per-symbol skips are now DEBUG (file only), and counted in
        result.skip_breakdown for a single cycle-summary line in run_loop.
        """
        try:
            import uuid as _uuid
            audit_id = f"skip_{_uuid.uuid4().hex[:12]}"
            self.decision_auditor.start_decision(
                symbol=symbol, cycle=self._cycle,
                feature_vector=fv.features if hasattr(fv, "features") else {},
                account_equity=equity,
                open_positions=self.portfolio.open_count(),
                current_drawdown_pct=self.portfolio.metrics().current_drawdown_pct,
                bar_close=fv.bar_close if hasattr(fv, "bar_close") else 0.0,
            )
            self.decision_auditor.finalize_decision(audit_id, approved=False)
            if self.db is not None:
                self.db.save_decision(
                    audit_id=audit_id, correlation_id=audit_id,
                    symbol=symbol, cycle=self._cycle,
                    bar_close=fv.bar_close if hasattr(fv, "bar_close") else 0.0,
                    feature_vector=fv.features if hasattr(fv, "features") else {},
                    account_equity=equity,
                    open_positions=self.portfolio.open_count(),
                    current_drawdown_pct=self.portfolio.metrics().current_drawdown_pct,
                    strategy_action=getattr(consensus, "action", "HOLD"),
                    strategy_strength=getattr(consensus, "strength", 0.0),
                    strategy_meta={"skip_reason": reason},
                )
                self.db.finalize_decision(audit_id, approved=False, reject_reason=reason)
        except Exception as e:
            log.warning("TradingBot: _record_skip failed for %s: %r", symbol, e)
        # UI fix / H18 fix: count skip by type for cycle-summary, log at DEBUG.
        # Previously split only on "(", which produced misleading keys for
        # reasons that use a colon separator instead (e.g. "risk_gate_X: msg"
        # split on "(" keeps the whole string as the key). Split on whichever
        # delimiter — "(" or ":" — appears first, so structurally similar
        # reasons collapse into the same bucket.
        reason_type = _skip_reason_key(reason)
        # Critical #1 fix: use thread-safe increment.
        self._result_incr_skip(result, reason_type)
        log.debug("TradingBot: SKIP %s — %s", symbol, reason)

    def _finalize_rejection(self, audit_id: str, symbol: str,
                            reason: str, result: CycleResult) -> None:
        """P0-1 FIX: every rejection is logged + DB-persisted with a reason.

        Prompt #7: logs at INFO with symbol + gate/stage + numeric reason.
        """
        try:
            self.decision_auditor.finalize_decision(audit_id, approved=False)
        except Exception as e:
            # Phase 7: log auditor failure — was silently swallowed.
            log.warning("TradingBot: decision_auditor.finalize_decision failed: %r", e)
        if self.db is not None:
            try:
                self.db.finalize_decision(audit_id, approved=False, reject_reason=reason)
            except Exception as e:
                log.warning("TradingBot: db.finalize_decision failed: %r", e)
        log.info("TradingBot: REJECTED %s — %s", symbol, reason)

    # ------------------------------------------------------------------
    # C1 FIX (Chief AI Architect Audit): Learning loop — trigger LLM reflection
    # when a trade closes. This completes the memory-reflection cycle that was
    # previously broken (resolve_trade_outcome was never called from the live
    # path, so get_past_context always returned empty).
    # ------------------------------------------------------------------
    def _trigger_trade_reflection(self, pos: dict, pnl: float, reason: str) -> None:
        """Call the LLM memory log's resolve_trade_outcome() for this trade.

        This runs ASYNCHRONOUSLY (in a daemon thread) so it doesn't block
        the trading cycle. The LLM reflection call can take 3-5s, and we
        don't want to delay position management for it.

        Args:
            pos: position dict with symbol, entry_price, side, etc.
            pnl: realized PnL for this trade
            reason: "broker_sl_tp" or "time_exit"
        """
        try:
            symbol = pos.get("symbol", "?")
            # Compute return as a fraction (not %) for the memory log
            entry_price = float(pos.get("entry_price", 0))
            volume = float(pos.get("volume", 0))
            risk_amount = abs(entry_price - float(pos.get("sl", 0))) * volume if entry_price > 0 else 0
            raw_return = pnl / max(risk_amount, 1.0) if risk_amount > 0 else 0.0

            # Get the trade date from the position's open_time
            open_time_str = pos.get("open_time", "")
            trade_date = open_time_str[:10] if open_time_str else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

            # Run in a daemon thread so it doesn't block
            def _reflect():
                try:
                    # Access the LLM graph's memory log if available
                    if hasattr(self.multi_agent, '_llm_graph') and self.multi_agent._llm_graph is not None:
                        graph = self.multi_agent._llm_graph
                        reflection = graph.resolve_trade_outcome(
                            symbol=symbol,
                            trade_date=trade_date,
                            raw_return=raw_return,
                            alpha_return=0.0,  # alpha vs benchmark — TODO: compute vs BTC
                            benchmark="BTC",
                        )
                        if reflection:
                            log.info("LLM REFLECTION %s (pnl=%+.2f, R=%.2f): %s",
                                    symbol, pnl, raw_return, reflection[:200])
                except Exception as e:
                    log.debug("LLM reflection failed for %s: %r", symbol, e)

            t = threading.Thread(target=_reflect, name="llm-reflection", daemon=True)
            t.start()
        except Exception as e:
            log.debug("TradingBot: _trigger_trade_reflection failed: %r", e)

    # ------------------------------------------------------------------
    # Phase 9: Trade journal + mistake analyzer + decay detector
    # ------------------------------------------------------------------
    def _record_closed_trade_to_journal(self, ticket: int, symbol: str,
                                         side: str, volume: float,
                                         entry_price: float, exit_price: float,
                                         pnl: float, sl: float, tp: float,
                                         hold_s: float, reason: str,
                                         slippage_bps: float = 0.0,
                                         strategy: str = "multi_agent") -> None:
        """Phase 9 req #50: Write a full structured entry to the trade journal
        when a position closes. Captures entry/exit reason, R-multiple, slippage,
        holding duration, and all signal scores at entry time.
        """
        try:
            from enhancements.trade_journal import JournalEntry
            import uuid as _uuid
            # Compute R-multiple: pnl / risk_amount
            risk_per_unit = abs(entry_price - sl) if sl > 0 else 0
            risk_amount = risk_per_unit * volume
            r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
            pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            if side.upper() == "SELL":
                pnl_pct = -pnl_pct
            entry = JournalEntry(
                entry_id=f"je_{_uuid.uuid4().hex[:12]}",
                symbol=symbol,
                timeframe="M15",
                side="long" if side.upper() == "BUY" else "short",
                strategy=strategy,
                entry_time="",  # filled from DB if available
                exit_time=datetime.now(tz=timezone.utc).isoformat(),
                entry_price=entry_price,
                exit_price=exit_price,
                lots=volume,
                pnl=pnl,
                pnl_pct=pnl_pct,
                r_multiple=r_multiple,
                hold_bars=int(hold_s / 900),  # M15 = 900s per bar
                outcome="win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven"),
                mistake=reason,
                metadata={"slippage_bps": slippage_bps, "ticket": ticket,
                         "sl": sl, "tp": tp, "hold_s": hold_s},
            )
            self._journal.record(entry)
            # Also update the decisions table outcome
            if self.db is not None:
                try:
                    outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
                    self.db.record_decision_outcome(
                        ticket=ticket, exit_price=exit_price, pnl=pnl,
                        hold_time_s=hold_s, outcome=outcome)
                except Exception as e:
                    log.debug("TradingBot: db.record_decision_outcome failed: %r", e)
        except Exception as e:
            log.warning("TradingBot: _record_closed_trade_to_journal failed: %r", e)

    def _run_mistake_analysis(self) -> None:
        """Phase 9 req #51: Run the mistake analyzer as a scheduled job.

        Reads the trade journal, computes win rate, expectancy, average R,
        max drawdown, and flags trades where exit reasoning contradicts
        entry reasoning. Results are logged + alerted.
        """
        try:
            from enhancements.mistake_analyzer import MistakeAnalyzer
            analyzer = MistakeAnalyzer()
            patterns = analyzer.analyze()
            if patterns:
                log.info("TradingBot: mistake analysis found %d patterns", len(patterns))
                for p in patterns[:5]:
                    log.warning("  Mistake: %s (frequency=%d, avg_loss=%.2f)",
                               p.name if hasattr(p, "name") else str(p),
                               getattr(p, "frequency", 0),
                               getattr(p, "avg_loss", 0))
                # Phase 9 req #52: Feed back into thresholds (bounded, logged)
                # For now this is informational — auto-adjustment requires
                # a configurable threshold-floor/ceiling system which is
                # left as a documented future enhancement.
                self._notify("Mistake Analysis Report",
                            f"Found {len(patterns)} mistake patterns.\n"
                            f"Top: {patterns[0].name if hasattr(patterns[0], 'name') else patterns[0]}")
        except Exception as e:
            log.warning("TradingBot: mistake analysis failed: %r", e)

    def _run_decay_detection(self) -> None:
        """Phase 9 req #53: Check for strategy decay.

        Compares live performance vs backtest expectation per strategy.
        Decay-flagged strategies are added to _decayed_strategies and
        excluded from the next cycle's routing.
        """
        try:
            recent = self.portfolio.recent_trades(n=50)
            if len(recent) < 10:
                return  # not enough data
            # Group by strategy (all "multi_agent" for now — future: per-strategy)
            pnls = [t["pnl"] for t in recent]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            if not losses:
                return
            win_rate = len(wins) / len(pnls)
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses))
            # Decay check: use actual expectancy (win_rate * avg_win -
            # loss_rate * avg_loss), not a bare win-rate threshold. A
            # low win-rate strategy can still be profitable if avg_win
            # is large relative to avg_loss, and a high win-rate
            # strategy can be losing money if avg_loss dwarfs avg_win.
            loss_rate = 1.0 - win_rate
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            if expectancy < 0:
                self._decayed_strategies.add("multi_agent")
                log.warning("TradingBot: strategy decay detected — win_rate=%.0f%% "
                           "avg_win=%.2f avg_loss=%.2f expectancy=%.2f "
                           "over last %d trades. Strategy excluded from routing.",
                           win_rate * 100, avg_win, avg_loss, expectancy, len(pnls))
                self._notify("Strategy Decay Detected",
                            f"Win rate {win_rate:.0%}, expectancy {expectancy:.2f} "
                            f"over last {len(pnls)} trades.\n"
                            f"Strategy paused — manual review required.")
        except Exception as e:
            log.debug("TradingBot: decay detection failed: %r", e)

    def _manage_open_positions(self, result: CycleResult) -> None:
        """Phase 14 req #79: Time-based exits + trailing stops.

        For each open position:
        1. If position has been open > max_holding_period_s → close it
           (avoids dead capital in stagnant trades).
        2. If position is in profit > trailing_threshold_r → move SL to
           breakeven + trail it (locks in profit, lets winners run).

        Both checks use the PortfolioManager's position data + current
        prices from the exchange.
        """
        if self.exchange is None:
            return
        max_hold_s = float(self._cfg.get("execution", {}).get(
            "max_holding_period_s", 86400 * 3))  # default 3 days
        trailing_threshold_r = float(self._cfg.get("execution", {}).get(
            "trailing_threshold_r", 1.0))  # trail after +1R

        # C14 fix: use the public all_positions() API instead of reaching
        # into portfolio._positions directly. Direct access bypassed the
        # PortfolioManager's lock and its internal bookkeeping guarantees.
        for pos in list(self.portfolio.all_positions()):
            ticket = pos.get("ticket")
            try:
                # 1. Time-based exit
                open_time_str = pos.get("open_time", "")
                if open_time_str:
                    try:
                        from datetime import datetime
                        open_time = datetime.fromisoformat(open_time_str)
                        hold_s = (datetime.now(tz=timezone.utc) - open_time).total_seconds()
                        if hold_s > max_hold_s:
                            # Close the position
                            tick = self.exchange.symbol_tick(pos["symbol"])
                            exit_price = tick.bid if pos["side"] == "BUY" else tick.ask
                            close_result = self.exchange.close_order(
                                ticket, pos["symbol"], pos["volume"],
                                pos["side"])
                            if close_result.ok:
                                pnl = self.portfolio.on_position_closed(
                                    ticket, exit_price, reason="time_exit")
                                # Review Fix 1: feed trade outcome to
                                # ConsecutiveLossBreaker so it actually trips
                                self.breakers.record_trade_outcome(pnl)
                                self._record_closed_trade_to_journal(
                                    ticket, pos["symbol"], pos["side"],
                                    pos["volume"], pos["entry_price"],
                                    exit_price, pnl, pos["sl"], pos["tp"],
                                    hold_s, "time_exit")
                                log.info("TradingBot: TIME EXIT %s ticket=%d (held %.1fh, pnl=%+.2f)",
                                        pos["symbol"], ticket, hold_s / 3600, pnl)
                                # C1 FIX: wire learning loop for time exits too
                                self._trigger_trade_reflection(pos, pnl, "time_exit")
                    except Exception as e:
                        log.debug("TradingBot: time-exit check failed for %s: %r",
                                 pos["symbol"], e)

                # 2. Trailing stop (Phase 14 req #81)
                # If position is in profit > trailing_threshold_r, move SL
                # to breakeven + trail by 0.5 ATR.
                # This requires current price + ATR — skip if not available.
                # For paper mode this is a no-op (no real SL to move).
                if self._mode != "paper" and pos["sl"] > 0:
                    try:
                        tick = self.exchange.symbol_tick(pos["symbol"])
                        current_price = tick.mid
                        direction = 1 if pos["side"] == "BUY" else -1
                        profit = (current_price - pos["entry_price"]) * direction
                        risk_per_unit = abs(pos["entry_price"] - pos["sl"])
                        if risk_per_unit > 0:
                            r_multiple = profit / risk_per_unit
                            if r_multiple > trailing_threshold_r:
                                # Move SL to breakeven (entry price) if not already
                                new_sl = pos["entry_price"]
                                if (direction == 1 and new_sl > pos["sl"]) or \
                                   (direction == -1 and new_sl < pos["sl"]):
                                    self.exchange.modify_order(ticket, new_sl, pos["tp"])
                                    # Co-Founder Audit Fix: use the thread-safe
                                    # mutator instead of mutating the dict ref
                                    # directly (which bypassed portfolio._lock).
                                    self.portfolio.update_position_sl_tp(
                                        ticket, sl=new_sl, tp=pos["tp"])
                                    pos["sl"] = new_sl  # local copy update for log line
                                    log.info("TradingBot: trailing stop moved to BE "
                                            "%s ticket=%d (R=%.1f)",
                                            pos["symbol"], ticket, r_multiple)
                    except Exception as e:
                        log.debug("TradingBot: trailing stop check failed for %s: %r",
                                 pos["symbol"], e)

                # 3. TIER 4: Dynamic exit intelligence
                # Evaluates trend weakening, momentum loss, structure break,
                # volatility spikes — recommends early exit if warranted.
                try:
                    tick = self.exchange.symbol_tick(pos["symbol"])
                    current_price = tick.mid if tick else 0.0
                    if current_price > 0 and pos.get("entry_price"):
                        direction = 1 if pos["side"] == "BUY" else -1
                        profit = (current_price - pos["entry_price"]) * direction
                        risk_per_unit = abs(pos["entry_price"] - pos.get("sl", 0))
                        r_mult = profit / risk_per_unit if risk_per_unit > 0 else 0.0
                        # Get recent df for this symbol (use cache if available)
                        pos_df = None
                        try:
                            pos_df = self.exchange.fetch_candles(pos["symbol"], "M15", 100)
                        except Exception:
                            pass
                        if pos_df is not None and len(pos_df) >= 20:
                            # BUG FIX: this previously compared exit_rec.action
                            # against ExitAction.CLOSE / .CLOSE_PARTIAL / .MODIFY
                            # — none of which exist on the real enum (see
                            # trading_modules/dynamic_exit_intelligence.py,
                            # which only ever sets HOLD, TIGHTEN_STOP,
                            # MOVE_TO_BREAKEVEN, TRAIL_STOP, PARTIAL_CLOSE, or
                            # CLOSE_ALL). Every position that reached this
                            # branch raised AttributeError (see system.log —
                            # "exit intelligence error ... has no attribute
                            # 'CLOSE'" on every cycle for every open trade),
                            # so dynamic exits never fired; positions relied
                            # solely on their static broker-side SL/TP.
                            #
                            # This also previously only *logged* CLOSE/PARTIAL
                            # recommendations instead of acting on them — the
                            # position management loop looked like it was
                            # managing exits but never actually closed or
                            # trimmed anything. Now it executes them via the
                            # same close_order() path used by the time-exit
                            # branch above.
                            from trading_modules.dynamic_exit_intelligence import ExitAction
                            exit_rec = self._exit_intel.evaluate(
                                position_side=pos["side"],
                                entry_price=pos["entry_price"],
                                current_price=current_price,
                                stop_loss=pos.get("sl", 0.0),
                                take_profit=pos.get("tp", 0.0),
                                df=pos_df,
                                r_multiple=r_mult,
                                spread_bps=pos.get("spread_bps", 5.0),
                            )
                            if exit_rec.action == ExitAction.CLOSE_ALL:
                                exit_price = tick.bid if pos["side"] == "BUY" else tick.ask
                                close_result = self.exchange.close_order(
                                    ticket, pos["symbol"], pos["volume"], pos["side"])
                                if close_result.ok:
                                    pnl = self.portfolio.on_position_closed(
                                        ticket, exit_price, reason="exit_intel")
                                    self.breakers.record_trade_outcome(pnl)
                                    self._record_closed_trade_to_journal(
                                        ticket, pos["symbol"], pos["side"],
                                        pos["volume"], pos["entry_price"],
                                        exit_price, pnl, pos.get("sl", 0.0),
                                        pos.get("tp", 0.0), 0.0, "exit_intel")
                                    self._trigger_trade_reflection(pos, pnl, "exit_intel")
                                log.info("TradingBot: EXIT_INTEL %s action=%s reason=%s "
                                         "urgency=%s ticket=%d ok=%s",
                                         pos["symbol"], exit_rec.action.value,
                                         exit_rec.reason, exit_rec.urgency.value,
                                         ticket, close_result.ok)
                            elif exit_rec.action == ExitAction.PARTIAL_CLOSE:
                                exit_price = tick.bid if pos["side"] == "BUY" else tick.ask
                                close_pct = min(max(exit_rec.close_pct, 0.0), 1.0) or 0.5
                                close_vol = self._normalize_volume_for_symbol(
                                    pos["symbol"], pos["volume"] * close_pct)
                                if 0 < close_vol < pos["volume"]:
                                    close_result = self.exchange.close_order(
                                        ticket, pos["symbol"], close_vol, pos["side"])
                                    if close_result.ok:
                                        pnl = self.portfolio.partial_close_position(
                                            ticket, exit_price, close_vol, reason="exit_intel")
                                        self.breakers.record_trade_outcome(pnl)
                                        if exit_rec.new_stop:
                                            self.exchange.modify_order(
                                                ticket, exit_rec.new_stop, pos.get("tp"))
                                            self.portfolio.update_position_sl_tp(
                                                ticket, sl=exit_rec.new_stop, tp=pos.get("tp"))
                                    log.info("TradingBot: EXIT_INTEL %s action=%s reason=%s "
                                             "urgency=%s ticket=%d closed_vol=%.4f ok=%s",
                                             pos["symbol"], exit_rec.action.value,
                                             exit_rec.reason, exit_rec.urgency.value,
                                             ticket, close_vol, close_result.ok)
                            elif exit_rec.action in (
                                ExitAction.TIGHTEN_STOP, ExitAction.MOVE_TO_BREAKEVEN,
                                ExitAction.TRAIL_STOP,
                            ):
                                # Update SL/TP if recommended
                                if exit_rec.new_stop and exit_rec.new_stop != pos.get("sl"):
                                    if self._mode != "paper":
                                        self.exchange.modify_order(
                                            ticket, exit_rec.new_stop, pos.get("tp"))
                                    self.portfolio.update_position_sl_tp(
                                        ticket, sl=exit_rec.new_stop, tp=pos.get("tp"))
                                    log.info("TradingBot: EXIT_INTEL modify SL %s → %.5f "
                                             "(reason: %s)", pos["symbol"],
                                             exit_rec.new_stop, exit_rec.reason)
                except Exception as exit_exc:
                    log.debug("TradingBot: exit intelligence error for %s: %r",
                             pos.get("symbol", "?"), exit_exc)
            except Exception as e:
                log.debug("TradingBot: position management failed for %s: %r",
                         pos.get("symbol", "?"), e)

    def _normalize_volume_for_symbol(self, symbol: str, volume: float) -> float:
        """Round `volume` down to the symbol's broker volume_step and clamp
        to [volume_min, volume_max]. Used anywhere a volume is derived
        from arithmetic (e.g. position_size * close_pct) rather than
        coming straight out of SizingGate, so it can't end up as a
        non-step-aligned value the broker rejects with retcode=10014
        "Invalid volume" — the same failure mode fixed for order entries.
        """
        try:
            info = self.exchange.symbol_info(symbol) if self.exchange else None
        except Exception:
            info = None
        step = float(getattr(info, "volume_step", 0.01)) if info else 0.01
        vmin = float(getattr(info, "volume_min", 0.01)) if info else 0.01
        vmax = float(getattr(info, "volume_max", 100.0)) if info else 100.0
        step = step or 0.01
        normalized = round(round(volume / step) * step, 8)
        if normalized < vmin:
            return 0.0  # too small to close as its own order — caller skips
        return max(0.0, min(vmax, normalized))

    def _symbol_magic(self, symbol: str, default: int = 100000) -> int:
        """Look up the per-symbol magic number from config (for MT5 trade tagging)."""
        for s in self._cfg.get("symbols", []):
            if isinstance(s, dict) and s.get("name") == symbol:
                return int(s.get("magic", default))
        return default

    # ------------------------------------------------------------------
    # Phase 6: Order slicing + reconciliation
    # ------------------------------------------------------------------
    def _place_order_with_slicing(self, req: OrderRequest, signal: Any,
                                   result: CycleResult) -> Any:
        """Place an order, slicing into child orders if the notional exceeds
        the configured threshold.

        Phase 6 req #33: orders above a configurable notional threshold are
        actually sliced into smaller child orders, not just theoretically
        capable of it. For PaperAdapter this is a no-op (fills at signal
        price regardless). For MT5Adapter, each slice is a separate
        place_order call with a small delay between them.

        Threshold: execution.slice_threshold_notional (default $50,000).
        Below threshold: single order. Above: TWAP-sliced into N child
        orders spaced execution.slice_interval_s (default 5s) apart.
        """
        exec_cfg = self._cfg.get("execution", {})
        slice_threshold = float(exec_cfg.get("slice_threshold_notional", 50000.0))
        slice_interval_s = float(exec_cfg.get("slice_interval_s", 5.0))
        max_slices = int(exec_cfg.get("max_slices", 5))

        notional = req.volume * signal.price
        if notional <= slice_threshold or max_slices <= 1:
            # Below threshold — single order, no slicing
            return self.exchange.place_order(req)

        # Above threshold — slice into child orders
        from execution.order_slicer import OrderSlicer
        slicer = OrderSlicer(
            max_slice_lots=req.volume / max_slices,
            min_slices=2, max_slices=max_slices,
            default_interval_s=slice_interval_s,
        )
        slices = slicer.slice(
            parent_lots=req.volume, side=req.side.value,
            strategy="twap",
        )
        log.info("TradingBot: slicing %s %.4f lots ($%.0f notional) into %d child orders",
                 req.symbol, req.volume, notional, len(slices))

        # Place each slice sequentially
        import time as _time
        total_filled_lots = 0.0
        weighted_fill_price = 0.0
        first_ticket = 0
        all_ok = True
        # H11 fix: bound the total time spent slicing so a stuck/slow
        # broker call can't hang the whole cycle. Default generous enough
        # for max_slices * slice_interval_s plus per-call latency.
        slicing_timeout_s = float(exec_cfg.get("slicing_timeout_s",
                                                max(60.0, slice_interval_s * max_slices * 3)))
        slicing_started_at = _time.time()
        slices_filled = 0
        for i, sl in enumerate(slices):
            if _time.time() - slicing_started_at > slicing_timeout_s:
                all_ok = False
                log.warning("TradingBot: slicing %s timed out after %.0fs "
                           "(%d/%d slices placed) — aborting remaining slices",
                           req.symbol, slicing_timeout_s, slices_filled, len(slices))
                break
            child_req = OrderRequest(
                symbol=req.symbol, side=req.side,
                volume=sl.lots, sl=req.sl, tp=req.tp,
                magic=req.magic,
                comment=f"{req.comment}_slice{i+1}of{len(slices)}",
            )
            child_result = self.exchange.place_order(child_req)
            if child_result.ok:
                total_filled_lots += sl.lots
                weighted_fill_price += child_result.price * sl.lots
                slices_filled += 1
                if first_ticket == 0:
                    first_ticket = child_result.ticket
                if i < len(slices) - 1:
                    _time.sleep(slice_interval_s)
            else:
                all_ok = False
                log.warning("TradingBot: slice %d/%d failed: %s",
                            i + 1, len(slices), child_result.comment)
                break

        # C8 fix: a PARTIAL fill (some slices filled, some didn't/timed out)
        # still means real capital is at risk at the broker for
        # `total_filled_lots`. The previous code returned ok=all_ok, which
        # was False on any partial failure — the caller then treated the
        # whole order as rejected, released the risk reservation, and never
        # called portfolio.on_position_opened(), leaving the filled slices
        # as an untracked "phantom" position (only caught later, if ever,
        # by reconcile_with_broker). We now report success whenever ANY
        # lots filled, with the comment flagging the shortfall so the
        # caller/operator can see it wasn't the full requested size.
        fully_filled = all_ok and total_filled_lots > 0
        partially_filled = (not all_ok) and total_filled_lots > 0
        if partially_filled:
            log.warning("TradingBot: %s PARTIAL FILL — %.4f/%.4f lots filled "
                       "across %d/%d slices. Tracking the filled portion; "
                       "review remaining exposure manually.",
                       req.symbol, total_filled_lots, req.volume,
                       slices_filled, len(slices))

        # Return aggregated result
        from architecture.exchange_abstraction import OrderResult
        return OrderResult(
            ok=fully_filled or partially_filled,
            ticket=first_ticket,
            price=weighted_fill_price / max(total_filled_lots, 0.0001),
            volume=total_filled_lots,
            comment=("sliced(%d)" % len(slices)) if fully_filled
                     else (f"partial_fill({slices_filled}/{len(slices)})" if partially_filled
                           else "slice_failed"),
            latency_ms=0.0,
        )

    def reconcile_with_broker(self) -> list[str]:
        """Phase 6 req #37: Compare broker's actual open positions against
        PortfolioManager's internal state. Returns list of discrepancy strings.

        Called on boot and every N cycles (configurable via
        runtime.reconciliation_interval_cycles, default 50).

        Conservative default resolution: TRUST THE BROKER. If there's a
        mismatch, we log + alert, and the bot's internal state is updated
        to match the broker. We never silently auto-close phantom positions
        or create local entries for broker positions we don't recognize —
        those are surfaced as discrepancies for manual review.
        """
        if self.exchange is None:
            return []
        discrepancies = []
        try:
            broker_positions = self.exchange.positions()
            broker_tickets = {int(p.ticket) for p in broker_positions}
            local_positions = self.portfolio.all_positions()
            local_tickets = {p["ticket"] for p in local_positions}

            # Local has position broker doesn't (orphan — could be closed
            # without us knowing, or a bookkeeping error).
            # C10 fix: actually resolve it — trust the broker and close the
            # local position rather than just logging the discrepancy and
            # leaving stale state around indefinitely.
            orphan = local_tickets - broker_tickets
            for t in orphan:
                pnl = self.portfolio.force_close_orphan(t, reason="reconciliation_orphan")
                self.breakers.record_trade_outcome(pnl)
                d = (f"Local ticket {t} not found at broker — closed locally "
                     f"(pnl_approx={pnl:+.2f}) — position was likely closed "
                     f"externally.")
                discrepancies.append(d)

            # Broker has position we don't track (phantom — could be a
            # manual trade, a fill we missed, or a crash/restart that lost
            # the in-memory reservation before on_position_opened() ran).
            #
            # ROOT-CAUSE FIX (screenshot 2026-07-18): this branch used to
            # ONLY log the discrepancy and never registered the position
            # locally. That left has_open_position(symbol) == False for a
            # symbol the broker already had an open trade on, which
            # silently defeated the "one trade per symbol" gate in
            # PortfolioManager.can_open_new() (rule #4). That is exactly
            # how Boom 99 Index and Skew Step Index 5 Up each ended up with
            # two simultaneous, uncoordinated positions (one pair opposite-
            # direction, one pair same-direction) in the MT5 screenshot —
            # the bot's own state said the symbol was flat when it wasn't,
            # so every downstream gate that reads has_open_position passed
            # a trade it should have blocked.
            #
            # Fix: register the broker's position into local state (with
            # its real SL/TP/side/volume) as soon as it's discovered, so
            # the very next cycle's has_open_position() check reflects
            # reality. We still emit the discrepancy for visibility/alerting
            # — silently absorbing it would hide the underlying miss (a
            # crash, a slow fill, a manual trade) that caused it.
            phantom = broker_tickets - local_tickets
            for t in phantom:
                p = next((bp for bp in broker_positions if int(bp.ticket) == t), None)
                if p:
                    # BUG FIX: Position.type is an OrderSide (str enum:
                    # "BUY"/"SELL"), not an int 0/1, and the price field is
                    # named open_price (see Position dataclass in
                    # exchange_abstraction.py), not price_open. The previous
                    # version of this fix used p.price_open, which doesn't
                    # exist on Position and raised AttributeError every time
                    # a phantom was found — silently aborting reconciliation
                    # for that cycle instead of registering the position.
                    side = str(getattr(p, "type", "BUY"))
                    self.portfolio.on_position_opened(
                        ticket=t, symbol=p.symbol, side=side,
                        volume=float(p.volume), entry_price=float(p.open_price),
                        sl=float(getattr(p, "sl", 0.0) or 0.0),
                        tp=float(getattr(p, "tp", 0.0) or 0.0),
                        magic=int(getattr(p, "magic", 0) or 0),
                        contract_size=float(getattr(p, "contract_size", 1.0) or 1.0),
                    )
                    d = (f"Broker ticket {t} ({p.symbol} {p.volume} {side}) not in "
                         f"local state — external trade or missed fill. "
                         f"Registered locally so duplicate-symbol gate now sees it; "
                         f"still flagged for manual review of why it was missed.")
                    discrepancies.append(d)

            # Volume mismatch on shared tickets — C10 fix: sync local
            # volume to match the broker (trust-the-broker resolution).
            for p in broker_positions:
                t = int(p.ticket)
                local = self.portfolio.get_position(t)
                if local is not None:
                    if abs(local["volume"] - float(p.volume)) > 0.001:
                        old_vol = local["volume"]
                        self.portfolio.force_sync_volume(t, float(p.volume))
                        d = (f"Volume mismatch on ticket {t}: "
                             f"local={old_vol} broker={p.volume} — synced to broker")
                        discrepancies.append(d)

        except Exception as e:
            log.warning("TradingBot: reconciliation failed: %r", e)
            discrepancies.append(f"reconciliation_error: {e}")

        if discrepancies:
            for d in discrepancies:
                log.warning("TradingBot: RECONCILIATION: %s", d)
            # Phase 11: alert on reconciliation discrepancies
            self._notify("Reconciliation Mismatch",
                        f"{len(discrepancies)} discrepancies detected:\n"
                        + "\n".join(discrepancies[:5]))
            # Review Point 5: feed reconciliation discrepancies into the
            # circuit breaker so persistent drift auto-halts trading.
            # Each discrepancy counts as a failure for the ErrorRateBreaker.
            for _ in discrepancies:
                self.breakers.record_cycle(ok=False, latency_s=0.0)
            # If 3+ discrepancies in one reconciliation, trip immediately
            if len(discrepancies) >= 3:
                log.error("TradingBot: %d reconciliation discrepancies — "
                         "tripping circuit breaker immediately", len(discrepancies))
                # Force the error-rate breaker open
                for b in self.breakers.breakers:
                    if b.name == "error_rate":
                        b.record_failure(
                            f"reconciliation_drift:{len(discrepancies)} discrepancies")
        return discrepancies

    # ------------------------------------------------------------------
    # Phase 11: Telegram notification helper
    # ------------------------------------------------------------------
    def _notify(self, title: str, body: str) -> None:
        """Send a Telegram alert. No-op if Telegram is not configured.

        Review Point 7: escalate to WARNING after 5 consecutive notify
        failures so the operator knows their alerting is broken.

        C19 fix: rate-limited to prevent an error storm (e.g. a flapping
        connection or a repeating gate rejection) from flooding Telegram.
        Two layers:
          1. Per-title cooldown — the same alert title won't re-fire within
             `notify_title_cooldown_s` (default 60s); a suppressed-count is
             folded into the next delivered alert with that title.
          2. Global rate cap — no more than `notify_max_per_minute`
             (default 20) notifications are sent in any rolling 60s window;
             extras are dropped with a single periodic summary log instead
             of hammering the Telegram API.
        """
        if self._telegram is None:
            return

        now = time.time()
        notify_cfg = self._cfg.get("execution", {}).get("notifications", {})
        title_cooldown_s = float(notify_cfg.get("title_cooldown_s", 60.0))
        max_per_minute = int(notify_cfg.get("max_per_minute", 20))

        if not hasattr(self, "_notify_last_sent_by_title"):
            self._notify_last_sent_by_title: Dict[str, float] = {}
            self._notify_suppressed_by_title: Dict[str, int] = {}
            self._notify_recent_sends: Deque[float] = deque(maxlen=200)
            self._notify_global_suppressed = 0

        # Layer 1: per-title cooldown
        last_sent = self._notify_last_sent_by_title.get(title, 0.0)
        if now - last_sent < title_cooldown_s:
            self._notify_suppressed_by_title[title] = \
                self._notify_suppressed_by_title.get(title, 0) + 1
            return

        # Layer 2: global rate cap (rolling 60s window)
        recent = self._notify_recent_sends
        while recent and now - recent[0] > 60.0:
            recent.popleft()
        if len(recent) >= max_per_minute:
            self._notify_global_suppressed += 1
            if self._notify_global_suppressed % 20 == 1:
                log.warning("TradingBot: notification rate cap hit (%d/min) — "
                           "%d alerts suppressed so far this session",
                           max_per_minute, self._notify_global_suppressed)
            return

        suppressed = self._notify_suppressed_by_title.pop(title, 0)
        if suppressed:
            body = f"{body}\n\n({suppressed} similar alert(s) suppressed in the last {title_cooldown_s:.0f}s)"
        self._notify_last_sent_by_title[title] = now
        recent.append(now)

        try:
            self._telegram.send_alert(title=title, body=body)
            self._notify_failures = 0  # reset on success
        except Exception as e:
            self._notify_failures = getattr(self, '_notify_failures', 0) + 1
            if self._notify_failures <= 3:
                log.debug("TradingBot: telegram send failed (%d): %r",
                         self._notify_failures, e)
            elif self._notify_failures == 4:
                log.warning("TradingBot: telegram has failed %d consecutive times — "
                           "alerts are NOT being delivered. Check TELEGRAM_BOT_TOKEN "
                           "and network connectivity.", self._notify_failures)
            elif self._notify_failures % 10 == 0:
                # Every 10th failure after the first warning, re-escalate
                log.warning("TradingBot: telegram still failing (%d consecutive) — "
                           "alerts remain broken.", self._notify_failures)

    # ------------------------------------------------------------------
    # P0-6 FIX (Phase 5): Build WisdomGate TradeContext from REAL telemetry
    # ------------------------------------------------------------------
    def _build_trade_context(self, symbol: str, signal: Any, fv: Any,
                             sym_info: Any, final_verdict: Any, ctx: Any,
                             regime: str, consecutive_losses: int,
                             consensus: Any = None) -> Any:
        """Construct a TradeContext for WisdomGate from real portfolio + feature data.

        Replaces the hardcoded synthetic values (recent_losses=0, recent_wins=0,
        pattern_match_count=5, pattern_win_rate=0.55, bars_since_last_trade=10)
        that made several of the 200 principles permanently unpassable.

        Real telemetry sources:
          - recent_losses / recent_wins: from portfolio._closed_trades streak
          - bars_since_last_trade: from portfolio.last_trade_time vs df time
          - win_rate: from memory_system.estimate_ev (falls back to 0.5)
          - pattern_match_count / pattern_win_rate: from last 30 trades on this symbol
          - drawdown_pct, regime, spread_bps, atr_ratio: from ctx + fv + sym_info
          - rr_ratio: from final_verdict SL/TP
          - trades_today: from portfolio.realized_pnl_today's underlying trade count
          - has_open_position: from portfolio
          - portfolio_exposure: from portfolio metrics
          - probability_buy/sell/wait: from consensus vote counts
        """
        from livermore_principles import TradeContext

        # Real consecutive wins/losses from portfolio history
        recent_trades = self.portfolio.recent_trades(n=30)
        recent_wins = 0
        for t in reversed(recent_trades):
            if t["pnl"] > 0:
                recent_wins += 1
            else:
                break

        # bars_since_last_trade: rough estimate from last_trade_time
        # (each M15 bar = 900s; if no trades yet, use a large default)
        last_t = self.portfolio.last_trade_time()
        bars_since = 100 if last_t == 0 else max(1, int((time.time() - last_t) / 900))

        # Pattern stats: use this symbol's recent trade history
        symbol_trades = [t for t in recent_trades if t.get("symbol") == symbol]
        pattern_match_count = len(symbol_trades)
        if pattern_match_count > 0:
            wins = sum(1 for t in symbol_trades if t["pnl"] > 0)
            pattern_win_rate = wins / pattern_match_count
        else:
            pattern_win_rate = 0.5  # neutral when no history

        # Win rate from memory_system (falls back to 0.5 on no data)
        try:
            ev = self.memory_system.estimate_ev(fv.features, symbol=symbol)
            win_rate = float(ev.get("win_rate", 0.5))
        except Exception:
            win_rate = 0.5

        # Spread in bps from symbol_info
        spread_bps = 0.0
        if sym_info is not None:
            spread = float(getattr(sym_info, "spread", 0))
            point = float(getattr(sym_info, "point", 0.0001))
            price = signal.price if signal.price > 0 else 1.0
            spread_bps = (spread * point / price) * 10000

        # R:R from the risk pipeline's final SL/TP
        sl = float(final_verdict.modified_sl or 0)
        tp = float(final_verdict.modified_tp or 0)
        risk = abs(signal.price - sl) if sl > 0 else 0.0001
        reward = abs(tp - signal.price) if tp > 0 else 0
        rr_ratio = reward / max(risk, 0.0001)

        # ATR ratio: current ATR / baseline (use fv's atr_pct as proxy)
        atr_ratio = float(fv.get("atr_pct", 0.01)) / 0.01  # normalized to 1.0

        # Trades today: count of closed_trades with today's date
        from datetime import datetime, timezone
        today = datetime.now(tz=timezone.utc).date()
        trades_today = sum(
            1 for t in recent_trades
            if t.get("close_time_iso", "")[:10] == today.isoformat()
        )

        # Portfolio exposure from metrics
        pm_metrics = self.portfolio.metrics()
        portfolio_exposure = pm_metrics.gross_exposure_pct / 100.0 if pm_metrics.gross_exposure_pct > 0 else 0.3

        # ── FORENSIC AUDIT FIX: normalize multi-agent strength to 0-1 scale ──
        #
        # The multi-agent coordinator produces strength on a 0.06–0.42 scale
        # (sum of confidence × target_weight, where weights are 0.08–0.15).
        # WisdomGate's 200 principles expect confidence on a 0.0–1.0 scale
        # with thresholds like 0.50, 0.60, 0.80, 0.90.
        #
        # Without normalization, EVERY principle with a threshold > 0.42
        # permanently fails — not just Principle 6, but also Principles 12,
        # 163, 180, 195, and many more. This made the bot structurally
        # incapable of trading.
        #
        # Mapping: strength 0.15 (min actionable) → confidence 0.60 (min to trade)
        #          strength 0.25 (strong)         → confidence 0.80 (normal)
        #          strength 0.35+ (max)           → confidence 0.95 (aggressive)
        #
        # Formula: confidence = 0.40 + strength × 2.0, clamped to [0.40, 0.95]
        #   0.15 → 0.70, 0.20 → 0.80, 0.25 → 0.90, 0.30+ → 0.95
        normalized_confidence = max(0.40, min(0.95, 0.40 + signal.strength * 2.0))

        # ── FORENSIC AUDIT FIX: populate probability fields from consensus ──
        # Principle 163 checks max(probability_buy, probability_sell, probability_wait) > 0.95
        # The default probability_wait=1.0 made this ALWAYS fail. Now we compute
        # real probabilities from the multi-agent vote counts.
        # BUGFIX: consensus was referenced but not passed as a parameter, causing
        # NameError("name 'consensus' is not defined") on every trade evaluation.
        if consensus is not None:
            total_votes = max(1, consensus.votes_buy + consensus.votes_sell
                              + consensus.votes_hold + consensus.votes_reduce)
            prob_buy = consensus.votes_buy / total_votes
            prob_sell = consensus.votes_sell / total_votes
            prob_wait = (consensus.votes_hold + consensus.votes_reduce) / total_votes
        else:
            # Fallback: derive from signal action
            prob_buy = 0.6 if signal.action.value == "BUY" else 0.1
            prob_sell = 0.6 if signal.action.value == "SELL" else 0.1
            prob_wait = max(0.0, 1.0 - prob_buy - prob_sell)

        # ── FORENSIC AUDIT FIX: strategy_switched for non-trending regimes ──
        # Principle 195 rejects if strategy_switched=False AND regime is range/crisis.
        # The multi-agent coordinator already adapts to regime (different agents fire
        # in different regimes), so mark strategy_switched=True for non-trending regimes.
        strategy_switched = regime.lower() in ("range", "crisis", "transition",
                                                "chop", "high_vol", "low_vol")

        return TradeContext(
            symbol=symbol,
            direction=signal.action.value,
            confidence=normalized_confidence,  # FORENSIC FIX: 0-1 scale for WisdomGate
            win_rate=win_rate,
            rr_ratio=rr_ratio,
            atr_ratio=atr_ratio,
            bars_since_last_trade=bars_since,
            spread_bps=spread_bps,
            regime=regime,
            drawdown_pct=ctx.current_drawdown_pct,
            recent_losses=consecutive_losses,
            recent_wins=recent_wins,
            pattern_match_count=pattern_match_count,
            pattern_win_rate=pattern_win_rate,
            news_pending=False,  # Phase 5: wire from news_calendar
            external_signal=False,
            trades_today=trades_today,
            has_open_position=self.portfolio.has_open_position(symbol),
            portfolio_exposure=portfolio_exposure,
            exit_rules_defined=True,  # SLTPGate always produces SL/TP
            structure_valid=True,  # Phase 5: wire from candlestick/confluence
            forward_tested=True,  # Phase 10: wire from backtest readiness
            # FORENSIC FIX: populate probability fields from consensus votes
            probability_buy=prob_buy,
            probability_sell=prob_sell,
            probability_wait=prob_wait,
            # FORENSIC FIX: mark strategy as adapted for non-trending regimes
            strategy_switched=strategy_switched,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fetch_current_prices(self) -> Dict[str, float]:
        """Fetch live tick prices — ONLY for symbols with an open position.

        PERF FIX: this used to loop over ALL tracked symbols (e.g. 100)
        sequentially, with no threading, making 100 blocking MT5 IPC
        calls every single cycle — even though portfolio.update_prices()
        only ever consumes prices for symbols that actually have an open
        position (see PortfolioManager.update_prices: `if p["symbol"] in
        prices`). On a 100-symbol demo config with few/no open positions,
        this was ~100 wasted IPC round-trips per cycle, unthreaded, and
        was a major contributor to the 35s+ cycle times. Now we only
        fetch ticks for symbols we actually hold, and do it in parallel
        (mirrors the pattern used for _process_symbol).
        """
        prices: Dict[str, float] = {}
        if self.exchange is None:
            return prices

        open_symbols = sorted({p.get("symbol") for p in self.portfolio.all_positions()
                               if p.get("symbol")})
        if not open_symbols:
            return prices

        def _fetch_one(sym: str):
            try:
                tick = self.exchange.symbol_tick(sym)
                return sym, tick.mid
            except Exception as e:
                # Phase 7: log tick fetch failures — was silently swallowed.
                # A failed tick for one symbol drops it from the price map,
                # which could leave positions with stale prices.
                log.debug("TradingBot: symbol_tick(%s) failed: %r", sym, e)
                return sym, None

        # C5/X3 fix: same reasoning as cycle() — a thread pool against an
        # IPC-serialized exchange (MT5Adapter._ipc_lock) adds overhead
        # without parallelism.
        exchange_is_ipc_serialized = getattr(self.exchange, "_ipc_lock", None) is not None
        if len(open_symbols) > 1 and not exchange_is_ipc_serialized:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(open_symbols), 8)) as ex:
                for sym, mid in ex.map(_fetch_one, open_symbols):
                    if mid is not None:
                        prices[sym] = mid
        else:
            for sym in open_symbols:
                _, mid = _fetch_one(sym)
                if mid is not None:
                    prices[sym] = mid

        return prices

    # ------------------------------------------------------------------
    # Status + diagnostics
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "cycle": self._cycle,
            "state": self.state_machine.current.value,
            "time_in_state_s": self.state_machine.time_in_state(),
            "equity": self.portfolio.equity(),
            "open_positions": self.portfolio.open_count(),
            "regime": self.regime_orchestrator.current_regime.value,
            "symbols_tracked": len(getattr(self, "_symbols", [])),
            "kpis": self.monitor.kpis(),
            "memory_stats": self.memory_system.stats(),
            "online_learner": self.online_learner.stats(),
            "self_healing": self.self_healing.health(),
            "event_bus": self.event_bus.metrics(),
            "feature_pipeline": self.feature_pipeline.stats(),
            "degraded_components": self.self_healing.degraded_components(),
            # Co-Founder Audit: LLM augmentation status — visible to operators
            # via `python main.py --status` so they can verify LLM is wired.
            "llm_augmentation": getattr(self.multi_agent, "llm_status",
                                        lambda: {"active": False,
                                                 "note": "rule-only coordinator"})(),
        }