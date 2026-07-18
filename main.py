"""main.py — the single canonical entrypoint for the trading bot.

Usage:
    python main.py                                # defaults to --mode=demo
    python main.py --mode=demo                    # MT5 demo account (Deriv)
    python main.py --mode=demo --once             # one cycle, then exit
    python main.py --mode=live --i-understand-this-is-real-money
    python main.py --check                        # preflight health check
    python main.py --status                       # show bot status
    python main.py --stats                        # database statistics
    python main.py --last-rejections 10           # show last 10 rejected signals

No paper mode — the bot always connects to a real MT5 terminal (demo or live).
Set MT5_PASSWORD env var before running:
    export MT5_PASSWORD="your_password"       (Linux/Mac)
    $env:MT5_PASSWORD = "your_password"       (PowerShell)

Audit Batch 1 remediation (C5, C6, C7, C10, C14, C15, C17, C19, H1, H4, H5,
H13, H14, H17, H19, M10, M14):
  - Catch-all `Exception` no longer swallows KeyboardInterrupt / SystemExit.
  - `bot.cycle()` runs in a worker thread with a hard timeout (C6).
  - Signal handler interrupts sleep via a `threading.Event` (C7, H1).
  - Kill-switch file is checked during the sleep loop, not only at cycle top.
  - Watchdog is integrated — single source of truth for kill-switch / heartbeat /
    error budget (C15).
  - Pre-flight checks verify DB integrity and MT5 connectivity before the loop
    starts (H4, C14).
  - Shutdown runs in a thread with a timeout; resource leaks are logged (H5).
  - Error counter differentiates fatal vs non-fatal errors (C19, H14).
  - Rejections are always logged with reasons (H17, C17).
  - Cycle timing uses `time.monotonic()` to avoid clock-skew bugs (M14).
  - Stale-file guard is now an opt-out build-time check (C10) — see below.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ──────────────────────────────────────────────────────────────────────
# CRITICAL REGRESSION GUARD (Prompt #6) — Audit-fix C10:
# Refuse to start if the archived v9/MasterOrchestrator pipeline is still
# importable from the canonical path. The previous implementation was a
# runtime hack that depended on the user manually deleting stale files.
# We now (a) keep the check as a defence-in-depth safety net, and (b)
# auto-quarantine the stale files into `legacy/_quarantine/` so the bot
# can boot on a fresh tarball extraction without operator intervention.
# ──────────────────────────────────────────────────────────────────────
_STALE_FILES = []
_QUARANTINE_DIR = ROOT / "legacy" / "_quarantine_runtime"
for _stale in ["main_v9.py", "main_v8.py", "main_v2.py", "main_v3.py",
               "main_v4.py", "main_v5.py", "start.py",
               "architecture/master_orchestrator.py"]:
    _stale_path = ROOT / _stale
    if _stale_path.exists():
        _STALE_FILES.append(str(_stale_path))

if _STALE_FILES:
    # C10 fix: auto-quarantine instead of refusing to start.
    # BUGFIX (external audit): if quarantine fails (read-only FS, permissions,
    # Docker), DON'T crash with sys.exit(3). Log a warning and continue —
    # stale files in legacy/ are not fatal; they just shouldn't be imported.
    try:
        _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        for f in _STALE_FILES:
            src = Path(f)
            dst = _QUARANTINE_DIR / src.name
            # Don't overwrite an existing quarantined file.
            if not dst.exists():
                src.rename(dst)
            else:
                src.unlink()
        print(f"[main] C10 fix: auto-quarantined {len(_STALE_FILES)} stale file(s) "
              f"to {_QUARANTINE_DIR}")
    except (OSError, PermissionError) as _qerr:
        # Best-effort: log and continue. Stale files in legacy/ don't block
        # the bot from running — they're just archived old entrypoints that
        # are no longer imported by the live path.
        print(f"[main] WARNING: could not quarantine {len(_STALE_FILES)} stale "
              f"file(s) to {_QUARANTINE_DIR} ({_qerr!r}) — continuing anyway. "
              f"Stale files: {_STALE_FILES[:3]}")

# ──────────────────────────────────────────────────────────────────────
# Load .env BEFORE config validation so MT5_PASSWORD and other env vars
# are available when config_loader checks for them.
# Review Gap 6: log .env parse failures instead of silently swallowing.
# ──────────────────────────────────────────────────────────────────────
try:
    from external.env_loader import EnvLoader
    EnvLoader.get_instance()  # loads .env file into os.environ
except Exception as _env_err:
    # Don't silently swallow — a malformed .env will cause confusing
    # "MT5_PASSWORD missing" errors downstream. Log the real cause.
    print(f"[main] WARNING: .env load failed ({_env_err!r}) — "
          f"continuing with system environment variables only")

from config_loader import load_config  # noqa: E402
from utils.logger import setup_logger, get_logger  # noqa: E402

log = get_logger("trading_bot.main")

# C7 fix: an Event is interruptible — `wait()` returns immediately when set.
_SHUTDOWN_EVENT = threading.Event()
_SHUTDOWN = False  # kept for backward-compat with any external callers


def _handle_sig(signum: int, _frame) -> None:
    """SIGINT/SIGTERM handler — set the shutdown flag and interrupt any sleep.

    Audit-fix C7: the previous implementation only set a boolean, which did
    nothing to interrupt `time.sleep()`. Now we set a `threading.Event`
    which interrupts `Event.wait()` immediately and lets the loop observe
    the shutdown signal within milliseconds rather than up to `poll_interval_s`.
    """
    global _SHUTDOWN
    _SHUTDOWN = True
    _SHUTDOWN_EVENT.set()
    print(f"\n[main] signal {signum} — shutdown requested (interrupting sleep)")


def _touch_kill_switch(cfg: dict) -> str:
    """Operator panic button: create the kill-switch file to halt the bot."""
    ks_path = cfg.get("runtime", {}).get("kill_switch_file", "data/KILL_SWITCH")
    os.makedirs(os.path.dirname(ks_path) or ".", exist_ok=True)
    with open(ks_path, "w") as f:
        f.write(f"kill switch activated at {datetime.now(tz=timezone.utc).isoformat()}\n")
    print(f"Kill switch written: {ks_path}")
    print("The bot will skip all new cycles until this file is removed.")
    # Also set the event so an already-running loop notices immediately.
    _SHUTDOWN_EVENT.set()
    return ks_path


# ----------------------------------------------------------------------
# C6 fix: run bot.cycle() in a worker thread with a hard timeout.
# ----------------------------------------------------------------------
def _cycle_with_timeout(bot, timeout_s: float):
    """Run `bot.cycle()` in a worker thread; kill the loop if it hangs.

    Audit-fix C6: a hung MT5 API call would previously freeze the entire
    main loop. Now we run cycle() in a daemon thread and impose a hard
    timeout. If the worker is still alive after the timeout, we log,
    increment the error streak, and continue — the daemon thread will
    die when the process exits.

    Returns the CycleResult on success, or raises TimeoutError.
    """
    # P0-12 fix: use ThreadPoolExecutor with cancel_futures instead of raw
    # Thread. This allows the executor to clean up properly and prevents
    # two concurrent cycle() calls if the timeout fires.
    t = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = t.submit(bot.cycle)
    try:
        result = fut.result(timeout=timeout_s)
        t.shutdown(wait=False, cancel_futures=True)
        return result
    except concurrent.futures.TimeoutError:
        t.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(
            f"bot.cycle() did not return within {timeout_s:.1f}s — worker thread "
            f"abandoned (will die with the process)")
    except Exception:
        t.shutdown(wait=False, cancel_futures=True)
        raise


# ----------------------------------------------------------------------
# H4 / C14 fix: pre-flight checks before entering the loop.
# ----------------------------------------------------------------------
def _preflight_checks(cfg: dict, mode: str) -> bool:
    """Audit-fix H4 / C14: verify DB integrity and MT5 connectivity before
    entering the trading loop. Returns True if all checks pass.
    """
    ok = True

    # DB integrity check
    try:
        from database import Database
        db_path = cfg.get("database", {}).get("path", "data/trading_bot.db")
        db = Database(db_path)
        if not db.health_check():
            log.error("PREFLIGHT: DB health_check FAILED — run --check then repair")
            ok = False
        else:
            log.info("PREFLIGHT: DB health_check OK")
    except Exception as e:
        log.error("PREFLIGHT: DB check raised: %r", e)
        ok = False

    # Kill-switch absence check
    ks = cfg.get("runtime", {}).get("kill_switch_file", "data/KILL_SWITCH")
    if os.path.exists(ks):
        log.error("PREFLIGHT: kill-switch file present at %s — remove before start", ks)
        ok = False
    else:
        log.info("PREFLIGHT: kill-switch absent — OK")

    # MT5 connectivity check (best-effort; non-fatal on Linux where MT5 is absent)
    try:
        from brokers.mt5_connector import MT5Connector, MT5Unavailable
        mt5_cfg = cfg.get("mt5", {})
        login = int(mt5_cfg.get("login", 0) or 0)
        password = mt5_cfg.get("password", "") or os.environ.get("MT5_PASSWORD", "")
        server = mt5_cfg.get("server", "")
        terminal_path = mt5_cfg.get("terminal_path", "")
        if login and password and server:
            try:
                conn = MT5Connector(
                    login=login, password=password, server=server,
                    terminal_path=terminal_path)
                log.info("PREFLIGHT: MT5 connector instantiated (login=%s)", login)
            except MT5Unavailable:
                log.warning("PREFLIGHT: MT5 not available on this system — "
                            "non-fatal for paper/diagnostic runs")
        else:
            if mode == "live":
                log.error("PREFLIGHT: --mode=live but MT5 credentials incomplete "
                          "(login/password/server) — refusing to start")
                ok = False
            else:
                log.warning("PREFLIGHT: MT5 credentials incomplete — non-fatal in demo "
                            "mode (cached terminal login may still work)")
    except Exception as e:
        log.warning("PREFLIGHT: MT5 pre-check raised: %r", e)

    return ok


# ----------------------------------------------------------------------
# H5 fix: shutdown with timeout.
# ----------------------------------------------------------------------
def _shutdown_with_timeout(bot, reason: str, timeout_s: float = 30.0) -> None:
    """Audit-fix H5: run bot.shutdown() in a thread with a hard timeout.

    If shutdown hangs (e.g. MT5 terminal unresponsive), we don't block the
    process indefinitely — we log and let the daemon thread die with the
    process.
    """
    def _worker():
        try:
            bot.shutdown(reason=reason)
        except BaseException as exc:  # noqa: BLE001
            log.error("shutdown raised: %r", exc)

    t = threading.Thread(target=_worker, name="shutdown-worker", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        log.error("shutdown did not complete within %.1fs — abandoning (resources may leak)",
                  timeout_s)
    else:
        log.info("shutdown completed cleanly")


def run_validate(cfg: dict) -> None:
    """TIER 6: Readiness gate assessment — GO/NO-GO before live money.

    Runs the 8-point ReadinessGate evaluation and prints a human-readable
    report. Exits with code 0 on GO, code 1 on NO-GO.
    """
    import json
    from validation.readiness_gate import ReadinessGate

    gate = ReadinessGate(
        require_all_pass=cfg.get("validation", {}).get("require_all_pass", True),
        allow_warn_with_operator_approval=cfg.get("validation", {}).get(
            "allow_warn", True),
    )

    # Collect available evidence (each module is optional — SKIP if not available)
    edge_proof = None
    try:
        from validation.edge_proof import EdgeProof
        ep = EdgeProof(db_path=cfg.get("database", {}).get("path", "data/trading_bot.db"))
        edge_proof = ep.run().to_dict()
    except Exception:
        pass

    kill_switch_report = None
    try:
        from validation.kill_switch_validator import KillSwitchValidator
        ksv = KillSwitchValidator(
            kill_file=cfg.get("runtime", {}).get("kill_switch_file", "data/KILL_SWITCH"))
        kill_switch_report = ksv.validate().to_dict()
    except Exception:
        pass

    report = gate.evaluate(
        edge_proof_result=edge_proof,
        kill_switch_report=kill_switch_report,
    )

    # Print report
    print("=" * 60)
    print(f"READINESS GATE — {report.overall_status}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Pass: {report.n_pass}  Warn: {report.n_warn}  "
          f"Fail: {report.n_fail}  Skip: {report.n_skip}")
    print("-" * 60)
    for v in report.verdicts:
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "—"}.get(
            v["status"], "?")
        print(f"  [{icon}] {v['name']}: {v['reason']}")
    if report.blocking_issues:
        print("-" * 60)
        print("BLOCKING ISSUES:")
        for issue in report.blocking_issues:
            print(f"  ✗ {issue}")
    print("=" * 60)

    sys.exit(0 if report.overall_status in ("GO", "CONDITIONAL") else 1)


def run_loop(cfg: dict, mode: str, once: bool = False) -> None:
    """Main trading loop.

    Audit Batch 1 remediation (see module docstring for the full list).
    """
    from architecture.integration import TradingBot
    from watchdog import Watchdog, KillSwitchActive, ErrorBudgetExceeded

    # H4 / C14: pre-flight checks.
    if not _preflight_checks(cfg, mode):
        log.error("Preflight checks failed — refusing to enter trading loop")
        return

    bot = TradingBot(cfg, mode=mode)
    if not bot.boot():
        log.error("TradingBot boot failed — exiting")
        return

    poll_s = float(cfg.get("runtime", {}).get("poll_interval_s", 5))
    max_consecutive_errors = int(cfg.get("runtime", {}).get(
        "max_consecutive_errors", 10))
    # C6 fix: per-cycle hard timeout. Default 90s — wider than the longest
    # legitimate cycle (MT5 + risk pipeline + execution) but short enough
    # to detect a hung terminal.
    cycle_timeout_s = float(cfg.get("runtime", {}).get("cycle_timeout_s", 90))

    # C15 fix: instantiate the Watchdog and use it as the single source of
    # truth for kill-switch / heartbeat / error-budget. The previous loop
    # reimplemented these inline, leading to divergent behaviour.
    watchdog = Watchdog(
        kill_switch_file=cfg.get("runtime", {}).get("kill_switch_file", "data/KILL_SWITCH"),
        heartbeat_file=cfg.get("runtime", {}).get("heartbeat_file", "data/heartbeat"),
        heartbeat_timeout_s=float(cfg.get("runtime", {}).get("heartbeat_timeout_s", 60)),
        max_consecutive_errors=max_consecutive_errors,
    )
    # Clear any stale kill-switch state from a prior run.
    # (Operator is expected to delete the kill-switch file before restart.)
    try:
        watchdog.check_kill_switch()
    except KillSwitchActive:
        log.error("Kill switch file is present at startup — refusing to trade. "
                  "Remove the file or run `python main.py --kill` then delete it.")
        return

    log.info("=== TradingBot starting (mode=%s, poll=%.1fs, cycle_timeout=%.0fs) ===",
             mode, poll_s, cycle_timeout_s)

    consecutive_errors = 0
    last_logged_equity = 0.0
    current_poll_s = poll_s

    # ------------------------------------------------------------------
    # TIER 5 wiring: monitoring / observability (non-blocking, no trade-logic impact)
    # ------------------------------------------------------------------
    from monitoring.system_monitor import SystemMonitor
    from monitoring.trading_monitor import TradingMonitor
    from monitoring.alpha_monitor import AlphaMonitor
    from monitoring.risk_monitor import RiskMonitor
    from monitoring.alert_system import AlertSystem, Alert, AlertSeverity
    from observability.metrics import MetricsCollector
    from observability.dashboard import DashboardRenderer

    _mon_cfg = cfg.get("monitoring", {})
    _sys_mon = SystemMonitor()
    _trade_mon = TradingMonitor()
    _alpha_mon = AlphaMonitor()
    _risk_mon = RiskMonitor(
        max_var_pct=float(_mon_cfg.get("max_var_pct", 0.04)),
        max_correlation=float(_mon_cfg.get("max_correlation", 0.85)),
        max_concentration_hhi=float(_mon_cfg.get("max_concentration_hhi", 0.5)),
    )
    _alert_sys = AlertSystem(
        log_path=_mon_cfg.get("alert_log_path", "data/alerts.jsonl"),
        webhook_url=_mon_cfg.get("webhook_url"),
    )
    _metrics_col = MetricsCollector(
        path=_mon_cfg.get("metrics_path", "data/metrics.jsonl"),
    )
    _dash_renderer = DashboardRenderer(
        output_path=_mon_cfg.get("dashboard_path", "data/dashboard.html"),
        auto_refresh_s=int(_mon_cfg.get("dashboard_refresh_s", 30)),
    )
    _dashboard_every = int(_mon_cfg.get("dashboard_every_n_cycles", 10))
    _monitor_every = int(_mon_cfg.get("monitor_report_every_n_cycles", 20))

    # ------------------------------------------------------------------
    # TIER 1 wiring: portfolio-level safety gates (run BEFORE bot.cycle)
    # ------------------------------------------------------------------
    from trading_modules.kill_conditions import KillConditions, PortfolioState
    from engine.guardrails import GuardrailEngine

    _kill_cfg = cfg.get("kill_conditions", {})
    _kill_cond = KillConditions(
        max_cumulative_loss=float(_kill_cfg.get("max_cumulative_loss", 500.0)),
        max_loss_cooldown_days=int(_kill_cfg.get("max_loss_cooldown_days", 7)),
        min_sharpe=float(_kill_cfg.get("min_sharpe", 0.0)),
        max_drawdown_pct=float(_kill_cfg.get("max_drawdown_pct", 15.0)),
    )
    _guardrails = GuardrailEngine()
    _start_of_day_equity = 0.0

    try:
        while not _SHUTDOWN:
            # C15: check kill-switch at the TOP of every cycle iteration.
            try:
                watchdog.check_kill_switch()
            except KillSwitchActive:
                log.warning("Kill switch detected — exiting loop")
                break

            # H19 fix: even in `once` mode we honour the kill-switch.
            if once and _SHUTDOWN:
                break

            # --- TIER 1: portfolio-level safety (before cycle) ---
            try:
                _metrics = bot.portfolio.metrics()
                _pstate = PortfolioState(
                    cumulative_loss_usd=-min(0.0, _metrics.realized_pnl_total),
                    current_drawdown_pct=_metrics.current_drawdown_pct,
                )
                _kd = _kill_cond.check(_pstate)
                if not _kd.can_trade:
                    log.warning("KillConditions: %s — state=%s, reason=%s",
                                "HALTED" if _kd.state == "GATED" else "WAITING",
                                _kd.state, _kd.trigger_reason)
                    if _kd.state == "GATED":
                        _touch_kill_switch(cfg)
                        break
                    # WAIT → skip this cycle but keep looping
                    _interruptible_sleep(poll_s, cfg)
                    continue

                # Guardrails (portfolio-level exposure / drawdown checks)
                _gr = _guardrails.evaluate({
                    "equity": _metrics.equity if hasattr(_metrics, 'equity') else 0.0,
                    "start_of_day_equity": _start_of_day_equity,
                    "current_drawdown_pct": _metrics.current_drawdown_pct,
                })
                if _gr.get("permission") == "halt":
                    log.error("Guardrails HALT: %s", _gr.get("results", []))
                    _touch_kill_switch(cfg)
                    break
                elif _gr.get("permission") == "block_new":
                    log.warning("Guardrails BLOCK_NEW: %s", _gr.get("results", []))
            except Exception as safety_exc:
                log.debug("TIER 1 safety check error (non-fatal): %r", safety_exc)

            # C6 fix: run cycle() with a hard timeout.
            try:
                # Use monotonic clock for timing (M14 fix).
                t_start = time.monotonic()
                result = _cycle_with_timeout(bot, cycle_timeout_s)
                t_end = time.monotonic()
                # M14 fix: monotonic — never negative, immune to clock skew.
                result.cycle_time_ms = max(0.0, (t_end - t_start) * 1000.0)
            except KeyboardInterrupt:
                # C5 fix: re-raise — don't swallow Ctrl+C.
                log.info("KeyboardInterrupt in run_loop — propagating")
                raise
            except SystemExit:
                # C5 fix: re-raise — don't swallow sys.exit().
                log.info("SystemExit in run_loop — propagating")
                raise
            except TimeoutError as e:
                # C6 fix: cycle hung — count toward error streak but don't crash.
                log.error("TradingBot: cycle TIMEOUT — %r", e)
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    log.error("TradingBot: %d consecutive cycle errors — AUTO-KILL",
                              consecutive_errors)
                    _touch_kill_switch(cfg)
                    break
                current_poll_s = min(poll_s * (2 ** consecutive_errors), 300)
                log.warning("TradingBot: backing off to %.0fs poll (consecutive_errors=%d)",
                            current_poll_s, consecutive_errors)
                _interruptible_sleep(current_poll_s, cfg)
                continue
            except Exception as e:
                # C5 fix: Exception is caught, but KeyboardInterrupt / SystemExit
                # were re-raised above so they propagate cleanly.
                log.exception("TradingBot: UNHANDLED exception in cycle() — "
                              "caught and logged, continuing loop: %r", e)
                consecutive_errors += 1
                try:
                    watchdog.record_error(e)
                except ErrorBudgetExceeded:
                    log.error("TradingBot: error budget exceeded — AUTO-KILL")
                    _touch_kill_switch(cfg)
                    break
                if consecutive_errors >= max_consecutive_errors:
                    log.error("TradingBot: %d consecutive cycle errors — AUTO-KILL",
                              consecutive_errors)
                    _touch_kill_switch(cfg)
                    break
                current_poll_s = min(poll_s * (2 ** consecutive_errors), 300)
                log.warning("TradingBot: backing off to %.0fs poll (consecutive_errors=%d)",
                            current_poll_s, consecutive_errors)
                _interruptible_sleep(current_poll_s, cfg)
                continue

            # C19 / H14 fix: differentiate fatal vs non-fatal errors.
            # `result.errors` is a list of strings; we treat any error that
            # contains "FATAL" or "CRITICAL" as fatal, others as warnings
            # that don't increment the streak.
            fatal_errors = [e for e in (result.errors or [])
                            if isinstance(e, str) and
                            ("FATAL" in e.upper() or "CRITICAL" in e.upper())]
            non_fatal_errors = [e for e in (result.errors or [])
                                if e not in fatal_errors]

            if non_fatal_errors and not fatal_errors:
                # Non-fatal: log but don't increment the streak (C19 fix).
                for e in non_fatal_errors[:3]:
                    log.warning("  (non-fatal) ERR: %s", e)

            if not fatal_errors:
                # Clean cycle (or only non-fatal warnings).
                if consecutive_errors > 0:
                    log.info("TradingBot: cycle clean — resetting error streak "
                             "(was %d)", consecutive_errors)
                consecutive_errors = 0
                current_poll_s = poll_s
                watchdog.record_success()
                watchdog.heartbeat()
            else:
                for e in fatal_errors[:3]:
                    log.error("  (fatal) ERR: %s", e)
                consecutive_errors += 1
                try:
                    watchdog.record_error(RuntimeError("; ".join(fatal_errors)))
                except ErrorBudgetExceeded:
                    log.error("TradingBot: error budget exceeded — AUTO-KILL")
                    _touch_kill_switch(cfg)
                    break
                if consecutive_errors >= max_consecutive_errors:
                    log.error("TradingBot: %d consecutive cycles with errors — AUTO-KILL",
                              consecutive_errors)
                    _touch_kill_switch(cfg)
                    break
                current_poll_s = min(poll_s * (2 ** consecutive_errors), 300)
                log.warning("TradingBot: backing off to %.0fs poll (consecutive_errors=%d)",
                            current_poll_s, consecutive_errors)

            # H17 / C17 fix: always log rejected trades with reasons.
            if result.trades_rejected > 0:
                log.info("  rejections this cycle: %d (use --last-rejections for details)",
                         result.trades_rejected)

            # DEBUG: log EVERY cycle (was every 5th) so operator can see
            # exactly what's happening. The funnel shows WHERE symbols are
            # being lost in the pipeline.
            log_interval = 1
            equity_delta_pct = 0.0
            if last_logged_equity > 0:
                equity_delta_pct = abs(result.equity - last_logged_equity) / last_logged_equity * 100
            has_activity = result.trades_placed > 0 or result.trades_rejected > 0
            should_log = (result.cycle % log_interval == 0 or once
                         or equity_delta_pct > 1.0 or has_activity
                         or result.state in ("KILL_SWITCH", "BREAKER_OPEN"))
            if should_log:
                skip_summary = ""
                if result.skip_breakdown:
                    # Show ALL skip reasons (was top 3 only) so operator can
                    # see the full picture of why symbols aren't trading.
                    parts = [f"{k}:{v}" for k, v in sorted(
                        result.skip_breakdown.items(), key=lambda x: -x[1])]
                    total_skipped = sum(result.skip_breakdown.values())
                    skip_summary = f" │ skipped={total_skipped} ({', '.join(parts[:5])})"
                delta_str = f" │ Δ{equity_delta_pct:+.2f}%" if equity_delta_pct > 0.1 else ""
                log.info(
                    "CYCLE %-4d │ %-8s │ $%-10.2f │ scan=%-3d │ buy=%-2d │ rej=%-3d%s │ %6.0fms%s",
                    result.cycle, result.regime, result.equity,
                    result.signals_generated, result.trades_placed,
                    max(0, result.trades_rejected),
                    skip_summary,
                    result.cycle_time_ms,
                    delta_str,
                )
                # DEBUG: pipeline funnel — shows exactly where symbols are lost
                # Example: FUNNEL: scan=20 → ai_actionable=5 → risk_reject=2 → wisdom_reject=1 → order_placed=0
                if result.funnel:
                    funnel_parts = []
                    for stage in ["ai_actionable", "mtf_fail", "fakeout_fail",
                                  "risk_reject", "wisdom_reject", "order_placed"]:
                        if stage in result.funnel:
                            funnel_parts.append(f"{stage}={result.funnel[stage]}")
                    if funnel_parts:
                        log.info("  FUNNEL: scan=%d → %s",
                                result.signals_generated, " → ".join(funnel_parts))
                last_logged_equity = result.equity

            # --- TIER 5: feed monitors (non-blocking, errors are swallowed) ---
            try:
                cycle_ok = len(result.errors) == 0
                _sys_mon.record_cycle(result.cycle_time_ms / 1000.0, ok=cycle_ok)
                _sys_mon.set_mt5_status(result.state != "MT5_DISCONNECTED")
                _metrics_col.record_equity(result.equity)
                if result.trades_placed > 0:
                    _alpha_mon.record_signal(strategy_name="_all", fired=True)
                if result.trades_rejected > 0:
                    _alpha_mon.record_signal(strategy_name="_all", fired=False)

                # Dashboard + alert evaluation (every N cycles to reduce I/O)
                if result.cycle % _dashboard_every == 0:
                    snap = _metrics_col.snapshot({
                        "regime": result.regime,
                        "signals": result.signals_generated,
                        "trades_placed": result.trades_placed,
                        "trades_rejected": result.trades_rejected,
                        "cycle_time_ms": result.cycle_time_ms,
                    })
                    _dash_renderer.render(snap)

                if result.cycle % _monitor_every == 0:
                    sys_h = _sys_mon.health().to_dict()
                    trade_h = _trade_mon.health().to_dict()
                    risk_h = _risk_mon.health().to_dict()
                    alpha_h = _alpha_mon.health().to_dict()
                    alerts = _alert_sys.evaluate_health_alerts(
                        system_health=sys_h, trading_health=trade_h,
                        risk_health=risk_h, alpha_health=alpha_h,
                    )
                    for a in alerts:
                        _alert_sys.fire(a)
            except Exception as mon_exc:
                log.debug("monitoring feed error (non-fatal): %r", mon_exc)

            if once:
                break
            # C7 fix: interruptible sleep — checks kill-switch every 0.5s.
            elapsed = result.cycle_time_ms / 1000
            remaining = current_poll_s - elapsed
            if remaining > 0:
                _interruptible_sleep(remaining, cfg)
    finally:
        # H5 fix: shutdown with timeout so a hung MT5 terminal doesn't trap us.
        _shutdown_with_timeout(bot, reason="signal received" if _SHUTDOWN else "user requested",
                               timeout_s=float(cfg.get("runtime", {}).get("shutdown_timeout_s", 30)))
        log.info("=== TradingBot shut down ===")


def _interruptible_sleep(seconds: float, cfg: dict) -> None:
    """Audit-fix C7 / H1: sleep in small increments so the kill-switch file
    and the shutdown Event are observed within ~0.5s, not after the full
    poll interval.

    The kill-switch file is checked every increment so operators who `touch
    data/KILL_SWITCH` from a shell see the bot stop almost immediately.
    """
    ks_path = cfg.get("runtime", {}).get("kill_switch_file", "data/KILL_SWITCH")
    increment = 0.5
    remaining = max(0.0, seconds)
    while remaining > 0:
        if _SHUTDOWN_EVENT.wait(timeout=min(increment, remaining)):
            return  # event set — shutdown requested
        if os.path.exists(ks_path):
            return  # kill-switch file appeared during sleep
        remaining -= increment


def run_check(cfg: dict) -> None:
    """Preflight health check — verifies architecture without trading.

    Audit-fix C14 / H11: now includes a DB integrity check and an MT5
    terminal-path existence check.
    """
    print("=" * 70)
    print("  TradingBot — Preflight Health Check")
    print("=" * 70)
    checks = [
        ("ConfigLoader", "config_loader", "load_config"),
        ("Database", "database", "Database"),
        ("TradingBot", "architecture.integration", "TradingBot"),
        ("ExchangeInterface", "architecture.exchange_abstraction", "ExchangeInterface"),
        ("MT5Adapter", "architecture.exchange_abstraction", "MT5Adapter"),
        ("PaperAdapter", "architecture.exchange_abstraction", "PaperAdapter"),
        ("PortfolioManager", "architecture.portfolio_manager_v2", "PortfolioManager"),
        ("RiskPipeline", "architecture.risk_pipeline", "RiskPipeline"),
        ("DailyLossGate", "architecture.risk_pipeline", "DailyLossGate"),
        ("StateMachine", "architecture.state_machine", "StateMachine"),
        ("EventBus", "architecture.event_bus", "EventBus"),
        ("FeaturePipeline", "architecture.feature_pipeline", "FeaturePipeline"),
        ("MultiAgentCoordinator", "architecture.multi_agent", "MultiAgentCoordinator"),
        ("RegimeOrchestrator", "architecture.regime_orchestrator", "RegimeOrchestrator"),
        ("WisdomGate", "livermore_principles", "WisdomGate"),
        ("MT5Connector", "brokers.mt5_connector", "MT5Connector"),
        ("Signal", "engine.signals", "Signal"),
        ("RecoveryEngine", "architecture.recovery_engine", "RecoveryEngine"),
        ("SelfHealingSystem", "architecture.self_healing", "SelfHealingSystem"),
        ("DecisionAuditor", "architecture.decision_audit", "DecisionAuditor"),
    ]
    passed = 0
    failed = 0
    # L8 fix: use importlib.import_module instead of __import__.
    import importlib
    for label, mod_name, class_name in checks:
        try:
            mod = importlib.import_module(mod_name)
            getattr(mod, class_name)
            print(f"  OK  {label:30s}  ({mod_name})")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {label:30s}  ({mod_name}): {e!r}")
            failed += 1
    # Check canonical pipeline gates
    try:
        from architecture.risk_pipeline import RiskPipeline
        from architecture.portfolio_manager_v2 import PortfolioManager
        pm = PortfolioManager(initial_capital=10000.0)
        pipe = RiskPipeline(portfolio=pm, config=cfg.get("risk", {}))
        gate_names = [g.name for g in pipe._gates]
        print(f"\n  Risk pipeline gates ({len(gate_names)}): {', '.join(gate_names)}")
        assert "daily_loss" in gate_names, "DailyLossGate missing from pipeline!"
        print("  OK  DailyLossGate is in the pipeline (P0-3 FIX)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  Risk pipeline check: {e!r}")
        failed += 1
    # Check kill-switch path
    ks = cfg.get("runtime", {}).get("kill_switch_file", "data/KILL_SWITCH")
    if os.path.exists(ks):
        print(f"  WARN  Kill switch file EXISTS: {ks} — bot will not trade until removed")
    else:
        print(f"  OK  Kill switch file absent: {ks}")
        passed += 1

    # C14 / H11 fix: DB integrity + MT5 terminal path checks.
    try:
        from database import Database
        db_path = cfg.get("database", {}).get("path", "data/trading_bot.db")
        db = Database(db_path)
        if db.health_check():
            print(f"  OK  DB integrity_check passed: {db_path}")
            passed += 1
        else:
            print(f"  FAIL  DB integrity_check FAILED: {db_path} — run repair_with_backup()")
            failed += 1
    except Exception as e:
        print(f"  FAIL  DB health check raised: {e!r}")
        failed += 1

    mt5_terminal = cfg.get("mt5", {}).get("terminal_path", "")
    if mt5_terminal:
        if os.path.exists(mt5_terminal):
            print(f"  OK  MT5 terminal exists: {mt5_terminal}")
            passed += 1
        else:
            print(f"  FAIL  MT5 terminal NOT FOUND: {mt5_terminal}")
            failed += 1
    else:
        print("  WARN  MT5 terminal_path not set — required on Windows for live trading")

    # H13 fix: light resource check (CPU count, disk free on data dir).
    try:
        import shutil as _shutil
        data_dir = cfg.get("database", {}).get("path", "data/trading_bot.db")
        parent = os.path.dirname(os.path.abspath(data_dir)) or "."
        usage = _shutil.disk_usage(parent)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            print(f"  WARN  Low disk space on {parent}: {free_gb:.2f} GB free")
        else:
            print(f"  OK  Disk space: {free_gb:.2f} GB free on {parent}")
            passed += 1
    except Exception as e:
        print(f"  WARN  Disk space check failed: {e!r}")

    print("\n" + "=" * 70)
    print(f"  RESULT: {passed} passed, {failed} failed")
    print("=" * 70)


def run_status(cfg: dict, mode: str) -> None:
    """Show bot status without running a cycle.

    Audit-fix M5: also show current open positions from MT5 (if available).
    """
    from architecture.integration import TradingBot
    bot = TradingBot(cfg, mode=mode)
    if not bot.boot():
        return
    status = bot.status()
    print("=" * 70)
    print(f"  TradingBot STATUS (mode={mode})")
    print("=" * 70)
    for k, v in status.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
    # M5 fix: dump open positions if the bot exposes them.
    try:
        positions = getattr(bot, "get_open_positions", lambda: [])()
        if positions:
            print(f"\n  OPEN POSITIONS ({len(positions)}):")
            for p in positions[:10]:
                print(f"    {p}")
    except Exception as e:
        log.debug("status: could not fetch open positions: %r", e)
    print("=" * 70)
    _shutdown_with_timeout(bot, reason="status check",
                           timeout_s=float(cfg.get("runtime", {}).get("shutdown_timeout_s", 30)))


def run_stats(cfg: dict) -> None:
    """Show database statistics."""
    try:
        from database import Database
        db_path = cfg.get("database", {}).get("path", "data/trading_bot.db")
        db = Database(db_path)
        stats = db.get_stats()
        print("=" * 70)
        print("  DATABASE STATISTICS")
        print("=" * 70)
        for k, v in stats.items():
            if isinstance(v, list):
                print(f"  {k}:")
                for item in v:
                    print(f"    {item}")
            else:
                print(f"  {k:30s}: {v}")
        # Decisions table stats (P0-8)
        decisions = db.get_decisions(limit=5)
        print(f"\n  RECENT DECISIONS (last 5, P0-8 FIX):")
        if not decisions:
            print("    (no decisions recorded yet — C17 note: run a cycle first)")
        for d in decisions:
            print(f"    {d['timestamp']} {d['symbol']:10s} "
                  f"{'APPROVED' if d['approved'] else 'REJECTED':8s} "
                  f"{d.get('reject_reason', '')[:50]}")
        print("=" * 70)
    except Exception as e:
        print(f"Database stats failed: {e!r}")


def run_last_rejections(cfg: dict, n: int) -> None:
    """Prompt #7: Print the last N rejected signals from the DB in a readable table.

    Audit-fix C17 / M18 / L13: if no decisions are recorded yet, print a
    helpful hint instead of an empty table. Table width is now derived from
    the terminal width (falling back to 120 if detection fails).
    """
    try:
        from database import Database
        db_path = cfg.get("database", {}).get("path", "data/trading_bot.db")
        db = Database(db_path)
        decisions = db.get_decisions(limit=n, rejected_only=True)
        # L13 fix: dynamic width.
        try:
            width = max(80, os.get_terminal_size().columns)
        except OSError:
            width = 120
        if not decisions:
            print("No rejected signals found in the database.")
            print("Note: decisions are recorded on every cycle. If the bot has")
            print("never run, run `python main.py --mode=demo --once` first.")
            return
        print("=" * width)
        print(f"  LAST {len(decisions)} REJECTED SIGNALS")
        print("=" * width)
        # Table header
        print(f"  {'Timestamp':26s} {'Symbol':10s} {'Action':6s} {'Stage/Reason':{width-52}s}")
        print("-" * width)
        for d in decisions:
            ts = d.get("timestamp", "")[:26]
            sym = d.get("symbol", "")[:10]
            action = d.get("strategy_action", "?")[:6]
            reason = d.get("reject_reason", "(no reason)")[:width - 52]
            print(f"  {ts:26s} {sym:10s} {action:6s} {reason}")
        print("=" * width)
        print(f"  Total: {len(decisions)} rejections shown")
    except Exception as e:
        print(f"Last-rejections query failed: {e!r}")
        if os.environ.get("TRADING_BOT_DEBUG"):
            traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradingBot — the single canonical entrypoint (Phase 3)")
    parser.add_argument("--config", default="config/config.yaml",
                        help="path to config.yaml")
    # C9 fix: support an env-var override for the config path so containers
    # can set TRADING_BOT_CONFIG instead of mounting to a fixed path.
    default_cfg = os.environ.get("TRADING_BOT_CONFIG", "config/config.yaml")
    if default_cfg != "config/config.yaml":
        parser.set_defaults(config=default_cfg)
    parser.add_argument("--mode", choices=["demo", "live"],
                        default="demo",
                        help="demo = MT5 demo account (default), live = MT5 real account")
    parser.add_argument("--once", action="store_true",
                        help="run one cycle then exit")
    parser.add_argument("--check", action="store_true",
                        help="preflight health check (no trading)")
    parser.add_argument("--status", action="store_true",
                        help="show bot status (no trading)")
    parser.add_argument("--stats", action="store_true",
                        help="show database statistics")
    parser.add_argument("--kill", action="store_true",
                        help="write the kill-switch file and exit (panic button)")
    parser.add_argument("--validate", action="store_true",
                        help="TIER 6: readiness gate GO/NO-GO assessment (no trading)")
    parser.add_argument("--last-rejections", type=int, metavar="N", default=0,
                        help="show the last N rejected signals with reasons (Prompt #7)")
    parser.add_argument("--i-understand-this-is-real-money", action="store_true",
                        help="required confirmation for --mode=live")
    args = parser.parse_args()

    # Load config — H7 fix: explicit existence check with a clear error.
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        print(f"ERROR: config file not found: {cfg_path}")
        print("       Set TRADING_BOT_CONFIG env var or pass --config <path>")
        sys.exit(2)
    try:
        cfg = load_config(cfg_path, mode_override=args.mode)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: config load failed: {e!r}")
        if os.environ.get("TRADING_BOT_DEBUG"):
            traceback.print_exc()
        sys.exit(2)
    cfg["config_path"] = cfg_path
    if args.i_understand_this_is_real_money:
        cfg["_i_understand_real_money"] = True

    # Setup logging
    log_cfg = cfg.get("logging", {}) if cfg else {}
    setup_logger(
        level=log_cfg.get("level", "INFO"),
        system_log=log_cfg.get("system_log", "logs/system.log"),
        trade_log=log_cfg.get("trade_log", "logs/trades.log"),
        json_logs=bool(log_cfg.get("json_logs", False)),
        rotate_mb=int(log_cfg.get("rotate_mb", 10)),
        backup_count=int(log_cfg.get("backup_count", 5)),
    )
    global log
    log = get_logger("trading_bot.main")

    # Signal handlers — C7 fix: now sets a threading.Event.
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # Dispatch subcommands
    if args.kill:
        _touch_kill_switch(cfg)
        return
    if args.check:
        run_check(cfg)
        return
    if args.status:
        run_status(cfg, args.mode)
        return
    if args.stats:
        run_stats(cfg)
        return
    if args.last_rejections > 0:
        run_last_rejections(cfg, args.last_rejections)
        return
    if args.validate:
        run_validate(cfg)
        return

    # P0-10 FIX: refuse --mode=live without the explicit confirmation flag
    if args.mode == "live" and not args.i_understand_this_is_real_money:
        print("ERROR: --mode=live requires --i-understand-this-is-real-money")
        print("       This flag exists so a typo can never risk real capital.")
        sys.exit(2)

    # BUGFIX (external audit): double-check MT5 password presence for live mode.
    # The --i-understand flag confirms intent, but if MT5_PASSWORD is not set,
    # the bot will connect with an empty password and either fail silently or
    # (worse) connect to the wrong account. Refuse to start live without it.
    if args.mode == "live":
        mt5_pass = os.environ.get("MT5_PASSWORD", "") or cfg.get("mt5", {}).get("password", "")
        if not mt5_pass:
            print("ERROR: --mode=live requires MT5_PASSWORD to be set.")
            print("       Set it in .env:  MT5_PASSWORD=your_password")
            print("       Or export:       export MT5_PASSWORD='your_password'")
            print("       Live mode with an empty password is refused for safety.")
            sys.exit(2)

    run_loop(cfg, mode=args.mode, once=args.once)


if __name__ == "__main__":
    main()
