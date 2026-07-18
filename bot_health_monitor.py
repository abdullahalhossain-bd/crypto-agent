#!/usr/bin/env python3
"""
bot_health_monitor.py — Real-time health monitor for the trading bot
=====================================================================
Run this in a SEPARATE terminal while the bot is running:

    python bot_health_monitor.py

It reads the bot's log file + database + state files and shows:
1. Is the bot alive? (heartbeat check)
2. Is MT5 connected? (last successful account_info)
3. Is equity healthy? (not $0, not in drawdown)
4. Are signals being generated? (per-cycle breakdown)
5. Are trades being placed? (or what's blocking them)
6. Are there errors? (last 10 errors with timestamps)
7. Are circuit breakers tripped?
8. What's the current regime?

Updates every 5 seconds. Press Ctrl+C to stop.
"""
import os
import sys
import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "trading_bot.db"
LOG_PATH = ROOT / "logs" / "system.log"
STATE_PATH = ROOT / "data" / "state.json"
HEARTBEAT_PATH = ROOT / "data" / "heartbeat"
KILL_SWITCH_PATH = ROOT / "data" / "KILL_SWITCH"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def color(text, code):
    """Color codes: 1=red, 2=green, 3=yellow, 4=blue, 5=magenta, 6=cyan"""
    return f"\033[9{code}m{text}\033[0m"


def check_heartbeat():
    """Check if bot is alive via heartbeat file."""
    try:
        if not HEARTBEAT_PATH.exists():
            return False, "heartbeat file missing", 0
        mtime = HEARTBEAT_PATH.stat().st_mtime
        age = time.time() - mtime
        if age > 60:
            return False, f"heartbeat stale ({age:.0f}s old)", age
        return True, "alive", age
    except Exception as e:
        return False, f"error: {e}", 0


def check_kill_switch():
    """Check if kill switch is active."""
    return KILL_SWITCH_PATH.exists()


def check_mt5_from_logs():
    """Check MT5 connection status from recent log entries."""
    if not LOG_PATH.exists():
        return "unknown", "no log file"
    try:
        # Read last 500 lines
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-500:]

        mt5_connected = False
        mt5_errors = 0
        last_mt5_msg = ""
        for line in reversed(lines):
            if "MT5 connected" in line or "account verified" in line:
                mt5_connected = True
                last_mt5_msg = line.strip()[:80]
                break
            if "Terminal: Not found" in line or "MT5 error" in line:
                mt5_errors += 1
                if not last_mt5_msg:
                    last_mt5_msg = line.strip()[:80]
            if mt5_errors > 5:
                return "DISCONNECTED", f"{mt5_errors} MT5 errors in recent logs"

        if mt5_errors > 0:
            return "DEGRADED", f"{mt5_errors} recent errors"
        if mt5_connected:
            return "CONNECTED", last_mt5_msg
        return "unknown", "no MT5 connection/error messages found in recent logs"
    except Exception as e:
        return "unknown", str(e)


def check_equity_from_logs():
    """Extract last known equity from log lines."""
    if not LOG_PATH.exists():
        return None, "no log file"
    try:
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-200:]
        for line in reversed(lines):
            if "CYCLE" in line and "$" in line:
                # Parse: CYCLE 1 │ unknown │ $9971.51 │ scan=20 │ ...
                parts = line.split("│")
                if len(parts) >= 3:
                    equity_str = parts[2].strip().replace("$", "").replace(",", "")
                    try:
                        eq = float(equity_str)
                        return eq, line.strip()[:100]
                    except ValueError:
                        pass
        return None, "no CYCLE line found"
    except Exception as e:
        return None, str(e)


def check_recent_errors():
    """Get last 10 error lines from log."""
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-1000:]
        errors = []
        for line in reversed(lines):
            if "│ ERROR" in line or "│ CRITICAL" in line:
                errors.append(line.strip()[:120])
                if len(errors) >= 10:
                    break
        return errors
    except Exception:
        return []


def check_circuit_breakers_from_logs():
    """Check if any circuit breaker is open."""
    if not LOG_PATH.exists():
        return []
    try:
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-500:]
        open_breakers = []
        for line in reversed(lines):
            if "breaker" in line.lower() and "OPEN" in line:
                open_breakers.append(line.strip()[:120])
        return open_breakers[-5:]  # last 5
    except Exception:
        return []


def check_db_stats():
    """Get database statistics."""
    if not DB_PATH.exists():
        return {"error": "DB not found"}
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=3.0)
        stats = {}
        try:
            cur = conn.execute("SELECT COUNT(*) FROM trades")
            stats["trades"] = cur.fetchone()[0]
        except Exception:
            stats["trades"] = "table missing"
        try:
            cur = conn.execute("SELECT COUNT(*) FROM decisions")
            stats["decisions"] = cur.fetchone()[0]
        except Exception:
            stats["decisions"] = "table missing"
        try:
            cur = conn.execute("SELECT COUNT(*) FROM decisions WHERE approved = 1")
            stats["approved"] = cur.fetchone()[0]
        except Exception:
            stats["approved"] = "?"
        try:
            cur = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE approved = 0"
            )
            stats["rejected"] = cur.fetchone()[0]
        except Exception:
            stats["rejected"] = "?"
        conn.close()
        return stats
    except Exception as e:
        return {"error": str(e)}


def check_last_cycle_stats():
    """Parse last CYCLE line from log."""
    if not LOG_PATH.exists():
        return None
    try:
        with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-200:]
        for line in reversed(lines):
            if "CYCLE" in line and "scan=" in line:
                return line.strip()
        return None
    except Exception:
        return None


def check_state_file():
    """Check state.json for daily loss halt."""
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, 'r') as f:
            state = json.load(f)
        halt_until = state.get("daily_loss_halted_until", 0)
        if halt_until > time.time():
            remaining_h = (halt_until - time.time()) / 3600
            return f"HALTED ({remaining_h:.1f}h remaining)"
        return None
    except Exception:
        return None


def display_health():
    """Main display loop."""
    clear_screen()
    print(color("=" * 70, 6))
    print(color("  🤖 TRADING BOT — HEALTH MONITOR", 6))
    print(color("=" * 70, 6))
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 1. Heartbeat
    alive, hb_msg, hb_age = check_heartbeat()
    if alive:
        print(color(f"  ✅ Bot Status: {hb_msg} (heartbeat {hb_age:.0f}s ago)", 2))
    else:
        print(color(f"  ❌ Bot Status: {hb_msg}", 1))

    # 2. Kill switch
    ks = check_kill_switch()
    if ks:
        print(color("  🛑 KILL SWITCH: ACTIVE — bot will not trade!", 1))
    else:
        print(color("  ✅ Kill Switch: absent (trading enabled)", 2))

    # 3. MT5 connection
    mt5_status, mt5_msg = check_mt5_from_logs()
    if mt5_status == "CONNECTED":
        print(color(f"  ✅ MT5: {mt5_status}", 2))
    elif mt5_status == "DEGRADED":
        print(color(f"  ⚠️  MT5: {mt5_status} — {mt5_msg}", 3))
    elif mt5_status == "DISCONNECTED":
        print(color(f"  ❌ MT5: {mt5_status} — {mt5_msg}", 1))
        print(color("     → Restart MT5 terminal + login to Deriv demo", 1))
    else:
        print(f"  ❓ MT5: {mt5_status} — {mt5_msg}")

    # 4. Equity
    equity, eq_msg = check_equity_from_logs()
    if equity is not None:
        if equity <= 0:
            print(color(f"  ❌ Equity: ${equity:.2f} — MT5 DISCONNECTED!", 1))
        elif equity < 9000:
            print(color(f"  ⚠️  Equity: ${equity:.2f} — drawdown detected", 3))
        else:
            print(color(f"  ✅ Equity: ${equity:.2f}", 2))
    else:
        print(f"  ❓ Equity: {eq_msg}")

    # 5. Daily loss halt
    halt = check_state_file()
    if halt:
        print(color(f"  🛑 Daily Loss Halt: {halt}", 1))

    # 6. Circuit breakers
    breakers = check_circuit_breakers_from_logs()
    if breakers:
        print(color(f"  ⚠️  Circuit Breakers OPEN ({len(breakers)}):", 3))
        for b in breakers[-3:]:
            print(f"     {b}")
    else:
        print(color("  ✅ Circuit Breakers: all closed", 2))

    print()
    print(color("─" * 70, 4))
    print(color("  📊 LAST CYCLE", 4))
    print(color("─" * 70, 4))
    last_cycle = check_last_cycle_stats()
    if last_cycle:
        print(f"  {last_cycle}")
    else:
        print("  (no cycle completed yet)")

    print()
    print(color("─" * 70, 4))
    print(color("  💾 DATABASE STATS", 4))
    print(color("─" * 70, 4))
    db_stats = check_db_stats()
    if "error" in db_stats:
        print(f"  ❌ {db_stats['error']}")
    else:
        print(f"  Total trades:     {db_stats.get('trades', '?')}")
        print(f"  Total decisions:  {db_stats.get('decisions', '?')}")
        print(f"  Approved:         {db_stats.get('approved', '?')}")
        print(f"  Rejected:         {db_stats.get('rejected', '?')}")

    print()
    print(color("─" * 70, 4))
    print(color("  ⚠️  RECENT ERRORS (last 10)", 4))
    print(color("─" * 70, 4))
    errors = check_recent_errors()
    if errors:
        for err in reversed(errors):
            print(f"  {err}")
    else:
        print(color("  ✅ No errors in recent logs", 2))

    print()
    print(color("─" * 70, 4))
    print(color("  💡 DIAGNOSTIC GUIDE", 4))
    print(color("─" * 70, 4))
    print("  If scan=0:          → MT5 disconnected, restart terminal")
    print("  If equity=$0:       → MT5 disconnected, restart terminal")
    print("  If hold_or_low_strength=X: → agents disagree, wait for trend")
    print("  If rej=X:           → risk pipeline rejecting, check reasons")
    print("  If buy=0 always:    → signals too weak OR risk too strict")
    print("  If MT5 errors:      → terminal crashed, restart + re-login")
    print()
    print(color("  Press Ctrl+C to stop monitor", 6))
    print()


def main():
    print("Starting health monitor (updates every 5s)...")
    time.sleep(1)
    try:
        while True:
            display_health()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main()