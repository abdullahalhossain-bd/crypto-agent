"""utils.logger
=====================================================================
Structured logging for the trading bot.

Two streams:
  - system.log : all runtime messages (connection, signals, errors)
  - trades.log : one line per order/trade event (sent, filled, closed)

Design choices:
  - RotatingFileHandler so logs never grow unbounded.
  - Optional JSON formatter for ingest into ELK / Loki.
  - Always logs to stdout as well, so headless runs are debuggable.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional


# ----------------------------------------------------------------------
# Formatters
# ----------------------------------------------------------------------
class _TextFormatter(logging.Formatter):
    """Human-readable, single-line, with UTC timestamp (ms precision) + level.

    Millisecond precision matters here: the bot can place several orders
    or evaluate several symbols within the same wall-clock second, and
    second-only timestamps made it impossible to reconstruct the true
    event order from the log file alone.
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d UTC | %(levelname)-7s | %(name)-28s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.converter = lambda *a: datetime.now(tz=timezone.utc).timetuple()

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        return super().format(record)


class _JsonFormatter(logging.Formatter):
    """One JSON object per line — friendly to log shippers."""

    RESERVED = {"name", "msg", "args", "levelname", "levelno", "created",
                "msecs", "relativeCreated", "exc_info", "exc_text",
                "stack_info", "asctime"}

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                  .isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self.RESERVED and not k.startswith("_"):
                try:
                    json.dumps(v)
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = str(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def setup_logger(
    name: str = "trading_bot",
    level: str = "INFO",
    system_log: str = "logs/system.log",
    trade_log: str = "logs/trades.log",
    json_logs: bool = False,
    rotate_mb: int = 10,
    backup_count: int = 5,
) -> tuple[logging.Logger, logging.Logger]:
    """Build (or rebuild) the system + trade loggers.

    UI fix: Console handler is INFO+ (clean — only cycle summaries, trades,
    errors). File handler is DEBUG (everything — per-symbol skips, internal
    diagnostics) for post-hoc debugging. This prevents 100 SKIP lines per
    cycle from spamming the console while keeping them in the log file.
    """
    # Make sure log dirs exist
    os.makedirs(os.path.dirname(system_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(trade_log) or ".", exist_ok=True)

    fmt_cls = _JsonFormatter if json_logs else _TextFormatter

    # Console formatter: cleaner, shorter — no logger name (just time/level/msg).
    # Color-codes the level so a scrolling terminal is scannable at a glance
    # (errors jump out red, warnings amber) without touching the file logs,
    # which stay plain text for grep/tooling. Disabled automatically when
    # stdout isn't a real terminal (e.g. redirected to a file, run under
    # systemd/supervisor) or when NO_COLOR is set, so piped output never
    # contains stray escape codes.
    _LEVEL_COLORS = {
        "DEBUG": "\033[2m",       # dim
        "INFO": "\033[0m",        # default
        "WARNING": "\033[33m",    # yellow
        "ERROR": "\033[31;1m",    # bold red
        "CRITICAL": "\033[41;97;1m",  # white-on-red
    }
    _RESET = "\033[0m"
    _use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    class _ConsoleFormatter(logging.Formatter):
        def __init__(self) -> None:
            super().__init__(
                fmt="%(asctime)s │ %(levelname)-7s │ %(message)s",
                datefmt="%H:%M:%S",
            )
            self.converter = lambda *a: datetime.now(tz=timezone.utc).timetuple()

        def format(self, record: logging.LogRecord) -> str:
            line = super().format(record)
            if _use_color:
                color = _LEVEL_COLORS.get(record.levelname, "")
                if color:
                    line = f"{color}{line}{_RESET}"
            return line

    # ---------------- system logger ----------------
    sys_logger = logging.getLogger(name)
    # Root level = DEBUG so file handler gets everything
    sys_logger.setLevel(logging.DEBUG)
    # remove old handlers (re-init scenario)
    for h in list(sys_logger.handlers):
        sys_logger.removeHandler(h)

    # Console: INFO+ only, clean format
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    sh.setFormatter(_ConsoleFormatter())
    sys_logger.addHandler(sh)

    # File: DEBUG+ (everything), full format with logger name
    fh = logging.handlers.RotatingFileHandler(
        system_log, maxBytes=rotate_mb * 1024 * 1024,
        backupCount=backup_count, encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt_cls())
    sys_logger.addHandler(fh)
    sys_logger.propagate = False

    # ---------------- trade logger ----------------
    tr_logger = logging.getLogger(name + ".trades")
    tr_logger.setLevel(logging.INFO)
    for h in list(tr_logger.handlers):
        tr_logger.removeHandler(h)
    tfh = logging.handlers.RotatingFileHandler(
        trade_log, maxBytes=rotate_mb * 1024 * 1024,
        backupCount=backup_count, encoding="utf-8",
    )
    tfh.setFormatter(fmt_cls())
    tr_logger.addHandler(tfh)
    tr_logger.propagate = False
    return sys_logger, tr_logger


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """Retrieve a child logger without re-initialising handlers.

    Auto-prefixes with 'trading_bot.' so the logger inherits the
    handlers attached by `setup_logger` (which configures the
    'trading_bot' parent). This keeps call sites short —
    `get_logger("engine.data_feed")` → logger "trading_bot.engine.data_feed".
    """
    if not name.startswith("trading_bot"):
        name = f"trading_bot.{name}"
    return logging.getLogger(name)


def get_trade_logger() -> logging.Logger:
    return logging.getLogger("trading_bot.trades")


def _kv_format(value: Any) -> str:
    """Render a single trade-log field value, quoting it if it contains
    whitespace so `key=value key2=value2` parsing stays unambiguous.

    BUG FIX: symbol names on this broker routinely contain spaces
    (e.g. "BTCUSD RSI Trend Down Index", "Crash 100 Index"). Emitting
    those unquoted — `symbol=Crash 100 Index side=SELL` — makes it
    impossible for any downstream parser (or a human with `awk`) to know
    where the symbol value ends and the next field begins. Quoting only
    when needed keeps the common case (numbers, tickers without spaces)
    exactly as before.
    """
    text = str(value)
    if " " in text or "\t" in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


def log_trade(event: str, **fields: Any) -> None:
    """Helper: emit one structured trade event."""
    msg = f"TRADE_EVENT={event}"
    if fields:
        msg += " " + " ".join(f"{k}={_kv_format(v)}" for k, v in fields.items())
    get_trade_logger().info(msg, extra={"event": event, **fields})
