"""external.telegram_bot
=====================================================================
Day 178-180 — Telegram bot for alerts + remote commands.

Two-way integration:
  1. ALERTS (bot → user):
     - Trade opened/closed
     - Kill switch activated
     - Drawdown breach
     - Daily summary
     - Error budget low

  2. COMMANDS (user → bot):
     /status     — current bot status (tier, equity, positions)
     /positions  — list open positions
     /pause      — pause trading (set tier to PAPER)
     /resume     — resume trading (restore previous tier)
     /kill       — activate kill switch
     /disarm     — disarm kill switch
     /promote    — promote capital tier (e.g. /promote MICRO)
     /stats      — show performance stats
     /help       — list commands

The bot runs a polling thread that listens for commands. Alerts are
sent via the send_message() method (also used by the notification system).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from external.env_loader import env
from utils.logger import get_logger

log = get_logger("external.telegram")


@dataclass
class TelegramCommand:
    command: str
    args: list[str]
    chat_id: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "args": list(self.args),
            "chat_id": self.chat_id,
            "timestamp": self.timestamp,
        }


# ----------------------------------------------------------------------
class TelegramBot:
    """Two-way Telegram bot."""

    def __init__(self) -> None:
        self.token = env.telegram_token
        self.chat_id = env.telegram_chat_id
        self.enabled = env.enable_telegram and bool(self.token)
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.token else ""
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._last_update_id = 0
        self._command_handlers: dict[str, Callable] = {}
        self._running = False
        # Authorization: only these chat ids may issue commands. Defaults to
        # just the configured owner chat id if no explicit list is set.
        self._authorized_chat_ids: set[str] = set(env.telegram_authorized_chat_ids)
        if self.enabled and not self._authorized_chat_ids:
            log.warning(
                "Telegram bot enabled but no TELEGRAM_CHAT_ID / "
                "TELEGRAM_AUTHORIZED_CHAT_IDS configured — all incoming "
                "commands will be rejected until this is set."
            )
        if self.enabled:
            self._register_default_commands()

    # ----------------------------------------------------------------
    # Sending messages
    # ----------------------------------------------------------------
    def send_message(self, text: str, chat_id: Optional[str] = None,
                       parse_mode: Optional[str] = None) -> bool:
        if not self.enabled:
            return False
        target_chat = chat_id or self.chat_id
        if not target_chat:
            return False
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text[:4096],  # Telegram limit
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        return self._http_post(url, payload)

    def send_alert(self, title: str, body: str,
                     severity: str = "INFO") -> bool:
        """Send a formatted alert message."""
        emoji = {"INFO": "ℹ️", "WARN": "⚠️", "CRITICAL": "🚨"}.get(severity, "ℹ️")
        text = f"{emoji} *{title}*\n\n{body}\n\n_{severity} • {datetime.now(tz=timezone.utc).isoformat()}_"
        return self.send_message(text, parse_mode="Markdown")

    # ----------------------------------------------------------------
    # Polling for commands
    # ----------------------------------------------------------------
    def start_polling(self) -> None:
        if not self.enabled or self._running:
            return
        self._running = True
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        log.info("Telegram bot polling started")

    def stop_polling(self) -> None:
        self._poll_stop.set()
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5.0)
        log.info("Telegram bot polling stopped")

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            try:
                self._poll_once()
            except Exception as e:  # noqa: BLE001
                log.warning("Telegram poll error: %r", e)
            self._poll_stop.wait(timeout=2.0)  # poll every 2s

    def _poll_once(self) -> None:
        url = (f"{self.base_url}/getUpdates"
               f"?offset={self._last_update_id + 1}&timeout=1")
        data = self._http_get(url)
        if not data or not data.get("ok"):
            return
        for update in data.get("result", []):
            self._last_update_id = update.get("update_id", self._last_update_id)
            message = update.get("message", {})
            text = message.get("text", "")
            chat_id = str(message.get("chat", {}).get("id", ""))
            if not text or not chat_id:
                continue
            # Parse command
            if text.startswith("/"):
                parts = text[1:].split()
                cmd = parts[0].lower() if parts else ""
                args = parts[1:] if len(parts) > 1 else []
                command = TelegramCommand(
                    command=cmd, args=args, chat_id=chat_id,
                    timestamp=datetime.now(tz=timezone.utc).isoformat(),
                )
                self._handle_command(command)

    # ----------------------------------------------------------------
    def _is_authorized(self, chat_id: str) -> bool:
        return chat_id in self._authorized_chat_ids

    # ----------------------------------------------------------------
    def _handle_command(self, command: TelegramCommand) -> None:
        if not self._is_authorized(command.chat_id):
            log.warning(
                "Rejected unauthorized Telegram command '/%s' from chat_id=%s",
                command.command, command.chat_id,
            )
            self.send_message(
                "⛔ Unauthorized. This bot is restricted to its configured owner(s).",
                chat_id=command.chat_id,
            )
            return
        handler = self._command_handlers.get(command.command)
        if handler:
            try:
                response = handler(command)
                if response:
                    self.send_message(response, chat_id=command.chat_id)
            except Exception as e:  # noqa: BLE001
                self.send_message(f"Error: {e!r}", chat_id=command.chat_id)
        else:
            self.send_message(
                f"Unknown command: /{command.command}\nUse /help for available commands",
                chat_id=command.chat_id,
            )

    # ----------------------------------------------------------------
    # Command registration
    # ----------------------------------------------------------------
    def register_command(self, command: str,
                          handler: Callable[[TelegramCommand], Optional[str]]) -> None:
        self._command_handlers[command.lower()] = handler

    def _register_default_commands(self) -> None:
        self.register_command("help", self._cmd_help)
        self.register_command("status", self._cmd_status)
        self.register_command("kill", self._cmd_kill)
        self.register_command("disarm", self._cmd_disarm)

    # ----------------------------------------------------------------
    # Default command handlers
    # ----------------------------------------------------------------
    @staticmethod
    def _cmd_help(cmd: TelegramCommand) -> str:
        return (
            "*Trading Bot Commands*\n\n"
            "/status — current bot status\n"
            "/positions — list open positions\n"
            "/pause — pause trading (set tier to PAPER)\n"
            "/resume — resume trading\n"
            "/kill — activate kill switch\n"
            "/disarm — disarm kill switch\n"
            "/promote TIER — promote capital tier\n"
            "/stats — show performance stats\n"
            "/help — this message"
        )

    def _cmd_status(self, cmd: TelegramCommand) -> str:
        # This would query the bot's actual state — placeholder
        return (
            "*Bot Status*\n\n"
            f"Time: {datetime.now(tz=timezone.utc).isoformat()}\n"
            f"Execution mode: {env.execution_mode}\n"
            f"Trading mode: {env.trading_mode}\n"
            f"Absolute safety: {env.absolute_safety}\n"
            f"Approval mode: {env.approval_mode}\n"
            f"Max lot: {env.max_lot}\n"
            f"Max open trades: {env.max_open_trades}\n"
            f"Daily loss limit: {env.daily_loss_limit_pct}%"
        )

    def _cmd_kill(self, cmd: TelegramCommand) -> str:
        # Create kill switch file
        kill_file = "data/KILL_SWITCH"
        os.makedirs("data", exist_ok=True)
        with open(kill_file, "w") as f:
            f.write(f"activated via Telegram by chat {cmd.chat_id} at {datetime.now(tz=timezone.utc).isoformat()}\n")
        return "🚨 Kill switch ACTIVATED. All trading halted."

    def _cmd_disarm(self, cmd: TelegramCommand) -> str:
        kill_file = "data/KILL_SWITCH"
        try:
            if os.path.isfile(kill_file):
                os.remove(kill_file)
            return "✅ Kill switch disarmed. Trading can resume."
        except Exception as e:  # noqa: BLE001
            return f"❌ Failed to disarm: {e!r}"

    # ----------------------------------------------------------------
    # HTTP helpers
    # ----------------------------------------------------------------
    def _http_post(self, url: str, payload: dict) -> bool:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(url, data=data,
                                          headers={"Content-Type": "application/json"},
                                          method="POST")
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                return resp.status == 200
        except (HTTPError, URLError, TimeoutError) as e:
            log.debug("Telegram POST failed: %r", e)
            return False

    @staticmethod
    def _http_get(url: str) -> Optional[dict]:
        try:
            req = urllib_request.Request(url, method="GET")
            with urllib_request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    return None
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            log.debug("Telegram GET failed: %r", e)
            return None

    # ----------------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "polling": self._running,
            "token_configured": bool(self.token),
            "chat_id_configured": bool(self.chat_id),
            "registered_commands": list(self._command_handlers.keys()),
        }