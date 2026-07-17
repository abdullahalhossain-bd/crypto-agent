"""enhancements.notification_system
=====================================================================
Day 146-148 — Multi-channel notification system.

Routes alerts to operators via:
  - Console (always on)
  - Log file (always on)
  - Webhook (generic HTTP POST)
  - Slack (incoming webhook)
  - Discord (webhook)
  - Email (SMTP)

Each channel is pluggable. Failure to send on one channel does NOT
block others. Retries with exponential backoff (max 3 attempts).

Severity-based routing:
  - INFO    → console + log only
  - WARN    → + webhook + Slack + Discord
  - CRITICAL→ + all channels + email
"""
from __future__ import annotations

import json
import os
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional
from urllib import request as urllib_request

from utils.logger import get_logger

log = get_logger("enhancements.notifications")


class NotificationChannel(str, Enum):
    CONSOLE = "console"
    LOG = "log"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DISCORD = "discord"
    EMAIL = "email"


class NotificationSeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass
class NotificationMessage:
    title: str
    body: str
    severity: NotificationSeverity = NotificationSeverity.INFO
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: str = ""

    def __post_init__(self) -> None:
        if not self.ts:
            self.ts = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "severity": self.severity.value,
            "category": self.category,
            "metadata": dict(self.metadata),
            "ts": self.ts,
        }

    def to_text(self) -> str:
        return f"[{self.severity.value}] {self.title}\n{self.body}"


# ----------------------------------------------------------------------
class NotificationSystem:
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        slack_webhook_url: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        email_smtp_host: Optional[str] = None,
        email_smtp_port: int = 587,
        email_username: Optional[str] = None,
        email_password: Optional[str] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
        log_path: str = "data/notifications.jsonl",
        max_retries: int = 3,
    ) -> None:
        self.webhook_url = webhook_url or os.environ.get("NOTIFY_WEBHOOK_URL")
        self.slack_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.discord_url = discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        self.email_host = email_smtp_host
        self.email_port = int(email_smtp_port)
        self.email_user = email_username
        self.email_pass = email_password
        self.email_from = email_from
        self.email_to = email_to
        self.log_path = log_path
        self.max_retries = int(max_retries)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # ----------------------------------------------------------------
    def send(self, message: NotificationMessage) -> dict[str, bool]:
        """Send to all channels appropriate for the severity.
        Returns dict of {channel: success}."""
        results: dict[str, bool] = {}
        # Always: console + log
        results["console"] = self._send_console(message)
        results["log"] = self._send_log(message)
        # WARN+ : webhook + slack + discord
        if message.severity in (NotificationSeverity.WARN, NotificationSeverity.CRITICAL):
            if self.webhook_url:
                results["webhook"] = self._send_webhook(message, self.webhook_url)
            if self.slack_url:
                results["slack"] = self._send_slack(message)
            if self.discord_url:
                results["discord"] = self._send_discord(message)
        # CRITICAL: + email
        if message.severity == NotificationSeverity.CRITICAL:
            if self.email_host and self.email_to:
                results["email"] = self._send_email(message)
        return results

    # ----------------------------------------------------------------
    # Channel implementations
    # ----------------------------------------------------------------
    @staticmethod
    def _send_console(message: NotificationMessage) -> bool:
        try:
            print(f"\n{'=' * 60}")
            print(f"  {message.title}")
            print(f"  [{message.severity.value}] {message.ts}")
            print(f"{'=' * 60}")
            print(message.body)
            print(f"{'=' * 60}\n")
            return True
        except Exception:  # noqa: BLE001
            return False

    def _send_log(self, message: NotificationMessage) -> bool:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(message.to_dict(), default=str) + "\n")
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("log notification failed: %r", e)
            return False

    def _send_webhook(self, message: NotificationMessage,
                        url: str) -> bool:
        """Generic HTTP POST webhook."""
        return self._http_post(url, message.to_dict())

    def _send_slack(self, message: NotificationMessage) -> bool:
        payload = {
            "text": f"*{message.title}*",
            "blocks": [
                {"type": "header", "text": {"type": "plain_text", "text": message.title}},
                {"type": "section", "text": {"type": "mrkdwn", "text": message.body}},
                {"type": "context", "elements": [
                    {"type": "mrkdwn", "text": f"_{message.severity.value} • {message.ts}_"}
                ]},
            ],
        }
        return self._http_post(self.slack_url, payload)

    def _send_discord(self, message: NotificationMessage) -> bool:
        color = {"INFO": 3447003, "WARN": 16776960, "CRITICAL": 15158332}.get(
            message.severity.value, 3447003
        )
        payload = {
            "embeds": [{
                "title": message.title,
                "description": message.body,
                "color": color,
                "timestamp": message.ts,
                "footer": {"text": f"severity={message.severity.value}"},
            }],
        }
        return self._http_post(self.discord_url, payload)

    def _send_email(self, message: NotificationMessage) -> bool:
        try:
            msg = MIMEText(message.body)
            msg["Subject"] = f"[{message.severity.value}] {message.title}"
            msg["From"] = self.email_from or self.email_user
            msg["To"] = self.email_to
            with smtplib.SMTP(self.email_host, self.email_port) as server:
                server.starttls()
                if self.email_user and self.email_pass:
                    server.login(self.email_user, self.email_pass)
                server.send_message(msg)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("email notification failed: %r", e)
            return False

    # ----------------------------------------------------------------
    def _http_post(self, url: str, payload: dict) -> bool:
        """HTTP POST with retries."""
        import time
        for attempt in range(1, self.max_retries + 1):
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib_request.Request(
                    url, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib_request.urlopen(req, timeout=5.0) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except Exception as e:  # noqa: BLE001
                log.debug("HTTP POST attempt %d/%d failed: %r",
                          attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        return False

    # ----------------------------------------------------------------
    # Convenience methods
    # ----------------------------------------------------------------
    def notify_trade_opened(self, symbol: str, side: str, lots: float,
                              price: float, strategy: str = "") -> None:
        self.send(NotificationMessage(
            title=f"Trade Opened: {side.upper()} {symbol}",
            body=(f"Strategy: {strategy}\n"
                  f"Symbol: {symbol}\nSide: {side}\n"
                  f"Lots: {lots}\nEntry: {price}"),
            severity=NotificationSeverity.INFO,
            category="trade",
        ))

    def notify_trade_closed(self, symbol: str, side: str, lots: float,
                              exit_price: float, pnl: float,
                              strategy: str = "") -> None:
        sev = NotificationSeverity.INFO if pnl >= 0 else NotificationSeverity.WARN
        self.send(NotificationMessage(
            title=f"Trade Closed: {side.upper()} {symbol} PnL={pnl:+.2f}",
            body=(f"Strategy: {strategy}\nSymbol: {symbol}\nSide: {side}\n"
                  f"Lots: {lots}\nExit: {exit_price}\nPnL: {pnl:+.2f}"),
            severity=sev,
            category="trade",
        ))

    def notify_kill_switch(self, reason: str) -> None:
        self.send(NotificationMessage(
            title="KILL SWITCH ACTIVATED",
            body=f"Reason: {reason}\nAll trading halted.\nManual review required.",
            severity=NotificationSeverity.CRITICAL,
            category="risk",
        ))

    def notify_drawdown_breach(self, drawdown_pct: float,
                                 threshold: float) -> None:
        self.send(NotificationMessage(
            title=f"DRAWDOWN BREACH: {drawdown_pct:.2%}",
            body=(f"Current drawdown: {drawdown_pct:.2%}\n"
                  f"Threshold: {threshold:.2%}\n"
                  f"Reducing position sizes and reviewing positions."),
            severity=NotificationSeverity.CRITICAL,
            category="risk",
        ))
