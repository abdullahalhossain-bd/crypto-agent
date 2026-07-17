"""external.env_loader
=====================================================================
Day 161 — Environment variable loader.

Loads .env file on startup and provides typed access to all config
values. Falls back to system environment variables if .env is missing.

Usage:
    from external.env_loader import env
    groq_keys = env.groq_keys        # list of all GROQ_API_KEY_*
    mt5_login = env.mt5_login
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger("external.env")


class EnvLoader:
    """Loads .env file and provides typed access to all variables."""

    _instance: Optional["EnvLoader"] = None

    def __init__(self, env_path: str = ".env") -> None:
        self.env_path = Path(env_path)
        self._values: dict[str, str] = {}
        self._load()

    # ----------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> "EnvLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ----------------------------------------------------------------
    def _load(self) -> None:
        # Load from .env file
        if self.env_path.is_file():
            with open(self.env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    self._values[key] = value
            log.info("Loaded %d env vars from %s", len(self._values), self.env_path)
        # Overlay with system environment (system takes precedence)
        for key in list(self._values.keys()):
            sys_val = os.environ.get(key)
            if sys_val is not None:
                self._values[key] = sys_val

    # ----------------------------------------------------------------
    def get(self, key: str, default: str = "") -> str:
        return self._values.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self._values.get(key, str(default)))
        except ValueError:
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        try:
            return float(self._values.get(key, str(default)))
        except ValueError:
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self._values.get(key, str(default)).lower()
        return val in ("true", "1", "yes", "on")

    def get_list(self, key: str, default: Optional[list[str]] = None,
                  sep: str = ",") -> list[str]:
        val = self._values.get(key, "")
        if not val:
            return default or []
        return [x.strip() for x in val.split(sep) if x.strip()]

    # ----------------------------------------------------------------
    # Typed accessors for known variables
    # ----------------------------------------------------------------
    @property
    def groq_keys(self) -> list[str]:
        """All Groq API keys (GROQ_API_KEY + GROQ_API_KEY_1..N)."""
        keys = []
        # Main key
        main = self.get("GROQ_API_KEY")
        if main:
            keys.append(main)
        # Numbered keys
        for i in range(1, 20):
            k = self.get(f"GROQ_API_KEY_{i}")
            if k and k not in keys:
                keys.append(k)
        return keys

    @property
    def groq_model(self) -> str:
        return self.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    @property
    def cerebras_api_key(self) -> str:
        return self.get("CEREBRAS_API_KEY")

    @property
    def cerebras_base_url(self) -> str:
        return self.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")

    @property
    def cerebras_model(self) -> str:
        return self.get("CEREBRAS_MODEL", "gpt-oss-120b")

    @property
    def sambanova_api_key(self) -> str:
        return self.get("SAMBANOVA_API_KEY")

    @property
    def sambanova_base_url(self) -> str:
        return self.get("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")

    @property
    def sambanova_model(self) -> str:
        return self.get("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct")

    @property
    def openrouter_api_key(self) -> str:
        return self.get("OPENROUTER_API_KEY")

    @property
    def openrouter_base_url(self) -> str:
        return self.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    @property
    def openrouter_model(self) -> str:
        return self.get("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")

    @property
    def openrouter_fallback_models(self) -> list[str]:
        out = []
        for i in range(1, 5):
            m = self.get(f"OPENROUTER_MODEL_FALLBACK_{i}")
            if m:
                out.append(m)
        return out

    @property
    def gemini_keys(self) -> list[str]:
        keys = []
        for i in range(1, 10):
            k = self.get(f"GEMINI_API_KEY_{i}")
            if k:
                keys.append(k)
        return keys

    @property
    def gemini_model(self) -> str:
        return self.get("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def hf_token(self) -> str:
        return self.get("HF_TOKEN")

    # ---- Token economy ----
    @property
    def loop_interval_sec(self) -> int:
        return self.get_int("LOOP_INTERVAL_SEC", 180)

    @property
    def max_llm_calls_per_cycle(self) -> int:
        return self.get_int("MAX_LLM_CALLS_PER_CYCLE", 4)

    @property
    def max_llm_calls_per_min(self) -> int:
        return self.get_int("MAX_LLM_CALLS_PER_MIN", 6)

    @property
    def llm_call_interval_sec(self) -> float:
        return self.get_float("LLM_CALL_INTERVAL_SEC", 2.0)

    @property
    def master_analyst_max_tokens(self) -> int:
        return self.get_int("MASTER_ANALYST_MAX_TOKENS", 800)

    @property
    def ai_analyst_max_tokens(self) -> int:
        return self.get_int("AI_ANALYST_MAX_TOKENS", 400)

    # ---- Safety ----
    @property
    def absolute_safety(self) -> bool:
        return self.get_bool("ABSOLUTE_SAFETY", True)

    @property
    def test_mode(self) -> bool:
        return self.get_bool("TEST_MODE", False)

    @property
    def trading_mode(self) -> str:
        return self.get("TRADING_MODE", "SAFE")

    @property
    def max_lot(self) -> float:
        return self.get_float("MAX_LOT", 0.10)

    @property
    def daily_loss_limit_pct(self) -> float:
        return self.get_float("DAILY_LOSS_LIMIT_PCT", 5.0)

    @property
    def max_open_trades(self) -> int:
        return self.get_int("MAX_OPEN_TRADES", 3)

    @property
    def simulation_mode(self) -> bool:
        return self.get_bool("SIMULATION_MODE", False)

    @property
    def approval_mode(self) -> int:
        return self.get_int("APPROVAL_MODE", 3)

    # ---- MT5 ----
    @property
    def mt5_login(self) -> int:
        return self.get_int("MT5_LOGIN", 0)

    @property
    def mt5_password(self) -> str:
        return self.get("MT5_PASSWORD")

    @property
    def mt5_server(self) -> str:
        return self.get("MT5_SERVER", "MetaQuotes-Demo")

    @property
    def mt5_path(self) -> str:
        return self.get("MT5_PATH")

    @property
    def mt5_investor(self) -> str:
        return self.get("MT5_INVESTOR")

    @property
    def execution_mode(self) -> str:
        return self.get("EXECUTION_MODE", "paper")

    # ---- Webhook ----
    @property
    def webhook_secret(self) -> str:
        return self.get("WEBHOOK_SECRET")

    @property
    def webhook_port(self) -> int:
        return self.get_int("WEBHOOK_PORT", 5000)

    # ---- Market Data ----
    @property
    def alpha_vantage_api_key(self) -> str:
        return self.get("ALPHA_VANTAGE_API_KEY")

    @property
    def alpha_vantage_base_url(self) -> str:
        return self.get("ALPHA_VANTAGE_BASE_URL", "https://www.alphavantage.co/query")

    @property
    def twelve_data_api_key(self) -> str:
        return self.get("TWELVE_DATA_API_KEY")

    @property
    def twelve_data_base_url(self) -> str:
        return self.get("TWELVE_DATA_BASE_URL", "https://api.twelvedata.com")

    @property
    def polygon_api_key(self) -> str:
        return self.get("POLYGON_API_KEY")

    @property
    def preferred_data_source(self) -> str:
        return self.get("PREFERRED_DATA_SOURCE", "twelve_data")

    # ---- News / Economic Calendar ----
    @property
    def newsapi_api_key(self) -> str:
        return self.get("NEWSAPI_API_KEY")

    @property
    def fred_api_key(self) -> str:
        return self.get("FRED_API_KEY")

    @property
    def tradermade_api_key(self) -> str:
        return self.get("TRADERMADE_API_KEY")

    @property
    def tradingeconomics_api_key(self) -> str:
        return self.get("TRADINGECONOMICS_API_KEY")

    # ---- Sentiment ----
    @property
    def oanda_api_key(self) -> str:
        return self.get("OANDA_API_KEY")

    @property
    def oanda_account_id(self) -> str:
        return self.get("OANDA_ACCOUNT_ID")

    @property
    def oanda_use_practice(self) -> bool:
        return self.get_bool("OANDA_USE_PRACTICE", True)

    @property
    def myfxbook_email(self) -> str:
        return self.get("MYFXBOOK_EMAIL")

    @property
    def myfxbook_password(self) -> str:
        return self.get("MYFXBOOK_PASSWORD")

    # ---- Retraining ----
    @property
    def retraining_interval(self) -> int:
        return self.get_int("RETRAINING_INTERVAL", 24)

    @property
    def performance_threshold(self) -> float:
        return self.get_float("PERFORMANCE_THRESHOLD", 0.55)

    @property
    def min_training_samples(self) -> int:
        return self.get_int("MIN_TRAINING_SAMPLES", 100)

    # ---- Logging ----
    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL", "INFO")

    # ---- Telegram ----
    @property
    def telegram_token(self) -> str:
        return self.get("TELEGRAM_TOKEN")

    @property
    def telegram_chat_id(self) -> str:
        """Primary/owner chat id — used as the default send target and as
        the default authorized command sender if TELEGRAM_AUTHORIZED_CHAT_IDS
        is not set."""
        return self.get("TELEGRAM_CHAT_ID")

    @property
    def telegram_authorized_chat_ids(self) -> list[str]:
        """Chat ids allowed to issue commands (e.g. /kill, /disarm).

        Falls back to just `telegram_chat_id` if not explicitly configured,
        so a single-owner setup keeps working without extra config.
        """
        configured = self.get_list("TELEGRAM_AUTHORIZED_CHAT_IDS")
        if configured:
            return configured
        owner = self.telegram_chat_id
        return [owner] if owner else []

    @property
    def enable_telegram(self) -> bool:
        return self.get_bool("ENABLE_TELEGRAM", True)

    # ---- Alert recipients ----
    @property
    def alert_recipients(self) -> list[str]:
        return self.get_list("ALERT_RECIPIENTS")

    @property
    def alert_webhook_url(self) -> str:
        return self.get("ALERT_WEBHOOK_URL")

    # ---- Paper balance ----
    @property
    def paper_balance(self) -> float:
        return self.get_float("PAPER_BALANCE", 10000)

    @property
    def backup_interval_min(self) -> int:
        return self.get_int("BACKUP_INTERVAL_MIN", 30)

    @property
    def recovery_cooldown_min(self) -> int:
        return self.get_int("RECOVERY_COOLDOWN_MIN", 5)

    # ----------------------------------------------------------------
    def all_keys(self) -> dict[str, str]:
        """Return all loaded env vars (for debugging — masks secrets)."""
        out = {}
        for k, v in self._values.items():
            if any(s in k.upper() for s in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
                out[k] = v[:8] + "..." if len(v) > 8 else "***"
            else:
                out[k] = v
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "n_groq_keys": len(self.groq_keys),
            "n_gemini_keys": len(self.gemini_keys),
            "has_cerebras": bool(self.cerebras_api_key),
            "has_sambanova": bool(self.sambanova_api_key),
            "has_openrouter": bool(self.openrouter_api_key),
            "has_mt5": bool(self.mt5_login),
            "has_twelve_data": bool(self.twelve_data_api_key),
            "has_alpha_vantage": bool(self.alpha_vantage_api_key),
            "has_polygon": bool(self.polygon_api_key),
            "has_newsapi": bool(self.newsapi_api_key),
            "has_fred": bool(self.fred_api_key),
            "has_myfxbook": bool(self.myfxbook_email),
            "execution_mode": self.execution_mode,
            "trading_mode": self.trading_mode,
            "absolute_safety": self.absolute_safety,
        }


# Singleton
env = EnvLoader.get_instance()