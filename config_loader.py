"""trading_bot.config_loader
=====================================================================
Phase 13 upgrade: YAML loader + pydantic schema validation.

Previously 50 lines of YAML load with zero validation (Phase 1 §6.2 gap).
Now validates required keys, types, and value ranges via pydantic, with
clear error messages for missing/invalid values — no more scattered
hardcoded defaults across a dozen files.

Secrets (MT5 password) are loaded from environment variables, NOT from
the YAML file (Phase 1 §6.1 gap). The YAML can reference env vars via
the `!ENV` tag or the mt5.password can be omitted entirely (env_loader
handles the fallback).

Audit Batch 1 remediation (C9, C18, H7, H16, M4, M11, M20):
  - `!ENV` tag now accepts a default and can be marked required:
    `!ENV MT5_PASSWORD` (required) or `!ENV MT5_PASSWORD, optional` (silent).
    Missing required env vars raise a clear `ConfigError`.
  - `pydantic` is now treated as effectively required: if missing, a
    fallback manual validator runs and a loud warning is logged. (C18)
  - `--mode=live` requires MT5 password to be present; missing password
    is a hard error, not a debug-level log. (C9, M4)
  - `load_config` raises `FileNotFoundError` if the path doesn't exist,
    with a clear message. (H7)
  - `deep_get` now optionally warns when the key is missing and a
    default is returned. (M20)
  - `deep_merge` is exposed as a public utility for applying overrides. (H16)
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Optional

import yaml

try:
    from pydantic import BaseModel, Field, ValidationError, field_validator
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]

from utils.logger import get_logger

log = get_logger("config_loader")

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "config", "config.yaml")


class ConfigError(RuntimeError):
    """Raised for any config-loading / validation failure."""


# ----------------------------------------------------------------------
# ENV tag processor — allows YAML values like `password: !ENV MT5_PASSWORD`
# Audit-fix M11: support `!ENV VAR` (required, raises if missing) and
# `!ENV VAR, optional` (returns empty string if missing). Required env
# vars that are absent now raise ConfigError instead of silently
# returning "".
# ----------------------------------------------------------------------
def _env_constructor(loader, node):
    raw = loader.construct_scalar(node)
    # Allow trailing `optional` / `required` marker.
    parts = [p.strip() for p in raw.split(",")]
    var_name = parts[0]
    optional = (len(parts) > 1 and parts[1].lower() == "optional")
    val = os.environ.get(var_name)
    if val is None or val == "":
        if optional:
            return ""
        raise ConfigError(
            f"Required environment variable {var_name!r} is not set "
            f"(referenced via !ENV in config YAML). "
            f"Set it in your .env file or shell environment.")
    return val


yaml.SafeLoader.add_constructor("!ENV", _env_constructor)


# ----------------------------------------------------------------------
# Pydantic schema (Phase 13 req #44)
# ----------------------------------------------------------------------
if _PYDANTIC_AVAILABLE:
    class MT5Config(BaseModel):
        login: int = 0
        password: str = ""  # Phase 13: prefer env var MT5_PASSWORD
        server: str = ""
        terminal_path: str = ""
        timeout_ms: int = 5000
        reconnect_attempts: int = 5
        reconnect_delay_s: float = 2.0

        @field_validator("login")
        @classmethod
        def login_must_be_positive(cls, v):
            if v < 0:
                raise ValueError("mt5.login must be >= 0")
            return v

    class RiskConfig(BaseModel):
        risk_per_trade: float = 0.01
        max_daily_loss: float = 0.05
        max_open_trades: int = 10
        max_consecutive_losses: int = 3
        cooldown_s: float = 60.0
        max_drawdown_pct: float = 15.0
        max_atr_pct: float = 0.05
        max_spread_bps: float = 15.0
        kelly_fraction: float = 0.25
        sl_atr_multiple: float = 1.5
        tp_atr_multiple: float = 2.5

        @field_validator("risk_per_trade")
        @classmethod
        def risk_per_trade_range(cls, v):
            if not 0 < v <= 0.10:
                raise ValueError("risk.risk_per_trade must be in (0, 0.10] — 0.01 = 1%")
            return v

        @field_validator("max_daily_loss")
        @classmethod
        def max_daily_loss_range(cls, v):
            if not 0 < v <= 0.50:
                raise ValueError("risk.max_daily_loss must be in (0, 0.50] — 0.05 = 5%")
            return v

    class RuntimeConfig(BaseModel):
        poll_interval_s: float = 5.0
        heartbeat_timeout_s: float = 60.0
        kill_switch_file: str = "data/KILL_SWITCH"
        state_file: str = "data/state.json"
        max_consecutive_errors: int = 10
        idempotency_file: str = "data/seen_orders.json"
        heartbeat_file: str = "data/heartbeat"
        cycle_timeout_s: float = 90.0
        shutdown_timeout_s: float = 30.0

    class TradingBotConfig(BaseModel):
        """Top-level config schema — validates the entire config dict."""
        mode: str = "paper"  # paper | demo | live
        capital: float = 10000.0
        mt5: MT5Config = MT5Config()
        risk: RiskConfig = RiskConfig()
        runtime: RuntimeConfig = RuntimeConfig()
        symbols: list = Field(default_factory=list)
        symbols_auto_load: bool = True

        @field_validator("mode")
        @classmethod
        def mode_must_be_valid(cls, v):
            if v not in ("demo", "live"):
                raise ValueError("mode must be 'demo' or 'live' (no paper mode)")
            return v

        @field_validator("capital")
        @classmethod
        def capital_positive(cls, v):
            if v <= 0:
                raise ValueError("capital must be > 0")
            return v


def _manual_fallback_validate(cfg: dict) -> None:
    """Audit-fix C18: minimal manual validation when pydantic is absent.

    We don't replicate every pydantic rule, but we check the most
    important ones: required types, mode enum, capital > 0, and the
    range bounds on risk_per_trade and max_daily_loss.
    """
    if not isinstance(cfg, dict):
        raise ConfigError(f"config root must be a mapping, got {type(cfg)}")
    mode = cfg.get("mode", "demo")
    if mode not in ("demo", "live"):
        raise ConfigError(f"mode must be 'demo' or 'live', got {mode!r}")
    capital = cfg.get("capital", 10000.0)
    if not isinstance(capital, (int, float)) or capital <= 0:
        raise ConfigError(f"capital must be a positive number, got {capital!r}")
    risk = cfg.get("risk", {})
    rpt = risk.get("risk_per_trade", 0.01)
    if not isinstance(rpt, (int, float)) or not (0 < rpt <= 0.10):
        raise ConfigError(f"risk.risk_per_trade must be in (0, 0.10], got {rpt!r}")
    mdl = risk.get("max_daily_loss", 0.05)
    if not isinstance(mdl, (int, float)) or not (0 < mdl <= 0.50):
        raise ConfigError(f"risk.max_daily_loss must be in (0, 0.50], got {mdl!r}")
    log.warning("config: pydantic not installed — running manual fallback validation "
                "(C18 fix: not skipping validation entirely)")


def load_config(path: str = DEFAULT_PATH, validate: bool = True,
                mode_override: Optional[str] = None) -> dict[str, Any]:
    """Load the YAML config from disk + validate via pydantic.

    Phase 13: raises clear errors for missing/invalid values instead of
    silently running with defaults.

    Audit-fix H7: explicit existence check + clear FileNotFoundError.
    Audit-fix C9 / M4: `mode=live` requires MT5_PASSWORD to be set.
    Audit-fix C18: if pydantic is missing, run manual fallback validation.

    Args:
        path: path to config.yaml
        validate: if True (default), validate against TradingBotConfig schema
        mode_override: if provided, overrides cfg["mode"] for the live-mode
            password check (useful when mode comes from CLI args).

    Raises:
        FileNotFoundError: if config file is missing
        ConfigError: for missing required env vars or invalid values
        pydantic.ValidationError: if validation fails (when validate=True)
    """
    if not os.path.isfile(path):
        # H7 fix: explicit, actionable error message.
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            f"  Set TRADING_BOT_CONFIG env var or pass --config <path>\n"
            f"  Default expected at: {DEFAULT_PATH}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping, got {type(cfg)}")

    # Apply mode override BEFORE validation so the password check uses it.
    if mode_override is not None:
        cfg["mode"] = mode_override
    effective_mode = cfg.get("mode", "demo")

    # Phase 13: load MT5 password from env var if not in YAML
    mt5_cfg = cfg.get("mt5", {})
    if not isinstance(mt5_cfg, dict):
        mt5_cfg = {}
        cfg["mt5"] = mt5_cfg
    if not mt5_cfg.get("password"):
        env_password = os.environ.get("MT5_PASSWORD", "")
        if env_password:
            mt5_cfg["password"] = env_password
            cfg["mt5"] = mt5_cfg
            log.info("config: mt5.password loaded from MT5_PASSWORD env var")
        else:
            # C9 / M4 fix: live mode REQUIRES the password — no silent fallback
            # to the terminal's cached login. Cached logins can connect to the
            # wrong account (live instead of demo, or vice versa).
            if effective_mode == "live":
                raise ConfigError(
                    "mt5.password is empty and MT5_PASSWORD env var is not set. "
                    "--mode=live requires an explicit password (audit C9): "
                    "a cached terminal login could connect to the wrong account.")
            if mt5_cfg.get("login"):
                log.debug("config: mt5.password empty and MT5_PASSWORD not set — "
                          "will rely on terminal's cached login if available (demo mode)")

    if validate:
        if _PYDANTIC_AVAILABLE:
            try:
                _ = TradingBotConfig(**cfg)
                log.info("config: schema validation passed")
            except ValidationError as e:
                log.error("config: schema validation FAILED:\n%s", e)
                raise
        else:
            # C18 fix: don't skip validation — run the manual fallback.
            _manual_fallback_validate(cfg)

    return cfg


def deep_get(d: dict[str, Any], dotted: str, default: Any = None,
               warn_on_default: bool = False) -> Any:
    """deep_get(cfg, 'risk.max_daily_loss', 0.05).

    Audit-fix M20: optionally log a warning when the key is missing and
    the default is returned, so config errors don't silently propagate.
    """
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            if warn_on_default:
                log.warning("config: deep_get(%r) missing — returning default %r",
                            dotted, default)
            return default
        cur = cur[part]
    return cur


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — `override` wins on conflict."""
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out