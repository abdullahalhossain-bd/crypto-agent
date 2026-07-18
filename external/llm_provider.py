"""external.llm_provider
=====================================================================
Day 162-165 — Multi-key LLM provider with automatic fallback.

Fallback chain (in order):
  1. Groq (6 keys, rotate on failure)
  2. Cerebras (fastest inference)
  3. SambaNova (free Llama)
  4. OpenRouter (catch-all, multiple free models)
  5. Gemini (Google)

Token economy:
  - Max N calls per cycle (default 4)
  - Max N calls per minute (default 6)
  - Min interval between calls (default 2.0s)
  - Rolling window tracking with auto-reset

All providers use OpenAI-compatible chat.completions.create API
shape, so the request code is shared. Only the URL + auth differ.

CRITICAL: This module is a GATEKEEPER for AI-assisted analysis.
It does NOT make trading decisions. It produces analysis text that
the strategy/risk layers can use as ONE input among many.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from external.env_loader import env
from utils.logger import get_logger

log = get_logger("external.llm")


# ----------------------------------------------------------------------
@dataclass
class LLMMessage:
    role: str          # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: float
    success: bool
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
        }


class LLMProviderError(RuntimeError):
    pass


# ----------------------------------------------------------------------
class LLMProvider:
    """Multi-provider LLM with fallback chain + token economy."""

    def __init__(self) -> None:
        self.providers = self._build_provider_chain()
        # Token economy
        self.max_per_cycle = env.max_llm_calls_per_cycle
        self.max_per_min = env.max_llm_calls_per_min
        self.min_interval = env.llm_call_interval_sec
        self._calls_this_cycle = 0
        self._calls_this_min: deque = deque(maxlen=200)  # timestamps
        self._last_call_ts = 0.0
        # Per-key rotation tracking
        self._groq_key_idx = 0
        self._gemini_key_idx = 0
        self._disabled_keys: set[str] = set()  # keys that returned 401/403
        # Guards all of the mutable state above. `chat()` may be invoked
        # concurrently (e.g. main loop + Telegram command handler thread);
        # without this lock, rate-limit counters and key-rotation indices
        # can race, allowing configured call limits to be exceeded.
        self._state_lock = threading.Lock()

    # ----------------------------------------------------------------
    def reset_cycle(self) -> None:
        """Call this at the start of each main-loop cycle.

        LLM PROVIDER AUDIT FIX: also re-enable disabled keys every cycle.
        Previously, once a key returned 401/403, it was disabled forever
        for the session — even if the auth issue was transient (rate limit,
        temporary server error). Now keys are re-enabled on each cycle
        reset, giving them a second chance. If the key is genuinely invalid,
        it will be re-disabled on the next call.
        """
        with self._state_lock:
            self._calls_this_cycle = 0
            if self._disabled_keys:
                n = len(self._disabled_keys)
                self._disabled_keys.clear()
                log.info("LLM: re-enabled %d previously disabled key(s)", n)

    # ----------------------------------------------------------------
    def chat(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 800,
        temperature: float = 0.3,
        prefer_provider: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat completion request with automatic fallback.

        Critical #1 fix: the previous implementation called `time.sleep()`
        while holding `self._state_lock`, which blocked ALL other threads
        from calling `chat()` for up to 5 seconds. It also did NOT re-check
        the rate limit after sleeping, so under concurrent access the limit
        could be silently exceeded.

        The fix releases the lock before sleeping, then re-acquires it and
        re-checks in a `while` loop until the limits are satisfied.
        """
        # ── Token-economy gate: check limits, sleep WITHOUT holding lock ──
        while True:
            with self._state_lock:
                # Per-cycle limit (hard cap — no waiting).
                if self._calls_this_cycle >= self.max_per_cycle:
                    return LLMResponse(
                        text="", provider="none", model="none",
                        tokens_used=0, latency_ms=0, success=False,
                        error=f"max calls per cycle ({self.max_per_cycle}) reached",
                    )
                # Purge old entries from the per-minute deque.
                now = time.time()
                while self._calls_this_min and self._calls_this_min[0] < now - 60:
                    self._calls_this_min.popleft()

                # Per-minute limit.
                if len(self._calls_this_min) >= self.max_per_min:
                    wait = 60 - (now - self._calls_this_min[0])
                    sleep_time = min(max(wait, 0.1), 5.0)
                else:
                    # Min interval between calls.
                    elapsed = now - self._last_call_ts
                    if elapsed < self.min_interval:
                        sleep_time = self.min_interval - elapsed
                    else:
                        # All limits satisfied — increment counters and proceed.
                        self._calls_this_cycle += 1
                        self._calls_this_min.append(time.time())
                        self._last_call_ts = time.time()
                        break  # exit the while loop, proceed to provider chain

            # Sleep OUTSIDE the lock so other threads aren't blocked.
            log.debug("LLM rate limit: sleeping %.1fs (outside lock)", sleep_time)
            time.sleep(sleep_time)
            # Loop back and re-check limits after sleeping.

        # Try each provider in chain
        chain = self.providers
        if prefer_provider:
            # Move preferred provider to front
            preferred = [p for p in chain if p["name"] == prefer_provider]
            rest = [p for p in chain if p["name"] != prefer_provider]
            chain = preferred + rest

        for provider in chain:
            if not provider["available"]:
                continue
            try:
                resp = self._call_provider(provider, messages, max_tokens, temperature)
                if resp.success:
                    return resp
                log.debug("LLM provider %s failed: %s", provider["name"], resp.error)
            except Exception as e:  # noqa: BLE001
                log.debug("LLM provider %s exception: %r", provider["name"], e)
                continue
        return LLMResponse(
            text="", provider="none", model="none",
            tokens_used=0, latency_ms=0, success=False,
            error="all providers failed",
        )

    # ----------------------------------------------------------------
    def _call_provider(
        self,
        provider: dict[str, Any],
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Call a single provider. Returns LLMResponse (success or failure)."""
        start = time.time()
        url = provider["base_url"] + "/chat/completions"
        # Build request body (OpenAI-compatible)
        body = {
            "model": provider["model"],
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        # Pick key (rotate for Groq/Gemini)
        key = provider["get_key"]()
        if not key:
            return LLMResponse(
                text="", provider=provider["name"], model=provider["model"],
                tokens_used=0, latency_ms=0, success=False,
                error="no API key available",
            )
        with self._state_lock:
            key_disabled = key in self._disabled_keys
        if key_disabled:
            return LLMResponse(
                text="", provider=provider["name"], model=provider["model"],
                tokens_used=0, latency_ms=0, success=False,
                error="key disabled (previous 401/403)",
            )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        # OpenRouter requires extra headers
        if provider["name"] == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/trading-bot"
            headers["X-Title"] = "Trading Bot"
        try:
            data = json.dumps(body).encode("utf-8")
            req = urllib_request.Request(url, data=data, headers=headers, method="POST")
            with urllib_request.urlopen(req, timeout=30.0) as resp:
                if resp.status != 200:
                    return LLMResponse(
                        text="", provider=provider["name"], model=provider["model"],
                        tokens_used=0, latency_ms=float((time.time() - start) * 1000),
                        success=False, error=f"HTTP {resp.status}",
                    )
                result = json.loads(resp.read().decode("utf-8"))
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens = int(result.get("usage", {}).get("total_tokens", 0))
            latency = float((time.time() - start) * 1000)
            return LLMResponse(
                text=text, provider=provider["name"], model=provider["model"],
                tokens_used=tokens, latency_ms=latency, success=True,
            )
        except HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")[:200]
            except Exception:  # noqa: BLE001
                pass
            # Disable key on auth errors
            if e.code in (401, 403):
                with self._state_lock:
                    self._disabled_keys.add(key)
                log.warning("LLM key disabled (HTTP %d): %s...", e.code, key[:12])
            return LLMResponse(
                text="", provider=provider["name"], model=provider["model"],
                tokens_used=0, latency_ms=float((time.time() - start) * 1000),
                success=False,
                error=f"HTTP {e.code}: {error_body}",
            )
        except (URLError, TimeoutError) as e:
            return LLMResponse(
                text="", provider=provider["name"], model=provider["model"],
                tokens_used=0, latency_ms=float((time.time() - start) * 1000),
                success=False, error=f"network: {e!r}",
            )
        except Exception as e:  # noqa: BLE001
            return LLMResponse(
                text="", provider=provider["name"], model=provider["model"],
                tokens_used=0, latency_ms=float((time.time() - start) * 1000),
                success=False, error=f"exception: {e!r}",
            )

    # ----------------------------------------------------------------
    def _build_provider_chain(self) -> list[dict[str, Any]]:
        """Build the fallback chain from env vars."""
        chain: list[dict[str, Any]] = []

        # 1. Groq (multiple keys)
        groq_keys = env.groq_keys
        if groq_keys:
            chain.append({
                "name": "groq",
                "base_url": "https://api.groq.com/openai/v1",
                "model": env.groq_model,
                "available": True,
                "get_key": self._next_groq_key,
            })

        # 2. Cerebras
        if env.cerebras_api_key:
            chain.append({
                "name": "cerebras",
                "base_url": env.cerebras_base_url,
                "model": env.cerebras_model,
                "available": True,
                "get_key": lambda: env.cerebras_api_key,
            })

        # 3. SambaNova
        if env.sambanova_api_key:
            chain.append({
                "name": "sambanova",
                "base_url": env.sambanova_base_url,
                "model": env.sambanova_model,
                "available": True,
                "get_key": lambda: env.sambanova_api_key,
            })

        # 4. OpenRouter
        if env.openrouter_api_key:
            # Try primary model, then fallbacks
            models = [env.openrouter_model] + env.openrouter_fallback_models
            for model in models:
                chain.append({
                    "name": "openrouter",
                    "base_url": env.openrouter_base_url,
                    "model": model,
                    "available": True,
                    "get_key": lambda: env.openrouter_api_key,
                })

        # 5. Gemini (via OpenAI-compatible proxy)
        if env.gemini_keys:
            chain.append({
                "name": "gemini",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "model": env.gemini_model,
                "available": True,
                "get_key": self._next_gemini_key,
            })

        log.info("LLM provider chain: %s",
                 [p["name"] for p in chain])
        return chain

    # ----------------------------------------------------------------
    def _next_groq_key(self) -> Optional[str]:
        with self._state_lock:
            keys = [k for k in env.groq_keys if k not in self._disabled_keys]
            if not keys:
                return None
            key = keys[self._groq_key_idx % len(keys)]
            self._groq_key_idx += 1
            return key

    def _next_gemini_key(self) -> Optional[str]:
        with self._state_lock:
            keys = [k for k in env.gemini_keys if k not in self._disabled_keys]
            if not keys:
                return None
            key = keys[self._gemini_key_idx % len(keys)]
            self._gemini_key_idx += 1
            return key

    # ----------------------------------------------------------------
    @property
    def stats(self) -> dict[str, Any]:
        with self._state_lock:
            calls_this_cycle = self._calls_this_cycle
            calls_this_min = len(self._calls_this_min)
            n_disabled_keys = len(self._disabled_keys)
        return {
            "calls_this_cycle": calls_this_cycle,
            "calls_this_min": calls_this_min,
            "max_per_cycle": self.max_per_cycle,
            "max_per_min": self.max_per_min,
            "n_providers": len(self.providers),
            "n_disabled_keys": n_disabled_keys,
            "providers": [p["name"] for p in self.providers],
        }

    def health_check(self) -> dict[str, bool]:
        """Quick check of which providers are configured."""
        return {
            "groq": any(p["name"] == "groq" for p in self.providers),
            "cerebras": any(p["name"] == "cerebras" for p in self.providers),
            "sambanova": any(p["name"] == "sambanova" for p in self.providers),
            "openrouter": any(p["name"] == "openrouter" for p in self.providers),
            "gemini": any(p["name"] == "gemini" for p in self.providers),
        }