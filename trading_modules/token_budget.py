"""
Token/Cost Budget Module — Agentic Loop Cost Protection
=========================================================

Soft client-side budget enforcer for agentic LLM loops. Caps both token
count and USD cost. When budget is exhausted, agent loops short-circuit
gracefully instead of burning unlimited API budget.

Thread-safe: methods are guarded by an internal Lock so a parallel
agent panel can safely share one budget instance.

Source: Orallexa (review #27) — token_budget.py (129 LOC) + Vibe-Trading

Usage:
    from token_budget import TokenBudget, guarded_call

    # Create budget
    budget = TokenBudget(cap_tokens=50000, cap_usd=5.0, label="deep_analysis")

    # Before each LLM call
    if budget.allow():
        response, record = llm.create(...)
        budget.consume(record)
    else:
        print("Budget exhausted — skipping remaining steps")

    # Or use the guarded_call helper
    response = guarded_call(budget, lambda: llm.create(...))
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """
    Soft client-side budget enforcer for agentic LLM loops.

    Either or both ceilings can be set; budget is exhausted as soon
    as ANY ceiling is hit. Set a ceiling to None to disable that gate.

    Attributes:
        cap_tokens: Maximum total tokens (prompt + completion). None = unlimited.
        cap_usd: Maximum total cost in USD. None = unlimited.
        label: Human-readable label for logging/debugging.
    """
    cap_tokens: Optional[int] = None
    cap_usd: Optional[float] = None
    label: str = "default"

    used_tokens: int = 0
    used_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def allow(self) -> bool:
        """Check if budget allows another call. Thread-safe."""
        with self._lock:
            if self.cap_tokens is not None and self.used_tokens >= self.cap_tokens:
                return False
            if self.cap_usd is not None and self.used_cost_usd >= self.cap_usd:
                return False
            return True

    def consume(self, record: dict | Any) -> None:
        """
        Consume budget from an LLM call record.

        Accepts either a dict with 'total_tokens' and 'estimated_cost_usd'
        or an object with those attributes.
        """
        with self._lock:
            tokens = self._extract(record, "total_tokens", "tokens", "usage_tokens")
            cost = self._extract(record, "estimated_cost_usd", "cost_usd", "cost")

            if tokens:
                self.used_tokens += int(tokens)
            if cost:
                self.used_cost_usd += float(cost)

            # Check if exhausted after consumption
            if not self._check_internal():
                logger.warning(
                    f"TokenBudget '{self.label}' exhausted: "
                    f"{self.used_tokens}/{self.cap_tokens} tokens, "
                    f"${self.used_cost_usd:.4f}/${self.cap_usd}"
                )

    def remaining_tokens(self) -> Optional[int]:
        """Remaining token budget, or None if unlimited."""
        if self.cap_tokens is None:
            return None
        with self._lock:
            return max(0, self.cap_tokens - self.used_tokens)

    def remaining_usd(self) -> Optional[float]:
        """Remaining USD budget, or None if unlimited."""
        if self.cap_usd is None:
            return None
        with self._lock:
            return max(0.0, self.cap_usd - self.used_cost_usd)

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return not self.allow()

    def report(self) -> dict:
        """Get budget usage report.

        Major #10 fix: compute all values within the same lock acquisition
        to ensure a consistent snapshot. The old code called
        remaining_tokens() and remaining_usd() which each acquired the
        lock independently — with RLock this is safe but produces
        inconsistent snapshots if another thread modifies usage between calls.
        """
        with self._lock:
            # Compute remaining inline (don't call methods that re-acquire).
            rem_tokens = None
            if self.cap_tokens is not None:
                rem_tokens = max(0, self.cap_tokens - self.used_tokens)
            rem_usd = None
            if self.cap_usd is not None:
                rem_usd = round(max(0.0, self.cap_usd - self.used_cost_usd), 6)
            return {
                "label": self.label,
                "used_tokens": self.used_tokens,
                "cap_tokens": self.cap_tokens,
                "remaining_tokens": rem_tokens,
                "used_cost_usd": round(self.used_cost_usd, 6),
                "cap_usd": self.cap_usd,
                "remaining_usd": rem_usd,
                "exhausted": not self._check_internal(),
                "token_pct": round(self.used_tokens / self.cap_tokens * 100, 1) if self.cap_tokens else None,
                "cost_pct": round(self.used_cost_usd / self.cap_usd * 100, 1) if self.cap_usd else None,
            }

    def reset(self) -> None:
        """Reset usage counters (for new run)."""
        with self._lock:
            self.used_tokens = 0
            self.used_cost_usd = 0.0

    def _check_internal(self) -> bool:
        """Internal check without lock (caller must hold lock)."""
        if self.cap_tokens is not None and self.used_tokens >= self.cap_tokens:
            return False
        if self.cap_usd is not None and self.used_cost_usd >= self.cap_usd:
            return False
        return True

    @staticmethod
    def _extract(record: Any, *keys: str) -> Optional[float]:
        """Extract a value from a record (dict or object)."""
        if isinstance(record, dict):
            for key in keys:
                if key in record:
                    try:
                        return float(record[key])
                    except (ValueError, TypeError):
                        continue
        else:
            for key in keys:
                if hasattr(record, key):
                    try:
                        return float(getattr(record, key))
                    except (ValueError, TypeError):
                        continue
        return None


def guarded_call(
    budget: TokenBudget,
    fn: Callable,
    *args,
    **kwargs,
) -> Optional[Any]:
    """
    Execute fn() if budget allows, consume budget from result.

    Returns None if budget exhausted (caller should handle gracefully).

    The function's return value should have total_tokens and
    estimated_cost_usd fields (dict or object attributes).
    """
    if not budget.allow():
        logger.info(f"Budget '{budget.label}' exhausted — skipping call")
        return None

    try:
        result = fn(*args, **kwargs)
        budget.consume(result)
        return result
    except Exception as e:
        logger.error(f"Guarded call failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# Cost estimation helpers
# ═══════════════════════════════════════════════════════════════

# Rough per-1M-token pricing (USD) as of 2026
MODEL_PRICING = {
    # OpenAI
    "gpt-5.5": {"input": 5.0, "output": 15.0},
    "gpt-5.4": {"input": 3.0, "output": 10.0},
    "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
    "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
    # Anthropic
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0},
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    # Google
    "gemini-3-pro": {"input": 1.25, "output": 5.0},
    "gemini-3-flash": {"input": 0.075, "output": 0.30},
    # Default
    "default": {"input": 1.0, "output": 3.0},
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate USD cost for an LLM call.

    Args:
        model: Model name (e.g., "gpt-5.4-mini")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        Estimated cost in USD
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    cost = (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )
    return round(cost, 6)
