"""scaling.capital_scaler
=====================================================================
Day 81-84 — Capital Scaler.

Increases (or decreases) the capital fraction based on EVIDENCE, not
hope. Scaling-up requires ALL of:

  - Stable Sharpe over rolling window (>= 1.0)
  - Low drawdown VOLATILITY (not just low DD)
  - Execution efficiency stable (slippage within tolerance)
  - No regime-sensitivity breakdown

Scaling-down triggers (any one):
  - Sharpe drops below 0.3
  - Drawdown volatility spikes
  - Live-vs-paper divergence exceeds threshold
  - Capital tier demotion event

The scaler outputs a target capital fraction; the actual allocation
is the min of (target, tier fraction) so the tier system remains a
hard ceiling.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from utils.logger import get_logger

log = get_logger("scaling.capital")


@dataclass
class ScalingDecision:
    action: str                  # "scale_up" | "scale_down" | "hold"
    current_fraction: float
    target_fraction: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "current_fraction": self.current_fraction,
            "target_fraction": self.target_fraction,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "ts": self.ts,
        }


# ----------------------------------------------------------------------
class CapitalScaler:
    def __init__(
        self,
        min_fraction: float = 0.001,
        max_fraction: float = 0.10,
        scale_up_step: float = 0.005,        # +0.5% per promotion
        scale_down_step: float = 0.25,        # cut by 25% on demotion (Major #2 fix: was 0.5)
        rolling_window: int = 500,
        min_sharpe_to_scale: float = 1.0,
        max_dd_volatility: float = 0.05,
        max_execution_slippage_bps: float = 15.0,
        max_divergence_pct: float = 0.01,
    ) -> None:
        self.min_fraction = float(min_fraction)
        self.max_fraction = float(max_fraction)
        self.scale_up_step = float(scale_up_step)
        self.scale_down_step = float(scale_down_step)
        self.rolling_window = int(rolling_window)
        self.min_sharpe = float(min_sharpe_to_scale)
        self.max_dd_vol = float(max_dd_volatility)
        self.max_slip_bps = float(max_execution_slippage_bps)
        self.max_divergence = float(max_divergence_pct)
        self.current_fraction = float(min_fraction)
        # Rolling history
        self._sharpes: list[float] = []
        self._drawdowns: list[float] = []
        self._slippages_bps: list[float] = []
        self._divergences: list[float] = []

    # ----------------------------------------------------------------
    def record_observation(
        self,
        sharpe: float,
        drawdown_pct: float,
        execution_slippage_bps: float,
        live_paper_divergence_pct: float,
    ) -> None:
        # Critical #3 fix: sanitize inputs for NaN/inf.
        def _safe(v: float, name: str) -> float:
            f = float(v) if v is not None else 0.0
            if not math.isfinite(f):
                log.warning("capital_scaler: non-finite %s=%r — using 0.0", name, v)
                return 0.0
            return f
        self._sharpes.append(_safe(sharpe, "sharpe"))
        self._drawdowns.append(_safe(drawdown_pct, "drawdown_pct"))
        self._slippages_bps.append(_safe(execution_slippage_bps, "slippage_bps"))
        self._divergences.append(_safe(live_paper_divergence_pct, "divergence_pct"))
        # Trim
        for lst in (self._sharpes, self._drawdowns, self._slippages_bps, self._divergences):
            if len(lst) > self.rolling_window:
                del lst[:-self.rolling_window]

    # ----------------------------------------------------------------
    def evaluate(self, tier_fraction: float) -> ScalingDecision:
        """Compute the next capital fraction. The tier_fraction is a
        hard ceiling."""
        evidence: dict[str, Any] = {
            "n_observations": len(self._sharpes),
            "tier_ceiling": tier_fraction,
            "current_fraction": self.current_fraction,
        }
        if len(self._sharpes) < 50:
            return ScalingDecision(
                action="hold",
                current_fraction=self.current_fraction,
                target_fraction=self.current_fraction,
                reason="insufficient observations",
                evidence=evidence,
                ts=time.time(),
            )
        recent_sharpe = float(np.mean(self._sharpes[-100:]))
        dd_vol = float(np.std(self._drawdowns[-100:])) if len(self._drawdowns) >= 100 else 0.0
        avg_slip = float(np.mean(self._slippages_bps[-100:]))
        avg_div = float(np.mean(self._divergences[-100:]))
        evidence.update({
            "recent_sharpe": recent_sharpe,
            "dd_volatility": dd_vol,
            "avg_slippage_bps": avg_slip,
            "avg_divergence_pct": avg_div,
        })

        # Critical #2 fix: if the tier ceiling has been lowered below
        # the current fraction (e.g. after a demotion), scale down
        # immediately — the tier system is a hard ceiling and must be
        # respected regardless of whether other scale-down triggers fire.
        if self.current_fraction > tier_fraction:
            return self._scale_down(
                f"tier ceiling lowered to {tier_fraction:.4f}", evidence)

        # Scale-DOWN checks (any one triggers)
        if recent_sharpe < 0.3:
            return self._scale_down("sharpe collapsed", evidence)
        if dd_vol > self.max_dd_vol * 2:
            return self._scale_down("drawdown volatility spiked", evidence)
        if avg_div > self.max_divergence:
            return self._scale_down("live-paper divergence too high", evidence)
        if avg_slip > self.max_slip_bps * 1.5:
            return self._scale_down("slippage exceeds tolerance", evidence)

        # Scale-UP checks (ALL must pass)
        scale_up_conditions = {
            "sharpe_ok": recent_sharpe >= self.min_sharpe,
            "dd_vol_ok": dd_vol <= self.max_dd_vol,
            "slippage_ok": avg_slip <= self.max_slip_bps,
            "divergence_ok": avg_div <= self.max_divergence,
        }
        evidence["scale_up_conditions"] = scale_up_conditions
        if all(scale_up_conditions.values()):
            return self._scale_up(tier_fraction, evidence)

        return ScalingDecision(
            action="hold",
            current_fraction=self.current_fraction,
            target_fraction=self.current_fraction,
            reason="scale-up conditions not all met",
            evidence=evidence,
            ts=time.time(),
        )

    # ----------------------------------------------------------------
    def _scale_up(self, tier_ceiling: float,
                  evidence: dict[str, Any]) -> ScalingDecision:
        target = min(self.max_fraction, tier_ceiling,
                     self.current_fraction + self.scale_up_step)
        if target <= self.current_fraction:
            return ScalingDecision("hold", self.current_fraction,
                                    self.current_fraction,
                                    "at tier ceiling or max fraction",
                                    evidence, time.time())
        self.current_fraction = target
        log.info("CAPITAL SCALE UP -> %.4f", target)
        return ScalingDecision(
            action="scale_up",
            current_fraction=self.current_fraction - self.scale_up_step,
            target_fraction=target,
            reason="all scale-up conditions met",
            evidence=evidence,
            ts=time.time(),
        )

    def _scale_down(self, reason: str,
                    evidence: dict[str, Any]) -> ScalingDecision:
        target = max(self.min_fraction,
                     self.current_fraction * (1.0 - self.scale_down_step))
        if target >= self.current_fraction:
            return ScalingDecision("hold", self.current_fraction,
                                    self.current_fraction,
                                    "already at min fraction",
                                    evidence, time.time())
        old = self.current_fraction
        self.current_fraction = target
        log.warning("CAPITAL SCALE DOWN -> %.4f (was %.4f) reason=%s",
                    target, old, reason)
        return ScalingDecision(
            action="scale_down",
            current_fraction=old,
            target_fraction=target,
            reason=reason,
            evidence=evidence,
            ts=time.time(),
        )

    # ----------------------------------------------------------------
    def manual_set(self, fraction: float, reason: str = "manual") -> None:
        old = self.current_fraction
        self.current_fraction = float(max(self.min_fraction,
                                          min(self.max_fraction, fraction)))
        log.info("CAPITAL MANUAL SET %.4f -> %.4f (%s)",
                 old, self.current_fraction, reason)
