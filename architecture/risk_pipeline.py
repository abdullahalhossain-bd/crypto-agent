"""architecture/risk_pipeline.py
=====================================================================
Enhanced Risk Engine Pipeline (Improvement #7)
=====================================================================
A modular, layered risk engine that processes every trade through
12 sequential gates. Each gate is independent, swappable, and emits
events to the EventBus. If ANY gate fails, the trade is rejected.

This is the v2 successor to engine/risk.py (9-layer pipeline). Adds:
    - Portfolio context awareness (uses PortfolioManager)
    - Real-time correlation check (don't add correlated position)
    - Margin utilization check (don't blow up the account)
    - Liquidity check (don't trade illiquid symbols)
    - News blackout check (don't trade around scheduled news)
    - Volatility regime check (size down in extreme vol)

The 12 Layers:
    1.  Validation Gate      — input integrity, signal schema check
    2.  Portfolio Gate       — exposure/heat/concentration limits
    3.  Correlation Gate     — block if too correlated with open positions
    4.  Market Regime Gate   — block if regime is hostile (e.g. chop)
    5.  Volatility Gate      — block if ATR% is extreme
    6.  Liquidity Gate       — block if spread/volume is too low
    7.  News Blackout Gate   — block N minutes around news events
    8.  Drawdown Gate        — block if current drawdown exceeds limit
    9.  Consecutive Loss Gate — block after N consecutive losses
    10. Cooldown Gate        — enforce minimum time between trades
    11. Sizing Gate          — dynamic position sizing (Kelly + ATR + confidence)
    12. SL/TP Gate           — compute stop-loss and take-profit

Each gate returns:
    (passed: bool, reason: str, modified_signal: Optional[Signal])
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from architecture.event_bus import EventBus, EventType, get_bus
from architecture.portfolio_manager_v2 import PortfolioManager, PortfolioMetrics
from utils.logger import get_logger

log = get_logger("trading_bot.architecture.risk_pipeline")


@dataclass
class RiskContext:
    """All inputs a risk gate might need."""
    signal: Any            # engine.signals.Signal
    df: pd.DataFrame       # latest OHLCV
    account_equity: float
    portfolio: PortfolioMetrics = None  # type: ignore
    symbol_info: Any = None            # exchange SymbolInfo
    current_prices: Dict[str, float] = field(default_factory=dict)
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    # P0-4 FIX (Phase 3): consecutive_losses MUST be computed from real trade
    # history by the caller (TradingBot._process_symbol) — never hardcoded to 0.
    # ConsecutiveLossGate can now actually trigger.
    consecutive_losses: int = 0
    current_drawdown_pct: float = 0.0
    last_trade_time: float = 0.0
    news_blackout_until: float = 0.0
    correlation_matrix: Optional[pd.DataFrame] = None
    # P0-3 FIX (Phase 3): daily P&L fields for DailyLossGate. The caller must
    # compute realized_pnl_today from the trade journal (sum of pnl for trades
    # closed since UTC midnight). When daily_loss_halted_until > time.time(),
    # the gate blocks ALL new entries regardless of pnl — this is the manual
    # override / cooldown path.
    realized_pnl_today: float = 0.0
    daily_loss_halted_until: float = 0.0
    # FIX-RP-01: scratch space for gates earlier in the pipeline to hand
    # real computed values (lots, risk_amount, sl/tp) to gates later in the
    # pipeline, so e.g. PortfolioGate validates the trade that will ACTUALLY
    # be placed rather than a hardcoded guess. Populated as the pipeline runs.
    pipeline_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskVerdict:
    """Output of a single risk gate."""
    gate_name: str
    passed: bool
    reason: str = ""
    modified_lots: Optional[float] = None
    modified_sl: Optional[float] = None
    modified_tp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskGate(ABC):
    """Abstract base class for all risk gates."""

    name: str = "abstract_gate"

    @abstractmethod
    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        ...


# ----------------------------------------------------------------------
# Concrete gates
# ----------------------------------------------------------------------
class ValidationGate(RiskGate):
    name = "validation"

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        sig = ctx.signal
        if sig is None:
            return RiskVerdict(self.name, False, "signal is None")
        if not hasattr(sig, "action") or not hasattr(sig, "strength"):
            return RiskVerdict(self.name, False, "signal missing fields")
        if ctx.df is None or ctx.df.empty:
            return RiskVerdict(self.name, False, "no OHLCV data")
        if len(ctx.df) < 50:
            return RiskVerdict(self.name, False,
                              f"insufficient bars ({len(ctx.df)} < 50)")
        if ctx.account_equity <= 0:
            return RiskVerdict(self.name, False, "equity <= 0")
        return RiskVerdict(self.name, True, "OK")


class PortfolioGate(RiskGate):
    """Validates portfolio-level exposure/heat/concentration limits.

    FIX-RP-01: this gate now runs AFTER SizingGate in the pipeline order
    (see RiskPipeline.__init__) and reads the REAL lots/risk amount that
    SizingGate computed via `ctx.pipeline_state`, instead of guessing a
    flat 2% risk figure that had no relationship to what actually gets
    sized and sent to the broker.

    FIX-RP-02: a RiskPipeline with no PortfolioManager wired in must FAIL,
    not silently substitute a no-op gate — there is no safe default for
    "skip portfolio risk checks" in a system meant to eventually carry
    real capital. See RiskPipeline.__init__ for the enforcement.
    """
    name = "portfolio"

    def __init__(self, portfolio: PortfolioManager):
        if portfolio is None:
            raise ValueError(
                "PortfolioGate requires a PortfolioManager instance — "
                "there is no safe way to run the risk pipeline without "
                "portfolio-level exposure/heat checks."
            )
        self._pm = portfolio

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        symbol = ctx.signal.symbol if hasattr(ctx.signal, "symbol") else ""
        price = getattr(ctx.signal, "price", 0.0)

        # Review Point 8: absolute max-position-count hard cap.
        # Independent of per-trade sizing — a final backstop that says
        # "never more than N open positions regardless of what each
        # individual gate says." Default 10 (configurable via risk.max_open_trades).
        open_count = self._pm.open_count()
        max_open = int(getattr(self._pm, '_max_open_positions', 10))
        if open_count >= max_open:
            return RiskVerdict(
                self.name, False,
                f"max_open_positions cap reached ({open_count}/{max_open}) — "
                f"absolute hard limit, no new entries until a position closes")

        # Prefer the REAL sizing output from SizingGate; only fall back to
        # a conservative estimate if sizing didn't run for some reason
        # (defensive — should not happen given the enforced gate order).
        sizing = ctx.pipeline_state.get("sizing")
        if sizing is not None:
            proposed_volume = sizing["lots"]
            proposed_risk = sizing["risk_amount"]
        else:
            log.warning("portfolio_gate: no sizing output found in pipeline_state — "
                       "falling back to conservative estimate; check gate order")
            proposed_risk = ctx.account_equity * 0.02
            proposed_volume = proposed_risk / max(price, 0.0001)

        contract_size = getattr(ctx.symbol_info, "contract_size", 1.0) if ctx.symbol_info else 1.0

        allowed, reason, reservation_id = self._pm.can_open_new(
            symbol=symbol,
            proposed_risk=proposed_risk,
            proposed_volume=proposed_volume,
            proposed_price=price,
            contract_size=contract_size,
        )
        if not allowed:
            return RiskVerdict(self.name, False, reason)
        return RiskVerdict(self.name, True, "OK",
                          metadata={"reservation_id": reservation_id})


class CorrelationGate(RiskGate):
    name = "correlation"

    def __init__(self, max_correlation: float = 0.85):
        self._max_corr = max_correlation

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        if ctx.correlation_matrix is None or ctx.correlation_matrix.empty:
            return RiskVerdict(self.name, True, "no correlation matrix")
        sym = ctx.signal.symbol if hasattr(ctx.signal, "symbol") else ""
        if sym not in ctx.correlation_matrix.columns:
            return RiskVerdict(self.name, True, "symbol not in matrix")
        # Check correlation with open positions
        for pos in ctx.open_positions:
            other = pos.get("symbol", "")
            if other == sym or other not in ctx.correlation_matrix.columns:
                continue
            corr = ctx.correlation_matrix.loc[sym, other]
            if abs(corr) > self._max_corr:
                return RiskVerdict(
                    self.name, False,
                    f"correlation with {other} = {corr:.2f} > {self._max_corr}"
                )
        return RiskVerdict(self.name, True, "OK")


class MarketRegimeGate(RiskGate):
    name = "regime"

    def __init__(self, blocked_regimes: Optional[set] = None,
                 block_unknown: bool = True):
        # C6 fix: 'unknown' is now included in the blocked set by default.
        # When regime detection fails (returns 'unknown'), the gate blocks
        # the trade rather than silently allowing it. Operators who want
        # to trade in unknown regimes can pass block_unknown=False.
        #
        # RISK PIPELINE AUDIT FIX: aligned blocked_regimes with the ACTUAL
        # regime strings returned by RegimeOrchestrator (MarketRegime enum):
        #   trend_up, trend_down, range, high_vol, low_vol,
        #   transition, crisis, chop, unknown
        # Previously "choppy" and "extreme_vol" were in the set but are
        # NEVER returned — they were legacy names. "crisis" and "transition"
        # were missing — now added.
        default_blocked = {"crisis", "transition", "chop", "unknown"}
        if blocked_regimes is None:
            blocked_regimes = default_blocked
        if block_unknown and "unknown" not in blocked_regimes:
            blocked_regimes = blocked_regimes | {"unknown"}
        self._blocked = blocked_regimes

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        # FIX-RP-04: a broken/erroring regime calculation is missing
        # information, not a green light — fail CLOSED (block the trade)
        # instead of silently passing it through as before.
        try:
            from utils.indicators import regime_detection
            regime = regime_detection(ctx.df).get("regime", "unknown")
        except Exception as e:  # noqa: BLE001
            log.error("risk_pipeline: regime gate calculation failed — "
                      "failing CLOSED: %r", e)
            return RiskVerdict(self.name, False,
                              f"regime check errored, blocking trade: {e}")
        if regime in self._blocked:
            return RiskVerdict(self.name, False,
                              f"regime {regime} is blocked")
        return RiskVerdict(self.name, True, f"regime={regime}")


class VolatilityGate(RiskGate):
    name = "volatility"

    def __init__(self, max_atr_pct: float = 0.05):
        """Block if ATR% > max_atr_pct (e.g. 5% of price)."""
        self._max_atr_pct = max_atr_pct

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        # FIX-RP-04: fail CLOSED on calculation error (see MarketRegimeGate
        # for the same reasoning) — a NaN column or indicator exception must
        # block the trade, not silently disable the volatility check.
        #
        # Major #1 fix: check pipeline_state for pre-computed atr_pct before
        # recomputing. The caller (_process_symbol) pre-computes ATR and
        # ATR% once and stores them in pipeline_state so multiple gates
        # don't each recompute the same indicator.
        try:
            atr_p = ctx.pipeline_state.get("atr_pct")
            if atr_p is None:
                from utils.indicators import atr_pct
                atr_p = float(atr_pct(ctx.df, 14).iloc[-1])
            else:
                atr_p = float(atr_p)
        except Exception as e:  # noqa: BLE001
            log.error("risk_pipeline: volatility gate calculation failed — "
                      "failing CLOSED: %r", e)
            return RiskVerdict(self.name, False,
                              f"vol check errored, blocking trade: {e}")
        if atr_p > self._max_atr_pct:
            return RiskVerdict(self.name, False,
                              f"ATR%={atr_p:.4f} > {self._max_atr_pct}")
        return RiskVerdict(self.name, True, f"ATR%={atr_p:.4f}",
                          metadata={"atr_pct": atr_p})


class LiquidityGate(RiskGate):
    """P0-7 FIX (Phase 3): Per-symbol volume calibration.

    The old hardcoded min_volume=100.0 silently rejected valid trades on
    Deriv synthetic pairs (Booms/Crashes/Step indices) that legitimately
    trade at lower tick volumes. Now defaults to auto-calibration: the
    gate takes the 20th percentile of the last 20 bars' volume as the
    floor, so a symbol only gets rejected if its CURRENT volume is
    abnormally low relative to its OWN recent history — not relative to
    an arbitrary global threshold. A hardcoded floor can still be set via
    config for operators who want a fixed minimum.
    """
    name = "liquidity"

    def __init__(self, min_volume: float = 0.0, max_spread_bps: float = 15.0,
                 auto_calibrate: bool = True):
        # min_volume=0 + auto_calibrate=True → floor becomes 20th pct of recent vol.
        # 15 bps spread cap catches genuine anomalies (crypto usually 1–5 bps).
        self._min_volume = float(min_volume)
        self._max_spread_bps = float(max_spread_bps)
        self._auto_calibrate = bool(auto_calibrate)

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        if ctx.df is None or ctx.df.empty:
            return RiskVerdict(self.name, False, "no data")
        if "volume" not in ctx.df.columns or len(ctx.df) < 20:
            return RiskVerdict(self.name, True, "insufficient volume history to calibrate")
        recent_vol = ctx.df["volume"].tail(20).astype(float)
        current_vol = float(recent_vol.iloc[-1])
        # Auto-calibrate floor: 20th percentile of recent volume.
        # A symbol is "illiquid right now" only if its current bar is in the
        # bottom 20% of its own last 20 bars — a relative, not absolute, test.
        floor = self._min_volume
        if self._auto_calibrate:
            pct20 = float(recent_vol.quantile(0.20))
            floor = max(floor, pct20 * 0.5)  # allow some slack below p20
        if current_vol < floor:
            return RiskVerdict(self.name, False,
                              f"current volume {current_vol:.0f} < calibrated floor {floor:.0f}")
        if ctx.symbol_info is not None:
            spread = getattr(ctx.symbol_info, "spread", 0)
            point = getattr(ctx.symbol_info, "point", 0.0001)
            spread_bps = (spread * point / max(ctx.signal.price, 0.0001)) * 10000
            if spread_bps > self._max_spread_bps:
                return RiskVerdict(self.name, False,
                                  f"spread {spread_bps:.1f}bps > {self._max_spread_bps}bps")
        return RiskVerdict(self.name, True, f"vol={current_vol:.0f} floor={floor:.0f}")


class NewsBlackoutGate(RiskGate):
    name = "news_blackout"

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        if ctx.news_blackout_until > time.time():
            remaining = ctx.news_blackout_until - time.time()
            return RiskVerdict(self.name, False,
                              f"news blackout active ({remaining:.0f}s remaining)")
        return RiskVerdict(self.name, True, "OK")


class DrawdownGate(RiskGate):
    name = "drawdown"

    def __init__(self, max_drawdown_pct: float = 15.0):
        # 15% is the industry-standard prop-firm halt threshold. At 1% risk/trade
        # this allows ~15 consecutive losses before halt — well beyond the
        # ConsecutiveLossGate's 3-loss trigger, so this is a true backstop.
        self._max_dd = max_drawdown_pct

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        if ctx.current_drawdown_pct > self._max_dd:
            return RiskVerdict(self.name, False,
                              f"drawdown {ctx.current_drawdown_pct:.2f}% > {self._max_dd}%")
        return RiskVerdict(self.name, True, "OK")


class DailyLossGate(RiskGate):
    """P0-3 FIX (Phase 3): Hard daily-loss halt — merged from engine/risk.py:RiskManager.

    Enforces `risk.max_daily_loss` on the canonical path (previously only
    enforced on Stack-A, dead from Stack-C). When realized P&L for the current
    UTC day falls below -(max_daily_loss * account_equity), the gate blocks ALL
    new entries and sets daily_loss_halted_until to next UTC midnight. Resumes
    automatically on the new UTC day, or immediately if an operator manually
    clears the halt via the kill-switch file rotation.

    This is a HARD halt — not a warning, not a size reduction. The bot must
    stop adding risk for the day when this trips.
    """
    name = "daily_loss"

    def __init__(self, max_daily_loss_pct: float = 0.05):
        # 5% daily loss = 5 losing trades at 1% risk each. Standard prop-firm
        # daily-loss rule. Configurable via risk.max_daily_loss in config.yaml.
        self._max_daily_loss_pct = float(max_daily_loss_pct)

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        # Manual override cooldown (set by operator or by a previous breach today)
        if ctx.daily_loss_halted_until > time.time():
            remaining_s = ctx.daily_loss_halted_until - time.time()
            return RiskVerdict(
                self.name, False,
                f"daily loss halt active ({remaining_s/3600:.1f}h remaining) "
                f"— new entries blocked until UTC midnight or manual reset",
            )
        # Compute daily loss as a fraction of equity
        equity = max(ctx.account_equity, 1e-9)
        daily_loss_pct = abs(min(ctx.realized_pnl_today, 0.0)) / equity
        if daily_loss_pct > self._max_daily_loss_pct:
            # Block for the rest of the UTC day. The caller (TradingBot) is
            # responsible for persisting this halt timestamp so a restart
            # doesn't reset it — that's handled in integration.py via the
            # state file.
            return RiskVerdict(
                self.name, False,
                f"daily loss {daily_loss_pct*100:.2f}% > {self._max_daily_loss_pct*100:.2f}% "
                f"halt — new entries blocked for the rest of the UTC day",
            )
        return RiskVerdict(self.name, True, f"daily_loss={daily_loss_pct*100:.2f}%")


class ConsecutiveLossGate(RiskGate):
    name = "consecutive_loss"

    def __init__(self, max_consecutive_losses: int = 3):
        # 3 consecutive losses = something is wrong with the current regime
        # or strategy. Pause and let the next cycle re-evaluate. Configurable
        # via risk.max_consecutive_losses.
        self._max_losses = int(max_consecutive_losses)

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        # P0-4 FIX (Phase 3): ctx.consecutive_losses is now computed from real
        # trade history by TradingBot._process_symbol (was hardcoded to 0).
        if ctx.consecutive_losses >= self._max_losses:
            return RiskVerdict(self.name, False,
                              f"consecutive losses {ctx.consecutive_losses} >= {self._max_losses}")
        return RiskVerdict(self.name, True, f"consecutive_losses={ctx.consecutive_losses}")


class CooldownGate(RiskGate):
    name = "cooldown"

    def __init__(self, cooldown_s: float = 60.0):
        # 60s prevents rapid-fire duplicate signals on the same symbol while
        # still allowing the bot to react to genuine new setups within a minute.
        self._cooldown_s = float(cooldown_s)

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        if ctx.last_trade_time > 0:
            elapsed = time.time() - ctx.last_trade_time
            if elapsed < self._cooldown_s:
                remaining = self._cooldown_s - elapsed
                return RiskVerdict(self.name, False,
                                  f"cooldown {remaining:.0f}s remaining")
        return RiskVerdict(self.name, True, "OK")


class SizingGate(RiskGate):
    name = "sizing"

    def __init__(self,
                 risk_per_trade: float = 0.02,
                 max_risk_per_trade: float = 0.05,
                 kelly_fraction: float = 0.25):
        self._risk_per_trade = risk_per_trade
        self._max_risk = max_risk_per_trade
        self._kelly_fraction = kelly_fraction

    @staticmethod
    def _fractional_kelly_multiplier(recent_trades: List[Dict[str, Any]],
                                     kelly_fraction: float,
                                     min_trades: int = 10) -> float:
        """FIX-RP-03: proper fractional-Kelly sizing multiplier.

        The previous implementation (`max(0, 2*wr-1) * kelly_fraction`,
        applied as `lots *= (1 + kelly)`) used only win rate, ignored the
        win/loss payout ratio entirely, and — because it was a `(1 + kelly)`
        multiplicative BONUS rather than a fraction of Kelly-optimal size —
        could only ever hold size flat or increase it; there was no way for
        it to shrink size when edge looked weak (wr < 0.5 just gave kelly=0).

        This version uses the full Kelly criterion f* = (b*p - q) / b,
        where b = avg_win / avg_loss, p = win rate, q = 1 - p, and returns
        a multiplier around 1.0 (not a raw fraction), so lots can move
        ABOVE OR BELOW the ATR-based baseline size depending on whether
        the recent edge is better or worse than assumed. The result is
        clamped to [0.25x, 1.5x] of baseline regardless of kelly_fraction,
        so a single bad trade sample can't zero out or explode sizing.
        """
        if not recent_trades or len(recent_trades) < min_trades:
            return 1.0  # not enough sample — use the ATR/equity baseline as-is

        pnls = [t.get("pnl", 0.0) for t in recent_trades]
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p < 0]
        if not wins or not losses:
            return 1.0  # can't estimate a payout ratio yet

        win_rate = len(wins) / len(pnls)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        if avg_loss <= 0:
            return 1.0
        b = avg_win / avg_loss
        p, q = win_rate, 1.0 - win_rate
        f_star = (b * p - q) / b  # full Kelly fraction of capital to risk

        # Multiplier = 1.0 at "baseline" edge (f* roughly matching the
        # configured risk_per_trade assumption); scale around 1.0 by the
        # kelly_fraction so this acts as a bounded tilt, not a full Kelly bet.
        multiplier = 1.0 + max(-1.0, min(1.0, f_star)) * kelly_fraction
        return max(0.25, min(1.5, multiplier))

    @staticmethod
    def _drawdown_scale_multiplier(current_drawdown_pct: float) -> float:
        """Phase 4 req #24: drawdown-based capital scaling.

        Reduces position size as drawdown from peak equity deepens:
          DD < 5%   → 1.0x (full size)
          DD 5-10%  → 0.75x
          DD 10-15% → 0.5x
          DD > 15%  → 0.25x (minimum — DrawdownGate will reject above 15%
                             by default, but if the threshold is raised
                             via config we still scale down rather than
                             trading full size into a deep drawdown)

        This is a defensive, monotonically-decreasing curve. The exact
        breakpoints are conservative — a trader who wants more aggressive
        scaling can override via config in a future phase.
        """
        dd = max(0.0, float(current_drawdown_pct))
        if dd < 5.0:
            return 1.0
        elif dd < 10.0:
            return 0.75
        elif dd < 15.0:
            return 0.5
        else:
            return 0.25

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        try:
            # Major #1 fix: use pre-computed ATR from pipeline_state if available.
            atr_val = ctx.pipeline_state.get("atr")
            if atr_val is None:
                from utils.indicators import atr as _atr_fn
                atr_val = float(_atr_fn(ctx.df, 14).iloc[-1])
            else:
                atr_val = float(atr_val)
            price = ctx.signal.price
            # CRITICAL FIX (Risk Pipeline Audit): SizingGate and SLTPGate MUST
            # use the SAME sl_distance formula. Previously SizingGate used
            # max(ATR*1.5, price*0.5%) but SLTPGate used only ATR*1.5 — so
            # when ATR was small (low volatility), SizingGate assumed a wider
            # stop than SLTPGate actually placed, making the actual risk per
            # trade HIGHER than configured.
            #
            # Now both gates use: sl_distance = max(ATR * sl_mult, price * 0.5%)
            # This ensures the sizing calculation matches the actual SL placed.
            sl_mult = 1.5  # must match SLTPGate's default
            sl_distance = max(atr_val * sl_mult, price * 0.005)  # 1.5×ATR or 0.5%
            risk_amount = ctx.account_equity * self._risk_per_trade

            # FIX-RP-03: fractional-Kelly multiplier that can size DOWN as
            # well as up (see _fractional_kelly_multiplier docstring).
            kelly_mult = self._fractional_kelly_multiplier(
                ctx.recent_trades, self._kelly_fraction)
            risk_amount *= kelly_mult

            # Phase 4 req #24: drawdown-based capital scaling — reduce size
            # as drawdown deepens. This is distinct from DrawdownGate (which
            # hard-rejects at the threshold); this scales size down BEFORE
            # the threshold is reached, preserving capital during rough patches.
            dd_mult = self._drawdown_scale_multiplier(ctx.current_drawdown_pct)
            risk_amount *= dd_mult

            # P0-8 fix: incorporate contract_size into the lot calculation.
            # For forex pairs (contract_size=100000), the risk per lot is
            # sl_distance * contract_size, not just sl_distance.
            contract_size = float(getattr(ctx.symbol_info, "contract_size", 1.0)
                                 if ctx.symbol_info else 1.0)
            lots = risk_amount / (sl_distance * contract_size)

            # Cap at max risk
            max_lots = (ctx.account_equity * self._max_risk) / (sl_distance * contract_size)
            lots = min(lots, max_lots)

            # Round to volume step
            if ctx.symbol_info is not None:
                step = getattr(ctx.symbol_info, "volume_step", 0.01)
                lots = round(lots / step) * step
                lots = max(lots, getattr(ctx.symbol_info, "volume_min", 0.01))
                lots = min(lots, getattr(ctx.symbol_info, "volume_max", 100.0))

            # FIX-RP-01: expose the REAL sizing output to later gates
            # (PortfolioGate) via pipeline_state instead of them guessing.
            ctx.pipeline_state["sizing"] = {
                "lots": lots, "risk_amount": risk_amount,
                "kelly_multiplier": kelly_mult,
                "drawdown_multiplier": dd_mult,
            }

            return RiskVerdict(self.name, True, f"lots={lots:.2f}",
                              modified_lots=lots,
                              metadata={"atr": atr_val, "sl_distance": sl_distance,
                                       "risk_amount": risk_amount,
                                       "kelly_multiplier": kelly_mult,
                                       "drawdown_multiplier": dd_mult})
        except Exception as e:
            return RiskVerdict(self.name, False, f"sizing failed: {e}")


class SLTPGate(RiskGate):
    name = "sl_tp"

    def __init__(self,
                 sl_atr_multiple: float = 1.5,
                 tp_atr_multiple: float = 2.5):
        self._sl_mult = sl_atr_multiple
        self._tp_mult = tp_atr_multiple

    def evaluate(self, ctx: RiskContext) -> RiskVerdict:
        try:
            # Major #1 fix: use pre-computed ATR from pipeline_state if available.
            atr_val = ctx.pipeline_state.get("atr")
            if atr_val is None:
                from utils.indicators import atr as _atr_fn
                atr_val = float(_atr_fn(ctx.df, 14).iloc[-1])
            else:
                atr_val = float(atr_val)
            price = ctx.signal.price
            action = ctx.signal.action.value if hasattr(ctx.signal.action, "value") \
                else str(ctx.signal.action)

            # CRITICAL FIX (Risk Pipeline Audit): use the SAME sl_distance
            # formula as SizingGate: max(ATR * sl_mult, price * 0.5%).
            # Previously SLTPGate used only ATR * sl_mult, which could be
            # tighter than what SizingGate assumed → actual risk > configured.
            sl_distance = max(atr_val * self._sl_mult, price * 0.005)
            tp_distance = max(atr_val * self._tp_mult, price * 0.0075)  # TP floor: 0.75%

            if action == "BUY":
                sl = price - sl_distance
                tp = price + tp_distance
            elif action == "SELL":
                sl = price + sl_distance
                tp = price - tp_distance
            else:
                return RiskVerdict(self.name, False, f"action {action} not tradable")

            # Adjust to stops_level
            if ctx.symbol_info is not None:
                stops_level = getattr(ctx.symbol_info, "stops_level", 0)
                point = getattr(ctx.symbol_info, "point", 0.0001)
                min_dist = stops_level * point
                if abs(price - sl) < min_dist:
                    sl = price - min_dist if action == "BUY" else price + min_dist
                if abs(tp - price) < min_dist:
                    tp = price + min_dist if action == "BUY" else price - min_dist

            return RiskVerdict(self.name, True, f"SL={sl:.5f} TP={tp:.5f}",
                              modified_sl=sl, modified_tp=tp,
                              metadata={"atr": atr_val})
        except Exception as e:
            return RiskVerdict(self.name, False, f"SL/TP failed: {e}")


# ----------------------------------------------------------------------
# The pipeline orchestrator
# ----------------------------------------------------------------------
class RiskPipeline:
    """Runs every trade through 12 sequential risk gates.

    Each gate's pass/fail is logged and emitted to EventBus. The first
    gate to fail short-circuits the pipeline (returns immediately).
    """

    def __init__(self,
                 portfolio: Optional[PortfolioManager] = None,
                 bus: Optional[EventBus] = None,
                 config: Optional[Dict[str, Any]] = None):
        self._bus = bus or get_bus()
        self._portfolio = portfolio
        cfg = config or {}
        # FIX-RP-02: no more silent ValidationGate substitution — if
        # `portfolio` is None, PortfolioGate's own constructor now raises
        # immediately (fail closed on misconfiguration, not fail open).
        #
        # FIX-RP-01: PortfolioGate moved to run AFTER SizingGate/SLTPGate so
        # it validates the REAL computed lots/risk/price, not a hardcoded
        # 2% guess made before sizing ever ran.
        self._gates: List[RiskGate] = [
            ValidationGate(),
            CorrelationGate(cfg.get("max_correlation", 0.85)),
            MarketRegimeGate(cfg.get("blocked_regimes", {"crisis", "transition", "chop", "unknown"})),
            VolatilityGate(cfg.get("max_atr_pct", 0.05)),
            LiquidityGate(
                min_volume=cfg.get("min_volume", 0.0),
                max_spread_bps=cfg.get("max_spread_bps", 15.0),
                auto_calibrate=cfg.get("liquidity_auto_calibrate", True),
            ),
            NewsBlackoutGate(),
            DrawdownGate(cfg.get("max_drawdown_pct", 15.0)),
            # P0-3 FIX (Phase 3): DailyLossGate merged from engine/risk.py.
            # Now enforced on the canonical path. Default 5% daily loss halt.
            DailyLossGate(cfg.get("max_daily_loss", 0.05)),
            ConsecutiveLossGate(cfg.get("max_consecutive_losses", 3)),
            CooldownGate(cfg.get("cooldown_s", 60.0)),
            SizingGate(cfg.get("risk_per_trade", 0.02),
                      cfg.get("max_risk_per_trade", 0.05),
                      cfg.get("kelly_fraction", 0.25)),
            SLTPGate(cfg.get("sl_atr_multiple", 1.5),
                    cfg.get("tp_atr_multiple", 2.5)),
            PortfolioGate(portfolio),
        ]
        self._last_verdicts: List[RiskVerdict] = []

    def evaluate(self, ctx: RiskContext) -> Tuple[bool, RiskVerdict, List[RiskVerdict]]:
        """Run all gates. Returns (approved, final_verdict, all_verdicts).

        NOTE on reservations (FIX-RP-01/FIX-PM-03): if approved is True,
        `final.metadata["reservation_id"]` holds the PortfolioManager
        reservation for this trade's exposure/heat. The caller MUST either:
          - commit it via `portfolio.on_position_opened(..., reservation_id=...)`
            once the order actually fills, or
          - release it via `portfolio.release_reservation(reservation_id)`
            if the order is rejected/abandoned after approval,
        otherwise the reserved capacity stays locked until it auto-expires
        (see PortfolioManager._prune_stale_reservations).
        """
        verdicts: List[RiskVerdict] = []
        for gate in self._gates:
            # FIX-RP-05: an unexpected exception from a gate must reject the
            # trade (fail closed), not propagate up and crash the caller's
            # trading loop — consistent with the fail-closed fix applied to
            # MarketRegimeGate/VolatilityGate above, generalized as a safety
            # net for any gate (including future ones).
            try:
                v = gate.evaluate(ctx)
            except Exception as e:  # noqa: BLE001
                log.error("risk_pipeline: gate %s raised unexpectedly — "
                         "failing CLOSED: %r", getattr(gate, "name", "?"), e)
                v = RiskVerdict(getattr(gate, "name", "unknown"), False,
                               f"gate raised unexpectedly: {e}")
            verdicts.append(v)
            self._bus.emit(
                EventType.RISK_LAYER_PASSED if v.passed else EventType.RISK_LAYER_FAILED,
                payload={
                    "gate": v.gate_name,
                    "passed": v.passed,
                    "reason": v.reason,
                    "signal_symbol": ctx.signal.symbol if hasattr(ctx.signal, "symbol") else "",
                },
                source="risk_pipeline",
            )
            if not v.passed:
                # Co-Founder Audit: lower this from INFO to DEBUG. The
                # universe pre-filter in TradingBot._universe_filter now
                # catches the common rejection reasons (liquidity, spread,
                # volatility) BEFORE the AI agents run, so a symbol that
                # reaches the risk pipeline is already a serious candidate.
                # Per-symbol REJECTED logs at INFO were flooding operator
                # consoles (100 symbols × multiple gates = hundreds of lines
                # per cycle). The cycle-summary line in main.py shows the
                # aggregated counts; per-symbol detail is available at DEBUG
                # when troubleshooting a specific rejection.
                log.debug("risk_pipeline: REJECTED at %s — %s",
                         v.gate_name, v.reason)
                self._last_verdicts = verdicts
                return False, v, verdicts

        # All gates passed — aggregate final values
        final_lots = next((v.modified_lots for v in reversed(verdicts)
                          if v.modified_lots is not None), 0.01)
        final_sl = next((v.modified_sl for v in reversed(verdicts)
                        if v.modified_sl is not None), 0.0)
        final_tp = next((v.modified_tp for v in reversed(verdicts)
                        if v.modified_tp is not None), 0.0)
        reservation_id = next((v.metadata.get("reservation_id") for v in reversed(verdicts)
                              if v.metadata.get("reservation_id") is not None), None)
        final = RiskVerdict(
            gate_name="pipeline",
            passed=True,
            reason=f"all {len(self._gates)} gates passed",
            modified_lots=final_lots,
            modified_sl=final_sl,
            modified_tp=final_tp,
            metadata={"reservation_id": reservation_id},
        )
        self._last_verdicts = verdicts
        return True, final, verdicts

    def last_verdicts(self) -> List[RiskVerdict]:
        # L19 fix: return a copy so callers can't mutate internal state.
        return list(self._last_verdicts)

    def reset(self) -> None:
        """H17/L20 fix: clear the pipeline's state for a fresh cycle.

        Previously `_last_verdicts` was retained across cycles, which could
        cause stale verdicts to leak into status reports or diagnostics.
        """
        self._last_verdicts = []