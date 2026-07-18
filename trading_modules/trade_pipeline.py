"""
Complete Trade Flow Pipeline
==============================

Orchestrates all 10 trading modules into a single end-to-end pipeline:

  Market Data → SMC Analysis → Pattern Detection → Confluence Gate
    → Kill Conditions → Coin Cooldown → Kelly Sizing
    → R-Multiple TP Plan → Signal Processing → Bias Adjustment
    → 5-Tier Rating → Final Trade Decision

This is the "one function call" entry point for the entire trading system.
Give it OHLCV data + an LLM analysis, and it returns a complete trade
decision with entry, stop, take-profits, position size, and risk assessment.

Usage:
    from trade_pipeline import TradePipeline, PipelineInput

    pipeline = TradePipeline()

    result = pipeline.execute(PipelineInput(
        symbol="BTCUSDT",
        ohlcv_df=df,                    # OHLCV DataFrame
        direction="BUY",
        llm_analysis="Based on...",     # LLM agent's analysis text
        account_equity=10000,
        mtf_trend={"H4": "bullish", "H1": "bullish"},
        pattern="Bullish Engulfing",
        pattern_rating=5,
    ))

    if result.approved:
        print(f"Entry: ${result.entry_price}")
        print(f"Stop: ${result.stop_loss}")
        print(f"Position: ${result.position_usd:.2f}")
        print(f"Take-profits: {[tp['price'] for tp in result.take_profits]}")
    else:
        print(f"Rejected: {result.rejection_reason}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from .smc_detector import SMCDetector, SMCResult
from .confluence_gate import ConfluenceGate, ConfluenceInput, ConfluenceResult
from .rating_system import RatingSystem
from .kill_conditions import KillConditions, PortfolioState
from .bias_tracker import BiasTracker
from .kelly_sizing import kelly_position_size
from .r_multiple_tp import RMultipleTP, Position
from .coin_cooldown import CoinCooldownManager
from .signal_processor import SignalProcessor, TradingDecision

logger = logging.getLogger(__name__)


@dataclass
class PipelineInput:
    """All inputs needed for the complete trade flow pipeline."""
    # Required
    symbol: str
    ohlcv_df: pd.DataFrame          # Must have open/high/low/close/volume columns
    direction: str                   # "BUY" or "SELL"
    account_equity: float

    # LLM Analysis (the agent's output)
    llm_analysis: str = ""

    # Market context
    mtf_trend: dict = field(default_factory=dict)
    pattern: str = ""
    pattern_rating: int = 0          # 1-5 stars

    # Optional overrides (auto-detected from OHLCV if not provided)
    rsi: Optional[float] = None
    volume_ratio: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None

    # Portfolio state (for kill conditions)
    cumulative_loss_usd: float = 0.0
    rolling_sharpe_14d: float = 1.0
    current_drawdown_pct: float = 0.0
    rolling_brier_30d: float = 0.20
    paper_trade_days: int = 0

    # Kelly parameters
    p_win: Optional[float] = None    # Auto from confidence if None
    avg_win_pct: float = 0.05        # 5% average win
    avg_loss_pct: float = 0.03       # 3% average loss

    # Configuration
    confluence_mode: str = "strict"  # "strict" or "weighted"


@dataclass
class TradeDecision:
    """Final trade decision with all computed values."""
    # Approval
    approved: bool = False
    rejection_reason: str = ""
    rejection_stage: str = ""        # Which stage rejected

    # Symbol & Direction
    symbol: str = ""
    direction: str = ""              # "BUY" / "SELL"
    rating: str = "Hold"             # 5-tier rating

    # Entry & Risk
    entry_price: float = 0.0
    stop_loss: float = 0.0
    risk_per_unit: float = 0.0
    risk_usd: float = 0.0

    # Position Sizing
    position_usd: float = 0.0
    position_size: float = 0.0       # In units (BTC, etc.)
    kelly_fraction: float = 0.0
    confidence: float = 0.5
    adjusted_confidence: float = 0.5

    # Take-Profit Plan
    take_profits: list = field(default_factory=list)
    # Each: {"r_multiple": 2.0, "price": X, "close_pct": 0.4, "new_stop": Y}

    # Confluence
    confluence_score: float = 0.0
    confluence_checks: dict = field(default_factory=dict)

    # SMC Analysis
    smc_trend: str = "unknown"
    smc_context: str = ""

    # Risk Assessment
    kill_state: str = "OK"
    bias_warnings: list = field(default_factory=list)
    all_warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
            "symbol": self.symbol,
            "direction": self.direction,
            "rating": self.rating,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "risk_per_unit": self.risk_per_unit,
            "risk_usd": round(self.risk_usd, 2),
            "position_usd": round(self.position_usd, 2),
            "position_size": round(self.position_size, 6),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "confidence": round(self.confidence, 4),
            "adjusted_confidence": round(self.adjusted_confidence, 4),
            "take_profits": self.take_profits,
            "confluence_score": round(self.confluence_score, 4),
            "confluence_checks": self.confluence_checks,
            "smc_trend": self.smc_trend,
            "kill_state": self.kill_state,
            "warnings": self.all_warnings,
        }

    def summary(self) -> str:
        """Generate a human-readable trade summary."""
        if not self.approved:
            return f"❌ REJECTED at {self.rejection_stage}: {self.rejection_reason}"

        lines = [
            f"✅ TRADE APPROVED: {self.direction} {self.symbol}",
            f"   Rating: {self.rating} ({self.confidence:.0%} → {self.adjusted_confidence:.0%} adjusted)",
            f"   Entry: ${self.entry_price:.2f}",
            f"   Stop Loss: ${self.stop_loss:.2f} (risk: ${self.risk_usd:.2f})",
            f"   Position: ${self.position_usd:.2f} ({self.position_size:.6f} units)",
            f"   Kelly: {self.kelly_fraction:.2%} of account",
            f"   SMC Trend: {self.smc_trend}",
            f"   Confluence: {self.confluence_score:.0%}",
            "",
            "   Take-Profit Plan:",
        ]

        for tp in self.take_profits:
            lines.append(
                f"     {tp['r_multiple']:.0f}R → ${tp['price']:.2f} "
                f"(close {tp['close_pct']:.0%}, stop→{tp['new_stop_label']})"
            )

        if self.all_warnings:
            lines.append("")
            lines.append("   ⚠️ Warnings:")
            for w in self.all_warnings:
                lines.append(f"     • {w}")

        return "\n".join(lines)


class TradePipeline:
    """
    Complete trade flow pipeline — orchestrates all 10 modules.

    Flow:
        1. SMC Analysis (BOS/CHoCH/Order Blocks/FVG/Liquidity)
        2. Auto-detect RSI & Volume from OHLCV
        3. Confluence Gate (8-check entry confirmation)
        4. Kill Conditions (4-gate risk protection)
        5. Coin Cooldown (revenge-trade prevention)
        6. Signal Processing (LLM output → structured JSON)
        7. Bias Adjustment (confidence correction)
        8. Kelly Sizing (Half-Kelly + drawdown-adjusted)
        9. R-Multiple TP Plan (2R/3R/5R + breakeven trail)
        10. 5-Tier Rating (institutional scale)
    """

    def __init__(
        self,
        smc_detector: Optional[SMCDetector] = None,
        confluence_gate: Optional[ConfluenceGate] = None,
        kill_conditions: Optional[KillConditions] = None,
        bias_tracker: Optional[BiasTracker] = None,
        coin_cooldown: Optional[CoinCooldownManager] = None,
        signal_processor: Optional[SignalProcessor] = None,
        rating_system: Optional[RatingSystem] = None,
        r_multiple_tp: Optional[RMultipleTP] = None,
    ):
        self.smc = smc_detector or SMCDetector()
        self.gate = confluence_gate or ConfluenceGate(require_all=True)
        self.kill = kill_conditions or KillConditions()
        self.bias = bias_tracker or BiasTracker()
        self.cooldown = coin_cooldown or CoinCooldownManager()
        self.signal = signal_processor or SignalProcessor()
        self.rating = rating_system or RatingSystem()
        self.tp = r_multiple_tp or RMultipleTP()

    def execute(self, inp: PipelineInput) -> TradeDecision:
        """
        Run the complete trade flow pipeline.

        Returns TradeDecision with approved=True only if ALL stages pass.
        """
        decision = TradeDecision(
            symbol=inp.symbol,
            direction=inp.direction,
        )

        # === Stage 1: SMC Analysis ===
        smc_result = self._run_smc(inp, decision)
        if smc_result is None:
            decision.rejection_stage = "SMC Analysis"
            decision.rejection_reason = "SMC analysis failed"
            return decision

        # === Stage 2: Auto-detect indicators from OHLCV ===
        rsi, vol_ratio, current_price = self._auto_detect_indicators(inp)
        if inp.entry_price is None:
            inp.entry_price = current_price
        if inp.rsi is None:
            inp.rsi = rsi
        if inp.volume_ratio is None:
            inp.volume_ratio = vol_ratio

        # Auto-detect stop loss from SMC if not provided
        if inp.stop_loss is None:
            inp.stop_loss = self._auto_stop_loss(inp, smc_result)

        # === Stage 3: Confluence Gate ===
        confluence = self._run_confluence(inp, smc_result, decision)
        if confluence is None or confluence.signal != "EXECUTE":
            decision.rejection_stage = "Confluence Gate"
            decision.rejection_reason = confluence.recommendation if confluence else "Confluence check failed"
            return decision

        # === Stage 4: Kill Conditions ===
        if not self._run_kill_conditions(inp, decision):
            return decision  # Rejection reason already set

        # === Stage 5: Coin Cooldown ===
        if not self._run_coin_cooldown(inp, decision):
            return decision

        # === Stage 6: Signal Processing (if LLM analysis provided) ===
        llm_decision = self._run_signal_processing(inp, decision)

        # === Stage 7: Bias Adjustment ===
        self._run_bias_adjustment(inp, llm_decision, decision)

        # === Stage 8: Kelly Sizing ===
        if not self._run_kelly_sizing(inp, decision):
            return decision

        # === Stage 9: R-Multiple TP Plan ===
        self._run_tp_plan(inp, decision)

        # === Stage 10: Final Rating ===
        self._run_rating(inp, llm_decision, decision)

        # === Approved! ===
        decision.approved = True
        decision.entry_price = inp.entry_price
        decision.stop_loss = inp.stop_loss
        decision.risk_per_unit = abs(inp.entry_price - inp.stop_loss)
        decision.risk_usd = decision.risk_per_unit * decision.position_size

        return decision

    # ═══════════════════════════════════════════════════════════════
    # Stage Implementations
    # ═══════════════════════════════════════════════════════════════

    def _run_smc(self, inp: PipelineInput, decision: TradeDecision) -> Optional[SMCResult]:
        """Stage 1: Run SMC analysis."""
        try:
            result = self.smc.analyze(inp.ohlcv_df, symbol=inp.symbol)
            decision.smc_trend = result.current_trend
            decision.smc_context = self.smc.get_confluence_context(
                result, float(inp.ohlcv_df['close'].iloc[-1])
            )
            return result
        except Exception as e:
            logger.error(f"SMC analysis failed: {e}")
            return None

    def _auto_detect_indicators(self, inp: PipelineInput) -> tuple:
        """Stage 2: Auto-detect RSI, volume ratio, and current price from OHLCV."""
        df = inp.ohlcv_df
        current_price = float(df['close'].iloc[-1])

        # Simple RSI (14-period)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi = float((100 - (100 / (1 + rs))).iloc[-1])
        rsi = max(0, min(100, rsi))

        # Volume ratio (current vs 20-bar average)
        vol = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))
        avg_vol = vol.rolling(20).mean().iloc[-1]
        current_vol = vol.iloc[-1]
        vol_ratio = float(current_vol / avg_vol) if avg_vol > 0 else 1.0

        return rsi, vol_ratio, current_price

    def _auto_stop_loss(self, inp: PipelineInput, smc: SMCResult) -> float:
        """Auto-detect stop loss from SMC zones or ATR."""
        current_price = inp.entry_price

        if inp.direction == "BUY":
            # For BUY: stop below nearest demand zone or 1.5% below entry
            if smc.nearest_demand_zone and smc.nearest_demand_zone < current_price:
                return smc.nearest_demand_zone * 0.998  # 0.2% buffer below zone
            return current_price * 0.985  # 1.5% stop
        else:
            # For SELL: stop above nearest supply zone or 1.5% above entry
            if smc.nearest_supply_zone and smc.nearest_supply_zone > current_price:
                return smc.nearest_supply_zone * 1.002  # 0.2% buffer above zone
            return current_price * 1.015  # 1.5% stop

    def _run_confluence(self, inp: PipelineInput, smc: SMCResult, decision: TradeDecision) -> Optional[ConfluenceResult]:
        """Stage 3: Run confluence gate."""
        # Check if at a key zone
        at_zone = False
        zone_type = ""
        if inp.direction == "BUY":
            if smc.nearest_demand_zone:
                dist = abs(inp.entry_price - smc.nearest_demand_zone) / inp.entry_price
                if dist < 0.02:  # Within 2% of zone
                    at_zone = True
                    zone_type = "demand"
        else:
            if smc.nearest_supply_zone:
                dist = abs(inp.entry_price - smc.nearest_supply_zone) / inp.entry_price
                if dist < 0.02:
                    at_zone = True
                    zone_type = "supply"

        # Determine structure break from SMC
        structure = ""
        if inp.direction == "BUY":
            if smc.bullish_bos:
                structure = "BOS"
            elif smc.bullish_choch:
                structure = "CHoCH"
        else:
            if smc.bearish_bos:
                structure = "BOS"
            elif smc.bearish_choch:
                structure = "CHoCH"

        confluence_input = ConfluenceInput(
            symbol=inp.symbol,
            direction=inp.direction,
            mtf_trend=inp.mtf_trend,
            at_key_zone=at_zone,
            zone_type=zone_type,
            liquidity_sweep=False,  # Would need separate detection
            pattern=inp.pattern,
            pattern_rating=inp.pattern_rating,
            volume_ratio=inp.volume_ratio or 1.0,
            rsi=inp.rsi or 50.0,
            structure_break=structure,
            candle_closed=True,  # Assume closed (pipeline runs after close)
        )

        result = self.gate.check(confluence_input)
        decision.confluence_score = result.score
        decision.confluence_checks = result.checks

        return result

    def _run_kill_conditions(self, inp: PipelineInput, decision: TradeDecision) -> bool:
        """Stage 4: Check kill conditions."""
        state = PortfolioState(
            cumulative_loss_usd=inp.cumulative_loss_usd,
            rolling_sharpe_14d=inp.rolling_sharpe_14d,
            current_drawdown_pct=inp.current_drawdown_pct,
            rolling_brier_30d=inp.rolling_brier_30d,
            paper_trade_days=inp.paper_trade_days,
        )

        kill_decision = self.kill.check(state)
        decision.kill_state = kill_decision.state

        if not kill_decision.can_trade:
            decision.rejection_stage = "Kill Conditions"
            decision.rejection_reason = kill_decision.trigger_reason or "Risk gate triggered"
            return False

        return True

    def _run_coin_cooldown(self, inp: PipelineInput, decision: TradeDecision) -> bool:
        """Stage 5: Check coin cooldown."""
        if self.cooldown.is_in_cooldown(inp.symbol):
            info = self.cooldown.get_cooldown_info(inp.symbol)
            decision.rejection_stage = "Coin Cooldown"
            decision.rejection_reason = f"{inp.symbol} in cooldown: {info.get('cooldown_reason', 'consecutive losses')}"
            return False
        return True

    def _run_signal_processing(self, inp: PipelineInput, decision: TradeDecision) -> Optional[TradingDecision]:
        """Stage 6: Process LLM analysis into structured decision."""
        if not inp.llm_analysis:
            return None

        llm_decision = self.signal.process(inp.llm_analysis, symbol=inp.symbol)
        decision.confidence = llm_decision.confidence
        return llm_decision

    def _run_bias_adjustment(self, inp: PipelineInput, llm_decision: Optional[TradingDecision], decision: TradeDecision):
        """Stage 7: Adjust confidence based on historical bias."""
        if llm_decision is None:
            decision.adjusted_confidence = inp.p_win or 0.5
            return

        # Get bias profile
        profile = self.bias.get_bias_profile()
        decision.adjusted_confidence = self.bias.adjust_confidence(
            llm_decision.confidence, direction=inp.direction
        )

        # Add bias warnings
        if profile.warnings:
            decision.bias_warnings = profile.warnings
            decision.all_warnings.extend(profile.warnings)

        # Use adjusted confidence as p_win for Kelly
        if inp.p_win is None:
            inp.p_win = decision.adjusted_confidence

    def _run_kelly_sizing(self, inp: PipelineInput, decision: TradeDecision) -> bool:
        """Stage 8: Compute Kelly position size."""
        p_win = inp.p_win or decision.adjusted_confidence or 0.5

        kelly_result = kelly_position_size(
            p_win=p_win,
            avg_win_pct=inp.avg_win_pct,
            avg_loss_pct=inp.avg_loss_pct,
            account_equity=inp.account_equity,
            current_drawdown_pct=inp.current_drawdown_pct,
        )

        if kelly_result.position_usd <= 0:
            decision.rejection_stage = "Kelly Sizing"
            decision.rejection_reason = "No edge detected or drawdown too high — position size = 0"
            return False

        decision.position_usd = kelly_result.position_usd
        decision.kelly_fraction = kelly_result.adjusted_kelly

        if kelly_result.warnings:
            decision.all_warnings.extend(kelly_result.warnings)

        # Compute position size in units
        if inp.entry_price > 0:
            decision.position_size = decision.position_usd / inp.entry_price

        return True

    def _run_tp_plan(self, inp: PipelineInput, decision: TradeDecision):
        """Stage 9: Create R-multiple take-profit plan."""
        position = Position(
            symbol=inp.symbol,
            direction="long" if inp.direction == "BUY" else "short",
            entry_price=inp.entry_price,
            stop_loss=inp.stop_loss,
            position_size=decision.position_size,
        )

        plan = self.tp.create_tp_plan(position)

        decision.take_profits = [
            {
                "r_multiple": level.r_multiple,
                "price": level.price,
                "close_pct": level.close_pct,
                "new_stop": level.new_stop,
                "new_stop_label": level.new_stop_label,
            }
            for level in plan.levels
        ]

    def _run_rating(self, inp: PipelineInput, llm_decision: Optional[TradingDecision], decision: TradeDecision):
        """Stage 10: Determine final 5-tier rating."""
        if llm_decision and llm_decision.rating:
            decision.rating = llm_decision.rating
        elif inp.pattern_rating > 0:
            direction = "bullish" if inp.direction == "BUY" else "bearish"
            decision.rating = self.rating.from_star_rating(inp.pattern_rating, direction)
        else:
            decision.rating = "Overweight" if inp.direction == "BUY" else "Underweight"