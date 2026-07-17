"""trading_modules.integration
=====================================================================
Integration bridge between the new trading_modules package and the
existing platform (engine, agents, validation, external).

This module provides adapter functions that:
  1. Convert platform Signal → trading_modules ConfluenceInput
  2. Convert trading_modules TradeDecision → platform ApprovedTrade
  3. Inject SMC analysis into the multi-agent graph
  4. Run confluence gate as pre-trade approval layer
  5. Apply edge_thesis + platt calibration to agent outputs
  6. Use verified_snapshot in analyst prompts

Usage from main loop:
    from trading_modules.integration import enhance_signal, pre_trade_gate

    # After strategy.evaluate() but before risk.evaluate():
    enhanced = enhance_signal(signal, df)
    if enhanced is None:
        continue  # confluence gate rejected

    approved = risk.evaluate(enhanced, df, equity)
"""
from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from utils.logger import get_logger
from engine.signals import Signal, Action

from .smc_detector import SMCDetector, SMCResult
from .confluence_gate import ConfluenceGate, ConfluenceInput, WeightedConfluenceGate
from .edge_thesis import EdgeThesisGate, EdgeThesis, EdgeCategory
from .platt_calibration import PlattCalibrator
from .verified_snapshot import build_verified_snapshot
from .rating_system import RatingSystem
from .kill_conditions import KillConditions, PortfolioState
from .bias_tracker import BiasTracker
from .kelly_sizing import kelly_position_size
from .r_multiple_tp import RMultipleTP, Position
from .coin_cooldown import CoinCooldownManager
from .signal_processor import SignalProcessor
from .triple_barrier import compute_labels
from .ml_models import build_features, MLModelTrainer
from .alpha_zoo import AlphaZoo

log = get_logger("trading_modules.integration")

# ═══════════════════════════════════════════════════════════════
# Singletons (instantiated on first use)
# ═══════════════════════════════════════════════════════════════

_smc_detector: Optional[SMCDetector] = None
_confluence_gate: Optional[ConfluenceGate] = None
_edge_gate: Optional[EdgeThesisGate] = None
_platt: Optional[PlattCalibrator] = None
_rating: Optional[RatingSystem] = None
_kill: Optional[KillConditions] = None
_bias: Optional[BiasTracker] = None
_cooldown: Optional[CoinCooldownManager] = None
_signal_proc: Optional[SignalProcessor] = None
_tp: Optional[RMultipleTP] = None


def _get_smc() -> SMCDetector:
    global _smc_detector
    if _smc_detector is None:
        _smc_detector = SMCDetector()
    return _smc_detector


def _get_gate(strict: bool = True) -> ConfluenceGate:
    global _confluence_gate
    if _confluence_gate is None:
        _confluence_gate = ConfluenceGate(require_all=strict)
    return _confluence_gate


def _get_edge_gate() -> EdgeThesisGate:
    global _edge_gate
    if _edge_gate is None:
        _edge_gate = EdgeThesisGate()
    return _edge_gate


def _get_platt() -> PlattCalibrator:
    global _platt
    if _platt is None:
        _platt = PlattCalibrator()
    return _platt


def _get_rating() -> RatingSystem:
    global _rating
    if _rating is None:
        _rating = RatingSystem()
    return _rating


def _get_kill() -> KillConditions:
    global _kill
    if _kill is None:
        _kill = KillConditions()
    return _kill


def _get_bias() -> BiasTracker:
    global _bias
    if _bias is None:
        _bias = BiasTracker()
    return _bias


def _get_cooldown() -> CoinCooldownManager:
    global _cooldown
    if _cooldown is None:
        _cooldown = CoinCooldownManager()
    return _cooldown


def _get_signal_proc() -> SignalProcessor:
    global _signal_proc
    if _signal_proc is None:
        _signal_proc = SignalProcessor()
    return _signal_proc


def _get_tp() -> RMultipleTP:
    global _tp
    if _tp is None:
        _tp = RMultipleTP()
    return _tp


# ═══════════════════════════════════════════════════════════════
# SMC Enhancement
# ═══════════════════════════════════════════════════════════════

def get_smc_context(df: pd.DataFrame, symbol: str) -> str:
    """
    Run SMC analysis and return context string for LLM prompts.

    Inject this into analyst agent prompts to give them SMC awareness:
      - Current trend (bullish/bearish/ranging)
      - Nearest demand/supply zones
      - BOS/CHoCH signals
      - Order blocks, FVGs, liquidity zones

    Usage:
        context = get_smc_context(df, "BTCUSD")
        prompt = f"\\n{context}\\n\\nBased on the above SMC data, analyze..."
    """
    detector = _get_smc()
    result = detector.analyze(df, symbol=symbol)
    current_price = float(df['close'].iloc[-1])
    return detector.get_confluence_context(result, current_price)


def get_smc_result(df: pd.DataFrame, symbol: str) -> SMCResult:
    """Get raw SMC result for programmatic use."""
    return _get_smc().analyze(df, symbol=symbol)


# ═══════════════════════════════════════════════════════════════
# Verified Snapshot for Anti-Confabulation
# ═══════════════════════════════════════════════════════════════

def get_verified_snapshot(df: pd.DataFrame, symbol: str) -> str:
    """
    Generate verified market-data snapshot for LLM prompts.

    Inject this into analyst prompts so they reference verified
    numbers instead of confabulated ones.

    Usage:
        snapshot = get_verified_snapshot(df, "BTCUSD")
        prompt = f"{snapshot}\\n\\nAnalyze using ONLY the verified data above..."
    """
    return build_verified_snapshot(df, symbol=symbol)


# ═══════════════════════════════════════════════════════════════
# Confluence Gate — Pre-Trade Approval
# ═══════════════════════════════════════════════════════════════

def pre_trade_gate(
    signal: Signal,
    df: pd.DataFrame,
    mtf_trend: Optional[dict] = None,
    pattern: str = "",
    pattern_rating: int = 0,
    strict: bool = False,
) -> Optional[Signal]:
    """
    Run confluence gate on a platform Signal.

    If the gate approves, returns the signal (possibly with adjusted strength).
    If the gate rejects, returns None.

    Args:
        signal: Platform Signal from strategy.evaluate()
        df: OHLCV DataFrame
        mtf_trend: Multi-timeframe trend dict {"H4": "bullish", "H1": "bullish"}
        pattern: Candlestick pattern name (if detected)
        pattern_rating: Pattern star rating (1-5)
        strict: If True, ALL checks must pass. If False, weighted (60% threshold).

    Returns:
        Signal if approved, None if rejected.
    """
    if signal.action == Action.HOLD:
        return signal  # Don't gate HOLD signals

    direction = "BUY" if signal.action == Action.BUY else "SELL"

    # Run SMC analysis for zone/structure context
    smc = _get_smc().analyze(df, symbol=signal.symbol)

    # Auto-detect RSI and volume
    close = df['close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi_val = float(50 + (100 * (gain - loss) / (gain + loss).replace(0, 1e-10)).iloc[-1])
    rsi_val = max(0, min(100, rsi_val))

    vol = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))
    avg_vol = vol.rolling(20).mean().iloc[-1]
    vol_ratio = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

    # Check if at key zone
    current_price = float(close.iloc[-1])
    at_zone = False
    zone_type = ""
    if direction == "BUY" and smc.nearest_demand_zone:
        dist = abs(current_price - smc.nearest_demand_zone) / current_price
        if dist < 0.02:
            at_zone = True
            zone_type = "demand"
    elif direction == "SELL" and smc.nearest_supply_zone:
        dist = abs(current_price - smc.nearest_supply_zone) / current_price
        if dist < 0.02:
            at_zone = True
            zone_type = "supply"

    # Determine structure break
    structure = ""
    if direction == "BUY":
        if smc.bullish_bos:
            structure = "BOS"
        elif smc.bullish_choch:
            structure = "CHoCH"
    else:
        if smc.bearish_bos:
            structure = "BOS"
        elif smc.bearish_choch:
            structure = "CHoCH"

    # Build confluence input
    confluence_input = ConfluenceInput(
        symbol=signal.symbol,
        direction=direction,
        mtf_trend=mtf_trend or {},
        at_key_zone=at_zone,
        zone_type=zone_type,
        liquidity_sweep=False,
        pattern=pattern,
        pattern_rating=pattern_rating,
        volume_ratio=vol_ratio,
        rsi=rsi_val,
        structure_break=structure,
        candle_closed=True,
    )

    # Run gate
    if strict:
        gate = _get_gate(strict=True)
    else:
        gate = WeightedConfluenceGate(min_score=0.50)

    result = gate.check(confluence_input)

    if result.signal != "EXECUTE":
        log.info(f"Confluence gate REJECTED {signal.symbol} {direction}: "
                 f"score={result.score:.0%}, failed={result.failed_checks}")
        return None

    # Approved — adjust signal strength by confluence score
    adjusted_strength = signal.strength * result.score
    log.info(f"Confluence gate APPROVED {signal.symbol} {direction}: "
             f"score={result.score:.0%}, strength {signal.strength:.2f}→{adjusted_strength:.2f}")

    # Return new signal with adjusted strength
    return Signal(
        symbol=signal.symbol,
        timeframe=signal.timeframe,
        action=signal.action,
        strength=adjusted_strength,
        price=signal.price,
        bar_time=signal.bar_time,
        meta={**(signal.meta or {}), 'confluence_score': result.score,
              'confluence_checks': result.checks},
    )


# ═══════════════════════════════════════════════════════════════
# Edge Thesis + Platt Calibration for Agent Outputs
# ═══════════════════════════════════════════════════════════════

def calibrate_agent_confidence(
    llm_output: str,
    raw_confidence: float,
    direction: str = "BUY",
) -> tuple[float, dict]:
    """
    Apply edge_thesis + platt calibration to LLM agent confidence.

    Pipeline:
      1. Extract edge thesis from LLM text
      2. Evaluate thesis quality → confidence downgrade if weak/missing
      3. Apply Platt calibration → fix systematic overconfidence

    Returns:
        (adjusted_confidence, metadata_dict)
    """
    # Step 1: Edge thesis
    edge_gate = _get_edge_gate()
    thesis = edge_gate.extract_from_text(llm_output)
    edge_result = edge_gate.evaluate(thesis, raw_confidence)

    # Step 2: Platt calibration
    platt = _get_platt()
    calibrated = platt.calibrate(edge_result.adjusted_confidence)

    # Step 3: Bias tracker adjustment
    bias = _get_bias()
    final = bias.adjust_confidence(calibrated, direction=direction)

    metadata = {
        "original_confidence": raw_confidence,
        "after_edge": edge_result.adjusted_confidence,
        "after_platt": calibrated,
        "after_bias": final,
        "edge_category": edge_result.category,
        "edge_quality": edge_result.quality_score,
        "edge_downgrade": edge_result.downgrade_pct,
        "platt_calibrated": platt.get_status().get("is_fitted", False),
    }

    return final, metadata


# ═══════════════════════════════════════════════════════════════
# Kill Conditions Check
# ═══════════════════════════════════════════════════════════════

def check_kill_conditions(
    cumulative_loss_usd: float = 0.0,
    rolling_sharpe_14d: float = 1.0,
    current_drawdown_pct: float = 0.0,
    rolling_brier_30d: float = 0.20,
    paper_trade_days: int = 0,
) -> bool:
    """
    Check if the system should be halted.

    Returns True if trading is allowed, False if killed.
    """
    state = PortfolioState(
        cumulative_loss_usd=cumulative_loss_usd,
        rolling_sharpe_14d=rolling_sharpe_14d,
        current_drawdown_pct=current_drawdown_pct,
        rolling_brier_30d=rolling_brier_30d,
        paper_trade_days=paper_trade_days,
    )
    decision = _get_kill().check(state)
    if not decision.can_trade:
        log.warning(f"KILL CONDITIONS TRIGGERED: {decision.trigger_reason}")
    return decision.can_trade


# ═══════════════════════════════════════════════════════════════
# Coin Cooldown Check
# ═══════════════════════════════════════════════════════════════

def check_coin_cooldown(symbol: str) -> bool:
    """
    Check if a symbol is in cooldown (revenge-trade prevention).

    Returns True if trading is allowed, False if in cooldown.
    """
    ccm = _get_cooldown()
    if ccm.is_in_cooldown(symbol):
        info = ccm.get_cooldown_info(symbol)
        log.warning(f"{symbol} in cooldown: {info.get('cooldown_reason')}")
        return False
    return True


def record_trade_outcome(symbol: str, pnl_usd: float) -> None:
    """Record a trade outcome for cooldown tracking."""
    result = "win" if pnl_usd > 0 else "loss" if pnl_usd < 0 else "breakeven"
    _get_cooldown().record_trade(symbol, pnl_usd=pnl_usd, result=result)


# ═══════════════════════════════════════════════════════════════
# R-Multiple Take-Profit Plan
# ═══════════════════════════════════════════════════════════════

def create_tp_plan(
    symbol: str,
    direction: str,
    entry_price: float,
    stop_loss: float,
    position_size: float,
) -> list[dict]:
    """
    Create R-multiple take-profit plan for a position.

    Returns list of TP levels:
      [{"r_multiple": 2.0, "price": X, "close_pct": 0.4, "new_stop": Y, "new_stop_label": "breakeven"}, ...]
    """
    tp = _get_tp()
    pos = Position(
        symbol=symbol,
        direction="long" if direction.upper() == "BUY" else "short",
        entry_price=entry_price,
        stop_loss=stop_loss,
        position_size=position_size,
    )
    plan = tp.create_tp_plan(pos)
    return [
        {
            "r_multiple": level.r_multiple,
            "price": level.price,
            "close_pct": level.close_pct,
            "new_stop": level.new_stop,
            "new_stop_label": level.new_stop_label,
        }
        for level in plan.levels
    ]


# ═══════════════════════════════════════════════════════════════
# Signal Processor — Parse LLM Output
# ═══════════════════════════════════════════════════════════════

def parse_llm_decision(llm_text: str, symbol: str = "") -> dict:
    """
    Parse LLM free-text output into structured decision.

    Returns dict with: action, target_price, confidence, risk_score,
    edge_thesis, rating
    """
    sp = _get_signal_proc()
    decision = sp.process(llm_text, symbol=symbol)
    return decision.to_dict()


# ═══════════════════════════════════════════════════════════════
# ML Models — Feature Building + Training
# ═══════════════════════════════════════════════════════════════

def get_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from OHLCV data."""
    return build_features(df)


def get_triple_barrier_labels(
    df: pd.DataFrame,
    upper_pct: float = 0.02,
    lower_pct: float = 0.01,
    max_holding: int = 5,
) -> pd.Series:
    """Compute triple-barrier labels for ML training."""
    return compute_labels(df, upper_pct=upper_pct, lower_pct=lower_pct,
                          max_holding=max_holding)


def train_ml_models(df: pd.DataFrame) -> dict:
    """
    Train RF + XGBoost + LR models on OHLCV data.

    Returns model leaderboard with accuracy, Sharpe, AUC.
    """
    features = build_features(df)
    labels = compute_labels(df)
    trainer = MLModelTrainer()
    results = trainer.train_all(features, labels)
    return {name: r.to_dict() for name, r in results.items()}


# ═══════════════════════════════════════════════════════════════
# Alpha Zoo — Cross-Sectional Factors
# ═══════════════════════════════════════════════════════════════

def compute_alpha(alpha_id: str, panel: dict) -> pd.DataFrame:
    """
    Compute a cross-sectional alpha factor.

    Args:
        alpha_id: Alpha ID (e.g., "alpha_001", "momentum_20d")
        panel: Dict of wide DataFrames: {"close": df, "open": df, ...}

    Returns:
        Wide DataFrame (date × assets) of alpha values
    """
    zoo = AlphaZoo()
    return zoo.compute(alpha_id, panel)


def list_available_alphas() -> list[str]:
    """List all available alpha IDs."""
    return AlphaZoo().list_alphas()


# ═══════════════════════════════════════════════════════════════
# Full Integration: enhance_signal
# ═══════════════════════════════════════════════════════════════

def enhance_signal(
    signal: Signal,
    df: pd.DataFrame,
    mtf_trend: Optional[dict] = None,
    pattern: str = "",
    pattern_rating: int = 0,
    cumulative_loss: float = 0.0,
    current_drawdown: float = 0.0,
) -> Optional[Signal]:
    """
    Full pre-trade enhancement pipeline.

    Combines: SMC analysis → Confluence gate → Kill conditions →
    Coin cooldown → Signal adjustment.

    This is the main integration point — call this AFTER strategy.evaluate()
    but BEFORE risk.evaluate().

    Returns enhanced Signal if approved, None if rejected.
    """
    # 1. Kill conditions
    if not check_kill_conditions(
        cumulative_loss_usd=cumulative_loss,
        current_drawdown_pct=current_drawdown,
    ):
        log.warning(f"Kill conditions active — rejecting {signal.symbol}")
        return None

    # 2. Coin cooldown
    if not check_coin_cooldown(signal.symbol):
        return None

    # 3. Confluence gate
    enhanced = pre_trade_gate(signal, df, mtf_trend, pattern, pattern_rating)
    if enhanced is None:
        return None

    return enhanced
