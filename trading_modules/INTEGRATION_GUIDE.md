# Trading Modules — Integration Guide

## Overview

**10 production-ready modules** (3,114 LOC) built from 30+ repository reviews.
All modules are tested, documented, and ready for integration into your trading platform.

## Modules Summary

| Module | LOC | Source Inspiration | What It Does |
|--------|-----|-------------------|--------------|
| `smc_detector.py` | 402 | Vibe-Trading `smc` skill | BOS/CHoCH/Order Block/FVG/Liquidity detection via `smartmoneyconcepts` |
| `confluence_gate.py` | 376 | User's candlestick guide | 8-check entry confirmation checklist |
| `rating_system.py` | 214 | TradingAgents v0.3.1 | 5-tier institutional rating (Buy/Overweight/Hold/Underweight/Sell) |
| `kill_conditions.py` | 253 | Orallexa | 4-gate risk protection (loss/Sharpe/DD/Brier) |
| `bias_tracker.py` | 452 | Orallexa | Prediction accuracy self-correction with confidence adjustment |
| `kelly_sizing.py` | 289 | Orallexa | Half-Kelly position sizing with drawdown protection |
| `r_multiple_tp.py` | 293 | NexusQuant | R-Multiple partial take-profit (2R/3R/5R + breakeven trail) |
| `coin_cooldown.py` | 239 | NexusQuant | Per-symbol loss tracking and revenge-trade prevention |
| `signal_processor.py` | 296 | TradingAgents-CN + v0.3.1 | Structured JSON extraction from LLM free-text |
| `cpcv.py` | 229 | Orallexa + López de Prado | Combinatorial Purged Cross-Validation |
| `__init__.py` | 71 | — | Package init with all exports |
| **Total** | **3,114** | | |

## Installation

```bash
pip install smartmoneyconcepts pandas numpy
```

## Quick Start

```python
from trading_modules import (
    SMCDetector,
    ConfluenceGate,
    ConfluenceInput,
    RatingSystem,
    KillConditions,
    PortfolioState,
    BiasTracker,
    kelly_position_size,
    RMultipleTP,
    Position,
    CoinCooldownManager,
    SignalProcessor,
    CPCV,
)
```

## Complete Trade Flow Example

```python
# === Step 1: SMC Analysis ===
detector = SMCDetector()
smc_result = detector.analyze(ohlcv_df, symbol="BTCUSDT")
# Detects: BOS, CHoCH, Order Blocks, FVG, Liquidity zones

# === Step 2: Confluence Gate ===
gate = ConfluenceGate()
confluence_input = ConfluenceInput(
    symbol="BTCUSDT",
    direction="BUY",
    mtf_trend={"H4": "bullish", "H1": "bullish"},
    at_key_zone=True,  # At demand zone from SMC
    zone_type="demand",
    liquidity_sweep=True,
    sweep_direction="downward",
    pattern="Bullish Engulfing",
    pattern_rating=5,
    volume_ratio=2.1,
    rsi=28.0,
    structure_break="BOS",  # From SMC detector
    candle_closed=True,
)
confluence_result = gate.check(confluence_input)

if confluence_result.signal != "EXECUTE":
    print(f"Waiting: {confluence_result.recommendation}")
    return

# === Step 3: Kill Conditions Check ===
kc = KillConditions()
portfolio_state = PortfolioState(
    cumulative_loss_usd=200,
    rolling_sharpe_14d=1.5,
    current_drawdown_pct=5.0,
    rolling_brier_30d=0.15,
    paper_trade_days=60,
)
kill_decision = kc.check(portfolio_state)
if not kill_decision.can_trade:
    print(f"KILL: {kill_decision.trigger_reason}")
    return

# === Step 4: Coin Cooldown Check ===
ccm = CoinCooldownManager()
if ccm.is_in_cooldown("BTCUSDT"):
    print("BTCUSDT in cooldown — skipping")
    return

# === Step 5: Position Sizing (Kelly) ===
kelly_result = kelly_position_size(
    p_win=0.65,
    avg_win_pct=0.05,
    avg_loss_pct=0.03,
    account_equity=10000,
    current_drawdown_pct=5.0,
)
position_usd = kelly_result.position_usd

# === Step 6: R-Multiple Take-Profit Plan ===
tp = RMultipleTP()
entry_price = 65000
stop_loss = 63500
position = Position(
    symbol="BTCUSDT",
    direction="long",
    entry_price=entry_price,
    stop_loss=stop_loss,
    position_size=position_usd / entry_price,
)
tp_plan = tp.create_tp_plan(position)
# 2R → close 40%, stop to breakeven
# 3R → close 30%, stop to 1R
# 5R → close 30%, stop to 3R

# === Step 7: Signal Processing (from LLM output) ===
sp = SignalProcessor()
llm_output = "Based on analysis... Rating: Overweight..."
decision = sp.process(llm_output, symbol="BTCUSDT")

# === Step 8: Bias-Aware Confidence Adjustment ===
bt = BiasTracker()
adjusted_confidence = bt.adjust_confidence(decision.confidence, direction="BUY")

# === Step 9: 5-Tier Rating ===
rs = RatingSystem()
rating = rs.parse(llm_output)
print(f"Final Rating: {rs.format_rating_label(rating, adjusted_confidence)}")
```

## Module Details

### 1. SMC Detector (`smc_detector.py`)

```python
detector = SMCDetector(swing_length=10)
result = detector.analyze(df, symbol="BTCUSDT")

result.current_trend     # "bullish" / "bearish" / "ranging"
result.bullish_bos       # True if Bullish BOS detected
result.bearish_choch     # True if Bearish CHoCH detected
result.order_blocks      # DataFrame of OB zones
result.fvg               # DataFrame of Fair Value Gaps
result.liquidity         # DataFrame of liquidity zones
result.nearest_demand_zone  # Closest demand below current price
result.nearest_supply_zone  # Closest supply above current price

# Generate context block for LLM prompts
context = detector.get_confluence_context(result, current_price)
```

### 2. Confluence Gate (`confluence_gate.py`)

```python
# Strict mode (ALL checks must pass)
gate = ConfluenceGate(require_all=True)

# Weighted mode (Orallexa-style adaptive)
gate = WeightedConfluenceGate(min_score=0.75)

result = gate.check(ConfluenceInput(
    symbol="BTCUSDT", direction="BUY",
    mtf_trend={"H4": "bullish", "H1": "bullish"},
    at_key_zone=True, zone_type="demand",
    liquidity_sweep=True, sweep_direction="downward",
    pattern="Bullish Engulfing", pattern_rating=5,
    volume_ratio=2.1, rsi=28.0,
    structure_break="BOS", candle_closed=True,
))

result.signal    # "EXECUTE" or "WAIT"
result.score     # 0.0 to 1.0
result.checks    # {check_name: bool}
```

### 3. Rating System (`rating_system.py`)

```python
rs = RatingSystem()

# Parse from LLM text
rating = rs.parse("...Rating: Overweight...")  # "Overweight"

# Full result with confidence
result = rs.parse_full("...Rating: Buy...")
result.rating       # "Buy"
result.confidence   # 0.90
result.direction    # "long"
result.is_actionable  # True (not Hold)

# Star → Rating mapping
rating = rs.from_star_rating(5, "bullish")  # "Buy"
```

### 4. Kill Conditions (`kill_conditions.py`)

```python
kc = KillConditions()
state = PortfolioState(
    cumulative_loss_usd=450,
    rolling_sharpe_14d=1.2,
    current_drawdown_pct=8.0,
    rolling_brier_30d=0.15,
    paper_trade_days=45,
)

decision = kc.check(state)
# decision.can_trade  → True/False
# decision.state      → "OK" / "WAIT" / "GATED"
# decision.trigger_reason → What failed

# Real-money gate (stricter)
ready = kc.is_ready_for_real_money(state)
```

### 5. Bias Tracker (`bias_tracker.py`)

```python
bt = BiasTracker()

# Record decisions
bt.record_decision("BTCUSDT", "BUY", 0.75, rsi=28, pattern="Bullish Engulfing")

# Record outcomes (5 days later)
bt.record_outcome("BTCUSDT", forward_return=0.032)

# Get bias profile
profile = bt.get_bias_profile()
# profile.overall_accuracy
# profile.overconfidence_pct
# profile.confidence_adjustment_factor

# Adjust confidence for new prediction
adjusted = bt.adjust_confidence(0.75, direction="BUY")

# Generate context for LLM prompt
bias_context = bt.get_bias_context_for_prompt()
```

### 6. Kelly Sizing (`kelly_sizing.py`)

```python
result = kelly_position_size(
    p_win=0.60,
    avg_win_pct=0.05,  # 5% avg win
    avg_loss_pct=0.03,  # 3% avg loss
    account_equity=10000,
    current_drawdown_pct=8.0,
    kelly_fraction=0.5,  # Half-Kelly
)

result.full_kelly      # 0.36 (full Kelly fraction)
result.adjusted_kelly  # 0.084 (after Half-Kelly + DD adjustment + cap)
result.position_usd    # $840
```

### 7. R-Multiple TP (`r_multiple_tp.py`)

```python
tp = RMultipleTP()
pos = Position(
    symbol="BTCUSDT", direction="long",
    entry_price=65000, stop_loss=63500,
    position_size=1.0,
)
plan = tp.create_tp_plan(pos)

# Check levels on price update
actions = tp.check_levels(pos, plan, current_price=68000)
for action in actions:
    if action["type"] == "PARTIAL_CLOSE":
        print(f"Close {action['close_pct']:.0%} at {action['r_multiple']}R")
        print(f"Move stop to {action['new_stop_label']}")
```

### 8. Coin Cooldown (`coin_cooldown.py`)

```python
ccm = CoinCooldownManager()

# Record trades
ccm.record_trade("BTCUSDT", pnl_usd=-100, result="loss")

# Check cooldown
if ccm.is_in_cooldown("BTCUSDT"):
    print("In cooldown — skip")

# Get loss penalty for signal scoring
penalty = ccm.get_loss_penalty("BTCUSDT")
adjusted_signal = raw_signal * (1.0 - penalty)
```

### 9. Signal Processor (`signal_processor.py`)

```python
sp = SignalProcessor()
decision = sp.process(llm_output_text, symbol="BTCUSDT")

decision.action         # "BUY" / "SELL" / "HOLD"
decision.target_price   # 72000.0
decision.confidence     # 0.75
decision.risk_score     # 0.30
decision.edge_thesis    # "funding rate divergence..."
decision.rating         # "Overweight"
```

### 10. CPCV (`cpcv.py`)

```python
cpcv = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)

for train_idx, test_idx in cpcv.split(df, label_horizon_days=5):
    model.train(df.iloc[train_idx])
    model.evaluate(df.iloc[test_idx])

# Get summary without running
summary = cpcv.get_split_summary(df)
# summary["total_folds"]  → 15
```

## File Locations

```
/home/z/my-project/download/trading_modules/
├── __init__.py              # Package exports
├── smc_detector.py          # SMC detection
├── confluence_gate.py       # Entry confirmation
├── rating_system.py         # 5-tier rating
├── kill_conditions.py       # Risk gates
├── bias_tracker.py          # Self-correction
├── kelly_sizing.py          # Position sizing
├── r_multiple_tp.py         # Partial take-profit
├── coin_cooldown.py         # Revenge-trade prevention
├── signal_processor.py      # LLM output parsing
└── cpcv.py                  # Cross-validation
```

## Sources

| Module | Source Repo | Review # |
|--------|------------|----------|
| smc_detector | Vibe-Trading `smc` skill | #23 |
| confluence_gate | User's candlestick guide | — |
| rating_system | TradingAgents v0.3.1 | #30 |
| kill_conditions | Orallexa | #27 |
| bias_tracker | Orallexa | #27 |
| kelly_sizing | Orallexa | #27 |
| r_multiple_tp | NexusQuant | #29 |
| coin_cooldown | NexusQuant | #29 |
| signal_processor | TradingAgents-CN + v0.3.1 | #24, #30 |
| cpcv | Orallexa + López de Prado | #27 |
