"""
Crypto Intelligence — On-Chain, Pump/Dump, Rug Pull Detection
================================================================

Crypto-specific analysis modules:
  1. On-chain whale tracking — large wallet movements
  2. Pump & Dump detection — abnormal price + volume patterns
  3. Rug Pull detection — liquidity withdrawal patterns
  4. Stablecoin flow — USDT/USDC mint/burn signals
  5. MEV/arbitrage detection — sandwich attacks, front-running

Usage:
    from trading_modules.crypto_intelligence import (
        PumpDumpDetector, RugPullDetector, WhaleTracker,
        StablecoinFlowAnalyzer
    )

    # Pump & Dump
    detector = PumpDumpDetector()
    result = detector.analyze(df)  # OHLCV
    # → {"is_pump": True, "confidence": 0.85, "stage": "distribution"}

    # Whale tracking
    tracker = WhaleTracker()
    tracker.record_transfer("0xabc...", 500, "BTC", "exchange")
    alert = tracker.check_whale_activity()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════
# Pump & Dump Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class PumpDumpResult:
    """Result of pump & dump analysis."""
    is_pump: bool = False
    is_dump: bool = False
    stage: str = "normal"  # normal / accumulation / pump / distribution / dump
    confidence: float = 0.0
    price_change_pct: float = 0.0
    volume_spike: float = 0.0
    duration_bars: int = 0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_pump": self.is_pump,
            "is_dump": self.is_dump,
            "stage": self.stage,
            "confidence": round(self.confidence, 4),
            "price_change_pct": round(self.price_change_pct, 2),
            "volume_spike": round(self.volume_spike, 2),
            "warnings": self.warnings,
        }


class PumpDumpDetector:
    """
    Detects pump & dump patterns in OHLCV data.

    Pump signals:
      - Price > +20% in < 1 hour (or 6 bars on M15)
      - Volume > 5x average
      - Followed by sharp reversal

    Stages:
      1. Accumulation: quiet buying before pump
      2. Pump: rapid price increase + volume spike
      3. Distribution: organizers sell into retail FOMO
      4. Dump: price crashes as supply overwhelms
    """

    PUMP_THRESHOLD = 0.15       # 15% in short window = pump
    VOLUME_SPIKE = 5.0          # 5x average volume
    DUMP_THRESHOLD = -0.20      # -20% after pump = dump
    WINDOW_BARS = 6             # How many bars to check for pump

    def analyze(self, df: pd.DataFrame) -> PumpDumpResult:
        """Analyze OHLCV for pump & dump patterns."""
        if len(df) < 50:
            return PumpDumpResult()

        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Recent price change
        recent = close.iloc[-self.WINDOW_BARS:]
        price_change = (recent.iloc[-1] / recent.iloc[0] - 1)

        # Volume spike
        avg_vol = volume.rolling(50).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        vol_spike = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Longer-term reversal check
        lookback = min(24, len(df) - 1)
        longer_change = (close.iloc[-1] / close.iloc[-lookback] - 1)

        result = PumpDumpResult(
            price_change_pct=float(price_change * 100),
            volume_spike=float(vol_spike),
        )

        # Pump detection
        if price_change > self.PUMP_THRESHOLD and vol_spike > self.VOLUME_SPIKE:
            result.is_pump = True
            result.stage = "pump"
            result.confidence = min(1.0, (price_change / self.PUMP_THRESHOLD) * (vol_spike / self.VOLUME_SPIKE) * 0.5)

            # Check if distribution starting (volume still high but price stalling)
            if abs(recent.iloc[-1] / recent.iloc[-3] - 1) < 0.02 and vol_spike > 3:
                result.stage = "distribution"
                result.warnings.append("Price stalling after pump — distribution phase likely")

        # Dump detection (after a pump)
        if longer_change < self.DUMP_THRESHOLD:
            result.is_dump = True
            result.stage = "dump"
            result.confidence = min(1.0, abs(longer_change / self.DUMP_THRESHOLD) * 0.7)
            result.warnings.append("Significant price drop detected — possible post-pump dump")

        # Accumulation detection (quiet volume increase before move)
        if not result.is_pump and not result.is_dump:
            vol_trend = volume.iloc[-10:].mean() / volume.iloc[-30:].mean()
            price_trend = close.iloc[-1] / close.iloc[-30] - 1
            if vol_trend > 1.5 and abs(price_trend) < 0.05:
                result.stage = "accumulation"
                result.warnings.append("Volume increasing with flat price — possible accumulation")

        return result


# ═══════════════════════════════════════════════════════════════
# Rug Pull Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class RugPullResult:
    """Result of rug pull analysis."""
    is_rug_pull: bool = False
    risk_level: str = "low"  # low / medium / high / critical
    liquidity_removed_pct: float = 0.0
    price_drop_pct: float = 0.0
    confidence: float = 0.0
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_rug_pull": self.is_rug_pull,
            "risk_level": self.risk_level,
            "liquidity_removed_pct": round(self.liquidity_removed_pct, 2),
            "price_drop_pct": round(self.price_drop_pct, 2),
            "confidence": round(self.confidence, 4),
            "warnings": self.warnings,
        }


class RugPullDetector:
    """
    Detects rug pull patterns — where developers remove liquidity
    causing price to crash to zero.

    Signals:
      - Sudden massive price drop (>50% in minutes)
      - Liquidity (volume) dries up completely
      - No recovery — price stays near zero
      - Often preceded by gradual price increase (luring buyers)
    """

    PRICE_CRASH_THRESHOLD = -0.50    # -50% = likely rug
    VOLUME_DRY_UP = 0.1              # Volume drops to 10% of average
    NO_RECOVERY_BARS = 12            # Price doesn't recover in 12 bars

    def analyze(self, df: pd.DataFrame) -> RugPullResult:
        """Analyze for rug pull patterns."""
        if len(df) < 50:
            return RugPullResult()

        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series(1, index=df.index)

        # Check for sudden crash
        recent_bars = min(6, len(df) - 1)
        price_drop = close.iloc[-1] / close.iloc[-recent_bars] - 1

        # Volume dry-up
        avg_vol = volume.rolling(50).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Check no recovery
        crash_low = close.iloc[-recent_bars:].min()
        current = close.iloc[-1]
        recovery = current / crash_low - 1 if crash_low > 0 else 0

        result = RugPullResult(
            price_drop_pct=float(price_drop * 100),
            liquidity_removed_pct=float((1 - vol_ratio) * 100) if vol_ratio < 1 else 0,
        )

        # Risk scoring
        risk_score = 0.0

        if price_drop < self.PRICE_CRASH_THRESHOLD:
            risk_score += 0.4
            result.warnings.append(f"Price crashed {price_drop*100:.1f}% — rug pull signal")

        if vol_ratio < self.VOLUME_DRY_UP:
            risk_score += 0.3
            result.warnings.append("Volume dried up — liquidity likely removed")

        if recovery < 0.05 and price_drop < -0.30:
            risk_score += 0.2
            result.warnings.append("No recovery after crash — confirmed rug pull")

        # Check for pre-crash pump (luring pattern)
        pre_crash = close.iloc[-recent_bars-20:-recent_bars]
        if len(pre_crash) > 0:
            pre_change = pre_crash.iloc[-1] / pre_crash.iloc[0] - 1
            if pre_change > 0.50:  # 50%+ before crash
                risk_score += 0.1
                result.warnings.append("Pre-crash pump detected — classic rug pull pattern")

        result.confidence = min(1.0, risk_score)
        result.is_rug_pull = risk_score >= 0.5

        if risk_score >= 0.7:
            result.risk_level = "critical"
        elif risk_score >= 0.5:
            result.risk_level = "high"
        elif risk_score >= 0.3:
            result.risk_level = "medium"
        else:
            result.risk_level = "low"

        return result


# ═══════════════════════════════════════════════════════════════
# Whale Tracking
# ═══════════════════════════════════════════════════════════════

@dataclass
class WhaleTransfer:
    """A large transfer event."""
    wallet: str
    amount: float
    asset: str
    destination: str  # "exchange" / "cold_wallet" / "defi" / "unknown"
    timestamp: str = ""
    usd_value: float = 0.0


class WhaleTracker:
    """
    Tracks large wallet transfers (whale movements).

    Alert when:
      - Large transfer TO exchange (potential sell pressure)
      - Large transfer FROM exchange (potential accumulation)
      - Multiple whales moving simultaneously (coordinated)
    """

    WHALE_THRESHOLD_USD = 1_000_000  # $1M+ = whale
    EXCHANGE_WALLETS = {"binance", "coinbase", "kraken", "okx", "bybit"}

    def __init__(self):
        self.transfers: deque = deque(maxlen=1000)
        self.wallet_registry: dict[str, str] = {}  # address → label

    def register_wallet(self, address: str, label: str) -> None:
        """Register a known wallet (exchange, whale, etc.)."""
        self.wallet_registry[address.lower()] = label

    def record_transfer(
        self,
        wallet: str,
        amount: float,
        asset: str,
        destination: str,
        price: float = 0.0,
    ) -> Optional[WhaleTransfer]:
        """Record a transfer. Returns WhaleTransfer if whale-sized."""
        usd_value = amount * price if price > 0 else amount * 50000  # Fallback estimate

        if usd_value < self.WHALE_THRESHOLD_USD:
            return None

        transfer = WhaleTransfer(
            wallet=wallet,
            amount=amount,
            asset=asset,
            destination=destination,
            timestamp=datetime.now(timezone.utc).isoformat(),
            usd_value=usd_value,
        )

        self.transfers.append(transfer)
        return transfer

    def check_whale_activity(self, lookback_minutes: int = 60) -> dict:
        """Check recent whale activity for signals."""
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_minutes * 60

        recent = [
            t for t in self.transfers
            if datetime.fromisoformat(t.timestamp).timestamp() > cutoff
        ]

        # Categorize
        to_exchange = [t for t in recent if t.destination in self.EXCHANGE_WALLETS]
        from_exchange = [t for t in recent if t.wallet in self.EXCHANGE_WALLETS]

        # Total flow
        inflow = sum(t.usd_value for t in to_exchange)
        outflow = sum(t.usd_value for t in from_exchange)
        net_flow = inflow - outflow

        signals = []
        if inflow > 10_000_000:
            signals.append(f"🚨 Large inflow to exchanges: ${inflow/1e6:.1f}M — potential sell pressure")
        if outflow > 10_000_000:
            signals.append(f"🟢 Large outflow from exchanges: ${outflow/1e6:.1f}M — potential accumulation")
        if len(recent) > 5:
            signals.append(f"⚠️ {len(recent)} whale transfers in {lookback_minutes}min — coordinated activity")

        return {
            "total_transfers": len(recent),
            "exchange_inflow_usd": round(inflow, 2),
            "exchange_outflow_usd": round(outflow, 2),
            "net_flow_usd": round(net_flow, 2),
            "signal": "bearish" if net_flow > 0 else "bullish" if net_flow < 0 else "neutral",
            "alerts": signals,
        }


# ═══════════════════════════════════════════════════════════════
# Stablecoin Flow Analyzer
# ═══════════════════════════════════════════════════════════════

class StablecoinFlowAnalyzer:
    """
    Analyzes stablecoin supply changes as market liquidity proxy.

    USDT minting → new capital entering crypto → bullish
    USDT burning → capital leaving crypto → bearish
    """

    def __init__(self):
        self.mint_events: deque = deque(maxlen=500)
        self.burn_events: deque = deque(maxlen=500)

    def record_mint(self, amount_usd: float, stablecoin: str = "USDT") -> None:
        """Record a stablecoin mint event."""
        self.mint_events.append({
            "amount": amount_usd,
            "stablecoin": stablecoin,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def record_burn(self, amount_usd: float, stablecoin: str = "USDT") -> None:
        """Record a stablecoin burn event."""
        self.burn_events.append({
            "amount": amount_usd,
            "stablecoin": stablecoin,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_flow_signal(self, lookback_hours: int = 24) -> dict:
        """Get stablecoin flow signal."""
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_hours * 3600

        recent_mints = [e for e in self.mint_events
                        if datetime.fromisoformat(e["timestamp"]).timestamp() > cutoff]
        recent_burns = [e for e in self.burn_events
                        if datetime.fromisoformat(e["timestamp"]).timestamp() > cutoff]

        total_mint = sum(e["amount"] for e in recent_mints)
        total_burn = sum(e["amount"] for e in recent_burns)
        net_flow = total_mint - total_burn

        if net_flow > 500_000_000:  # >$500M net minting
            signal = "strong_bullish"
        elif net_flow > 100_000_000:
            signal = "bullish"
        elif net_flow < -500_000_000:
            signal = "strong_bearish"
        elif net_flow < -100_000_000:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "net_flow_usd": round(net_flow, 2),
            "total_minted": round(total_mint, 2),
            "total_burned": round(total_burn, 2),
            "n_mint_events": len(recent_mints),
            "n_burn_events": len(recent_burns),
            "signal": signal,
            "lookback_hours": lookback_hours,
        }
