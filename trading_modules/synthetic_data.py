"""
Synthetic Data Generation — time-series augmentation for training
===================================================================

Generate synthetic OHLCV data for:
    1. Strategy backtesting without overfitting to one historical path
    2. Stress-testing under different market regimes
    3. Data augmentation for ML training
    4. Bootstrapping confidence intervals

Methods:
    1. Block Bootstrap         — resample contiguous blocks (preserves autocorrelation)
    2. MBM (Model-Based)       — fit GBM + jump-diffusion, simulate
    3. GAN-style (simplified)  — train + sample (without deep learning)
    4. Time-series augmentation — jitter, scale, time-warp, mixup

Usage:
    from trading_modules.synthetic_data import (
        block_bootstrap, simulate_gbm, augment_jitter, augment_mixup
    )
    # Generate 100 alternative paths
    for _ in range(100):
        synthetic = block_bootstrap(df, block_size=20, n_samples=500)
        backtest(strategy, synthetic)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. Block Bootstrap
# ──────────────────────────────────────────────────────────────────────
def block_bootstrap(
    df: pd.DataFrame, block_size: int = 20, n_samples: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Block bootstrap — resample contiguous blocks to preserve autocorrelation.

    Args:
        df: OHLCV dataframe
        block_size: # of contiguous bars per block
        n_samples: total # of bars in synthetic series
        seed: RNG seed

    Returns:
        Synthetic OHLCV dataframe of length n_samples
    """
    if df is None or df.empty:
        return df
    rng = np.random.default_rng(seed)
    n = len(df)
    n_blocks = (n_samples + block_size - 1) // block_size
    # Sample starting indices for each block
    start_indices = rng.integers(0, n - block_size + 1, n_blocks)
    # Concatenate blocks
    block_arrays = [np.arange(start, start + block_size) for start in start_indices]
    indices = np.concatenate(block_arrays)[:n_samples]
    synthetic = df.iloc[indices].reset_index(drop=True)
    # Adjust timestamps to be sequential
    if "time" in synthetic.columns:
        base_time = pd.Timestamp.utcnow()
        synthetic["time"] = pd.date_range(
            end=base_time, periods=n_samples, freq="15min", tz="UTC",
        )
    return synthetic


# ──────────────────────────────────────────────────────────────────────
# 2. Geometric Brownian Motion (GBM)
# ──────────────────────────────────────────────────────────────────────
def simulate_gbm(
    initial_price: float, mu: float, sigma: float,
    n_steps: int = 500, dt: float = 1.0 / 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate Geometric Brownian Motion.

    dS = μ S dt + σ S dW

    Args:
        initial_price: starting price
        mu: annualized drift
        sigma: annualized volatility
        n_steps: # of bars to simulate
        dt: time step (1/252 = daily, 1/(252*6.5) = hourly for 6.5h trading day)
        seed: RNG seed

    Returns:
        OHLCV dataframe with simulated prices
    """
    rng = np.random.default_rng(seed)
    # Generate returns: r_t = (μ - σ²/2) dt + σ √dt Z
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    returns = drift + diffusion * rng.standard_normal(n_steps)
    # Cumulative price
    prices = initial_price * np.exp(np.cumsum(returns))
    # Build OHLCV
    # For each bar, generate OHLC around the close
    opens = np.roll(prices, 1)
    opens[0] = initial_price
    # High/low: random spread around min/max of open and close
    spread = np.abs(prices - opens) + np.abs(rng.normal(0, sigma * prices * np.sqrt(dt)))
    highs = np.maximum(opens, prices) + spread * rng.uniform(0.1, 0.5, n_steps)
    lows = np.minimum(opens, prices) - spread * rng.uniform(0.1, 0.5, n_steps)
    volumes = rng.uniform(100, 1000, n_steps)
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_steps, freq="15min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs,
        "low": lows, "close": prices, "volume": volumes,
    })


# ──────────────────────────────────────────────────────────────────────
# 3. Jump Diffusion (Merton model)
# ──────────────────────────────────────────────────────────────────────
def simulate_jump_diffusion(
    initial_price: float, mu: float, sigma: float,
    jump_lambda: float = 5.0, jump_mu: float = -0.05, jump_sigma: float = 0.05,
    n_steps: int = 500, dt: float = 1.0 / 252,
    seed: int = 42,
) -> pd.DataFrame:
    """Merton jump-diffusion model.

    dS = μ S dt + σ S dW + J

    Where J is a compound Poisson process modeling sudden jumps (e.g., flash crashes).

    Args:
        initial_price: starting price
        mu: drift
        sigma: diffusive volatility
        jump_lambda: jump intensity (avg jumps per year)
        jump_mu: mean jump size (negative = crash)
        jump_sigma: jump size volatility
    """
    rng = np.random.default_rng(seed)
    # Diffusive component
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    diff_returns = drift + diffusion * rng.standard_normal(n_steps)
    # Jump component
    jump_returns = np.zeros(n_steps)
    for t in range(n_steps):
        # Poisson: probability of jump in this step
        if rng.random() < jump_lambda * dt:
            # Jump size: log-normal
            jump_size = rng.normal(jump_mu, jump_sigma)
            jump_returns[t] = jump_size
    # Combined returns
    returns = diff_returns + jump_returns
    prices = initial_price * np.exp(np.cumsum(returns))
    # Build OHLCV
    opens = np.roll(prices, 1); opens[0] = initial_price
    spread = np.abs(prices - opens) + np.abs(rng.normal(0, sigma * prices * np.sqrt(dt)))
    highs = np.maximum(opens, prices) + spread * rng.uniform(0.1, 0.5, n_steps)
    lows = np.minimum(opens, prices) - spread * rng.uniform(0.1, 0.5, n_steps)
    volumes = rng.uniform(100, 1000, n_steps) + np.abs(jump_returns) * 10000  # higher vol on jumps
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_steps, freq="15min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs,
        "low": lows, "close": prices, "volume": volumes,
    })


# ──────────────────────────────────────────────────────────────────────
# 4. Time-Series Augmentation
# ──────────────────────────────────────────────────────────────────────
def augment_jitter(
    df: pd.DataFrame, noise_std: float = 0.001, seed: int = 42,
) -> pd.DataFrame:
    """Add Gaussian noise to prices (data augmentation for ML).

    Args:
        df: OHLCV dataframe
        noise_std: noise standard deviation (as fraction of price)
        seed: RNG seed
    """
    rng = np.random.default_rng(seed)
    result = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in result.columns:
            noise = rng.normal(0, noise_std * result[col].values, len(result))
            result[col] = result[col] * (1 + noise)
    return result


def augment_scaling(
    df: pd.DataFrame, scale_range: tuple[float, float] = (0.95, 1.05),
    seed: int = 42,
) -> pd.DataFrame:
    """Scale all prices by a random factor (data augmentation).

    Args:
        df: OHLCV dataframe
        scale_range: (min_scale, max_scale)
    """
    rng = np.random.default_rng(seed)
    scale = rng.uniform(*scale_range)
    result = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in result.columns:
            result[col] = result[col] * scale
    return result


def augment_time_warp(
    df: pd.DataFrame, warp_std: float = 0.1, seed: int = 42,
) -> pd.DataFrame:
    """Time-warp augmentation — randomly stretch/compress time segments.

    Useful for making ML models invariant to timing variations.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    # Generate random time warp
    warp_factors = 1 + rng.normal(0, warp_std, n)
    warp_factors = np.clip(warp_factors, 0.5, 2.0)
    # Apply warping by resampling
    new_indices = np.cumsum(warp_factors).astype(int)
    new_indices = np.clip(new_indices, 0, n - 1)
    return df.iloc[new_indices].reset_index(drop=True)


def augment_mixup(
    df1: pd.DataFrame, df2: pd.DataFrame, alpha: float = 0.5,
) -> pd.DataFrame:
    """Mixup augmentation — linearly combine two series.

    mixed = α × df1 + (1 - α) × df2

    Args:
        df1, df2: two OHLCV dataframes (same length)
        alpha: mixing coefficient (0..1)
    """
    n = min(len(df1), len(df2))
    result = df1.iloc[:n].copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df1.columns and col in df2.columns:
            result[col] = alpha * df1[col].iloc[:n].values + (1 - alpha) * df2[col].iloc[:n].values
    return result


# ──────────────────────────────────────────────────────────────────────
# 5. Synthetic regime generator
# ──────────────────────────────────────────────────────────────────────
def generate_regime_switching(
    initial_price: float, n_steps: int = 1000,
    regimes: Optional[list[dict]] = None,
    dt: float = 1.0 / 252, seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic data with regime switches.

    Args:
        initial_price: starting price
        n_steps: total bars
        regimes: list of regime dicts with: {mu, sigma, n_bars}
                 (default: 3 regimes — trending, ranging, volatile)
    """
    if regimes is None:
        regimes = [
            {"mu": 0.15, "sigma": 0.2, "n_bars": 300},   # trending up
            {"mu": 0.0, "sigma": 0.1, "n_bars": 300},    # ranging
            {"mu": -0.10, "sigma": 0.4, "n_bars": 400},  # volatile down
        ]
    rng = np.random.default_rng(seed)
    all_returns: list[float] = []
    for regime in regimes:
        r = regime["n_bars"]
        drift = (regime["mu"] - 0.5 * regime["sigma"] ** 2) * dt
        diffusion = regime["sigma"] * np.sqrt(dt)
        rets = drift + diffusion * rng.standard_normal(r)
        all_returns.extend(rets.tolist())
    # Truncate or pad to n_steps
    if len(all_returns) > n_steps:
        all_returns = all_returns[:n_steps]
    elif len(all_returns) < n_steps:
        all_returns.extend([0.0] * (n_steps - len(all_returns)))
    prices = initial_price * np.exp(np.cumsum(np.array(all_returns)))
    # Build OHLCV
    opens = np.roll(prices, 1); opens[0] = initial_price
    spread = np.abs(prices - opens) + np.abs(rng.normal(0, 0.01 * prices))
    highs = np.maximum(opens, prices) + spread * 0.3
    lows = np.minimum(opens, prices) - spread * 0.3
    volumes = rng.uniform(100, 1000, n_steps)
    times = pd.date_range(end=pd.Timestamp.utcnow(), periods=n_steps, freq="15min", tz="UTC")
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs,
        "low": lows, "close": prices, "volume": volumes,
    })


__all__ = [
    "block_bootstrap", "simulate_gbm", "simulate_jump_diffusion",
    "augment_jitter", "augment_scaling", "augment_time_warp", "augment_mixup",
    "generate_regime_switching",
]
