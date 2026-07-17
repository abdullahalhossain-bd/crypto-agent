"""
Signal Processing — FFT, wavelet, EMD, noise filtering
========================================================

Pure-numpy implementations of signal processing techniques for:
    1. FFT decomposition        — extract dominant cycles
    2. Simple wavelet transform — Haar wavelet multi-resolution analysis
    3. Empirical Mode Decomposition (EMD) — decompose into IMFs
    4. Noise filtering          — low-pass, high-pass, band-pass
    5. Trend-cycle decomposition — separate trend from cycle

These are useful for:
    - Cycle detection (e.g., 20-bar, 50-bar cycles)
    - Denoising price series
    - Identifying dominant frequencies

Usage:
    from trading_modules.signal_processing import (
        fft_decompose, haar_wavelet, emd_decompose, lowpass_filter
    )
    components = fft_decompose(df["close"], n_components=3)
    trend, cycle, noise = haar_wavelet(df["close"], level=3)
    imfs = emd_decompose(df["close"].to_numpy())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 1. FFT Decomposition
# ──────────────────────────────────────────────────────────────────────
@dataclass
class FFTResult:
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    dominant_freqs: list[float]
    dominant_periods: list[float]
    reconstructed: np.ndarray
    n_components_kept: int

    def to_dict(self) -> dict:
        return {
            "n_frequencies": len(self.frequencies),
            "dominant_freqs": [round(f, 6) for f in self.dominant_freqs],
            "dominant_periods": [round(p, 2) for p in self.dominant_periods],
            "n_components_kept": self.n_components_kept,
        }


def fft_decompose(
    series: np.ndarray, n_components: int = 5,
) -> FFTResult:
    """Decompose a time series using FFT and reconstruct from top components.

    Args:
        series: 1-D array
        n_components: # of dominant frequencies to keep

    Returns:
        FFTResult with reconstructed series using only top components
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 8:
        return FFTResult(
            frequencies=np.zeros(0), amplitudes=np.zeros(0), phases=np.zeros(0),
            dominant_freqs=[], dominant_periods=[],
            reconstructed=x.copy(), n_components_kept=0,
        )
    # Detrend (remove mean)
    x_centered = x - x.mean()
    # FFT
    fft_vals = np.fft.fft(x_centered)
    frequencies = np.fft.fftfreq(n)
    amplitudes = np.abs(fft_vals) / n
    phases = np.angle(fft_vals)
    # Find top n_components by amplitude (excluding DC which is 0)
    # Use only positive frequencies (first half)
    n_half = n // 2
    pos_amps = amplitudes[:n_half]
    pos_freqs = frequencies[:n_half]
    # Sort by amplitude
    top_indices = np.argsort(pos_amps)[::-1][:n_components]
    dominant_freqs = [float(pos_freqs[i]) for i in top_indices if pos_freqs[i] > 0]
    dominant_periods = [float(1 / f) if f > 0 else 0.0 for f in dominant_freqs]
    # Reconstruct using only top components
    fft_filtered = np.zeros_like(fft_vals)
    for idx in top_indices:
        fft_filtered[idx] = fft_vals[idx]
        # Mirror to negative frequency
        if idx > 0:
            fft_filtered[n - idx] = fft_vals[n - idx]
    reconstructed = np.real(np.fft.ifft(fft_filtered)) + x.mean()
    return FFTResult(
        frequencies=frequencies, amplitudes=amplitudes, phases=phases,
        dominant_freqs=dominant_freqs, dominant_periods=dominant_periods,
        reconstructed=reconstructed, n_components_kept=n_components,
    )


# ──────────────────────────────────────────────────────────────────────
# 2. Haar Wavelet Transform (multi-resolution analysis)
# ──────────────────────────────────────────────────────────────────────
def haar_wavelet(
    series: np.ndarray, level: int = 3,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Haar wavelet multi-resolution decomposition.

    Returns:
        (approximation, [details_level_1, details_level_2, ...])
        The approximation is the smooth trend; details are finer scales.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    # Pad to power of 2
    target_len = 2 ** int(np.ceil(np.log2(max(n, 2 ** level))))
    if target_len > n:
        x_padded = np.concatenate([x, np.full(target_len - n, x[-1])])
    else:
        x_padded = x[:target_len]
    details: list[np.ndarray] = []
    current = x_padded.copy()
    for _ in range(level):
        n_curr = len(current)
        if n_curr < 2:
            break
        # Haar transform: averages and differences
        even = current[0::2]
        odd = current[1::2]
        # Pad if odd length
        if len(odd) < len(even):
            odd = np.append(odd, even[-1])
        approx = (even + odd) / 2.0
        detail = (even - odd) / 2.0
        details.append(detail)
        current = approx
    # Truncate back to original length
    current = current[:n] if len(current) > n else current
    details = [d[:n] if len(d) > n else d for d in details]
    return current, details


def haar_reconstruct(
    approximation: np.ndarray, details: list[np.ndarray],
) -> np.ndarray:
    """Reconstruct the original series from Haar wavelet components."""
    current = approximation.copy()
    for detail in reversed(details):
        # Upsample approximation and add detail
        n = len(current)
        # Interpolate to 2x length
        upsampled = np.zeros(2 * n)
        upsampled[0::2] = current
        upsampled[1::2] = current  # simple repetition
        if len(detail) >= 2 * n:
            upsampled = upsampled + detail[:2 * n]
        else:
            # Pad detail
            padded_detail = np.concatenate([detail, np.zeros(2 * n - len(detail))])
            upsampled = upsampled + padded_detail
        current = upsampled
    return current


# ──────────────────────────────────────────────────────────────────────
# 3. Empirical Mode Decomposition (EMD)
# ──────────────────────────────────────────────────────────────────────
def emd_decompose(
    series: np.ndarray, max_imfs: int = 10, max_iter: int = 100,
) -> list[np.ndarray]:
    """Empirical Mode Decomposition — decompose into Intrinsic Mode Functions.

    EMD adaptively decomposes any signal into a set of IMFs (each representing
    a simple oscillatory mode) plus a residual trend.

    Args:
        series: 1-D array
        max_imfs: maximum # of IMFs to extract
        max_iter: max sifting iterations per IMF

    Returns:
        List of IMFs (last element is the residual)
    """
    x = np.asarray(series, dtype=float)
    imfs: list[np.ndarray] = []
    residual = x.copy()
    for _ in range(max_imfs):
        h = residual.copy()
        for _ in range(max_iter):
            # Find local maxima and minima
            maxima_idx = []
            minima_idx = []
            for i in range(1, len(h) - 1):
                if h[i] > h[i - 1] and h[i] > h[i + 1]:
                    maxima_idx.append(i)
                if h[i] < h[i - 1] and h[i] < h[i + 1]:
                    minima_idx.append(i)
            # Need at least 2 extrema to interpolate
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            # Upper envelope (cubic spline through maxima)
            try:
                from scipy.interpolate import CubicSpline
                # Extend boundaries
                max_x = [0] + maxima_idx + [len(h) - 1]
                max_y = [h[0]] + [h[i] for i in maxima_idx] + [h[-1]]
                min_x = [0] + minima_idx + [len(h) - 1]
                min_y = [h[0]] + [h[i] for i in minima_idx] + [h[-1]]
                cs_upper = CubicSpline(max_x, max_y)
                cs_lower = CubicSpline(min_x, min_y)
                upper = cs_upper(np.arange(len(h)))
                lower = cs_lower(np.arange(len(h)))
            except ImportError:
                # Fallback to linear interpolation
                max_x = np.array([0] + maxima_idx + [len(h) - 1])
                max_y = np.array([h[0]] + [h[i] for i in maxima_idx] + [h[-1]])
                min_x = np.array([0] + minima_idx + [len(h) - 1])
                min_y = np.array([h[0]] + [h[i] for i in minima_idx] + [h[-1]])
                upper = np.interp(np.arange(len(h)), max_x, max_y)
                lower = np.interp(np.arange(len(h)), min_x, min_y)
            mean = (upper + lower) / 2.0
            h_new = h - mean
            # Stop criterion: standard deviation small
            sd = np.sum((h - h_new) ** 2) / max(np.sum(h ** 2), 1e-10)
            h = h_new
            if sd < 0.05:
                break
        # Check if h is an IMF (at least 2 extrema, zero crossings roughly equal)
        if len(maxima_idx) < 2 and len(minima_idx) < 2:
            break
        imfs.append(h)
        residual = residual - h
        # Stop if residual is monotonic
        if len(np.diff(np.sign(np.diff(residual)))) == 0:
            break
    imfs.append(residual)  # residual is the last element
    return imfs


# ──────────────────────────────────────────────────────────────────────
# 4. Noise Filtering
# ──────────────────────────────────────────────────────────────────────
def lowpass_filter(
    series: np.ndarray, cutoff_pct: float = 0.1,
) -> np.ndarray:
    """Low-pass filter using FFT — keep only low-frequency components.

    Args:
        series: 1-D array
        cutoff_pct: fraction of frequencies to keep (0.1 = keep lowest 10%)
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 8:
        return x
    fft_vals = np.fft.fft(x)
    n_keep = max(1, int(n * cutoff_pct))
    # Zero out high frequencies
    fft_filtered = np.zeros_like(fft_vals)
    fft_filtered[:n_keep] = fft_vals[:n_keep]
    fft_filtered[-n_keep:] = fft_vals[-n_keep:]  # mirror for real signal
    return np.real(np.fft.ifft(fft_filtered))


def highpass_filter(
    series: np.ndarray, cutoff_pct: float = 0.1,
) -> np.ndarray:
    """High-pass filter — keep only high-frequency components."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 8:
        return x - x.mean()
    fft_vals = np.fft.fft(x)
    n_remove = max(1, int(n * cutoff_pct))
    fft_filtered = fft_vals.copy()
    fft_filtered[:n_remove] = 0
    fft_filtered[-n_remove:] = 0
    return np.real(np.fft.ifft(fft_filtered))


def bandpass_filter(
    series: np.ndarray, low_pct: float = 0.1, high_pct: float = 0.3,
) -> np.ndarray:
    """Band-pass filter — keep only middle-frequency components."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 8:
        return x
    fft_vals = np.fft.fft(x)
    n_low = max(1, int(n * low_pct))
    n_high = max(n_low + 1, int(n * high_pct))
    fft_filtered = np.zeros_like(fft_vals)
    fft_filtered[n_low:n_high] = fft_vals[n_low:n_high]
    fft_filtered[-n_high:-n_low] = fft_vals[-n_high:-n_low]
    return np.real(np.fft.ifft(fft_filtered))


# ──────────────────────────────────────────────────────────────────────
# 5. Trend-Cycle Decomposition (Hodrick-Prescott filter approximation)
# ──────────────────────────────────────────────────────────────────────
def hodrick_prescott(
    series: np.ndarray, lambda_: float = 1600,
) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter — decompose into trend and cyclical component.

    Args:
        series: 1-D array
        lambda_: smoothing parameter (1600 for quarterly, 100 for annual,
                 129600 for monthly, 14400 for daily)

    Returns:
        (trend, cycle)
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 4:
        return x, np.zeros_like(x)
    # HP filter solves: min sum((y_t - τ_t)^2 + λ * sum((τ_{t+1} - 2τ_t + τ_{t-1})^2))
    # Solution: τ = (I + λ * D'D)^{-1} y, where D is the second-difference operator
    # Build D matrix (n-2 × n)
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    A = np.eye(n) + lambda_ * D.T @ D
    try:
        trend = np.linalg.solve(A, x)
    except np.linalg.LinAlgError:
        trend = x
    cycle = x - trend
    return trend, cycle


__all__ = [
    "FFTResult", "fft_decompose",
    "haar_wavelet", "haar_reconstruct",
    "emd_decompose",
    "lowpass_filter", "highpass_filter", "bandpass_filter",
    "hodrick_prescott",
]
