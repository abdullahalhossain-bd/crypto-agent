"""
Information Theory — entropy, divergence, mutual information
=============================================================

Pure-Python implementations of information-theoretic measures used for:
    - Feature selection (mutual information between feature and target)
    - Regime uncertainty (Shannon entropy of regime probabilities)
    - Information flow (transfer entropy between assets)
    - Distribution comparison (KL divergence, Jensen-Shannon)

Functions:
    1. shannon_entropy(p)            — H(p) = -Σ p log p
    2. kl_divergence(p, q)           — D_KL(p || q)
    3. jensen_shannon_divergence(p, q) — symmetric KL
    4. mutual_information(x, y)      — I(X; Y)
    5. transfer_entropy(x, y, k=1)   — T_{X→Y} (info flow from X to Y)
    6. entropy_rate(series)          — entropy per time step

Usage:
    from trading_modules.information_theory import (
        shannon_entropy, kl_divergence, mutual_information, transfer_entropy
    )
    # Regime uncertainty
    H = shannon_entropy(regime_probabilities)
    # Feature selection
    mi = mutual_information(feature_values, returns)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def shannon_entropy(probabilities: np.ndarray, base: float = 2.0) -> float:
    """Shannon entropy: H(p) = -Σ p_i log(p_i).

    Args:
        probabilities: array of probabilities (will be normalized)
        base: log base (2 = bits, e = nats)

    Returns:
        Entropy in bits (or nats if base=e)
    """
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]  # remove zeros (0 log 0 = 0)
    if len(p) == 0:
        return 0.0
    p = p / p.sum()  # normalize
    if base == 2.0:
        return float(-np.sum(p * np.log2(p)))
    elif base == np.e:
        return float(-np.sum(p * np.log(p)))
    else:
        return float(-np.sum(p * np.log(p) / np.log(base)))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence: D_KL(p || q) = Σ p log(p/q).

    Asymmetric — D_KL(p||q) ≠ D_KL(q||p).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(p) != len(q):
        raise ValueError("p and q must have same length")
    # Normalize
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum <= 0 or q_sum <= 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    p_safe = p + eps
    q_safe = q + eps
    return float(np.sum(p * np.log(p_safe / q_safe)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence — symmetric version of KL.

    JSD(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)
    where m = 0.5 * (p + q)
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(p) != len(q):
        raise ValueError("p and q must have same length")
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def mutual_information(
    x: np.ndarray, y: np.ndarray, n_bins: int = 10,
) -> float:
    """Mutual information between two continuous variables.

    I(X; Y) = H(X) + H(Y) - H(X, Y)

    Discretizes both variables into n_bins and computes the joint histogram.

    Args:
        x, y: 1-D arrays of same length
        n_bins: # of bins for discretization

    Returns:
        MI in bits
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 10:
        return 0.0
    x = x[:n]; y = y[:n]
    # Remove NaN
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 10:
        return 0.0
    # Discretize
    x_bins = np.linspace(x.min(), x.max() + 1e-10, n_bins + 1)
    y_bins = np.linspace(y.min(), y.max() + 1e-10, n_bins + 1)
    x_disc = np.digitize(x, x_bins) - 1
    y_disc = np.digitize(y, y_bins) - 1
    x_disc = np.clip(x_disc, 0, n_bins - 1)
    y_disc = np.clip(y_disc, 0, n_bins - 1)
    # Joint histogram
    joint, _, _ = np.histogram2d(x_disc, y_disc, bins=n_bins)
    joint = joint / joint.sum() if joint.sum() > 0 else joint
    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    # MI = sum_{i,j} p(i,j) log [p(i,j) / (p(i) p(j))]
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return float(max(0.0, mi))


def transfer_entropy(
    x: np.ndarray, y: np.ndarray, k: int = 1, l: int = 1,
    n_bins: int = 5,
) -> float:
    """Transfer entropy from X to Y.

    T_{X→Y} = I(Y_t ; X_{t-k} | Y_{t-l})

    Measures how much the past of X helps predict Y beyond Y's own past.

    Args:
        x, y: 1-D arrays (time series)
        k: lag for X (default 1)
        l: lag for Y (default 1)
        n_bins: discretization bins

    Returns:
        Transfer entropy in bits. Positive = X influences Y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < max(k, l) + 20:
        return 0.0
    x = x[:n]; y = y[:n]
    # Build lagged arrays
    # We need: Y_t, Y_{t-l}, X_{t-k}
    min_idx = max(k, l)
    y_t = y[min_idx:]
    y_past = y[min_idx - l: n - l]
    x_past = x[min_idx - k: n - k]
    # Equalize lengths
    m = min(len(y_t), len(y_past), len(x_past))
    if m < 20:
        return 0.0
    y_t = y_t[:m]; y_past = y_past[:m]; x_past = x_past[:m]
    # Discretize
    def discretize(arr, bins):
        edges = np.linspace(arr.min(), arr.max() + 1e-10, bins + 1)
        return np.clip(np.digitize(arr, edges) - 1, 0, bins - 1)
    y_t_d = discretize(y_t, n_bins)
    y_past_d = discretize(y_past, n_bins)
    x_past_d = discretize(x_past, n_bins)
    # Compute conditional MI: I(Y_t ; X_past | Y_past)
    # = H(Y_t, Y_past) + H(X_past, Y_past) - H(Y_t, X_past, Y_past) - H(Y_past)
    # Simplified using joint probabilities
    # P(Y_t, X_past, Y_past)
    from collections import Counter
    counts_3 = Counter(zip(y_t_d, x_past_d, y_past_d))
    counts_2_yt_yp = Counter(zip(y_t_d, y_past_d))
    counts_2_xp_yp = Counter(zip(x_past_d, y_past_d))
    counts_1_yp = Counter(y_past_d)
    total = m
    te = 0.0
    for (yt, xp, yp), c3 in counts_3.items():
        p3 = c3 / total
        c_yt_yp = counts_2_yt_yp.get((yt, yp), 0)
        c_xp_yp = counts_2_xp_yp.get((xp, yp), 0)
        c_yp = counts_1_yp.get(yp, 0)
        if c_yt_yp > 0 and c_xp_yp > 0 and c_yp > 0:
            p_yt_yp = c_yt_yp / total
            p_xp_yp = c_xp_yp / total
            p_yp = c_yp / total
            # TE = sum p(yt, xp, yp) * log[ p(yt, xp, yp) * p(yp) / (p(yt, yp) * p(xp, yp)) ]
            te += p3 * np.log2((p3 * p_yp) / (p_yt_yp * p_xp_yp))
    return float(max(0.0, te))


def entropy_rate(series: np.ndarray, n_bins: int = 10, order: int = 2) -> float:
    """Approximate entropy rate — entropy per time step.

    Uses a simple Markov approximation: H(X_t | X_{t-1}, ..., X_{t-order+1}).

    Args:
        series: 1-D array
        n_bins: discretization bins
        order: Markov order (default 2)
    """
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < order + 20:
        return 0.0
    # Discretize
    edges = np.linspace(series.min(), series.max() + 1e-10, n_bins + 1)
    disc = np.clip(np.digitize(series, edges) - 1, 0, n_bins - 1)
    # Build (state, next) pairs
    from collections import Counter
    state_counts: Counter = Counter()
    next_counts: Counter = Counter()
    for i in range(len(disc) - order):
        state = tuple(disc[i:i + order])
        nxt = disc[i + order]
        state_counts[(state, nxt)] += 1
        next_counts[state] += 1
    # H(next | state) = sum p(state, next) * log[1 / p(next | state)]
    h = 0.0
    total = sum(state_counts.values())
    for (state, nxt), c in state_counts.items():
        p_state_next = c / total
        p_next_given_state = c / next_counts[state]
        h += p_state_next * (-np.log2(p_next_given_state))
    return float(h)


__all__ = [
    "shannon_entropy", "kl_divergence", "jensen_shannon_divergence",
    "mutual_information", "transfer_entropy", "entropy_rate",
]
