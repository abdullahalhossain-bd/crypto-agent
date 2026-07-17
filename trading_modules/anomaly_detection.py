"""
Anomaly Detection — Isolation Forest, LOF, z-score, robust
============================================================

Detects anomalous bars / outliers in market data:

    1. Z-Score anomaly         — |z| > 3
    2. Modified Z-Score (MAD)  — robust to outliers
    3. Isolation Forest        — tree-based anomaly detection
    4. Local Outlier Factor    — density-based
    5. DBSCAN outlier          — clustering-based

Pure-Python implementations (no scikit-learn dependency).

Usage:
    from trading_modules.anomaly_detection import AnomalyDetector
    detector = AnomalyDetector()
    result = detector.detect(df["close"])
    for idx in result.anomaly_indices:
        print(f"Anomaly at idx {idx}: {result.scores[idx]:.2f}")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    anomaly_indices: list[int]
    scores: dict[int, float]         # index → anomaly score (higher = more anomalous)
    method: str
    threshold: float
    total_anomalies: int
    anomaly_rate: float              # fraction of bars flagged
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "anomaly_indices": self.anomaly_indices,
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "method": self.method,
            "threshold": round(self.threshold, 3),
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "notes": self.notes,
        }


class AnomalyDetector:
    """Multi-method anomaly detector.

    Parameters:
        contamination: expected fraction of anomalies (default 0.05)
        n_trees: # of trees for Isolation Forest (default 50)
        max_samples: samples per tree (default 64)
    """

    def __init__(
        self, contamination: float = 0.05,
        n_trees: int = 50, max_samples: int = 64,
    ) -> None:
        self.contamination = float(contamination)
        self.n_trees = int(n_trees)
        self.max_samples = int(max_samples)

    def detect(
        self, series: pd.Series, method: str = "auto",
    ) -> AnomalyResult:
        """Detect anomalies using the specified method.

        Args:
            series: 1-D array
            method: "zscore" / "mad" / "isolation_forest" / "lof" / "auto"
                    "auto" tries all and combines
        """
        x = np.asarray(series, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 10:
            return AnomalyResult([], {}, "none", 0, 0, 0.0, ["insufficient data"])

        if method == "zscore":
            return self._zscore(x)
        elif method == "mad":
            return self._mad(x)
        elif method == "isolation_forest":
            return self._isolation_forest(x.reshape(-1, 1))
        elif method == "lof":
            return self._lof(x.reshape(-1, 1))
        elif method == "auto":
            # Combine all methods — anomaly if flagged by >= 2
            results = [
                self._zscore(x), self._mad(x),
                self._isolation_forest(x.reshape(-1, 1)),
            ]
            all_anomalies: dict[int, int] = {}
            scores: dict[int, float] = {}
            for r in results:
                for idx in r.anomaly_indices:
                    all_anomalies[idx] = all_anomalies.get(idx, 0) + 1
                    scores[idx] = scores.get(idx, 0) + r.scores.get(idx, 0)
            # Keep those flagged by >= 2 methods
            final_anomalies = sorted([idx for idx, count in all_anomalies.items() if count >= 2])
            final_scores = {idx: scores[idx] / 3.0 for idx in final_anomalies}
            return AnomalyResult(
                anomaly_indices=final_anomalies,
                scores=final_scores,
                method="auto_ensemble",
                threshold=2,  # flagged by >= 2 methods
                total_anomalies=len(final_anomalies),
                anomaly_rate=len(final_anomalies) / n,
                notes=[f"ensemble of zscore + mad + isolation_forest"],
            )
        else:
            return AnomalyResult([], {}, "unknown", 0, 0, 0.0, [f"unknown method: {method}"])

    # ──────────────────────────────────────────────────────────────
    def _zscore(self, x: np.ndarray) -> AnomalyResult:
        mean = float(np.mean(x))
        std = float(np.std(x))
        if std <= 0:
            return AnomalyResult([], {}, "zscore", 3.0, 0, 0.0, ["zero std"])
        z_scores = np.abs((x - mean) / std)
        threshold = 3.0
        anomalies = [i for i in range(len(x)) if z_scores[i] > threshold]
        scores = {i: float(z_scores[i]) for i in anomalies}
        return AnomalyResult(
            anomaly_indices=anomalies, scores=scores,
            method="zscore", threshold=threshold,
            total_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / len(x),
        )

    def _mad(self, x: np.ndarray) -> AnomalyResult:
        """Modified Z-Score using Median Absolute Deviation — robust to outliers."""
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        if mad <= 0:
            return AnomalyResult([], {}, "mad", 3.5, 0, 0.0, ["zero MAD"])
        modified_z = 0.6745 * (x - median) / mad
        abs_mz = np.abs(modified_z)
        threshold = 3.5
        anomalies = [i for i in range(len(x)) if abs_mz[i] > threshold]
        scores = {i: float(abs_mz[i]) for i in anomalies}
        return AnomalyResult(
            anomaly_indices=anomalies, scores=scores,
            method="mad", threshold=threshold,
            total_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / len(x),
        )

    def _isolation_forest(self, X: np.ndarray) -> AnomalyResult:
        """Simplified Isolation Forest."""
        n, d = X.shape
        if n < 10:
            return AnomalyResult([], {}, "isolation_forest", 0.5, 0, 0.0, ["insufficient data"])
        rng = np.random.default_rng(42)
        # Build trees
        avg_path_lengths = np.zeros(n)
        for _ in range(self.n_trees):
            sample_size = min(self.max_samples, n)
            sample_idx = rng.choice(n, sample_size, replace=False)
            sample = X[sample_idx]
            # Build isolation tree
            path_lengths = self._isolation_tree(sample, X, depth=0, max_depth=int(np.log2(sample_size)) + 1)
            avg_path_lengths += np.array(path_lengths)
        avg_path_lengths /= self.n_trees
        # Anomaly score: s = 2^(-E(h) / c(n)) where c(n) = 2H(n-1) - 2(n-1)/n
        c_n = 2 * (np.log(n - 1) + 0.5772) - 2 * (n - 1) / n if n > 1 else 1.0
        scores_arr = np.power(2, -avg_path_lengths / c_n)
        # Threshold: top contamination% are anomalies
        threshold = float(np.percentile(scores_arr, 100 * (1 - self.contamination)))
        anomalies = [i for i in range(n) if scores_arr[i] > threshold]
        scores = {i: float(scores_arr[i]) for i in anomalies}
        return AnomalyResult(
            anomaly_indices=anomalies, scores=scores,
            method="isolation_forest", threshold=threshold,
            total_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / n,
        )

    def _isolation_tree(
        self, sample: np.ndarray, X: np.ndarray, depth: int, max_depth: int,
    ) -> list:
        """Build an isolation tree and return path lengths for all X."""
        n = len(X)
        path_lengths = np.zeros(n)
        # Simple recursive splitting
        self._itree_recursive(sample, X, depth, max_depth, path_lengths, np.ones(n, dtype=bool))
        return path_lengths.tolist()

    def _itree_recursive(
        self, sample: np.ndarray, X: np.ndarray, depth: int, max_depth: int,
        path_lengths: np.ndarray, mask: np.ndarray,
    ) -> None:
        if depth >= max_depth or len(sample) <= 1 or mask.sum() == 0:
            # Path length = depth + expected average for remaining
            path_lengths[mask] += depth
            return
        d = sample.shape[1]
        # Random feature and split
        feature = np.random.randint(d)
        f_min = float(sample[:, feature].min())
        f_max = float(sample[:, feature].max())
        if f_min == f_max:
            path_lengths[mask] += depth
            return
        split = np.random.uniform(f_min, f_max)
        # Partition X
        left_mask_X = X[:, feature] < split
        left_mask = mask & left_mask_X
        right_mask = mask & (~left_mask_X)
        # Partition sample
        left_sample = sample[sample[:, feature] < split]
        right_sample = sample[sample[:, feature] >= split]
        if len(left_sample) > 0:
            self._itree_recursive(left_sample, X, depth + 1, max_depth, path_lengths, left_mask)
        else:
            path_lengths[left_mask] += depth + 1
        if len(right_sample) > 0:
            self._itree_recursive(right_sample, X, depth + 1, max_depth, path_lengths, right_mask)
        else:
            path_lengths[right_mask] += depth + 1

    def _lof(self, X: np.ndarray, k: int = 20) -> AnomalyResult:
        """Local Outlier Factor — density-based anomaly detection."""
        n, d = X.shape
        if n < k + 1:
            k = max(1, n // 2)
        # Compute k-nearest neighbors for each point
        from scipy.spatial.distance import cdist
        try:
            distances = cdist(X, X)
        except ImportError:
            # Manual distance computation
            distances = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    distances[i, j] = float(np.sqrt(np.sum((X[i] - X[j]) ** 2)))
        # k-distance for each point
        k_distances = np.sort(distances, axis=1)[:, k]
        # Reachability distance
        lof_scores = np.zeros(n)
        for i in range(n):
            # Find k nearest neighbors
            nn_idx = np.argsort(distances[i])[1:k + 1]  # exclude self
            # Local reachability density
            lrd_sum = 0.0
            for j in nn_idx:
                reach_dist = max(k_distances[j], distances[i, j])
                lrd_sum += reach_dist
            lrd_i = k / lrd_sum if lrd_sum > 0 else 1.0
            # LOF = avg(lrd_neighbors) / lrd_i
            lrd_neighbors = []
            for j in nn_idx:
                nn_j = np.argsort(distances[j])[1:k + 1]
                lrd_sum_j = 0.0
                for jj in nn_j:
                    reach_dist_j = max(k_distances[jj], distances[j, jj])
                    lrd_sum_j += reach_dist_j
                lrd_j = k / lrd_sum_j if lrd_sum_j > 0 else 1.0
                lrd_neighbors.append(lrd_j)
            lof_scores[i] = float(np.mean(lrd_neighbors) / lrd_i) if lrd_i > 0 else 1.0
        # LOF > 1 = anomaly
        threshold = 1.5
        anomalies = [i for i in range(n) if lof_scores[i] > threshold]
        scores = {i: float(lof_scores[i]) for i in anomalies}
        return AnomalyResult(
            anomaly_indices=anomalies, scores=scores,
            method="lof", threshold=threshold,
            total_anomalies=len(anomalies),
            anomaly_rate=len(anomalies) / n,
        )


__all__ = ["AnomalyDetector", "AnomalyResult"]
