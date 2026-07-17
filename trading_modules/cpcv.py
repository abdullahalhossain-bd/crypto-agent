"""
CPCV Module — Combinatorial Purged Cross-Validation
=====================================================

López de Prado's gold-standard CV method for financial time series.

Why CPCV over standard walk-forward:
  1. Label overlap: When entry and exit span N days, train and test
     samples whose holding windows overlap have label leakage.
  2. Post-test contamination: The N days AFTER a test window carry
     information from that window. Training on those days injects
     future leakage.

CPCV fixes:
  - Split data into N groups by time
  - Iterate ALL C(N, k) combinations of k groups as test set
    (not just N sliding windows)
  - Purge: remove train samples whose label window overlaps test
  - Embargo: remove train samples in a δ×n_samples window after test

Source: Orallexa (review #27) — cpcv.py + López de Prado (2018) ch. 12
        TradingAgents v0.3.1 (review #30) — look-ahead filtering
        ml4t-3e (review #18) — walk-forward CV methodology

Usage:
    from cpcv import CPCV
    import pandas as pd

    cpcv = CPCV(n_groups=6, n_test_groups=2, embargo_pct=0.01)
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=500, freq='D'),
        'feature': ...,
        'label': ...,
        'return': ...,
    })

    splits = cpcv.split(df, label_horizon_days=5)

    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"Fold {i}: train={len(train_idx)}, test={len(test_idx)}")
        # Train model on df.iloc[train_idx]
        # Evaluate on df.iloc[test_idx]
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Iterator, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CPCV:
    """
    Combinatorial Purged Cross-Validation.

    Parameters:
        n_groups: Number of time groups to split data into (default: 6)
        n_test_groups: Number of groups in each test set (default: 2)
        embargo_pct: Fraction of total samples to embargo after each
                     test group (default: 0.01 = 1%)
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ):
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if n_test_groups < 1 or n_test_groups >= n_groups:
            raise ValueError("n_test_groups must be in [1, n_groups-1]")

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct

    def split(
        self,
        df: pd.DataFrame,
        label_horizon_days: int = 1,
        date_col: str = "date",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate CPCV train/test splits.

        Args:
            df: DataFrame sorted by date (oldest first)
            label_horizon_days: Number of days a label spans (for purging)
            date_col: Name of the date column

        Yields:
            (train_indices, test_indices) for each fold
        """
        n = len(df)
        if n < self.n_groups:
            logger.warning(f"Not enough samples ({n}) for {self.n_groups} groups")
            yield np.arange(n), np.arange(n)
            return

        # Create group boundaries
        group_boundaries = np.array_split(np.arange(n), self.n_groups)

        # Compute embargo size
        embargo_size = max(1, int(n * self.embargo_pct))

        # Compute label window size (for purging)
        if date_col in df.columns:
            dates = pd.to_datetime(df[date_col])
            label_window = label_horizon_days
        else:
            label_window = label_horizon_days
            dates = None

        # Generate all C(n_groups, n_test_groups) combinations
        # Major #7 fix: safeguard against combinatorial explosion.
        # For large n_groups, the number of combinations can be enormous
        # (e.g. C(20,5) = 15504). Cap at a reasonable limit and sample.
        from math import comb
        total_combinations = comb(self.n_groups, self.n_test_groups)
        MAX_FOLDS = 1000  # safety cap
        if total_combinations > MAX_FOLDS:
            logger.warning(
                f"CPCV: {total_combinations} folds from C({self.n_groups}, "
                f"{self.n_test_groups}) — capping to {MAX_FOLDS} random folds "
                f"to prevent combinatorial explosion"
            )
            import random
            all_combos = list(combinations(range(self.n_groups), self.n_test_groups))
            selected_combos = random.sample(all_combos, MAX_FOLDS)
            total_combinations = MAX_FOLDS
        else:
            selected_combos = combinations(range(self.n_groups), self.n_test_groups)

        logger.info(
            f"CPCV: {total_combinations} folds from C({self.n_groups}, {self.n_test_groups}) "
            f"with {label_window}d purge + {embargo_size} sample embargo"
        )

        for test_groups in selected_combos:
            # Collect test indices
            test_idx = np.concatenate([group_boundaries[g] for g in test_groups])

            # Collect train indices (all groups not in test)
            train_groups = [g for g in range(self.n_groups) if g not in test_groups]
            train_idx = np.concatenate([group_boundaries[g] for g in train_groups])

            # Purge: remove train samples whose label window overlaps test
            train_idx = self._purge(train_idx, test_idx, label_window, n)

            # Embargo: remove train samples immediately after test
            train_idx = self._embargo(train_idx, test_idx, embargo_size, n)

            yield train_idx, test_idx

    def _purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        label_window: int,
        n: int,
    ) -> np.ndarray:
        """
        Remove train samples whose label window overlaps test samples.

        If a train sample at position i has a label that extends to
        i + label_window, and any test sample falls in [i, i+label_window],
        that train sample is purged.
        """
        if label_window <= 0:
            return train_idx

        test_set = set(test_idx)
        purge_set = set()

        for i in train_idx:
            # Label window: [i, i + label_window]
            for j in range(i, min(i + label_window + 1, n)):
                if j in test_set:
                    purge_set.add(i)
                    break

        return np.array([i for i in train_idx if i not in purge_set])

    def _embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        embargo_size: int,
        n: int,
    ) -> np.ndarray:
        """
        Remove train samples in the embargo zone after each test sample.

        The embargo zone is the [test_idx, test_idx + embargo_size] range
        after each test sample, preventing post-test contamination.
        """
        if embargo_size <= 0:
            return train_idx

        embargo_set = set()
        for t in test_idx:
            for j in range(t + 1, min(t + embargo_size + 1, n)):
                embargo_set.add(j)

        return np.array([i for i in train_idx if i not in embargo_set])

    def get_split_summary(self, df: pd.DataFrame, date_col: str = "date") -> dict:
        """
        Get summary statistics about the CPCV splits without running full CV.

        Useful for understanding the data layout before training.
        """
        n = len(df)
        group_boundaries = np.array_split(np.arange(n), self.n_groups)
        total_combinations = len(list(combinations(range(self.n_groups), self.n_test_groups)))

        groups = []
        for i, bounds in enumerate(group_boundaries):
            start_idx = bounds[0]
            end_idx = bounds[-1]
            if date_col in df.columns:
                start_date = str(df[date_col].iloc[start_idx])[:10]
                end_date = str(df[date_col].iloc[end_idx])[:10]
            else:
                start_date = f"idx:{start_idx}"
                end_date = f"idx:{end_idx}"

            groups.append({
                "group": i,
                "start": start_date,
                "end": end_date,
                "samples": len(bounds),
            })

        return {
            "total_samples": n,
            "n_groups": self.n_groups,
            "n_test_groups": self.n_test_groups,
            "total_folds": total_combinations,
            "embargo_pct": self.embargo_pct,
            "groups": groups,
        }
