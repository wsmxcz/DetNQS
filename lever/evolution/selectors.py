# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant selection policies for evolutionary CI.

Implements score-based selection strategies: top-k ranking, threshold
filtering, and cumulative-mass selection.

File: lever/evolution/selectors.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import numpy as np

from ..dtypes import ScoreResult


class TopKSelector:
    """
    Select top-k determinants by score ranking.

    Uses partial sort to extract k highest-scoring configurations,
    automatically handling edge cases where k exceeds available candidates.
    """

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """
        Extract k determinants with highest scores.

        Args:
            score_result: Scored determinant collection

        Returns:
            Selected determinants ordered by descending score
        """
        scores, dets = score_result.scores, score_result.dets
        if dets.size == 0:
            return dets

        n_select = min(self.k, len(scores))

        # Use argsort on scores; for small k this is effectively O(n log k)
        top_indices = np.argsort(scores)[-n_select:]
        return dets[top_indices]


class ThresholdSelector:
    """
    Select determinants exceeding a score threshold with optional size cap.

    Applies absolute score cutoff followed by top-k reduction if needed,
    ensuring both quality and quantity constraints are satisfied.
    """

    def __init__(
        self,
        threshold: float,
        max_size: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_size = max_size

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """
        Filter determinants by threshold with optional size limiting.

        Args:
            score_result: Scored determinant collection

        Returns:
            Selected determinants meeting threshold criteria
        """
        scores, dets = score_result.scores, score_result.dets
        if dets.size == 0:
            return dets

        # First pass: threshold filtering
        passing_mask = scores >= self.threshold
        passing_indices = np.where(passing_mask)[0]

        if passing_indices.size == 0:
            return np.empty((0, dets.shape[1]), dtype=dets.dtype)

        # Second pass: enforce size constraint if needed
        if self.max_size is not None and passing_indices.size > self.max_size:
            passing_scores = scores[passing_indices]
            top_k_local = np.argsort(passing_scores)[-self.max_size:]
            passing_indices = passing_indices[top_k_local]

        return dets[passing_indices]


class CumulativeMassSelector:
    """
    Select determinants until a cumulative mass threshold is reached.

    The selector assumes scores are non-negative "weights". For amplitude-
    based scores, a common choice is |ψ_i|². The selected set captures at
    least `mass_threshold` of the total weight.
    """

    def __init__(self, mass_threshold: float = 0.999) -> None:
        """
        Args:
            mass_threshold: Target cumulative mass in (0, 1]. For example,
                            0.999 means the selected space captures 99.9%
                            of the total score mass.
        """
        if not 0.0 < mass_threshold <= 1.0:
            raise ValueError(
                f"mass_threshold must be in (0, 1], got {mass_threshold}"
            )
        self.mass_threshold = mass_threshold

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """
        Select determinants by descending mass until mass_threshold is met.

        Args:
            score_result: Scored determinant collection

        Returns:
            New S-space determinants as ndarray (n, 2)
        """
        scores, dets = score_result.scores, score_result.dets
        if dets.size == 0:
            return dets

        # Use absolute values to be robust to signed scores
        weights = np.abs(np.asarray(scores, dtype=float))
        total_mass = weights.sum()

        # Degenerate case: no meaningful mass
        if total_mass <= 0.0:
            return np.empty((0, dets.shape[1]), dtype=dets.dtype)

        # Normalize to probability-like weights
        probs = weights / total_mass

        # Sort determinants by decreasing probability
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Prefix sum to get cumulative mass
        cumulative_mass = np.cumsum(sorted_probs)

        # Find the first index where cumulative mass exceeds the threshold
        cutoff_idx = np.searchsorted(
            cumulative_mass, self.mass_threshold, side="right"
        )

        # Ensure at least one determinant is selected when possible
        if cutoff_idx == 0 and len(sorted_indices) > 0:
            cutoff_idx = 1

        return dets[sorted_indices[:cutoff_idx]]


__all__ = ["TopKSelector", "ThresholdSelector", "CumulativeMassSelector"]