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
    """Select top-k determinants by score using partial selection algorithm.
    
    Uses argpartition for O(n) average complexity instead of full O(n log n) sort.
    """

    def __init__(self, k: int) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """Select k determinants with highest scores.
        
        Args:
            score_result: Scored determinant collection
            
        Returns:
            Selected determinants ordered by descending score
        """
        scores, dets = score_result.scores, score_result.dets
        if dets.size == 0:
            return dets

        n_select = min(self.k, len(scores))

        if n_select == len(scores):
            top_indices = np.argsort(scores)[::-1]  # Full sort when k = n
        else:
            # Partial selection: partition then sort only top-k
            partition_indices = np.argpartition(scores, -n_select)[-n_select:]
            top_indices = partition_indices[np.argsort(scores[partition_indices])[::-1]]
        
        return dets[top_indices]


class ThresholdSelector:
    """Select determinants exceeding score threshold with optional size cap."""

    def __init__(
        self,
        threshold: float,
        max_size: int | None = None,
    ) -> None:
        self.threshold = threshold
        self.max_size = max_size

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """Filter determinants by threshold with size limiting.
        
        Args:
            score_result: Scored determinant collection
            
        Returns:
            Selected determinants meeting threshold criteria
        """
        scores, dets = score_result.scores, score_result.dets
        if dets.size == 0:
            return dets

        passing_mask = scores >= self.threshold
        
        if not np.any(passing_mask):
            return np.empty((0, dets.shape[1]), dtype=dets.dtype)

        if self.max_size is not None:
            passing_indices = np.where(passing_mask)[0]
            
            if passing_indices.size > self.max_size:
                # Apply top-k selection on threshold-passed candidates
                passing_scores = scores[passing_indices]
                local_top_k = np.argpartition(passing_scores, -self.max_size)[-self.max_size:]
                sorted_local = local_top_k[np.argsort(passing_scores[local_top_k])[::-1]]
                return dets[passing_indices[sorted_local]]
        
        return dets[passing_mask]


class CumulativeMassSelector:
    """Select determinants until cumulative mass threshold is reached.
    
    Assumes scores are non-negative weights. For amplitude-based scores,
    commonly uses |ψ_i|². Selected set captures at least mass_threshold
    of total weight: Σ_{selected} w_i ≥ threshold × Σ_{all} w_i.
    """

    def __init__(
        self, 
        mass_threshold: float = 0.999,
        max_size: int | None = None,
    ) -> None:
        if not 0.0 < mass_threshold <= 1.0:
            raise ValueError(f"mass_threshold must be in (0, 1], got {mass_threshold}")
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive")
        
        self.mass_threshold = mass_threshold
        self.max_size = max_size

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """Select determinants by descending mass until threshold met.
        
        Args:
            score_result: Scored determinant collection
            
        Returns:
            Selected determinants as ndarray (n, 2)
        """
        scores, dets = score_result.scores, score_result.dets
        if dets.size == 0:
            return dets

        weights = np.abs(np.asarray(scores, dtype=float))
        total_mass = weights.sum()

        if total_mass <= 0.0:
            return np.empty((0, dets.shape[1]), dtype=dets.dtype)

        probs = weights / total_mass
        n = len(probs)
        
        if self.max_size is not None and self.max_size < n:
            # Partial selection for large arrays
            partition_idx = np.argpartition(probs, -self.max_size)[-self.max_size:]
            sorted_indices = partition_idx[np.argsort(probs[partition_idx])[::-1]]
            sorted_probs = probs[sorted_indices]
        else:
            # Full sort for smaller arrays
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]

        cumulative_mass = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumulative_mass, self.mass_threshold, side="right")

        if cutoff_idx == 0 and len(sorted_indices) > 0:
            cutoff_idx = 1  # Ensure at least one selection

        return dets[sorted_indices[:cutoff_idx]]


__all__ = ["TopKSelector", "ThresholdSelector", "CumulativeMassSelector"]
