# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant selection policies for evolutionary CI.

Implements score-based selection strategies: top-k ranking and threshold
filtering with optional size constraints.

File: lever/evolution/selectors.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

import numpy as np

from ..utils.dtypes import ScoreResult


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
        n_select = min(self.k, len(scores))
        
        # Partition via argsort: O(n log k) for small k
        top_indices = np.argsort(scores)[-n_select:]
        
        return dets[top_indices]


class ThresholdSelector:
    """
    Select determinants exceeding score threshold with optional size cap.
    
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
        
        # First pass: threshold filtering
        passing_mask = scores >= self.threshold
        passing_indices = np.where(passing_mask)[0]
        
        # Second pass: enforce size constraint if needed
        if self.max_size is not None and len(passing_indices) > self.max_size:
            passing_scores = scores[passing_indices]
            top_k_local = np.argsort(passing_scores)[-self.max_size:]
            passing_indices = passing_indices[top_k_local]
        
        return dets[passing_indices]


__all__ = ["TopKSelector", "ThresholdSelector"]
