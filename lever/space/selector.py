# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant selection strategies based on importance scores.

Provides configurable selectors for pruning the determinant space:
  - TopKSelector: Select k largest scores (works in log domain)
  - TopFractionSelector: Select by cumulative probability mass
  - ThresholdSelector: Select by normalized probability cutoff

All selectors accept log amplitude scores log|ψ| as input.

File: lever/selector.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Selector(Protocol):
    """Protocol for determinant selection strategies."""

    def select(self, scores: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """
        Select determinant subset based on importance scores.

        Args:
            scores: Log amplitudes log|ψ| [N]
            dets: Determinant bitstrings [N, 2]

        Returns:
            Selected determinants [K, 2] where K ≤ N
        """
        ...


class TopKSelector:
    """Select k determinants with highest scores."""

    def __init__(self, k: int) -> None:
        """Initialize with fixed selection size."""
        self.k = int(k)

    def select(self, scores: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """
        Select top-k determinants by descending score.

        Works directly on input scores (e.g., log amplitudes) without
        normalization. Uses argpartition for O(n) average complexity.

        Args:
            scores: Importance metric [N] (e.g., log|ψ|)
            dets: Determinant bitstrings [N, 2]

        Returns:
            Top-k determinants sorted by descending score
        """
        if dets.size == 0:
            return dets

        n = scores.shape[0]
        k = min(self.k, n)
        if k <= 0:
            return dets[:0]

        # Partial sort: O(n) average, O(n log k) for final sort
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return dets[idx]


class TopFractionSelector:
    """Select top fraction of determinants by cumulative probability."""

    def __init__(
        self,
        fraction: float,
        min_k: int = 1,
        max_k: int | None = None,
    ) -> None:
        """
        Initialize fraction-based selector with bounds.

        Args:
            fraction: Target cumulative probability in (0.0, 1.0]
            min_k: Minimum selection size
            max_k: Maximum selection size (unbounded if None)
        """
        self.fraction = float(fraction)
        self.min_k = int(min_k)
        self.max_k = int(max_k) if max_k is not None else None

    def select(self, scores: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """
        Select determinants by cumulative probability mass.

        Converts log amplitudes to normalized probabilities via stable
        exponentiation, then selects minimal subset reaching target fraction.

        Algorithm: Normalize via log-sum-exp trick, sort by probability,
        find minimal k where cumsum >= fraction.

        Args:
            scores: Log amplitudes log|ψ| [N]
            dets: Determinant bitstrings [N, 2]

        Returns:
            Determinants [K, 2] where cumsum(prob) >= fraction
        """
        if dets.size == 0:
            return dets

        log_amp = np.asarray(scores, dtype=np.float64)

        # Convert to normalized probabilities: prob = |ψ|² / Σ|ψ|²
        # Use log-sum-exp for numerical stability
        log_prob = 2.0 * log_amp
        log_prob_max = np.max(log_prob)
        log_prob_shifted = log_prob - log_prob_max
        prob_unnorm = np.exp(log_prob_shifted)
        prob = prob_unnorm / np.sum(prob_unnorm)

        # Sort descending and compute CDF
        idx = np.argsort(prob)[::-1]
        cdf = np.cumsum(prob[idx])

        # Find minimal k where cumulative mass >= fraction
        k = int(np.searchsorted(cdf, self.fraction, side="left")) + 1

        # Apply bounds: [min_k, max_k]
        n = log_amp.shape[0]
        k = max(k, self.min_k)
        if self.max_k is not None:
            k = min(k, self.max_k)
        k = min(k, n)

        if k <= 0:
            return dets[:0]

        return dets[idx[:k]]


class ThresholdSelector:
    """Select determinants with normalized probability above threshold."""

    def __init__(self, threshold: float, max_size: int | None = None) -> None:
        """
        Initialize threshold-based selector.

        Args:
            threshold: Minimum normalized probability for inclusion
            max_size: Optional hard limit on selection size
        """
        self.threshold = float(threshold)
        self.max_size = int(max_size) if max_size is not None else None

    def select(self, scores: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """
        Select determinants with normalized probability >= threshold.

        Converts log amplitudes to probabilities via stable exponentiation,
        then applies threshold filter.

        Args:
            scores: Log amplitudes log|ψ| [N]
            dets: Determinant bitstrings [N, 2]

        Returns:
            Determinants meeting probability threshold, optionally truncated
        """
        if dets.size == 0:
            return dets

        log_amp = np.asarray(scores, dtype=np.float64)

        # Convert to normalized probabilities via log-sum-exp
        log_prob = 2.0 * log_amp
        log_prob_max = np.max(log_prob)
        log_prob_shifted = log_prob - log_prob_max
        prob_unnorm = np.exp(log_prob_shifted)
        prob = prob_unnorm / np.sum(prob_unnorm)

        # Apply probability threshold
        mask = prob >= self.threshold
        if not np.any(mask):
            return dets[:0]

        passed_idx = np.nonzero(mask)[0]
        if self.max_size is None or passed_idx.size <= self.max_size:
            return dets[passed_idx]

        # Truncate to max_size by probability ranking
        local_prob = prob[passed_idx]
        k = self.max_size
        idx_local = np.argpartition(local_prob, -k)[-k:]
        idx_local = idx_local[np.argsort(local_prob[idx_local])[::-1]]
        return dets[passed_idx[idx_local]]


__all__ = ["Selector", "TopKSelector", "TopFractionSelector", "ThresholdSelector"]