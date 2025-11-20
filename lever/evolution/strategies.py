# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evolution strategies for determinant space selection.

Provides single-stage, two-stage, and mass-locking strategies
for iterative CI space evolution.

File: lever/evolution/strategies.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import EvolutionStrategy, Scorer, Selector

if TYPE_CHECKING:
    from ..dtypes import OuterCtx, PsiCache, ScoreResult


class BasicStrategy(EvolutionStrategy):
    """Single-stage evolution: score → select."""

    def __init__(self, scorer: Scorer, selector: Selector) -> None:
        self.scorer = scorer
        self.selector = selector

    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """Apply scoring and selection using cached amplitudes."""
        scores = self.scorer.score(ctx, psi_cache)
        return self.selector.select(scores)


class TwoStageStrategy(EvolutionStrategy):
    """
    Two-stage evolution: independent core and frontier selection.
    
    Enables hybrid criteria (e.g., amplitude-based core, PT2-based frontier).
    Final space: S_new = S_core ∪ S_frontier
    """

    def __init__(
        self,
        core_scorer: Scorer,
        core_selector: Selector,
        frontier_scorer: Scorer,
        frontier_selector: Selector,
    ) -> None:
        self.core_scorer = core_scorer
        self.core_selector = core_selector
        self.frontier_scorer = frontier_scorer
        self.frontier_selector = frontier_selector

    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """Select core and frontier determinants independently."""
        s_core = self.core_selector.select(
            self.core_scorer.score(ctx, psi_cache)
        )
        s_frontier = self.frontier_selector.select(
            self.frontier_scorer.score(ctx, psi_cache)
        )
        return self._unique_union(s_core, s_frontier)

    @staticmethod
    def _unique_union(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Return unique union of two determinant arrays."""
        if arr1.size == 0:
            return arr2
        if arr2.size == 0:
            return arr1
        return np.unique(np.concatenate([arr1, arr2], axis=0), axis=0)


class MassLockingStrategy(EvolutionStrategy):
    """
    ASCI-inspired mass-locking evolution.
    
    Algorithm:
      1. Lock S-space determinants with cumulative probability mass
         ∑|ψᵢ|² ≥ threshold (sorted by |ψᵢ|² descending)
      2. Expand frontier using scorer (typically PT2 energy contribution)
      3. Final space: S_new = S_locked ∪ S_frontier
    
    Parameters:
        mass_threshold: Cumulative probability threshold ∈ (0, 1)
        frontier_scorer: Scorer for frontier selection (e.g., E2Scorer)
        frontier_selector: Selector for frontier space
    """

    def __init__(
        self,
        mass_threshold: float,
        frontier_scorer: Scorer,
        frontier_selector: Selector,
    ) -> None:
        if not 0 < mass_threshold < 1:
            raise ValueError(
                f"mass_threshold must be in (0, 1), got {mass_threshold}"
            )
        
        self.mass_threshold = mass_threshold
        self.frontier_scorer = frontier_scorer
        self.frontier_selector = frontier_selector

    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """Apply mass-locking with frontier expansion."""
        s_locked = self._lock_core(ctx, psi_cache)
        s_frontier = self.frontier_selector.select(
            self.frontier_scorer.score(ctx, psi_cache)
        )
        return self._unique_union(s_locked, s_frontier)

    def _lock_core(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """Lock determinants with cumulative |ψᵢ|² ≥ threshold."""
        psi_s = np.asarray(psi_cache.psi_s)
        current_s = ctx.space.s_dets
        
        probs = np.abs(psi_s) ** 2
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])
        
        cutoff = np.searchsorted(cumsum, self.mass_threshold, side="right")
        return current_s[sorted_idx[: cutoff + 1]]

    @staticmethod
    def _unique_union(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Return unique union of two determinant arrays."""
        if arr1.size == 0:
            return arr2
        if arr2.size == 0:
            return arr1
        return np.unique(np.concatenate([arr1, arr2], axis=0), axis=0)


class CumulativeMassStrategy(EvolutionStrategy):
    """
    Selects a new S-space by including determinants with the highest
    probability mass until a cumulative threshold is reached.

    This strategy ensures that the new core space captures a specified
    fraction of the total wavefunction's norm, based on the amplitudes
    from the previous cycle.
    """

    def __init__(self, mass_threshold: float = 0.999) -> None:
        """
        Args:
            mass_threshold: The target cumulative probability mass, a value
                            between 0.0 and 1.0. For example, 0.999 means
                            the new S-space will capture 99.9% of the total
                            probability mass from the S ∪ C space.
        """
        if not 0.0 < mass_threshold <= 1.0:
            raise ValueError(
                f"mass_threshold must be in (0, 1], got {mass_threshold}"
            )
        self.mass_threshold = mass_threshold

    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """
        Constructs the new S-space based on cumulative probability mass.

        Algorithm:
          1. Combine all determinants from the S and C spaces.
          2. Calculate the probability |ψᵢ|² for each determinant.
          3. Sort determinants in descending order of their probability.
          4. Compute the cumulative sum of normalized probabilities.
          5. Find the cutoff point where the cumulative sum exceeds the
             mass_threshold.
          6. The new S-space consists of all determinants up to this cutoff.
        """
        # 1. Combine determinants and get amplitudes from the cache
        all_dets = np.concatenate([ctx.space.s_dets, ctx.space.c_dets], axis=0)
        psi_all = np.asarray(psi_cache.psi_all)

        # Handle the edge case of an empty space
        if all_dets.shape[0] == 0:
            return all_dets

        # 2. Calculate probabilities and normalize them
        probs = np.abs(psi_all) ** 2
        total_mass = np.sum(probs)
        if total_mass < 1e-12: # Avoid division by zero if wavefunction is null
             return np.empty((0, 2), dtype=np.uint64)
        
        normalized_probs = probs / total_mass

        # 3. Sort determinants by descending probability
        sorted_indices = np.argsort(normalized_probs)[::-1]
        sorted_dets = all_dets[sorted_indices]
        sorted_probs = normalized_probs[sorted_indices]

        # 4. Compute cumulative sum and find the cutoff
        cumulative_mass = np.cumsum(sorted_probs)
        
        # np.searchsorted finds the index where the threshold would be inserted
        # to maintain order. 'right' side gives us the first index *after*
        # the condition is met, so the slice will include all necessary dets.
        cutoff_idx = np.searchsorted(
            cumulative_mass, self.mass_threshold, side="right"
        )
        
        # Ensure at least one determinant is selected if possible
        if cutoff_idx == 0 and len(sorted_dets) > 0:
            cutoff_idx = 1
            
        # 5. Select the new S-space
        new_s_dets = sorted_dets[:cutoff_idx]

        return new_s_dets

__all__ = ["BasicStrategy", "TwoStageStrategy", "MassLockingStrategy", "CumulativeMassStrategy"]