# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evolution strategies with OuterCtx integration.

Provides single-stage, two-stage, and mass-locking determinant selection
strategies for iterative CI space evolution.

File: lever/evolution/strategies.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import EvolutionStrategy, Scorer, Selector
from ..utils.dtypes import ScoreResult

if TYPE_CHECKING:
    from ..utils.dtypes import OuterCtx, PsiCache

__all__ = ["BasicStrategy", "TwoStageStrategy", "MassLockingStrategy"]


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


__all__ = ["BasicStrategy", "TwoStageStrategy", "MassLockingStrategy"]