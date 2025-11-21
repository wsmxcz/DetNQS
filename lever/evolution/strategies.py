# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evolution strategies for determinant space selection.

Provides single-stage and two-stage strategies for iterative CI
space evolution, plus mass-locking inspired protocols.

File: lever/evolution/strategies.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import EvolutionStrategy, Scorer, Selector

if TYPE_CHECKING:
    from ..dtypes import OuterCtx, PsiCache


def _unique_union(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Return unique union of two determinant arrays."""
    if arr1.size == 0:
        return arr2
    if arr2.size == 0:
        return arr1
    return np.unique(np.concatenate([arr1, arr2], axis=0), axis=0)


class BasicStrategy(EvolutionStrategy):
    """Single-stage evolution: score → select."""

    def __init__(self, scorer: Scorer, selector: Selector) -> None:
        self.scorer = scorer
        self.selector = selector

    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """
        Apply scoring and selection using cached amplitudes.

        Returns:
            New S-space determinants as ndarray (n, 2)
        """
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
        return _unique_union(s_core, s_frontier)


class MassLockingStrategy(EvolutionStrategy):
    """
    ASCI-inspired mass-locking evolution.

    Algorithm:
      1. Lock S-space determinants with cumulative probability mass
         ∑|ψᵢ|² ≥ threshold (sorted by |ψᵢ|² descending)
      2. Expand frontier using a second scorer (e.g., PT2 contribution)
      3. Final space: S_new = S_locked ∪ S_frontier

    Parameters:
        mass_threshold: Cumulative probability threshold ∈ (0, 1)
        frontier_scorer: Scorer for frontier selection
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
        return _unique_union(s_locked, s_frontier)

    def _lock_core(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """Lock determinants with cumulative |ψᵢ|² ≥ threshold."""
        psi_s = np.asarray(psi_cache.psi_s)
        current_s = ctx.space.s_dets

        if current_s.size == 0:
            return current_s

        probs = np.abs(psi_s) ** 2
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = np.cumsum(probs[sorted_idx])

        cutoff = np.searchsorted(cumsum, self.mass_threshold, side="right")

        # Ensure at least one determinant is selected when possible
        if cutoff == 0 and sorted_idx.size > 0:
            cutoff = 1

        return current_s[sorted_idx[: cutoff]]


__all__ = ["BasicStrategy", "TwoStageStrategy", "MassLockingStrategy"]