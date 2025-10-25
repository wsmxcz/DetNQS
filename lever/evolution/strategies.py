# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-level evolution strategies with OuterCtx integration.

File: lever/evolution/strategies.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import EvolutionStrategy, Scorer, Selector

if TYPE_CHECKING:
    from ..engine import OuterCtx
    from ..engine.utils import PyTree


# --- Single-Stage Strategy ---

class BasicStrategy(EvolutionStrategy):
    """
    Single-stage evolution: score → select.
  
    Applies one scorer to entire active space (S ∪ C), then selects new S.
    """

    def __init__(self, scorer: Scorer, selector: Selector) -> None:
        self.scorer = scorer
        self.selector = selector

    def evolve(self, ctx: OuterCtx, params: PyTree) -> np.ndarray:
        """Apply scoring and selection to produce new S-space."""
        scores = self.scorer.score(ctx, params)
        return self.selector.select(scores)


# --- Two-Stage Strategy ---

class TwoStageStrategy(EvolutionStrategy):
    """
    Two-stage evolution: separate core and frontier selection.
  
    Enables hybrid criteria (e.g., amplitude-based core + PT2-based frontier).
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

    def evolve(self, ctx: OuterCtx, params: PyTree) -> np.ndarray:
        """Select core and frontier independently, return their union."""
        # Stage 1: Core selection
        s_core = self.core_selector.select(
            self.core_scorer.score(ctx, params)
        )

        # Stage 2: Frontier selection
        s_frontier = self.frontier_selector.select(
            self.frontier_scorer.score(ctx, params)
        )

        # Combine unique determinants
        return self._unique_union(s_core, s_frontier)

    @staticmethod
    def _unique_union(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Return unique union of two determinant arrays."""
        if arr1.size == 0:
            return arr2
        if arr2.size == 0:
            return arr1
        return np.unique(np.concatenate([arr1, arr2], axis=0), axis=0)


# --- Mass-Locking Strategy ---

class MassLockingStrategy(EvolutionStrategy):
    """
    ASCI-inspired mass-locking evolution strategy.
  
    Algorithm:
      1. Core Locking: Retain S-space determinants with cumulative probability
         mass ∑|ψᵢ|² > threshold (sorted by |ψᵢ|² descending)
      2. Frontier Expansion: Select from full active space (S ∪ C) using
         specified scorer (typically PT2 energy contribution)
      3. Final space: S_new = S_locked ∪ S_frontier
  
    Parameters:
        mass_threshold: Cumulative probability mass threshold ∈ (0, 1)
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
            raise ValueError(f"mass_threshold must be in (0, 1), got {mass_threshold}")
      
        self.mass_threshold = mass_threshold
        self.frontier_scorer = frontier_scorer
        self.frontier_selector = frontier_selector

    def evolve(self, ctx: OuterCtx, params: PyTree) -> np.ndarray:
        """Apply mass-locking core selection + frontier expansion."""
        # Stage 1: Lock core by cumulative probability mass
        s_locked = self._lock_core(ctx, params)

        # Stage 2: Expand frontier from active space
        s_frontier = self.frontier_selector.select(
            self.frontier_scorer.score(ctx, params)
        )

        # Combine unique determinants
        return self._unique_union(s_locked, s_frontier)

    def _lock_core(self, ctx: OuterCtx, params: PyTree) -> np.ndarray:
        """
        Lock determinants with cumulative |ψᵢ|² > threshold.
      
        Returns:
            Array of locked determinants from current S-space
        """
        import jax.numpy as jnp
        
        # Compute ψ_S from cached logpsi_fn
        log_all = ctx.logpsi_fn(params)
        psi_s = jnp.exp(log_all[:ctx.space.n_s])
        psi_s_cpu = np.asarray(psi_s)
        
        current_s = ctx.space.s_dets

        # Compute probability mass |ψᵢ|²
        probs = np.abs(psi_s_cpu) ** 2

        # Sort by probability (descending)
        sorted_idx = np.argsort(probs)[::-1]

        # Find cutoff where cumulative mass exceeds threshold
        cumsum = np.cumsum(probs[sorted_idx])
        cutoff = np.searchsorted(cumsum, self.mass_threshold, side="right")

        # Return locked determinants (include determinant at cutoff)
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
