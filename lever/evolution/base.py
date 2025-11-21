# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Space evolution protocols for determinant selection.

Defines interfaces for:
  - Scoring: Determinant importance evaluation
  - Selection: Core space subset selection
  - Evolution: Complete cycle orchestration

File: lever/evolution/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from ..dtypes import ScoreResult

if TYPE_CHECKING:
    from ..dtypes import OuterCtx, PsiCache


class Scorer(Protocol):
    """
    Determinant importance evaluator.

    Uses converged wavefunction amplitudes to compute scores that drive
    space selection in the next iteration.
    """

    def score(self, ctx: OuterCtx, psi_cache: PsiCache) -> ScoreResult:
        """
        Evaluate determinant importance from cached wavefunction.

        Args:
            ctx: Outer cycle context (space/Hamiltonian info)
            psi_cache: Cached wavefunction amplitudes

        Returns:
            ScoreResult with scores and candidate determinants
        """
        ...


class Selector(Protocol):
    """
    Determinant subset selector.

    Chooses a new core space from scored candidates based on
    ranking, thresholds, or mass-based criteria.
    """

    def select(self, result: ScoreResult) -> object:
        """
        Select new S-space from scored determinants.

        Args:
            result: Scored determinants from Scorer

        Returns:
            Selected determinants as ndarray with shape (n, 2)
        """
        ...


class EvolutionStrategy(Protocol):
    """
    Complete evolution orchestrator.

    Combines scoring and selection into a single evolution cycle,
    optionally with custom logic (e.g. hybrid criteria).
    """

    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> object:
        """
        Execute a single evolution cycle.

        Args:
            ctx: Converged outer context
            psi_cache: Cached wavefunction amplitudes

        Returns:
            New S-space determinants (n, 2) array
        """
        ...


__all__ = ["ScoreResult", "Scorer", "Selector", "EvolutionStrategy"]