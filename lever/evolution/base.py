# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Space evolution protocols with OuterCtx integration.

Defines interfaces for:
  - Scoring: Determinant importance evaluation
  - Selection: Core space subset selection
  - Evolution: Complete cycle orchestration

File: lever/evolution/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np
from ..utils.dtypes import ScoreResult

if TYPE_CHECKING:
    from ..utils.dtypes import OuterCtx, PyTree, PsiCache

# --- Evolution Protocols ---

class Scorer(Protocol):
    """
    Determinant importance evaluator.
    
    Computes importance scores from converged wavefunction amplitudes
    to guide space selection in next iteration.
    """
    
    def score(self, ctx: OuterCtx, psi_cache: PsiCache) -> ScoreResult:
        """
        Evaluate determinant importance from cached wavefunction.
        
        Args:
            ctx: Outer cycle context (space/Hamiltonian info)
            psi_cache: Cached wavefunction amplitudes
        
        Returns:
            Scored determinants with metadata
        """
        ...


class Selector(Protocol):
    """
    Determinant subset selector.
    
    Chooses new core space from scored candidates based on
    importance thresholds or size constraints.
    """

    def select(self, result: ScoreResult) -> np.ndarray:
        """
        Select new S-space from scored determinants.
      
        Args:
            result: Scored determinants from Scorer
      
        Returns:
            Selected determinants (N×M array)
        """
        ...


class EvolutionStrategy(Protocol):
    """
    Complete evolution orchestrator.
    
    Combines scoring and selection into single evolution cycle,
    potentially with custom logic (e.g., adaptive thresholds).
    """
    
    def evolve(self, ctx: OuterCtx, psi_cache: PsiCache) -> np.ndarray:
        """
        Execute single evolution cycle.
        
        Args:
            ctx: Converged outer context
            psi_cache: Cached wavefunction amplitudes
        
        Returns:
            New S-space determinants (N×M array)
        """
        ...


__all__ = ["ScoreResult", "Scorer", "Selector", "EvolutionStrategy"]
