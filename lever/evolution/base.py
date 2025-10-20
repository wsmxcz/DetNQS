# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational space evolution protocols for adaptive CI.

Defines core abstractions for determinant importance scoring, selection,
and iterative space refinement in LEVER's S_k → S_{k+1} evolution cycle.

File: lever/evolution/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..engine import Evaluator


# --- Data Structures ---

class ScoreResult(NamedTuple):
    """
    Scored determinant container.
  
    Attributes:
        scores: Importance measure for each determinant (1D array)
        dets: Corresponding determinant bitmasks (2D array)
        metadata: Diagnostic information dictionary
    """
    scores: np.ndarray
    dets: np.ndarray
    metadata: dict


# --- Evolution Protocols ---

class Scorer(Protocol):
    """
    Determinant importance evaluator.
  
    Computes scalar scores measuring variational significance of each
    determinant in the active space (S ∪ C) for guiding selection.
    """

    def compute_scores(self, evaluator: Evaluator) -> ScoreResult:
        """
        Evaluate determinant importance from current wavefunction state.
      
        Args:
            evaluator: Lazy evaluation context with ψ(S∪C) and Hamiltonian
      
        Returns:
            ScoreResult with importance scores and determinant indices
        """
        ...


class Selector(Protocol):
    """
    Determinant subset selector.
  
    Implements selection policy (threshold-based, top-k, etc.) to choose
    core space S_{k+1} from scored candidates.
    """

    def select(self, score_result: ScoreResult) -> np.ndarray:
        """
        Choose new core space from scored determinants.
      
        Args:
            score_result: Scorer output with importance measures
      
        Returns:
            Selected determinant bitmasks (2D array)
        """
        ...


class EvolutionStrategy(Protocol):
    """
    Complete space evolution orchestrator.
  
    Coordinates Scorer(s) and Selector(s) to execute one refinement
    step: S_k → S_{k+1} at each optimization convergence point.
    """

    def evolve(self, evaluator: Evaluator) -> np.ndarray:
        """
        Execute single evolution cycle.
      
        Args:
            evaluator: Converged wavefunction state from optimization
      
        Returns:
            New core space S_{k+1} as determinant bitmasks
        """
        ...


__all__ = ["ScoreResult", "Scorer", "Selector", "EvolutionStrategy"]