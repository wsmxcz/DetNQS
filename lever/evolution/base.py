# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Space evolution protocols with OuterCtx integration.

File: lever/evolution/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..engine import OuterCtx
    from ..engine.utils import PyTree


# --- Data Structures ---

class ScoreResult(NamedTuple):
    """Scored determinant container."""
    scores: np.ndarray
    dets: np.ndarray
    meta: dict


# --- Evolution Protocols ---

class Scorer(Protocol):
    """Determinant importance evaluator."""

    def score(self, ctx: OuterCtx, params: PyTree) -> ScoreResult:
        """
        Compute importance scores from converged state.
      
        Args:
            ctx: Outer cycle context with space/features/Hamiltonian
            params: Converged neural network parameters
      
        Returns:
            ScoreResult with importance measures
        """
        ...


class Selector(Protocol):
    """Determinant subset selector."""

    def select(self, result: ScoreResult) -> np.ndarray:
        """
        Choose new core space from scored determinants.
      
        Args:
            result: Scorer output
      
        Returns:
            Selected determinants
        """
        ...


class EvolutionStrategy(Protocol):
    """Complete evolution orchestrator."""

    def evolve(self, ctx: OuterCtx, params: PyTree) -> np.ndarray:
        """
        Execute single evolution cycle.
      
        Args:
            ctx: Converged outer context
            params: Converged parameters
      
        Returns:
            New S-space determinants
        """
        ...


__all__ = ["ScoreResult", "Scorer", "Selector", "EvolutionStrategy"]
