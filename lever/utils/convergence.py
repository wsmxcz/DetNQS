# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convergence control utilities for inner and outer optimization loops.

Provides simple state objects to encapsulate early-stopping logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class InnerConvergence:
    """
    Convergence controller for inner optimization (host-side).
    
    Tracks consecutive steps where |value_k - value_{k-1}| < tol.
    Converges when streak reaches patience.
    """

    tol: float
    patience: int

    last_value: float = field(default=float("inf"), init=False)
    streak: int = field(default=0, init=False)

    @property
    def enabled(self) -> bool:
        """Check if early stopping is active."""
        return self.tol > 0.0 and self.patience > 0

    def update(self, value: float) -> Tuple[bool, float]:
        """
        Update with new scalar value and check convergence.
        
        Args:
            value: Current optimization metric (e.g., energy)
        
        Returns:
            (converged, delta): Whether patience criterion is met and current delta
        """
        delta = abs(value - self.last_value)
        
        if delta < self.tol:
            self.streak += 1
        else:
            self.streak = 0
        
        self.last_value = value
        return self.streak >= self.patience, delta


@dataclass
class OuterConvergence:
    """
    Sliding-window convergence for outer cycles.
    
    Requires all pairwise deltas in the last (patience + 1) values
    to be below tol.
    """

    tol: float
    patience: int

    history: List[float] = field(default_factory=list, init=False)

    def update(self, value: float) -> Tuple[bool, float]:
        """
        Append a new outer energy and check convergence.
        
        Args:
            value: Energy at the end of current cycle
        
        Returns:
            (converged, max_delta): Whether sliding window converged and max delta
        """
        self.history.append(value)
        
        if self.patience <= 0 or len(self.history) <= self.patience:
            return False, float("inf")

        # Check last (patience + 1) values
        window = self.history[-(self.patience + 1) :]
        deltas = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
        max_delta = max(deltas)
        
        return max_delta < self.tol, max_delta


__all__ = ["InnerConvergence", "OuterConvergence"]