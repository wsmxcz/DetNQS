# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step size control rules for variational optimization.

Implements strategies for computing step size α along search direction:
  - ConstantRule: α = lr (fixed learning rate)
  - LineSearchRule: Armijo backtracking (Phase 5 placeholder)
  - TrustRegionRule: Adaptive damping (Phase 5 placeholder)

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import RuleState

if TYPE_CHECKING:
    from ..utils.dtypes import PyTree


@dataclass
class ConstantRule:
    """
    Fixed step size: α = lr.
    
    Simplest stateless rule, suitable for well-conditioned problems.
    Update: θ_{k+1} = θ_k + α·d_k where d_k is search direction.
    
    Attributes:
        learning_rate: Fixed step size α > 0
    """
    learning_rate: float = 1e-3
    
    def __call__(
        self,
        direction: PyTree,
        state: RuleState,
        energy: float
    ) -> tuple[float, RuleState]:
        """
        Return constant step size.
        
        Args:
            direction: Search direction (unused)
            state: Rule state (unchanged)
            energy: Current energy (unused)
            
        Returns:
            (α, state): Fixed learning rate and unmodified state
        """
        return self.learning_rate, state


@dataclass
class LineSearchRule:
    """
    Backtracking line search with Armijo condition (Phase 5 placeholder).
    
    Will implement: α_k = β^m · α_0 where β ∈ (0,1) and m satisfies
    f(θ + α_k·d) ≤ f(θ) + c·α_k·∇f^T·d with Armijo constant c ∈ (0,1).
    
    Currently delegates to fixed initial_step.
    
    Attributes:
        initial_step: Starting trial step α_0
        backtrack_factor: Reduction factor β ∈ (0,1)
    """
    initial_step: float = 1.0
    backtrack_factor: float = 0.5
    
    def __call__(
        self,
        direction: PyTree,
        state: RuleState,
        energy: float
    ) -> tuple[float, RuleState]:
        """Placeholder: returns initial_step without backtracking."""
        return self.initial_step, state


__all__ = ["ConstantRule", "LineSearchRule"]
