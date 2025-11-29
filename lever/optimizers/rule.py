# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step size control for variational optimization.

Implements adaptive strategies for computing step size α along search direction:
  - ConstantRule: Fixed or scheduled learning rate α = lr(t)
  - LineSearchRule: Armijo backtracking (Phase 5 placeholder)

File: lever/optim/step_size.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Union

import jax.numpy as jnp

from .base import RuleState

if TYPE_CHECKING:
    from ..dtypes import PyTree

# Optax-compatible schedule: step → scalar
Schedule = Callable[[int], jnp.ndarray | float]
ScalarOrSchedule = Union[float, Schedule]


@dataclass
class ConstantRule:
    """
    Fixed or scheduled step size: α_t = lr(t).
    
    Supports:
      - Constant: lr = 0.001
      - Optax schedules: lr = optax.cosine_decay_schedule(init_value, decay_steps)
    
    Attributes:
        learning_rate: Scalar constant or callable schedule
    """
    learning_rate: ScalarOrSchedule = 1e-3
    
    def __call__(
        self,
        direction: PyTree,
        state: RuleState,
        energy: float,
        step: int,
    ) -> tuple[jnp.ndarray, RuleState]:
        """
        Compute step size at iteration t.
        
        Args:
            direction: Search direction (unused)
            state: Rule state (unchanged)
            energy: Current energy (unused)
            step: Global iteration index (0-based)
        
        Returns:
            (α_t, state): Step size and unchanged state
        """
        lr = self.learning_rate
        if callable(lr):
            lr = lr(step)
        return jnp.asarray(lr), state


@dataclass
class LineSearchRule:
    """
    Backtracking line search with Armijo condition (Phase 5 placeholder).
    
    Algorithm:
      Find α_k = β^m · α_0 satisfying Armijo condition:
        E(θ + α_k·d) ≤ E(θ) + c·α_k·⟨∇E, d⟩
      where β ∈ (0,1) is backtrack factor, c ∈ (0,1) is Armijo constant.
    
    Current implementation: Returns initial_step without backtracking.
    
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
        energy: float,
        step: int,
    ) -> tuple[float, RuleState]:
        """
        Compute step size via line search.
        
        Args:
            direction: Search direction
            state: Rule state
            energy: Current energy E(θ)
            step: Global iteration index
        
        Returns:
            (α, state): Step size and updated state
        """
        # TODO: Implement Armijo backtracking
        return self.initial_step, state


__all__ = ["ConstantRule", "LineSearchRule"]
