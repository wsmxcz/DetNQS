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
from typing import TYPE_CHECKING, Callable, Union

import jax.numpy as jnp

from .base import RuleState

if TYPE_CHECKING:
    from ..dtypes import PyTree

# Optax schedule is Callable[[step], ArrayLike]【turn9view0†L147-L164】
Schedule = Callable[[int], jnp.ndarray | float]
ScalarOrSchedule = Union[float, Schedule]


@dataclass
class ConstantRule:
    """
    Fixed or scheduled step size: α = lr(step).
    
    learning_rate can be:
      - a float / ArrayLike constant
      - an optax schedule: Callable[[int], ArrayLike]
        (e.g. optax.cosine_decay_schedule, optax.sgdr_schedule, etc.)【turn9view0†L87-L93】【turn9view0†L121-L123】
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
        Return step size α(step).
        
        Args:
            direction: Search direction (unused)
            state: Rule state (unchanged)
            energy: Current energy (unused for constant/scheduled lr)
            step: Global iteration index (0-based)
        """
        lr = self.learning_rate
        if callable(lr):
            # schedule(step) → scalar / 0-D array
            lr = lr(step)
        # jnp.asarray makes sure we always return an ArrayLike scalar
        lr = jnp.asarray(lr)
        return lr, state


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
        energy: float,
        step: int,
    ) -> tuple[float, RuleState]:
        """Placeholder: returns initial_step without backtracking."""
        return self.initial_step, state


__all__ = ["ConstantRule", "LineSearchRule"]
