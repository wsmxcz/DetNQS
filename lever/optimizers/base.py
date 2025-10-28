# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base optimizer protocol and composable framework.

File: lever/optimizers/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Any, Protocol

import flax.struct
import jax

from ..engine.geometry import GeometryTape
from ..utils.dtypes import PyTree


class OptimizerState(Protocol):
    """Optimizer state protocol (analogous to optax.OptState)."""
    step: int


@flax.struct.dataclass
class BaseOptimizerState:
    """Minimal optimizer state for stateless optimizers."""
    step: int = 0


class Optimizer(Protocol):
    """
    Optimizer protocol compatible with Optax interface.
    
    Key difference: `update` accepts optional `tape` and `energy` kwargs
    for geometry-aware optimization (SR, LM).
    """
    
    def init(self, params: PyTree) -> OptimizerState:
        """Initialize optimizer state."""
        ...
    
    def update(
        self,
        grad: PyTree,
        state: OptimizerState,
        params: PyTree,
        *,
        tape: GeometryTape | None = None,
        energy: float | None = None,
    ) -> tuple[PyTree, OptimizerState]:
        """
        Compute parameter updates.
        
        Args:
            grad: Parameter gradient
            state: Optimizer state
            params: Current parameters
            tape: Geometry tape (optional, for SR/LM)
            energy: Current energy (optional, for line search)
            
        Returns:
            (updates, new_state): Updates to apply via optax.apply_updates
        """
        ...


class ComposableOptimizer:
    """
    Base optimizer implementation via direction + rule composition.
    
    Workflow:
      1. direction_provider(grad, tape) → δ
      2. update_rule(δ, params, energy) → α
      3. updates = α * δ
    """
    
    def __init__(self, direction_provider, update_rule):
        """
        Args:
            direction_provider: Callable (grad, tape) → direction
            update_rule: Callable (direction, params, energy) → step_size
        """
        self.direction = direction_provider
        self.rule = update_rule
    
    def init(self, params: PyTree) -> BaseOptimizerState:
        """Initialize with zero step counter."""
        return BaseOptimizerState(step=0)
    
    def update(
        self,
        grad: PyTree,
        state: BaseOptimizerState,
        params: PyTree,
        *,
        tape: GeometryTape | None = None,
        energy: float | None = None,
    ) -> tuple[PyTree, BaseOptimizerState]:
        """Compute updates via direction × rule."""
        # Compute search direction (may use tape for SR/LM)
        direction = self.direction(grad, tape)
        
        # Compute step size (may use energy for line search)
        step_size = self.rule(direction, params, energy)
        
        # Apply step size to direction
        updates = jax.tree.map(lambda d: step_size * d, direction)
        
        # Update state
        new_state = BaseOptimizerState(step=state.step + 1)
        
        return updates, new_state


def create_optimizer(direction_provider, update_rule) -> Optimizer:
    """
    Factory for composable optimizers.
    
    Example:
        opt = create_optimizer(
            GradientDirection(),
            ConstantRule(learning_rate=1e-3)
        )
    
    Args:
        direction_provider: Direction computation strategy
        update_rule: Step size computation strategy
        
    Returns:
        Optimizer instance
    """
    return ComposableOptimizer(direction_provider, update_rule)


__all__ = [
    "Optimizer",
    "OptimizerState",
    "BaseOptimizerState",
    "ComposableOptimizer",
    "create_optimizer",
]
