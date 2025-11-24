# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base protocols and state definitions for LEVER optimizers.

Defines functional interfaces following Optax conventions:
  - DirectionProvider: gradient → search direction (g → δ)
  - UpdateRule: direction → step size (δ → α)
  - Optimizer: DirectionProvider + UpdateRule

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import flax.struct

if TYPE_CHECKING:
    from ..dtypes import PyTree, GeometryTape


# ============================================================================
# State Definitions
# ============================================================================

@flax.struct.dataclass
class DirectionState:
    """
    Base state for direction computation.
    
    Subclasses extend with algorithm-specific fields (e.g., CG history).
    Empty base enables stateless providers (e.g., gradient descent).
    """
    pass


@flax.struct.dataclass
class RuleState:
    """
    Base state for update rules.
    
    Subclasses extend with algorithm-specific fields (e.g., line search).
    Empty base enables stateless rules (e.g., constant learning rate).
    """
    pass


@flax.struct.dataclass
class OptimizerState:
    """
    Complete optimizer state: direction + rule components + step counter.
    
    Attributes:
        direction_state: DirectionProvider state
        rule_state: UpdateRule state
        step: Global iteration counter
    """
    direction_state: DirectionState
    rule_state: RuleState
    step: int = 0


# ============================================================================
# Provider Protocols
# ============================================================================

class DirectionProvider(Protocol):
    """
    Computes search direction δ from gradient g.
    
    Functional interface: stateless, thread-safe, JIT-compatible.
    First-order: δ = -g (ignores tape)
    Second-order: δ = -F⁻¹g where F is QGT (requires tape)
    """
    
    def __call__(
        self,
        grad: PyTree,
        state: DirectionState,
        *,
        tape: GeometryTape | None = None
    ) -> tuple[PyTree, DirectionState]:
        """
        Compute search direction from gradient.
        
        Args:
            grad: Gradient ∇E (PyTree structure)
            state: Algorithm state (e.g., CG history)
            tape: Geometry tape for QGT computation (optional)
            
        Returns:
            direction: Search direction δ
            new_state: Updated algorithm state
        """
        ...


class UpdateRule(Protocol):
    """
    Computes step size α from search direction δ.
    
    Functional interface: stateless, thread-safe, JIT-compatible.
    Simple: α = const or α = f(step)
    Adaptive: α = argmin E(θ - αδ) via line search
    """
    
    def __call__(
        self,
        direction: PyTree,
        state: RuleState,
        energy: float,
        step: int,
    ) -> tuple[float, RuleState]:
        """
        Compute step size from search direction.
        
        Args:
            direction: Search direction δ
            state: Algorithm state (e.g., previous α)
            energy: Current energy E(θ) for adaptive rules
            step: Global iteration counter (0-based)
            
        Returns:
            step_size: Scalar α
            new_state: Updated algorithm state
        """
        ...


# ============================================================================
# Optimizer Class
# ============================================================================

class Optimizer:
    """
    Optimizer = DirectionProvider ∘ UpdateRule.
    
    Follows Optax functional API:
      - init(params) → state
      - update(grad, state, **ctx) → (updates, new_state)
    
    Update formula: θ_{t+1} = θ_t - α_t δ_t
      where δ_t = direction(∇E_t) and α_t = rule(δ_t, E_t)
    
    Example:
        >>> opt = Optimizer(
        ...     direction=SRDirection(backend="matvec"),
        ...     rule=ConstantRule(lr=1e-3)
        ... )
        >>> state = opt.init(params)
        >>> updates, state = opt.update(grad, state, params,
        ...                              tape=tape, energy=E)
        >>> params = optax.apply_updates(params, updates)
    """
    
    def __init__(
        self,
        direction: DirectionProvider,
        rule: UpdateRule
    ) -> None:
        """
        Initialize optimizer with direction and rule strategies.
        
        Args:
            direction: Search direction provider (Gradient/SR/LM)
            rule: Step size controller (Constant/LineSearch)
        """
        self.direction = direction
        self.rule = rule
    
    def init(self, params: PyTree) -> OptimizerState:
        """
        Initialize optimizer state (structure inference only).
        
        Args:
            params: Initial parameters
            
        Returns:
            Initial state with step=0
        """
        return OptimizerState(
            direction_state=DirectionState(),
            rule_state=RuleState(),
            step=0
        )
    
    def update(
        self,
        grad: PyTree,
        state: OptimizerState,
        params: PyTree,  # API compatibility, unused
        *,
        tape: GeometryTape | None = None,
        energy: float = 0.0
    ) -> tuple[PyTree, OptimizerState]:
        """
        Compute parameter updates: Δθ = -α·δ.
        
        Two-stage process:
          1. δ = direction(g, state_dir, tape)
          2. α = rule(δ, state_rule, E)
          3. Δθ = α·δ
        
        Args:
            grad: Gradient ∇E
            state: Current optimizer state
            params: Current parameters (unused)
            tape: Geometry tape for QGT (optional)
            energy: Current energy E(θ) (optional)
            
        Returns:
            updates: Parameter updates Δθ
            new_state: Updated optimizer state
        """
        # Compute search direction δ
        direction, new_dir_state = self.direction(
            grad,
            state.direction_state,
            tape=tape
        )
        
        # Compute step size α
        step_size, new_rule_state = self.rule(
            direction,
            state.rule_state,
            energy,
            state.step,
        )
        
        # Scale direction: Δθ = α·δ
        from ..utils.jax_utils import tree_scale
        updates = tree_scale(direction, step_size)
        
        # Update state
        new_state = OptimizerState(
            direction_state=new_dir_state,
            rule_state=new_rule_state,
            step=state.step + 1
        )
        
        return updates, new_state


__all__ = [
    "DirectionProvider",
    "DirectionState",
    "UpdateRule",
    "RuleState",
    "OptimizerState",
    "Optimizer",
]
