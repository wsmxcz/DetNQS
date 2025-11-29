# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer factory with unified Optax-compatible interface.

Provides both LEVER (QGT-based) and first-order optimizers.
All factories return Optax-compatible GradientTransformation objects.

File: lever/optim/factory.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Literal

import optax
from ..utils.config_utils import capture_config


# ============================================================================
# Learning Rate Schedules
# ============================================================================

@capture_config
def cosine_decay_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0,
):
    """
    Cosine annealing schedule: η(t) = α + (η₀ - α) · [1 + cos(πt/T)] / 2.
    
    Args:
        init_value: Initial learning rate η₀
        decay_steps: Total steps T
        alpha: Minimum learning rate (default: 0)
    """
    return optax.cosine_decay_schedule(
        init_value=init_value,
        decay_steps=decay_steps,
        alpha=alpha,
    )


@capture_config
def exponential_decay_schedule(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    staircase: bool = False,
):
    """
    Exponential decay: η(t) = η₀ · γ^(t/T).
    
    Args:
        init_value: Initial learning rate η₀
        transition_steps: Decay period T
        decay_rate: Decay factor γ
        staircase: Use floor(t/T) instead of t/T
    """
    return optax.exponential_decay(
        init_value=init_value,
        transition_steps=transition_steps,
        decay_rate=decay_rate,
        staircase=staircase,
    )


# ============================================================================
# First-Order Optimizers
# ============================================================================

@capture_config
def adam(
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> optax.GradientTransformation:
    """
    Adam optimizer with optional weight decay (AdamW).
    
    Update rule: θ ← θ - α · m̂ / (√v̂ + ε)
    where m̂, v̂ are bias-corrected first/second moment estimates.
    
    Args:
        learning_rate: Step size α
        weight_decay: L2 regularization λ (enables AdamW if > 0)
        **kwargs: b1, b2, eps, etc.
    """
    if weight_decay > 0:
        return optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    return optax.adam(learning_rate=learning_rate, **kwargs)


@capture_config
def sgd(
    learning_rate: float = 1e-2,
    momentum: float = 0.0,
    **kwargs
) -> optax.GradientTransformation:
    """
    Stochastic gradient descent with momentum.
    
    Update rule: v ← β·v + g, θ ← θ - α·v
    
    Args:
        learning_rate: Step size α
        momentum: Momentum coefficient β ∈ [0, 1)
        **kwargs: nesterov, etc.
    """
    return optax.sgd(learning_rate=learning_rate, momentum=momentum, **kwargs)


# ============================================================================
# Quantum Natural Gradient Optimizers
# ============================================================================

@capture_config
def sr(
    damping: float = 1e-4,
    backend: Literal["matvec", "dense"] = "matvec",
    learning_rate: float = 1.0,
    cg_maxiter: int = 100,
    cg_tol: float = 1e-4
):
    """
    Stochastic reconfiguration (quantum natural gradient).
    
    Solves (S + λI)δ = -g where S is the quantum geometric tensor (QGT).
    Implements imaginary-time evolution with metric-aware updates.
    
    Args:
        damping: Diagonal regularization λ for numerical stability
        backend: QGT computation strategy
            - "matvec": Matrix-free conjugate gradient (memory efficient)
            - "dense": Direct Cholesky decomposition (faster for small nets)
        learning_rate: Step size multiplier (typically 1.0 for natural gradient)
        cg_maxiter: Max CG iterations (matvec backend only)
        cg_tol: CG convergence tolerance (matvec backend only)
        
    Returns:
        LEVER Optimizer with Optax-compatible interface
        
    Example:
        >>> opt = sr(damping=1e-4, backend="matvec")
        >>> state = opt.init(params)
        >>> updates, state = opt.update(grad, state, params, tape=tape, energy=E)
    """
    from .base import Optimizer
    from .direction import SRDirection
    from .rule import ConstantRule
    
    direction = SRDirection(
        backend=backend,
        damping=damping,
        cg_maxiter=cg_maxiter,
        cg_tol=cg_tol
    )
    rule = ConstantRule(learning_rate=learning_rate)
    
    return Optimizer(direction, rule)


@capture_config
def lm(damping: float = 1e-3, learning_rate: float = 1.0):
    """
    Linear method optimizer (approximate second-order).
    
    Solves (H + λI)δ = -g where H approximates the energy Hessian.
    Note: Simplified implementation; full second-order features pending.
    
    Args:
        damping: Diagonal regularization λ
        learning_rate: Step size multiplier
    """
    from .base import Optimizer
    from .direction import LMDirection
    from .rule import ConstantRule
    
    direction = LMDirection(damping=damping)
    rule = ConstantRule(learning_rate=learning_rate)
    
    return Optimizer(direction, rule)


__all__ = ["cosine_decay_schedule", "exponential_decay_schedule", 
           "adam", "sgd", "sr", "lm"]
