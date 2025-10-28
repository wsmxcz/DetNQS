# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optimizer factory functions with unified interface.

Provides consistent API for creating both LEVER (QGT-based) and Optax
(first-order) optimizers. All factories return Optax-compatible objects.

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Literal

import optax


def adam(
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs
) -> optax.GradientTransformation:
    """
    Create Adam optimizer (delegates to Optax).
    
    Args:
        learning_rate: Step size α
        weight_decay: L2 regularization coefficient λ
        **kwargs: Additional arguments (b1, b2, eps, etc.)
        
    Returns:
        Optax optimizer instance
    """
    if weight_decay > 0:
        return optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    return optax.adam(learning_rate=learning_rate, **kwargs)


def sgd(
    learning_rate: float = 1e-2,
    momentum: float = 0.0,
    **kwargs
) -> optax.GradientTransformation:
    """
    Create SGD optimizer (delegates to Optax).
    
    Args:
        learning_rate: Step size α
        momentum: Momentum coefficient β
        **kwargs: Additional arguments
        
    Returns:
        Optax optimizer instance
    """
    return optax.sgd(learning_rate=learning_rate, momentum=momentum, **kwargs)


def sr(
    damping: float = 1e-4,
    backend: Literal["matvec", "dense"] = "matvec",
    learning_rate: float = 1.0,
    cg_maxiter: int = 100,
    cg_tol: float = 1e-5
):
    """
    Create stochastic reconfiguration / natural gradient optimizer.
    
    Solves (S + λI)δ = -g where S is quantum geometric tensor (QGT).
    Implements imaginary-time evolution dynamics with metric correction.
    
    Args:
        damping: Diagonal regularization λ for stability
        backend: QGT computation strategy
            - "matvec": Matrix-free CG solver (memory efficient)
            - "dense": Direct Cholesky solve (faster for small systems)
        learning_rate: Step size multiplier (typically 1.0 for SR)
        cg_maxiter: Max conjugate gradient iterations (matvec only)
        cg_tol: CG convergence tolerance (matvec only)
        
    Returns:
        LEVER Optimizer with Optax-compatible interface
        
    Example:
        >>> opt = sr(damping=1e-4, backend="matvec")
        >>> state = opt.init(params)
        >>> # Update step requires tape and energy context
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


def lm(damping: float = 1e-3, learning_rate: float = 1.0):
    """
    Create linear method optimizer.
    
    Solves linear system with energy Hessian approximation:
    (H + λI)δ = -g where H ≈ ∂²E/∂θ².
    
    Note: Simplified implementation. Full second-order features pending.
    
    Args:
        damping: Diagonal regularization λ
        learning_rate: Step size multiplier
        
    Returns:
        LEVER Optimizer instance
    """
    from .base import Optimizer
    from .direction import LMDirection
    from .rule import ConstantRule
    
    direction = LMDirection(damping=damping)
    rule = ConstantRule(learning_rate=learning_rate)
    
    return Optimizer(direction, rule)


__all__ = ["adam", "sgd", "sr", "lm"]
