# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified optimizer interface for variational quantum algorithms.

Provides high-level factories (adam, sgd, sr, lm) and low-level components
(DirectionProvider, UpdateRule, Optimizer) for parameter optimization.

Stochastic Reconfiguration (SR):
  - Solves natural gradient: δθ = S⁻¹·g where S = ⟨∂θψ|∂θψ⟩
  - Backends: 'dense' (stable), 'matvec' (memory-efficient)

Example:
    >>> from lever.optimizers import sr, adam, cosine_decay_schedule
    >>> 
    >>> # Natural gradient with damping
    >>> opt = sr(damping=1e-4, backend="matvec")
    >>> 
    >>> # Adam with cosine decay schedule
    >>> lr = cosine_decay_schedule(init_value=1e-3, decay_steps=400)
    >>> opt = adam(learning_rate=lr, weight_decay=1e-4)

Author: Zheng (Alex) Che <wsmxcz@gmail.com>
Date: November 2025
"""

# Recommended user-facing API
from .factories import (
    adam,
    lm,
    sgd,
    sr,
    # Learning rate schedules
    cosine_decay_schedule,
    exponential_decay_schedule,
)

# Advanced components
from .base import (
    DirectionProvider,
    DirectionState,
    Optimizer,
    OptimizerState,
    RuleState,
    UpdateRule,
)
from .direction import GradientDirection, LMDirection, SRDirection, SRState
from .linalg import solve_cg, solve_cholesky
from .rule import ConstantRule, LineSearchRule

__all__ = [
    # High-level factories
    "adam",
    "lm",
    "sgd",
    "sr",
    # Learning rate schedules
    "cosine_decay_schedule",
    "exponential_decay_schedule",
    # Base abstractions
    "DirectionProvider",
    "DirectionState",
    "Optimizer",
    "OptimizerState",
    "RuleState",
    "UpdateRule",
    # Direction implementations
    "GradientDirection",
    "LMDirection",
    "SRDirection",
    "SRState",
    # Update rules
    "ConstantRule",
    "LineSearchRule",
    # Linear algebra solvers
    "solve_cg",
    "solve_cholesky",
]
