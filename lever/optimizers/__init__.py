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
    >>> from lever.optimizers import sr, adam
    >>> 
    >>> # Natural gradient with damping
    >>> opt = sr(damping=1e-4, backend="matvec")
    >>> 
    >>> # Standard Adam with weight decay
    >>> opt = adam(learning_rate=5e-4, weight_decay=1e-4)

Author: Zheng (Alex) Che <wsmxcz@gmail.com>
Date: November 2025
"""

# Recommended user-facing API
from .factories import adam, lm, sgd, sr

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
