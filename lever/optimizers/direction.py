# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Direction providers for optimization.

Phase 1: Gradient descent
Phase 2+: SR, LM

File: lever/optimizers/direction.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..engine.geometry import GeometryTape
from ..utils.dtypes import PyTree


class GradientDirection:
    """
    Standard gradient descent direction: δ = -∇E.
    
    Does not consume geometry tape (zero QGT overhead).
    Compatible with first-order methods (Adam, SGD).
    """
    
    def __call__(
        self,
        grad: PyTree,
        tape: GeometryTape | None = None
    ) -> PyTree:
        """
        Return negative gradient.
        
        Args:
            grad: Parameter gradient
            tape: Ignored in Phase 1
            
        Returns:
            Direction: -grad
        """
        return jax.tree.map(jnp.negative, grad)


__all__ = ["GradientDirection"]
