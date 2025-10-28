# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Composable optimizers for variational quantum Monte Carlo.

Provides Optax-compatible interface with support for:
  - Gradient descent (Adam, SGD)
  - Natural gradient (SR)
  - Linear method (LM)

File: lever/optimizers/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from .base import Optimizer, OptimizerState, create_optimizer
from .direction import GradientDirection
from .rule import ConstantRule

__all__ = [
    "Optimizer",
    "OptimizerState", 
    "create_optimizer",
    "GradientDirection",
    "ConstantRule",
]
