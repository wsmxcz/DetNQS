# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for LEVER framework.

Provides JAX operations, space manipulations, feature transformations,
and logging infrastructure.

File: lever/utils/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from .jax_utils import (
    tree_add,
    tree_dot,
    tree_norm,
    tree_scale,
    tree_sub,
)
__all__ = [
    # JAX utilities
    "tree_dot",
    "tree_scale",
    "tree_add",
    "tree_sub",
    "tree_norm",
]
