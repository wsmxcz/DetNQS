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

from .features import (
    compute_normalized_amplitudes,
    create_psi_cache,
    dets_to_features,
    masks_to_vecs,
)
from .jax_utils import (
    tree_add,
    tree_dot,
    tree_norm,
    tree_scale,
    tree_sub,
)
from .space_utils import (
    count_overlaps,
    get_space_overlap,
    merge_spaces,
    remove_overlaps,
)

__all__ = [
    # Features
    "masks_to_vecs",
    "dets_to_features",
    "compute_normalized_amplitudes",
    "create_psi_cache",
    
    # JAX utilities
    "tree_dot",
    "tree_scale",
    "tree_add",
    "tree_sub",
    "tree_norm",
    
    # Space operations
    "remove_overlaps",
    "merge_spaces",
    "get_space_overlap",
    "count_overlaps",
    
]
