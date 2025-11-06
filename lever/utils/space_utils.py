# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant space manipulation utilities.

Provides set operations on determinant collections represented as
NumPy uint64 arrays with shape (n_dets, 2) encoding [α-string, β-string].

File: lever/utils/space_utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def remove_overlaps(
    target: NDArray[np.uint64],
    exclude: NDArray[np.uint64]
) -> NDArray[np.uint64]:
    """
    Remove determinants from target that appear in exclude set.
    
    Uses set-based lookup with O(|exclude| + |target|) complexity.
    Preserves original ordering of target determinants.
    
    Args:
        target: Determinants to filter, shape (n, 2)
        exclude: Determinants to remove, shape (m, 2)
        
    Returns:
        Filtered determinants, shape (k≤n, 2)
    """
    if len(target) == 0:
        return np.empty((0, 2), dtype=np.uint64)
    
    if len(exclude) == 0:
        return target.copy()
    
    # Build exclusion set from (α, β) tuples
    exclude_set = {(int(d[0]), int(d[1])) for d in exclude}
    
    # Filter preserving order
    filtered = [d for d in target if (int(d[0]), int(d[1])) not in exclude_set]
    
    return (
        np.array(filtered, dtype=np.uint64)
        if filtered
        else np.empty((0, 2), dtype=np.uint64)
    )


def merge_spaces(
    *spaces: NDArray[np.uint64],
    sort: bool = False
) -> NDArray[np.uint64]:
    """
    Merge multiple determinant spaces with deduplication.
    
    Concatenates input arrays and removes duplicates based on
    (α, β) tuple uniqueness.
    
    Args:
        *spaces: Variable number of determinant arrays, each shape (n_i, 2)
        sort: Whether to sort output by (α, β) lexicographic order
        
    Returns:
        Merged unique determinants, shape (n_unique, 2)
    """
    # Filter empty arrays
    non_empty = [s for s in spaces if len(s) > 0]
    
    if not non_empty:
        return np.empty((0, 2), dtype=np.uint64)
    
    if len(non_empty) == 1:
        return np.unique(non_empty[0], axis=0)
    
    # Concatenate and deduplicate
    merged = np.concatenate(non_empty, axis=0)
    unique = np.unique(merged, axis=0)
    
    if sort:
        # Lexicographic sort: (α, β)
        idx = np.lexsort((unique[:, 1], unique[:, 0]))
        return unique[idx]
    
    return unique


def get_space_overlap(
    space_a: NDArray[np.uint64],
    space_b: NDArray[np.uint64]
) -> NDArray[np.uint64]:
    """
    Find determinants present in both spaces (set intersection).
    
    Args:
        space_a: First determinant space, shape (n, 2)
        space_b: Second determinant space, shape (m, 2)
        
    Returns:
        Overlapping determinants, shape (k, 2)
    """
    if len(space_a) == 0 or len(space_b) == 0:
        return np.empty((0, 2), dtype=np.uint64)
    
    set_b = {(int(d[0]), int(d[1])) for d in space_b}
    overlap = [d for d in space_a if (int(d[0]), int(d[1])) in set_b]
    
    return (
        np.array(overlap, dtype=np.uint64)
        if overlap
        else np.empty((0, 2), dtype=np.uint64)
    )


def count_overlaps(
    space_a: NDArray[np.uint64],
    space_b: NDArray[np.uint64]
) -> int:
    """
    Count determinants present in both spaces.
    
    Fast counting without materializing overlap array.
    
    Args:
        space_a: First determinant space, shape (n, 2)
        space_b: Second determinant space, shape (m, 2)
        
    Returns:
        Number of overlapping determinants
    """
    if len(space_a) == 0 or len(space_b) == 0:
        return 0
    
    set_b = {(int(d[0]), int(d[1])) for d in space_b}
    return sum(1 for d in space_a if (int(d[0]), int(d[1])) in set_b)


__all__ = [
    "remove_overlaps",
    "merge_spaces",
    "get_space_overlap",
    "count_overlaps",
]
