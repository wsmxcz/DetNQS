#!/usr/bin/env python3
# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant subspace management for variational calculations.

Manages two determinant spaces:
  - S: Selected space (variational optimization)
  - C: Connected space (perturbative corrections)

Evolution workflow:
  1. Score all determinants in T = S ∪ C
  2. Select top-scored configurations as new S
  3. Rebuild C via Hamiltonian connections

File: lever/space/detspace.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .selector import Selector


@dataclass(eq=False)
class DetSpace:
    """
    Determinant subspace container for S ∪ C.
    
    Attributes:
        S_dets: Selected determinants [n_S, 2], uint64 bitstrings
        C_dets: Connected determinants [n_C, 2], None if not built
    
    Invariants:
        - S_dets.shape = (n_S, 2)
        - C_dets.shape = (n_C, 2) or None
        - T = S ∪ C (disjoint union)
    """

    S_dets: np.ndarray
    C_dets: Optional[np.ndarray] = field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Space dimensions
    # -------------------------------------------------------------------------

    @property
    def size_S(self) -> int:
        """Dimension of S-space: |S|."""
        return int(self.S_dets.shape[0])

    @property
    def size_C(self) -> int:
        """Dimension of C-space: |C| (zero if unbuilt)."""
        if self.C_dets is None:
            return 0
        return int(self.C_dets.shape[0])

    @property
    def size_T(self) -> int:
        """Total dimension: |T| = |S| + |C|."""
        return self.size_S + self.size_C

    @property
    def has_C(self) -> bool:
        """Check if C-space exists."""
        return self.C_dets is not None and self.size_C > 0

    # -------------------------------------------------------------------------
    # Index ranges
    # -------------------------------------------------------------------------

    @property
    def S_indices(self) -> np.ndarray:
        """S-space index range: [0, n_S)."""
        return np.arange(self.size_S, dtype=np.int32)

    @property
    def C_indices(self) -> np.ndarray:
        """C-space index range in T: [n_S, n_S + n_C)."""
        return np.arange(self.size_S, self.size_T, dtype=np.int32)

    @property
    def T_dets(self) -> np.ndarray:
        """
        Full T-space determinants: [S_dets; C_dets].
        
        Returns:
            Concatenated array [n_T, 2] if C exists, else S_dets
        """
        if self.C_dets is None or self.C_dets.size == 0:
            return self.S_dets
        return np.concatenate([self.S_dets, self.C_dets], axis=0)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @classmethod
    def initialize(cls, dets: np.ndarray) -> "DetSpace":
        """
        Create initial space from determinant(s).
        
        Args:
            dets: Single determinant [2] or batch [n, 2]
        
        Returns:
            DetSpace with S_dets = dets, C_dets = None
        
        Raises:
            ValueError: Invalid shape
        """
        arr = np.asarray(dets, dtype=np.uint64)

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(
                    f"Single determinant requires shape (2,), got {arr.shape}"
                )
            arr = arr[None, :]
        elif arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(
                    f"Batch determinants require shape (n, 2), got {arr.shape}"
                )
        else:
            raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")

        return cls(S_dets=arr, C_dets=None)

    # -------------------------------------------------------------------------
    # Space evolution
    # -------------------------------------------------------------------------

    def evolve(self, selector: Selector, scores_T: np.ndarray) -> "DetSpace":
        """
        Update S-space via scoring and selection on T = S ∪ C.
        
        Algorithm:
          1. Concatenate T_dets = [S_dets; C_dets]
          2. Select S_new ⊂ T via selector(scores_T, T_dets)
          3. Return DetSpace(S_new, C_dets=None)
        
        Args:
            selector: Selection strategy (TopK/TopFraction/Threshold)
            scores_T: Importance scores on T-space [n_T]
        
        Returns:
            New DetSpace with updated S, unbuilt C
        
        Notes:
            - C_dets reset to None (rebuild via Hamiltonian in next cycle)
            - Selector determines new |S| based on scores
        """
        T_dets = self.T_dets
        new_S = selector.select(scores_T, T_dets) if T_dets.shape[0] > 0 else T_dets
        return DetSpace(S_dets=new_S, C_dets=None)
