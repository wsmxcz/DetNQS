# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Determinant subspace management for variational calculations.

Manages two determinant spaces:
  - V (Variational space)
  - P (Perturbative space)

Evolution workflow:
  1. Score all determinants in T = V ∪ P
  2. Select top-scored configurations as new V
  3. Rebuild P via Hamiltonian connections

File: detnqs/space/detspace.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .selector import Selector


@dataclass(eq=False)
class DetSpace:
    """
    Determinant subspace container for target space T_k = V_k ∪ P_k.
  
    Attributes:
        V_dets: Variational determinants [n_V, 2], uint64 bitstrings
        P_dets: Perturbative determinants [n_P, 2], None if not built
  
    Invariants:
        - V_dets.shape = (n_V, 2), n_V >= 1
        - P_dets.shape = (n_P, 2) or None
        - V and P are disjoint
    """

    V_dets: np.ndarray
    P_dets: Optional[np.ndarray] = field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Space dimensions
    # -------------------------------------------------------------------------

    @property
    def size_V(self) -> int:
        """Dimension of variational space |V_k|."""
        return int(self.V_dets.shape[0])

    @property
    def size_P(self) -> int:
        """Dimension of perturbative space |P_k| (zero if unbuilt)."""
        if self.P_dets is None:
            return 0
        return int(self.P_dets.shape[0])

    @property
    def size_T(self) -> int:
        """Total dimension |T_k| = |V_k| + |P_k|."""
        return self.size_V + self.size_P

    @property
    def has_P(self) -> bool:
        """Check if perturbative space exists."""
        return self.P_dets is not None and self.size_P > 0

    # -------------------------------------------------------------------------
    # Index ranges
    # -------------------------------------------------------------------------

    @property
    def V_indices(self) -> np.ndarray:
        """Index range [0, n_V) for V-space within T."""
        return np.arange(self.size_V, dtype=np.int32)

    @property
    def P_indices(self) -> np.ndarray:
        """Index range [n_V, n_V + n_P) for P-space within T."""
        return np.arange(self.size_V, self.size_T, dtype=np.int32)

    @property
    def T_dets(self) -> np.ndarray:
        """
        Full target space T_k = V_k ∪ P_k.
      
        Returns:
            Concatenated array [n_T, 2] if P exists, else V_dets
        """
        if self.P_dets is None or self.P_dets.size == 0:
            return self.V_dets
        return np.concatenate([self.V_dets, self.P_dets], axis=0)

    def iter_T(self, block_size: int) -> Iterable[np.ndarray]:
        """
        Yield T_k as CPU blocks without materializing full array.
      
        Useful for streaming H2D transfers when T is too large for VRAM.
        Blocks are contiguous views/slices of underlying arrays.
      
        Args:
            block_size: Maximum size of each determinant block
          
        Yields:
            Determinant blocks [<=block_size, 2] in uint64 format
        """
        bs = int(block_size)
        if bs <= 0:
            raise ValueError("block_size must be positive")

        # Yield V blocks
        for i in range(0, self.size_V, bs):
            yield self.V_dets[i : i + bs]

        # Yield P blocks if present
        if self.P_dets is not None and self.size_P > 0:
            for i in range(0, self.size_P, bs):
                yield self.P_dets[i : i + bs]

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    @classmethod
    def initialize(cls, dets: np.ndarray) -> "DetSpace":
        """
        Create initial space from seed determinant(s).
      
        Args:
            dets: Single determinant [2] or batch [n, 2]
      
        Returns:
            DetSpace with V_dets = dets, P_dets = None
      
        Raises:
            ValueError: If shape is invalid
        """
        arr = np.asarray(dets, dtype=np.uint64)

        if arr.ndim == 1:
            if arr.shape[0] != 2:
                raise ValueError(f"Expected shape (2,), got {arr.shape}")
            arr = arr[None, :]
        elif arr.ndim == 2:
            if arr.shape[1] != 2:
                raise ValueError(f"Expected shape (n, 2), got {arr.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D")

        return cls(V_dets=arr, P_dets=None)

    # -------------------------------------------------------------------------
    # Space evolution
    # -------------------------------------------------------------------------

    def evolve(self, selector: Selector, scores_T: Any) -> "DetSpace":
        """
        Update variational space via scoring and selection on T_k.
      
        Algorithm:
          1. Compute scores w(x) for all x in T_k = V_k ∪ P_k
          2. Select V_{k+1} ⊂ T_k via selector strategy
          3. Return DetSpace(V_{k+1}, P=None) for next outer iteration
      
        Args:
            selector: Selection strategy (TopK/TopFraction/Threshold)
            scores_T: Importance scores on T_k [n_T] or streaming factory
      
        Returns:
            New DetSpace with updated V, unbuilt P (rebuild via H in next cycle)
      
        Notes:
            - P_dets reset to None; must rebuild via Hamiltonian connections
            - scores_T can be ndarray (legacy) or callable factory (streaming)
            - Selector determines new |V_{k+1}| based on score distribution
        """
        if isinstance(scores_T, np.ndarray):
            # Legacy path: materialize T on CPU
            T_dets = self.T_dets
            new_V = selector.select(scores_T, T_dets) if T_dets.shape[0] > 0 else T_dets
        else:
            # Streaming path: selector consumes (scores_block, dets_block) from factory
            new_V = selector.select(scores_T, None)
        return DetSpace(V_dets=new_V, P_dets=None)