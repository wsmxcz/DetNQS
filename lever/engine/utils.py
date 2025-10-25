# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core data structures and utility functions for the LEVER engine.

File: lever/engine/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Type Aliases
# ============================================================================

PyTree = Any
MatVecOp = Callable[[PyTree], PyTree]

# ============================================================================
# Hilbert Space & Hamiltonian Data Structures
# ============================================================================

@dataclass(frozen=True, slots=True)
class HamOp:
    """
    Sparse Hamiltonian block in COO format (host memory).
    
    Used for Numba kernel invocations, not JAX-jitted computations.
    """
    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    vals: NDArray[np.float64]
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        return self.vals.size


@dataclass(frozen=True, slots=True)
class SpaceRep:
    """
    Hilbert space representation (host memory).
    
    Static within single optimization cycle.
    """
    s_dets: NDArray[np.uint64]
    c_dets: NDArray[np.uint64]
    H_diag_S: NDArray[np.float64]
    H_diag_C: NDArray[np.float64]

    @property
    def size_S(self) -> int:
        return self.s_dets.shape[0]

    @property
    def size_C(self) -> int:
        return self.c_dets.shape[0]


# ============================================================================
# JAX-Compatible Data Structures
# ============================================================================

class Contractions(NamedTuple):
    """
    Hamiltonian-wavefunction products.
    
    For EFFECTIVE mode: N_SC/N_CS/N_CC are None.
    """
    N_SS: jnp.ndarray
    N_SC: jnp.ndarray | None
    N_CS: jnp.ndarray | None
    N_CC: jnp.ndarray | None


class GradientResult(NamedTuple):
    """Energy and gradient computation result."""
    gradient: PyTree
    energy_elec: jnp.ndarray


class SOperatorMetadata(NamedTuple):
    """S-operator (Quantum Geometric Tensor) diagnostics."""
    total_norm: float
    mean_deriv_residual: float


class SubspaceMetadata(NamedTuple):
    """Linear Method subspace projection diagnostics."""
    subspace_dim: int
    condition_number: float
    min_eigenvalue: float
    jitter: float


# ============================================================================
# Utility Functions
# ============================================================================

def masks_to_vecs(dets: jnp.ndarray, n_orbitals: int) -> jnp.ndarray:
    """
    Convert determinant bitmasks to spin-orbital occupancy vectors.
    
    Args:
        dets: shape (..., 2), dtype uint64
        n_orbitals: Number of spatial orbitals
    
    Returns:
        Occupancy vectors, shape (..., 2*n_orbitals), dtype float32
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")

    dets = dets.astype(jnp.uint64)
    alpha_mask, beta_mask = dets[..., 0], dets[..., 1]

    orb_indices = jnp.arange(n_orbitals, dtype=jnp.uint64)
    alpha_occ = ((alpha_mask[..., None] >> orb_indices) & 1).astype(jnp.float32)
    beta_occ = ((beta_mask[..., None] >> orb_indices) & 1).astype(jnp.float32)

    return jnp.concatenate([alpha_occ, beta_occ], axis=-1)
