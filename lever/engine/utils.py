# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core data structures and utility functions for the LEVER engine.

This module serves as the central repository for all custom data types
(e.g., HamOp, SpaceRep) and common helper functions used across the engine.
Centralizing these definitions prevents circular dependencies and promotes
a consistent data model.

File: lever/engine/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025 (Refactored)
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

# A PyTree is a nested structure of containers (e.g., dicts, lists, tuples)
# and leaves (e.g., JAX arrays). Used for model parameters and gradients.
PyTree = Any
MatVecOp = Callable[[PyTree], PyTree]

# ============================================================================
# Hilbert Space & Hamiltonian Data Structures
# ============================================================================

@dataclass(frozen=True, slots=True)
class HamOp:
    """
    Sparse Hamiltonian block in COO format, residing on the host (CPU).

    This structure is designed to be passed to Numba kernels and is not
    intended for direct use in JAX-jitted computations.

    Attributes:
        rows: Row indices (int32).
        cols: Column indices (int32).
        vals: Matrix elements H_ij (float64).
        shape: Logical dimensions (n_rows, n_cols).
    """
    rows: NDArray[np.int32]
    cols: NDArray[np.int32]
    vals: NDArray[np.float64]
    shape: tuple[int, int]

    @property
    def nnz(self) -> int:
        """Number of stored non-zero elements."""
        return self.vals.size


@dataclass(frozen=True, slots=True)
class SpaceRep:
    """
    Hilbert space representation for S and C subspaces, residing on the host.

    Stores determinant lists and diagonal Hamiltonian elements, which are
    static within a single optimization cycle.

    Attributes:
        s_dets: S-space determinants, shape (|S|, 2), uint64.
        c_dets: C-space determinants, shape (|C|, 2), uint64.
        H_diag_S: Diagonal elements for S-space, shape (|S|,).
        H_diag_C: Diagonal elements for C-space, shape (|C|,).
    """
    s_dets: NDArray[np.uint64]
    c_dets: NDArray[np.uint64]
    H_diag_S: NDArray[np.float64]
    H_diag_C: NDArray[np.float64]

    @property
    def size_S(self) -> int:
        """Dimension of S-space."""
        return self.s_dets.shape[0]

    @property
    def size_C(self) -> int:
        """Dimension of C-space."""
        return self.c_dets.shape[0]


# ============================================================================
# JAX-Compatible Data Structures for Computation
# ============================================================================

class Contractions(NamedTuple):
    """
    Hamiltonian-wavefunction contraction results as JAX arrays.
    These are the outputs of the Numba kernels, brought back into the JAX world.
    """
    N_SS: jnp.ndarray  # H_SS @ psi_S
    N_SC: jnp.ndarray  # H_SC @ psi_C
    N_CS: jnp.ndarray  # H_CS @ psi_S
    N_CC: jnp.ndarray  # H_CC & psi_C


class GradientResult(NamedTuple):
    """Output of the unified gradient and energy computation."""
    gradient: PyTree
    energy_elec: jnp.ndarray


class SOperatorMetadata(NamedTuple):
    """Diagnostics for the S-operator (Quantum Geometric Tensor) construction."""
    total_norm: float
    mean_deriv_residual: float


class SubspaceMetadata(NamedTuple):
    """Diagnostics for Linear Method subspace projection."""
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

    Expands compact uint64 alpha/beta configurations into explicit binary
    occupancy vectors for use as neural network features.

    Args:
        dets: Determinant configurations, shape (..., 2).
        n_orbitals: Number of spatial orbitals.

    Returns:
        Occupancy vectors, shape (..., 2 * n_orbitals).
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected last dimension of dets to be 2, got {dets.shape}")

    dets = dets.astype(jnp.uint64)
    alpha_mask, beta_mask = dets[..., 0], dets[..., 1]

    # Use bitwise operations to extract occupancy for each orbital.
    orb_indices = jnp.arange(n_orbitals, dtype=jnp.uint64)
    alpha_occ = ((alpha_mask[..., None] >> orb_indices) & 1).astype(jnp.float32)
    beta_occ = ((beta_mask[..., None] >> orb_indices) & 1).astype(jnp.float32)

    return jnp.concatenate([alpha_occ, beta_occ], axis=-1)