# Copyright 2025 The LEVER Authors
# SPDX-License-Identifier: Apache-2.0

"""
Determinant encoding utilities for LEVER.

Provides CPU/GPU functions to convert 64-bit determinant bitstrings into
spin-orbital index representations and binary feature vectors used by neural models.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import jax.numpy as jnp


def bitstrings_to_indices(
    bitstrings: np.ndarray,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    dtype: Any = np.uint8,
) -> np.ndarray:
    """
    Decode determinant bitstrings into spin-orbital indices.
    
    Each determinant is represented as [alpha_word, beta_word] 64-bit integers.
    Converts to spin-orbital indices following:
        alpha: 0..n_orb-1
        beta: n_orb..2*n_orb-1
    
    Args:
        bitstrings: [N, 2] uint64 array of determinant encodings
        n_orb: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
        dtype: Output dtype (default uint8 for memory efficiency)

    Returns:
        occ_so: [N, n_alpha+n_beta] array of spin-orbital indices
    """
    bitstrings = np.asarray(bitstrings, dtype=np.uint64)
    if bitstrings.ndim != 2 or bitstrings.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), got {bitstrings.shape}")

    n_det = bitstrings.shape[0]
    n_elec = n_alpha + n_beta
    
    # Validate dtype range
    max_index = 2 * n_orb - 1
    if np.issubdtype(np.dtype(dtype), np.integer):
        if max_index > np.iinfo(dtype).max:
            raise ValueError(
                f"dtype {dtype} cannot hold max index {max_index}"
            )

    # Bit masks for each spatial orbital
    masks = np.uint64(1) << np.arange(n_orb, dtype=np.uint64)

    alpha_bits = bitstrings[:, 0:1]  # (N, 1)
    beta_bits = bitstrings[:, 1:2]

    # Compute occupation matrices
    alpha_occ = (alpha_bits & masks) != 0  # (N, n_orb)
    beta_occ = (beta_bits & masks) != 0

    # Extract occupied orbital indices
    rows_a, cols_a = np.nonzero(alpha_occ)
    rows_b, cols_b = np.nonzero(beta_occ)

    # Validate electron counts
    if cols_a.size != n_det * n_alpha:
        raise ValueError(
            f"Alpha count mismatch: {cols_a.size} != {n_det * n_alpha}"
        )
    if cols_b.size != n_det * n_beta:
        raise ValueError(
            f"Beta count mismatch: {cols_b.size} != {n_det * n_beta}"
        )

    # Reshape to per-determinant indices
    alpha_idx = cols_a.reshape(n_det, n_alpha)
    beta_idx = cols_b.reshape(n_det, n_beta)

    # Pack spin-orbital indices: alpha then beta shifted by n_orb
    occ_so = np.empty((n_det, n_elec), dtype=dtype)
    occ_so[:, :n_alpha] = alpha_idx.astype(dtype, copy=False)
    occ_so[:, n_alpha:] = (beta_idx + n_orb).astype(dtype, copy=False)
    return occ_so


def bitstrings_to_vectors(
    bitstrings: jnp.ndarray,
    n_orb: int,
    dtype: Any = jnp.float64,
) -> jnp.ndarray:
    """
    Convert determinant bitstrings to binary occupation features.
    
    Extracts occupation numbers via bitwise AND with orbital masks.
    Output format: [α_1...α_n β_1...β_n] where α_i, β_i ∈ {0,1}
    
    Args:
        bitstrings: [N, 2] uint64 array, columns = [alpha, beta]
        n_orb: Number of spatial orbitals (must be ≤ 64)
        dtype: Output feature dtype

    Returns:
        Occupation features [N, 2*n_orb] with binary values
    """
    alpha_bits, beta_bits = bitstrings[:, 0], bitstrings[:, 1]
    masks = jnp.uint64(1) << jnp.arange(n_orb, dtype=jnp.uint64)

    def extract(bits: jnp.ndarray) -> jnp.ndarray:
        occupied = (bits[:, None] & masks) > 0
        return occupied.astype(dtype)

    alpha_features = extract(alpha_bits)  # [N, n_orb]
    beta_features = extract(beta_bits)    # [N, n_orb]
    
    return jnp.concatenate([alpha_features, beta_features], axis=1)
