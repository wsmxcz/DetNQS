# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic feature transformations for quantum determinants.

Converts bitmask representations to neural network input features and
computes normalized wavefunction amplitudes. All operations are pure
functions without stochastic sampling.

File: lever/engine/features.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from ..dtypes import PyTree, PsiCache


# ============================================================================
# Bitmask → Occupancy Vector Conversion
# ============================================================================

def masks_to_vecs(dets: jnp.ndarray, n_orb: int) -> jnp.ndarray:
    """
    Convert determinant bitmasks to occupancy vectors (device version).
    
    Extracts occupation numbers from uint64 bitmasks via bitwise operations:
        occ[i] = (mask >> i) & 1
    
    Args:
        dets: Determinant bitmasks [..., 2] with [α, β] spin channels
        n_orb: Number of spatial orbitals
    
    Returns:
        Occupancy vectors [..., 2*n_orb] in float32
    
    Raises:
        ValueError: If dets.shape[-1] ≠ 2
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")
    
    dets = dets.astype(jnp.uint64)
    alpha, beta = dets[..., 0], dets[..., 1]
    
    orb_idx = jnp.arange(n_orb, dtype=jnp.uint64)
    alpha_occ = ((alpha[..., None] >> orb_idx) & 1).astype(jnp.float32)
    beta_occ = ((beta[..., None] >> orb_idx) & 1).astype(jnp.float32)
    
    return jnp.concatenate([alpha_occ, beta_occ], axis=-1)


def masks_to_vecs_numpy(dets: np.ndarray, n_orb: int) -> np.ndarray:
    """
    Convert determinant bitmasks to occupancy vectors (host version).
    
    Pure NumPy implementation avoids JIT recompilation overhead for
    varying (n_s, n_c) dimensions during runtime determinant updates.
    
    Args:
        dets: Determinant bitmasks [..., 2] with [α, β] spin channels
        n_orb: Number of spatial orbitals
    
    Returns:
        Occupancy vectors [..., 2*n_orb] in float32
    
    Raises:
        ValueError: If dets.shape[-1] ≠ 2
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")
    
    dets = dets.astype(np.uint64, copy=False)
    alpha, beta = dets[..., 0], dets[..., 1]
    
    orb_idx = np.arange(n_orb, dtype=np.uint64)
    alpha_occ = ((alpha[..., None] >> orb_idx) & 1).astype(np.float32)
    beta_occ = ((beta[..., None] >> orb_idx) & 1).astype(np.float32)
    
    return np.concatenate([alpha_occ, beta_occ], axis=-1)


def dets_to_features(dets: np.ndarray, n_orb: int) -> jnp.ndarray:
    """
    Convert host determinants to device feature tensor.
    
    Performs bitmask extraction on CPU then transfers to accelerator,
    minimizing JIT compilation overhead.
    
    Args:
        dets: Host determinant bitmasks [n, 2]
        n_orb: Number of spatial orbitals
    
    Returns:
        Device feature tensor [n, 2*n_orb]
    """
    if len(dets) == 0:
        return jnp.empty((0, 2 * n_orb), dtype=jnp.float32)
    
    feats_np = masks_to_vecs_numpy(dets, n_orb)
    return jax.device_put(feats_np)


# ============================================================================
# Wavefunction Amplitude Computation
# ============================================================================

def compute_normalized_amplitudes(
    dets: np.ndarray,
    params: PyTree,
    log_psi_fn: Callable,
    n_orb: int,
    eps: float = 1e-14,
) -> np.ndarray:
    """
    Compute L2-normalized wavefunction amplitudes on fixed determinant set.
    
    Normalization: |ψ̃ᵢ| = |ψᵢ| / √(Σⱼ |ψⱼ|²)
    
    Args:
        dets: Determinant bitmasks [n, 2]
        params: Neural network parameters
        log_psi_fn: Model evaluator (params, features) → log(ψ)
        n_orb: Number of spatial orbitals
        eps: Numerical threshold for zero norm protection
    
    Returns:
        Normalized amplitudes [n] on host
    """
    features = dets_to_features(dets, n_orb)
    log_psi = log_psi_fn(params, features)
    psi = jnp.abs(jnp.exp(log_psi))
    
    norm = jnp.linalg.norm(psi)
    norm = jnp.where(norm > eps, norm, 1.0)
    normalized = psi / norm
    
    return np.array(jax.device_get(normalized))


def create_psi_cache(
    log_psi: jnp.ndarray,
    n_s: int,
    n_c: int,
) -> PsiCache:
    """
    Create cached wavefunction amplitudes with space partitioning metadata.
    
    Stores both log and linear amplitudes for efficient gradient computation
    and probability calculations.
    
    Args:
        log_psi: Log wavefunction values [n_s + n_c]
        n_s: S-space dimension
        n_c: C-space dimension
    
    Returns:
        PsiCache with full amplitudes and partitioning info
    """
    from ..dtypes import PsiCache
    
    psi_all = jnp.exp(log_psi)
    return PsiCache(
        log_psi_all=log_psi,
        psi_all=psi_all,
        n_s=n_s,
    )


__all__ = [
    "masks_to_vecs",
    "masks_to_vecs_numpy",
    "dets_to_features",
    "compute_normalized_amplitudes",
    "create_psi_cache",
]
