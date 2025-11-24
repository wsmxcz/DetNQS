# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Feature transformation and wavefunction utilities for LEVER engine.

Provides:
  - Determinant → feature vector conversion
  - Bootstrap amplitude computation  
  - Wavefunction cache creation

File: lever/utils/features.py
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


def masks_to_vecs(dets: jnp.ndarray, n_orb: int) -> jnp.ndarray:
    """
    Convert determinant bitmasks to occupancy vectors (JAX version).
    
    This version operates on JAX arrays and is suitable for use inside
    JIT-compiled code. For host-side preprocessing, prefer the NumPy
    implementation `masks_to_vecs_numpy` to avoid extra JIT compilations.
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")

    dets = dets.astype(jnp.uint64)
    alpha, beta = dets[..., 0], dets[..., 1]
    
    # Extract occupations via bitshift: (det >> k) & 1
    orb_idx = jnp.arange(n_orb, dtype=jnp.uint64)
    alpha_occ = ((alpha[..., None] >> orb_idx) & 1).astype(jnp.float32)
    beta_occ = ((beta[..., None] >> orb_idx) & 1).astype(jnp.float32)
    
    return jnp.concatenate([alpha_occ, beta_occ], axis=-1)


def masks_to_vecs_numpy(dets: np.ndarray, n_orb: int) -> np.ndarray:
    """
    Host-side determinant → feature conversion using NumPy only.
    
    This avoids XLA compilation for bit operations and is used in the
    compilation pipeline where shapes change across outer cycles.
    
    Args:
        dets: (..., 2) uint64 bitmasks [α-string, β-string]
        n_orb: Number of spatial orbitals M
    
    Returns:
        (..., 2M) occupancy vectors in [0,1] as float32 NumPy array
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")

    dets = dets.astype(np.uint64, copy=False)
    alpha = dets[..., 0]
    beta = dets[..., 1]

    orb_idx = np.arange(n_orb, dtype=np.uint64)  # [M]

    # (..., 1) >> [M] → (..., M)
    alpha_occ = ((alpha[..., None] >> orb_idx) & 1).astype(np.float32)
    beta_occ = ((beta[..., None] >> orb_idx) & 1).astype(np.float32)

    return np.concatenate([alpha_occ, beta_occ], axis=-1)


def dets_to_features(
    dets: np.ndarray,
    n_orb: int
) -> jnp.ndarray:
    """
    Convert CPU determinants to device feature tensor.
    
    Host-side bit operations (NumPy) are used to avoid JAX compiling
    the bit-manipulation kernels for every new (S, C) shape.
    
    Args:
        dets: NumPy determinants, shape (n, 2)
        n_orb: Number of spatial orbitals
        
    Returns:
        JAX features, shape (n, 2·n_orb)
    """
    if len(dets) == 0:
        return jnp.empty((0, 2 * n_orb), dtype=jnp.float32)
    
    # Compute features on CPU, then transfer a dense float32 tensor once
    feats_np = masks_to_vecs_numpy(dets, n_orb)
    return jax.device_put(feats_np)


def compute_normalized_amplitudes(
    dets: np.ndarray,
    params: PyTree,
    log_psi_fn: Callable,
    n_orb: int,
    eps: float = 1e-14
) -> np.ndarray:
    """
    Compute L2-normalized wavefunction amplitudes.
    
    Bootstrap utility: evaluates log ψ on determinants and returns
    |ψᵢ|/||ψ||₂ for dynamic screening initialization.
    
    Args:
        dets: Determinant bitmasks, shape (n, 2)
        params: Neural network parameters
        log_psi_fn: Model evaluation function
        n_orb: Number of spatial orbitals
        eps: Numerical threshold for zero norm
        
    Returns:
        Normalized amplitudes |ψᵢ|/√(Σ|ψⱼ|²), shape (n,)
    """
    features = dets_to_features(dets, n_orb)
    log_psi = log_psi_fn(params, features)
    psi = jnp.abs(jnp.exp(log_psi))
    
    norm = jnp.linalg.norm(psi)
    normalized = psi / jnp.where(norm > eps, norm, 1.0)
    
    return np.array(jax.device_get(normalized))


def create_psi_cache(
    log_psi: jnp.ndarray,
    n_s: int,
    n_c: int
) -> PsiCache:
    """
    Create wavefunction cache from log amplitudes.
    
    Computes ψ = exp(log ψ) and packages into PsiCache structure.
    
    Args:
        log_psi: Log wavefunction, shape (n_s + n_c,)
        n_s: S-space dimension
        n_c: C-space dimension
        
    Returns:
        PsiCache with amplitudes and space partitioning
    """
    from ..dtypes import PsiCache
    
    psi_all = jnp.exp(log_psi)
    
    return PsiCache(
        log_psi_all=log_psi,
        psi_all=psi_all,
        n_s=n_s
    )


__all__ = [
    "masks_to_vecs",
    "masks_to_vecs_numpy",
    "dets_to_features",
    "compute_normalized_amplitudes",
    "create_psi_cache",
]