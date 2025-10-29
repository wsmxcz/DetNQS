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
    from .dtypes import PyTree, PsiCache


def masks_to_vecs(dets: jnp.ndarray, n_orb: int) -> jnp.ndarray:
    """
    Convert determinant bitmasks to occupancy vectors.
    
    Maps bit representations to neural network features:
      |det⟩ = |i₁...iₐ⟩⊗|j₁...jᵦ⟩ → [n₁ᵅ,...,nₘᵅ, n₁ᵝ,...,nₘᵝ]
    where nₖᵠ ∈ {0,1} denotes spin-σ occupation of orbital k.
    
    Args:
        dets: (..., 2) uint64 bitmasks [α-string, β-string]
        n_orb: Number of spatial orbitals M
    
    Returns:
        (..., 2M) occupancy vectors in [0,1]
    
    Example:
        >>> dets = jnp.array([[0b1011, 0b0110]])  # 4 orbitals
        >>> masks_to_vecs(dets, 4)
        array([[1,1,0,1, 0,1,1,0]], dtype=float32)
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


def dets_to_features(
    dets: np.ndarray,
    n_orb: int
) -> jnp.ndarray:
    """
    Convert CPU determinants to device feature tensor.
    
    Wrapper around masks_to_vecs with device transfer.
    
    Args:
        dets: NumPy determinants, shape (n, 2)
        n_orb: Number of spatial orbitals
        
    Returns:
        JAX features, shape (n, 2·n_orb)
    """
    if len(dets) == 0:
        return jnp.empty((0, 2 * n_orb), dtype=jnp.float32)
    
    dets_dev = jax.device_put(dets)
    return masks_to_vecs(dets_dev, n_orb)


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
    
    Computes ψ = exp(log ψ) and precomputes S/C space norms:
      ||ψₛ||² = Σᵢ₌₁ⁿˢ |ψᵢ|²
      ||ψ_c||² = Σᵢ₌ₙₛ₊₁ⁿˢ⁺ⁿᶜ |ψᵢ|²
    
    Args:
        log_psi: Log wavefunction, shape (n_s + n_c,)
        n_s: S-space dimension
        n_c: C-space dimension
        
    Returns:
        PsiCache with amplitudes and precomputed norms
    """
    from .dtypes import PsiCache
    
    psi_all = jnp.exp(log_psi)
    
    psi_s_norm_sq = jnp.sum(jnp.abs(psi_all[:n_s])**2)
    psi_c_norm_sq = jnp.sum(jnp.abs(psi_all[n_s:])**2)
    
    return PsiCache(
        log_all=log_psi,
        psi_all=psi_all,
        n_s=n_s,
        n_c=n_c
    )


__all__ = [
    "masks_to_vecs",
    "dets_to_features",
    "compute_normalized_amplitudes",
    "create_psi_cache",
]
