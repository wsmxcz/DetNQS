# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical utility functions for LEVER engine.

Contains pure numerical transformations without data structure definitions

File: lever/utils/features.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import jax.numpy as jnp


def masks_to_vecs(dets: jnp.ndarray, n_orb: int) -> jnp.ndarray:
    """
    Convert determinant bitmasks to occupancy vectors.
    
    Maps bit representations to neural network input features:
      |det⟩ = |i₁...i_α⟩⊗|j₁...j_β⟩ → [n₁^α,...,n_M^α, n₁^β,...,n_M^β]
    where n_k^σ ∈ {0,1} denotes spin-σ occupation of orbital k.
    
    Args:
        dets: (..., 2) uint64 bitmasks [α-string, β-string]
        n_orb: Number of spatial orbitals M
    
    Returns:
        (..., 2·M) occupancy vectors in [0,1]
    
    Example:
        >>> dets = jnp.array([[0b1011, 0b0110]])  # HF state for 4 orbitals
        >>> masks_to_vecs(dets, 4)
        [[1,1,0,1, 0,1,1,0]]  # [α₀,α₁,α₂,α₃, β₀,β₁,β₂,β₃]
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")

    dets = dets.astype(jnp.uint64)
    alpha, beta = dets[..., 0], dets[..., 1]
    
    # Extract bit k from all determinants: (det >> k) & 1
    orb_idx = jnp.arange(n_orb, dtype=jnp.uint64)
    alpha_occ = ((alpha[..., None] >> orb_idx) & 1).astype(jnp.float32)
    beta_occ = ((beta[..., None] >> orb_idx) & 1).astype(jnp.float32)
    
    return jnp.concatenate([alpha_occ, beta_occ], axis=-1)


__all__ = ["masks_to_vecs"]
