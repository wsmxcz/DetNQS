# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Geometry tape for gradient and QGT computation via single linearization.

Implements single-linearization paradigm: capture Jacobian once, reuse for
both gradient (VJP) and QGT operations (JVP) without redundant evaluation.

Core algorithm:
  1. Linearize: (log ψ, jvp_fn) = jax.linearize(logpsi_fn, params)
  2. Transpose: vjp_fn = jax.linear_transpose(jvp_fn, params)
  3. Center: O'_k = O_k - ⟨O_k⟩ where O_k = ∂_k log ψ

File: lever/engine/geometry.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import vmap
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ..utils.jax_utils import tree_dot, tree_scale, tree_sub, tree_add
from ..dtypes import GeometryTape

if TYPE_CHECKING:
    from ..dtypes import PyTree, LogPsiFn


def prepare_tape(
    params: PyTree,
    logpsi_fn: LogPsiFn,
    num_eps: float = 1e-12
) -> GeometryTape:
    """
    Create geometry tape via single linearization.
    
    Captures Jacobian O = ∂_θ log ψ and computes centered mean ⟨O⟩ for
    gradient and QGT operations. Centering removes gauge freedom (global phase).
    
    Args:
        params: Network parameters θ
        logpsi_fn: Wavefunction evaluator θ → log ψ [n_samples]
        num_eps: Numerical stability threshold
        
    Returns:
        GeometryTape with linearization and statistics
    """
    # Phase 1: Single linearization (key optimization)
    log_psi, jvp_fn = jax.linearize(logpsi_fn, params)
    
    # Phase 2: Derive VJP via transpose
    vjp_fn = jax.linear_transpose(jvp_fn, params)
    
    # Phase 3: Compute normalized weights w_i = |ψ_i|² / Σ|ψ_j|²
    s = 2.0 * jnp.real(log_psi)
    s = s - jnp.max(s)  # Log-sum-exp stabilization
    weights = jnp.exp(s)
    weights = weights / jnp.maximum(jnp.sum(weights), num_eps)
    
    # Phase 4: Compute centered mean ⟨O⟩ = J† @ w
    cotangent = jnp.asarray(weights, dtype=log_psi.dtype)
    mean_T = vjp_fn(cotangent)[0]
    centered_mean = jax.tree.map(jnp.conj, mean_T)
    
    return GeometryTape(
        jvp_fn=jvp_fn,
        vjp_fn=vjp_fn,
        log_psi=log_psi,
        weights=weights,
        centered_mean=centered_mean
    )


def qgt_matvec(
    tape: GeometryTape,
    v: PyTree,
    diag_shift: float = 1e-4
) -> PyTree:
    """
    Compute QGT matrix-vector product: (S + λI) @ v.
    
    Implements centered covariance via JVP/VJP without explicit matrix:
      S @ v = J† @ diag(w) @ J @ v - ⟨O⟩ * (⟨O⟩ · v)
    where J = ∂_θ log ψ is the Jacobian.
    
    Algorithm:
      1. u = J @ v              (JVP)
      2. r = J† @ (w * u)       (VJP with weights)
      3. r -= ⟨O⟩ * (⟨O⟩ · v)   (centering)
      4. r += λ * v             (regularization)
    
    Args:
        tape: Geometry tape from prepare_tape
        v: Direction vector (PyTree)
        diag_shift: Diagonal regularization λ
        
    Returns:
        Product (S + λI) @ v in same PyTree structure
    """
    # Forward JVP
    u = tape.jvp_fn(v)

    # Backward VJP with weights
    weighted_u = tape.weights * u
    result_T = tape.vjp_fn(jnp.conj(weighted_u))[0]
    result = jax.tree.map(jnp.conj, result_T)

    # Centering: r -= ⟨O⟩ * (⟨O⟩ · v)
    v_dot_mean = tree_dot(tape.centered_mean, v)
    result = tree_sub(result, tree_scale(tape.centered_mean, v_dot_mean))

    # Regularization: r += λ * v
    result = tree_add(result, tree_scale(v, diag_shift))
    return result


def qgt_dense(
    tape: GeometryTape,
    diag_shift: float = 1e-4,
    symmetrize: bool = True
) -> jnp.ndarray:
    """
    Compute explicit QGT matrix S via vmapped JVP.
    
    Constructs full Jacobian O = ∂_θ log ψ by applying JVP to identity matrix,
    then computes centered covariance S = O'† @ diag(w) @ O'.
    
    Algorithm:
      1. Build Jacobian: O = vmap(jvp_fn)(I)
      2. Center: O' = O - ⟨O⟩
      3. Compute: S = O'† @ diag(w) @ O' + λI
    
    Args:
        tape: Geometry tape from prepare_tape
        diag_shift: Diagonal regularization λ
        symmetrize: Force Hermitian symmetry
        
    Returns:
        Dense QGT matrix [n_params, n_params]
        
    Warning:
        Memory-intensive for large models. Use qgt_matvec + CG instead.
    """
    # Flatten parameter structure
    flat_mean, unravel_fn = ravel_pytree(tape.centered_mean)
    n_params = flat_mean.size
    
    # Build Jacobian via vmapped JVP
    identity = jnp.eye(n_params, dtype=flat_mean.dtype)
    O = vmap(lambda e: tape.jvp_fn(unravel_fn(e)), in_axes=0)(identity).T
    
    # Center Jacobian
    O_centered = O - jnp.conj(flat_mean)[None, :]
    
    # Compute weighted covariance
    sqrt_w = jnp.sqrt(tape.weights)[:, None]
    weighted_O = sqrt_w * O_centered
    S = weighted_O.conj().T @ weighted_O
    
    # Symmetrize and regularize
    if symmetrize:
        S = (S + S.conj().T) / 2.0
    S = S + diag_shift * jnp.eye(n_params, dtype=S.dtype)
    
    return S


__all__ = ["prepare_tape", "qgt_matvec", "qgt_dense"]
