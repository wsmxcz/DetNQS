# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Geometry information provider for variational optimization.

Captures linearized wavefunction information in a single forward pass,
enabling reuse across gradient and QGT computations.

File: lever/engine/geometry.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Callable

import flax.struct
import jax
import jax.numpy as jnp

from ..utils.dtypes import PyTree


@flax.struct.dataclass
class GeometryTape:
    """
    Per-step linearization context for VMC optimization.
    
    Records Jacobian information from a single forward pass:
      - jvp_fn: JVP operator v → J @ v
      - vjp_fn: VJP operator c → J^T @ c
      - weights: Probability distribution |ψ|²/||ψ||² over determinant space
      - log_psi: Wavefunction logarithm
    
    Phase 1: Only provides gradient computation.
    Phase 2+: Will add QGT matvec/dense providers.
    """
    jvp_fn: Callable[[PyTree], jnp.ndarray]  # params → J @ params
    vjp_fn: Callable[[jnp.ndarray], PyTree]  # cotangent → J^T @ cotangent
    weights: jnp.ndarray                      # |ψ|²/||ψ||²
    log_psi: jnp.ndarray                      # log(ψ)
    
    def compute_grad(self, cotangent: jnp.ndarray) -> PyTree:
        """
        Compute parameter gradient via VJP with conjugate cotangent.
        
        Formula: ∇E = J^† @ conj(c)
        where c = weights * (E_loc - E)
        
        Args:
            cotangent: Weighted local energy deviations
            
        Returns:
            Parameter gradient tree
        """
        # VJP expects conjugate cotangent for holomorphic gradient
        grad_conj = self.vjp_fn(jnp.conj(cotangent))
        # Conjugate back to get true gradient
        return jax.tree.map(jnp.conj, grad_conj)


def prepare_tape(
    logpsi_fn: Callable[[PyTree], jnp.ndarray],
    params: PyTree,
    weights: jnp.ndarray
) -> GeometryTape:
    """
    Create geometry tape from wavefunction evaluator.
    
    Performs single linearization to capture JVP/VJP operators.
    
    Args:
        logpsi_fn: Neural network forward pass (params → log ψ)
        params: Current parameters
        weights: Probability weights |ψ|²/||ψ||² (pre-computed)
        
    Returns:
        GeometryTape with cached linearization
    """
    # Single forward pass with linearization
    log_psi, jvp_fn = jax.linearize(logpsi_fn, params)
    
    # Derive VJP from JVP via transpose
    vjp_fn = jax.linear_transpose(jvp_fn, log_psi)
    
    return GeometryTape(
        jvp_fn=jvp_fn,
        vjp_fn=vjp_fn,
        weights=weights,
        log_psi=log_psi
    )


__all__ = ["GeometryTape", "prepare_tape"]
