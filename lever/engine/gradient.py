# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Energy and gradient computation for variational quantum Monte Carlo.

Provides mode-specific energy functionals and their gradients:
  - EFFECTIVE:  E = ⟨ψ_S|H_eff|ψ_S⟩ / ‖ψ_S‖²
  - PROXY:      E = ⟨ψ|H_proxy|ψ⟩ / ‖ψ‖²
  - ASYMMETRIC: E = ⟨ψ_S|H|ψ⟩ / ‖ψ_S‖²

File: lever/engine/gradient.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..config import ComputeMode
from ..dtypes import GradResult, GeometryTape

if TYPE_CHECKING:
    from ..dtypes import OuterCtx, PyTree


def compute_energy_and_grad(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float = 1e-12
) -> GradResult:
    """
    Compute variational energy and gradient from geometry tape.
    
    Dispatches to mode-specific kernel based on ctx.compute_mode.
    Uses single linearization captured in tape for efficiency.
    
    Args:
        params: Neural network parameters
        tape: Geometry tape with linearization context
        ctx: Outer context with Hamiltonian and operators
        num_eps: Numerical stability threshold
        
    Returns:
        GradResult with gradient and total energy (including E_nuc)
    """
    mode = ctx.compute_mode
    
    if mode == ComputeMode.EFFECTIVE:
        return _effective_kernel(params, tape, ctx, num_eps)
    elif mode == ComputeMode.PROXY:
        return _proxy_kernel(params, tape, ctx, num_eps)
    elif mode == ComputeMode.ASYMMETRIC:
        return _asymmetric_kernel(params, tape, ctx, num_eps)
    else:
        raise ValueError(f"Unknown compute mode: {mode}")


def _effective_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float
) -> GradResult:
    """
    S-space energy and gradient with effective Hamiltonian.
    
    Energy:   E = ⟨ψ_S|H_eff|ψ_S⟩ / ‖ψ_S‖²
    Gradient: ∇E = 2 Re[⟨O_k (E_loc - E)⟩] (variance-reduced)
    """
    psi_s = jnp.exp(tape.log_psi)
    
    # Matrix-vector product: N = H_eff @ ψ_S
    # spmv_fn returns tuple or Contractions, extract n_ss
    result = ctx.spmv_fn(psi_s)
    n_ss = result.n_ss if hasattr(result, 'n_ss') else result[0]
    
    # Energy: E = ⟨ψ_S|N⟩ / ⟨ψ_S|ψ_S⟩
    numerator = jnp.vdot(psi_s, n_ss)
    denominator = jnp.vdot(psi_s, psi_s)
    energy = (numerator / jnp.maximum(denominator, num_eps)).real
    
    # Local energy: E_loc[i] = N[i] / ψ_S[i]
    e_loc_s = jnp.where(
        jnp.abs(psi_s) >= num_eps,
        n_ss / psi_s,
        0.0
    )
    
    # Variance-reduced cotangent: cot = w * (E_loc - E)
    cotangent = tape.weights * (e_loc_s - energy)
    cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
    
    # Gradient via VJP: ∇E = conj(VJP[conj(cot)])
    (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
    gradient = jax.tree.map(jnp.conj, grad_conj)
    
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


def _proxy_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float
) -> GradResult:
    """
    Full T-space energy and gradient with proxy Hamiltonian.
    
    Energy: E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / (‖ψ_S‖² + ‖ψ_C‖²)
    """
    n_s = ctx.space.n_s
    psi_all = jnp.exp(tape.log_psi)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
    
    # Block SpMV: N = H_proxy @ [ψ_S; ψ_C]
    result = ctx.spmv_fn(psi_s, psi_c)
    
    # Extract contractions (handle both tuple and Contractions object)
    if hasattr(result, 'n_ss'):
        n_ss, n_sc, n_cs, n_cc = result.n_ss, result.n_sc, result.n_cs, result.n_cc
    else:
        n_ss, n_sc, n_cs, n_cc = result
    
    # Energy: E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / ‖ψ‖²
    num_s = jnp.vdot(psi_s, n_ss + n_sc)
    num_c = jnp.vdot(psi_c, n_cs + n_cc)
    denom = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
    energy = ((num_s + num_c) / jnp.maximum(denom, num_eps)).real
    
    # Local energies for S and C spaces
    e_loc_s = jnp.where(
        jnp.abs(psi_s) >= num_eps,
        (n_ss + n_sc) / psi_s,
        0.0
    )
    e_loc_c = jnp.where(
        jnp.abs(psi_c) >= num_eps,
        (n_cs + n_cc) / psi_c,
        0.0
    )
    e_loc_all = jnp.concatenate([e_loc_s, e_loc_c])
    
    # Full-space cotangent
    cotangent = tape.weights * (e_loc_all - energy)
    cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
    
    (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
    gradient = jax.tree.map(jnp.conj, grad_conj)
    
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


def _asymmetric_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float
) -> GradResult:
    """
    Asymmetric kernel: S-space normalization with full Hamiltonian.
    
    Energy: E = ⟨ψ_S|N_S⟩ / ⟨ψ_S|ψ_S⟩ where N from full H @ [ψ_S; ψ_C]
    """
    n_s = ctx.space.n_s
    
    # Full wavefunction (already computed in tape from full evaluator)
    psi_all = jnp.exp(tape.log_psi)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
    
    # SpMV: N = H @ [ψ_S; ψ_C]
    result = ctx.spmv_fn(psi_s, psi_c)
    
    if hasattr(result, 'n_ss'):
        n_ss, n_sc = result.n_ss, result.n_sc
    else:
        n_ss, n_sc = result[0], result[1]
    
    # Energy: E = ⟨ψ_S|N_S⟩ / ⟨ψ_S|ψ_S⟩ (S-space normalization)
    numerator = jnp.vdot(psi_s, n_ss + n_sc)
    denominator = jnp.vdot(psi_s, psi_s)
    energy = (numerator / jnp.maximum(denominator, num_eps)).real
    
    # Local energy on S-space
    e_loc_s = jnp.where(
        jnp.abs(psi_s) >= num_eps,
        (n_ss + n_sc) / psi_s,
        0.0
    )
    
    # Use tape from S-normalized logψ for gradient
    cotangent = tape.weights * (e_loc_s - energy)
    cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
    
    (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
    gradient = jax.tree.map(jnp.conj, grad_conj)
    
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


__all__ = ["compute_energy_and_grad"]
