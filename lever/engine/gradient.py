# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational quantum Monte Carlo energy and gradient computation.

Mode-specific energy functionals:
  - EFFECTIVE:  E = ⟨ψ_S|H_eff|ψ_S⟩ / ⟨ψ_S|ψ_S⟩
  - PROXY:      E = ⟨ψ|H_proxy|ψ⟩ / ⟨ψ|ψ⟩
  - ASYMMETRIC: E = ⟨ψ_S|Hψ⟩ / ⟨ψ_S|ψ_S⟩

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


# ============================================================================
# Main Entry
# ============================================================================

def compute_energy_and_grad(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float = 1e-12
) -> GradResult:
    """
    Compute energy and variance-reduced gradient: ∇E = 2Re[⟨O_k(E_loc - E)⟩].
    
    Args:
        params: Network parameters (unused, kept for interface)
        tape: Geometry tape with VJP context
        ctx: Hamiltonian and operator context
        num_eps: Stability threshold for division
        
    Returns:
        GradResult(grad, energy) with nuclear repulsion included
    """
    dispatch = {
        ComputeMode.EFFECTIVE: _effective_kernel,
        ComputeMode.PROXY: _proxy_kernel,
        ComputeMode.ASYMMETRIC: _asymmetric_kernel,
    }
    
    kernel = dispatch.get(ctx.compute_mode)
    if kernel is None:
        raise ValueError(f"Unknown compute mode: {ctx.compute_mode}")
    
    return kernel(params, tape, ctx, num_eps)


# ============================================================================
# Mode-Specific Kernels
# ============================================================================

def _effective_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float
) -> GradResult:
    """
    S-space effective Hamiltonian: E = ⟨ψ_S|H_eff|ψ_S⟩ / ‖ψ_S‖².
    """
    psi_s = jnp.exp(tape.log_psi)
    
    # SpMV: N_S = H_eff @ ψ_S
    result = ctx.spmv_fn(psi_s)
    n_ss = result.n_ss if hasattr(result, 'n_ss') else result[0]
    
    # Energy: ⟨ψ_S|N_S⟩ / ⟨ψ_S|ψ_S⟩
    energy = _compute_energy(psi_s, n_ss, num_eps)
    
    # Local energy: E_loc[i] = N_S[i] / ψ_S[i]
    e_loc = _safe_division(n_ss, psi_s, num_eps)
    
    # Gradient via VJP with variance reduction
    gradient = _compute_gradient(tape, e_loc, energy)
    
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


def _proxy_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float
) -> GradResult:
    """
    Full T-space proxy: E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / ‖ψ‖².
    """
    n_s = ctx.space.n_s
    psi_all = jnp.exp(tape.log_psi)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
    
    # Block SpMV: [N_S; N_C] = H_proxy @ [ψ_S; ψ_C]
    result = ctx.spmv_fn(psi_s, psi_c)
    n_ss, n_sc, n_cs, n_cc = _unpack_contractions(result)
    
    # Total contractions per subspace
    ns_total = n_ss + n_sc
    nc_total = n_cs + n_cc
    
    # Energy: (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / (‖ψ_S‖² + ‖ψ_C‖²)
    num_s = jnp.vdot(psi_s, ns_total)
    num_c = jnp.vdot(psi_c, nc_total)
    denom = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
    energy = ((num_s + num_c) / jnp.maximum(denom, num_eps)).real
    
    # Local energies for both subspaces
    e_loc_s = _safe_division(ns_total, psi_s, num_eps)
    e_loc_c = _safe_division(nc_total, psi_c, num_eps)
    e_loc_all = jnp.concatenate([e_loc_s, e_loc_c])
    
    gradient = _compute_gradient(tape, e_loc_all, energy)
    
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


def _asymmetric_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float
) -> GradResult:
    """
    Asymmetric: S-normalization with full H: E = ⟨ψ_S|Hψ⟩ / ‖ψ_S‖².
    """
    n_s = ctx.space.n_s
    psi_all = jnp.exp(tape.log_psi)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
    
    # SpMV: [N_S; N_C] = H @ [ψ_S; ψ_C]
    result = ctx.spmv_fn(psi_s, psi_c)
    n_ss, n_sc = _unpack_contractions(result)[:2]
    
    # Energy: ⟨ψ_S|N_S⟩ / ⟨ψ_S|ψ_S⟩ (S-normalized)
    ns_total = n_ss + n_sc
    energy = _compute_energy(psi_s, ns_total, num_eps)
    
    # S-space local energy only
    e_loc = _safe_division(ns_total, psi_s, num_eps)
    
    gradient = _compute_gradient(tape, e_loc, energy)
    
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_energy(
    psi: jnp.ndarray,
    n: jnp.ndarray,
    eps: float
) -> float:
    """Rayleigh quotient: ⟨ψ|N⟩ / ⟨ψ|ψ⟩."""
    numerator = jnp.vdot(psi, n)
    denominator = jnp.vdot(psi, psi)
    return (numerator / jnp.maximum(denominator, eps)).real


def _safe_division(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    eps: float
) -> jnp.ndarray:
    """Element-wise division with zero protection."""
    return jnp.where(
        jnp.abs(denominator) >= eps,
        numerator / denominator,
        0.0
    )


def _compute_gradient(
    tape: GeometryTape,
    e_loc: jnp.ndarray,
    energy: float
) -> PyTree:
    """
    Variance-reduced gradient: ∇E = conj(VJP[conj(w * (E_loc - E))]).
    
    Uses linearization captured in tape.vjp_fn.
    """
    cotangent = tape.weights * (e_loc - energy)
    cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
    
    (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
    return jax.tree.map(jnp.conj, grad_conj)


def _unpack_contractions(result):
    """Extract contractions from SpMV result (tuple or object)."""
    if hasattr(result, 'n_ss'):
        return result.n_ss, result.n_sc, result.n_cs, result.n_cc
    return result


__all__ = ["compute_energy_and_grad"]
