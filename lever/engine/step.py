# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JIT-compiled VMC optimization step kernels with unified mode abstraction.

Provides ModeKernel factory for ASYMMETRIC, PROXY, and EFFECTIVE modes.
Each mode implements energy E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ with mode-specific Hamiltonian
and variance-reduced gradient ∇E via VJP with weighted cotangent vectors.

File: lever/engine/step.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax

from ..dtypes import GradResult, InnerState
from ..config import ComputeMode

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..dtypes import OuterCtx, PyTree


# ============================================================================
# Mode Kernel Factory
# ============================================================================

class ModeKernel:
    """Factory for mode-specific energy/gradient kernels."""
    
    @staticmethod
    def make(ctx: OuterCtx, num_eps: float) -> Callable[[PyTree], GradResult]:
        """
        Create mode-specific kernel via Enum dispatch.
        
        Args:
            ctx: Outer context with logpsi_fn, spmv_fn, mode specification
            num_eps: Numerical stability threshold for division
            
        Returns:
            Energy/gradient function: params → (grad, energy)
        """
        mode = ctx.mode
        
        if mode is ComputeMode.ASYMMETRIC:
            return _make_asym_kernel(ctx, num_eps)
        elif mode is ComputeMode.PROXY:
            return _make_proxy_kernel(ctx, num_eps)
        elif mode is ComputeMode.EFFECTIVE:
            return _make_eff_kernel(ctx, num_eps)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# ASYMMETRIC Mode: Non-variational S-norm
# ============================================================================

def _make_asym_kernel(ctx: OuterCtx, num_eps: float) -> Callable:
    """
    ASYMMETRIC mode: E = ⟨ψ_S|H|ψ⟩ / ||ψ_S||² (non-Hermitian numerator).
    
    Algorithm:
      1. Forward: log(ψ) → ψ = exp(log(ψ)) for S ∪ C
      2. SpMV: N = H_SS·ψ_S + H_SC·ψ_C
      3. Energy: E = ⟨ψ_S|N⟩ / ||ψ_S||²
      4. Local energy: E_loc[i] = N[i]/ψ_S[i]
      5. Cotangent: c_S = w_S·(E_loc - E), c_C = 0 (blocks C-space backprop)
      6. Backward: VJP with conj(c) → ∇E
      
    S-space optimization only; C amplitudes propagate but don't train.
    """
    n_s, n_c = ctx.space.n_s, ctx.space.n_c
    logpsi_fn = ctx.logpsi_fn
    spmv_fn = ctx.spmv_fn
    e_nuc = ctx.e_nuc
    
    def energy_grad_fn(params: PyTree) -> GradResult:
        # Forward VJP: params → log(ψ)
        log_all, vjp_fn = jax.vjp(logpsi_fn, params)
        psi_all = jnp.exp(log_all)
        psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
        
        # SpMV: N = H·ψ in S-space
        N = spmv_fn(psi_s, psi_c)
        
        # Energy: E = ⟨ψ_S|H·ψ⟩ / ||ψ_S||²
        num = jnp.vdot(psi_s, N.n_ss + N.n_sc)
        denom = jnp.vdot(psi_s, psi_s)
        energy = (num / jnp.maximum(denom, num_eps)).real
        
        # Local energy: E_loc = (H·ψ)/ψ_S
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            (N.n_ss + N.n_sc) / psi_s,
            0.0
        )
        
        # Variance-reduced cotangent (S-only, zero C to block gradients)
        weights_s = jnp.exp(2.0 * jnp.real(log_all[:n_s]))
        weights_s /= jnp.maximum(jnp.sum(weights_s), num_eps)
        cot_s = weights_s * (e_loc_s - energy)
        cot_c = jnp.zeros(n_c, dtype=cot_s.dtype)
        cot_full = jnp.concatenate([cot_s, cot_c])
        cot_full = jnp.asarray(cot_full, dtype=log_all.dtype)
        
        # Backward VJP: c → ∇E
        (grad_conj,) = vjp_fn(jnp.conj(cot_full))
        grad = jax.tree.map(jnp.conj, grad_conj)
        
        return GradResult(grad=grad, energy=energy + e_nuc)
    
    return energy_grad_fn


# ============================================================================
# PROXY Mode: Variational Hermitian surrogate
# ============================================================================

def _make_proxy_kernel(ctx: OuterCtx, num_eps: float) -> Callable:
    """
    PROXY mode: E = ⟨ψ|H̃|ψ⟩ / ||ψ||² (Hermitian, full-space).
    
    Algorithm:
      1. Surrogate H̃ = [H_SS, H_SC; H_CS, diag(H_CC)]
      2. Full-space SpMV for S ∪ C
      3. Energy: E = [⟨ψ_S|H_S⟩ + ⟨ψ_C|H_C⟩] / ||ψ||²
      4. Cotangent: c = w·(E_loc - E) for both S and C
      5. VJP: c → ∇E with full-space optimization
      
    Standard Rayleigh quotient gradient descent.
    """
    n_s = ctx.space.n_s
    logpsi_fn = ctx.logpsi_fn
    spmv_fn = ctx.spmv_fn
    e_nuc = ctx.e_nuc
    
    def energy_grad_fn(params: PyTree) -> GradResult:
        log_all, vjp_fn = jax.vjp(logpsi_fn, params)
        psi_all = jnp.exp(log_all)
        psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
        
        # Full-space SpMV
        N = spmv_fn(psi_s, psi_c)
        
        # Energy: E = ⟨ψ|H̃·ψ⟩ / ||ψ||²
        num_s = jnp.vdot(psi_s, N.n_ss + N.n_sc)
        num_c = jnp.vdot(psi_c, N.n_cs + N.n_cc)
        denom = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
        energy = ((num_s + num_c) / jnp.maximum(denom, num_eps)).real
        
        # Local energies for S and C
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            (N.n_ss + N.n_sc) / psi_s,
            0.0
        )
        e_loc_c = jnp.where(
            jnp.abs(psi_c) >= num_eps,
            (N.n_cs + N.n_cc) / psi_c,
            0.0
        )
        e_loc_all = jnp.concatenate([e_loc_s, e_loc_c])
        
        # Full-space variance-reduced cotangent
        weights = jnp.exp(2.0 * jnp.real(log_all))
        weights /= jnp.maximum(jnp.sum(weights), num_eps)
        cot_full = weights * (e_loc_all - energy)
        cot_full = jnp.asarray(cot_full, dtype=log_all.dtype)
        
        # VJP: c → ∇E
        (grad_conj,) = vjp_fn(jnp.conj(cot_full))
        grad = jax.tree.map(jnp.conj, grad_conj)
        
        return GradResult(grad=grad, energy=energy + e_nuc)
    
    return energy_grad_fn


# ============================================================================
# EFFECTIVE Mode: S-space downfolded Hamiltonian
# ============================================================================

def _make_eff_kernel(ctx: OuterCtx, num_eps: float) -> Callable:
    """
    EFFECTIVE mode: E = ⟨ψ_S|H_eff|ψ_S⟩ / ||ψ_S||².
    
    Algorithm:
      1. Use S-only logpsi closure for hot path (training)
      2. H_eff = H_SS + H_SC·D⁻¹·H_CS (C-space downfolded)
      3. S-space SpMV: N = H_eff @ ψ_S
      4. Energy: E = ⟨ψ_S|N⟩ / ||ψ_S||²
      5. Cotangent: c_S = w_S·(E_loc - E), no C-space
      6. VJP: c_S → ∇E (S-only optimization)
      
    C amplitudes computed separately for evolution/analysis.
    """
    # Extract S-only closure for optimization hot path
    if isinstance(ctx.logpsi_fn, tuple):
        logpsi_s_fn, _ = ctx.logpsi_fn
    else:
        logpsi_s_fn = ctx.logpsi_fn  # Backward compatibility
    
    spmv_fn = ctx.spmv_fn
    e_nuc = ctx.e_nuc
    
    def energy_grad_fn(params: PyTree) -> GradResult:
        # S-space forward VJP
        log_s, vjp_fn = jax.vjp(logpsi_s_fn, params)
        psi_s = jnp.exp(log_s)
        
        # SpMV: N = H_eff @ ψ_S
        N = spmv_fn(psi_s)
        
        # Energy: E = ⟨ψ_S|H_eff|ψ_S⟩ / ||ψ_S||²
        num = jnp.vdot(psi_s, N.n_ss)
        denom = jnp.vdot(psi_s, psi_s)
        energy = (num / jnp.maximum(denom, num_eps)).real
        
        # Local energy
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            N.n_ss / psi_s,
            0.0
        )
        
        # Variance-reduced cotangent (S-only)
        weights_s = jnp.exp(2.0 * jnp.real(log_s))
        weights_s /= jnp.maximum(jnp.sum(weights_s), num_eps)
        cot_s = weights_s * (e_loc_s - energy)
        cot_s = jnp.asarray(cot_s, dtype=log_s.dtype)
        
        # VJP: c_S → ∇E (no C-space)
        (grad_conj,) = vjp_fn(jnp.conj(cot_s))
        grad = jax.tree.map(jnp.conj, grad_conj)
        
        return GradResult(grad=grad, energy=energy + e_nuc)
    
    return energy_grad_fn


# ============================================================================
# Step Function Factories
# ============================================================================

def create_step_fn(
    ctx: OuterCtx,
    optimizer,
    num_eps: float
) -> Callable[[InnerState, None], tuple[InnerState, jnp.ndarray]]:
    """
    Create single VMC optimization step: gradient → parameter update.
    
    Args:
        ctx: Outer context with mode specification
        optimizer: Optax optimizer (AdamW, etc.)
        num_eps: Numerical stability threshold
        
    Returns:
        Step function: (state, _) → (new_state, energy)
    """
    energy_grad_fn = ModeKernel.make(ctx, num_eps)
    
    def _step(state: InnerState, _unused) -> tuple[InnerState, jnp.ndarray]:
        """Single step: compute gradient and apply optimizer update."""
        import optax
        
        # Compute variance-reduced gradient
        result = energy_grad_fn(state.params)
        
        # Apply optimizer update
        updates, new_opt = optimizer.update(
            result.grad, state.opt_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)
        
        new_state = InnerState(
            params=new_params,
            opt_state=new_opt,
            step=state.step + 1
        )
        
        return new_state, result.energy
    
    return _step


def create_scan_fn(
    ctx: OuterCtx,
    optimizer,
    num_eps: float
) -> Callable[[InnerState, int], tuple[InnerState, jnp.ndarray]]:
    """
    Create JIT-compiled scan wrapper for multi-step execution.
    
    Uses lax.scan for XLA loop unrolling. For CPU with callbacks,
    single-step JIT + Python loop may outperform.
    
    Args:
        ctx: Outer context
        optimizer: Optax optimizer
        num_eps: Numerical threshold
        
    Returns:
        Scan function: (state, n_steps) → (final_state, energy_history)
    """
    step_fn = create_step_fn(ctx, optimizer, num_eps)
    
    def _scan_wrapper(state: InnerState, n_steps: int):
        """Execute n_steps via lax.scan with XLA compilation."""
        final_state, energies = lax.scan(
            step_fn,
            state,
            None,
            length=n_steps
        )
        return final_state, energies
    
    return jax.jit(_scan_wrapper, static_argnames=['n_steps'])


__all__ = ["ModeKernel", "create_step_fn", "create_scan_fn"]
