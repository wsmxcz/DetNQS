# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JIT-compiled VMC optimization step kernels.

Assembles pure JAX step functions from closures (logpsi, spmv)
for efficient lax.scan execution in variational Monte Carlo.

File: lever/engine/step.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax

from .utils import GradResult, InnerState, StepResult

if TYPE_CHECKING:
    from collections.abc import Callable
    from .utils import OuterCtx, PyTree


# ============================================================================
# Energy & Gradient Computation
# ============================================================================

def _compute_energy_grad_eff(
    ctx: OuterCtx,
    params: PyTree,
    eps: float
) -> GradResult:
    """
    EFFECTIVE mode: Energy E = ⟨ψ_S|H_eff|ψ_S⟩ / ||ψ_S||².
    
    Variance-reduced gradient via VJP with cotangent:
        ∂E/∂θ = 2·Re[⟨(E_loc - E)·ψ_S | ∂ψ_S/∂θ⟩]
    where E_loc[i] = (H_eff·ψ_S)[i]/ψ_S[i].
    
    Args:
        ctx: Outer context with logpsi_fn, spmv_fn, space metadata
        params: Neural network parameters
        eps: Numerical stability threshold
        
    Returns:
        GradResult with variance-reduced gradient and energy
    """
    n_s = ctx.space.n_s
    
    # Forward VJP: params → log(ψ)
    log_all, vjp_fn = jax.vjp(ctx.logpsi_fn, params)
    psi_all = jnp.exp(log_all)
    psi_s = psi_all[:n_s]
    
    # SpMV: N = H_eff @ ψ_S
    N = ctx.spmv_fn(psi_s)
    
    # Energy: E = ⟨ψ_S|N⟩ / ||ψ_S||²
    num = jnp.vdot(psi_s, N.n_ss)
    denom = jnp.vdot(psi_s, psi_s)
    energy = (num / jnp.maximum(denom, eps)).real
    
    # Local energy: E_loc = N / ψ_S (element-wise)
    e_loc = jnp.where(jnp.abs(psi_s) >= eps, N.n_ss / psi_s, 0.0)
    
    # Variance-reduced cotangent: w·(E_loc - E) with w = |ψ_S|²/Σ|ψ_S|²
    weights = jnp.exp(2.0 * jnp.real(log_all[:n_s]))
    weights /= jnp.maximum(jnp.sum(weights), eps)
    cot_s = weights * (e_loc - energy)
    
    # Zero-pad C-space (features present but unused in optimization)
    n_c = ctx.feat_c.shape[0]
    cot_full = jnp.concatenate([cot_s, jnp.zeros(n_c, dtype=cot_s.dtype)])
    
    # Backward VJP: cotangent → gradient
    (grad_conj,) = vjp_fn(jnp.conj(cot_full))
    grad = jax.tree.map(jnp.conj, grad_conj)
    
    return GradResult(grad=grad, energy=energy)


def _compute_energy_grad_proxy(
    ctx: OuterCtx,
    params: PyTree,
    eps: float
) -> GradResult:
    """
    PROXY mode: Energy E = ⟨ψ|H̃|ψ⟩ / ||ψ||² over full T-space.
    
    Variance-reduced gradient with E_loc = (H̃·ψ)/ψ for both S- and C-spaces.
    
    Args:
        ctx: Outer context
        params: Neural network parameters
        eps: Numerical stability threshold
        
    Returns:
        GradResult with full T-space gradient and energy
    """
    n_s = ctx.space.n_s
    
    # Forward VJP
    log_all, vjp_fn = jax.vjp(ctx.logpsi_fn, params)
    psi_all = jnp.exp(log_all)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
    
    # Full T-space SpMV
    N = ctx.spmv_fn(psi_s, psi_c)
    
    # Energy: E = [⟨ψ_S|H_S⟩ + ⟨ψ_C|H_C⟩] / ||ψ||²
    num_s = jnp.vdot(psi_s, N.n_ss + N.n_sc)
    num_c = jnp.vdot(psi_c, N.n_cs + N.n_cc)
    denom = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
    energy = ((num_s + num_c) / jnp.maximum(denom, eps)).real
    
    # Local energies
    e_loc_s = jnp.where(jnp.abs(psi_s) >= eps, (N.n_ss + N.n_sc) / psi_s, 0.0)
    e_loc_c = jnp.where(jnp.abs(psi_c) >= eps, (N.n_cs + N.n_cc) / psi_c, 0.0)
    e_loc_all = jnp.concatenate([e_loc_s, e_loc_c])
    
    # Variance-reduced cotangent
    weights = jnp.exp(2.0 * jnp.real(log_all))
    weights /= jnp.maximum(jnp.sum(weights), eps)
    cot_full = weights * (e_loc_all - energy)
    
    # Backward VJP
    (grad_conj,) = vjp_fn(jnp.conj(cot_full))
    grad = jax.tree.map(jnp.conj, grad_conj)
    
    return GradResult(grad=grad, energy=energy)


# ============================================================================
# Step Kernel Factory
# ============================================================================

def create_step_fn(
    ctx: OuterCtx,
    optimizer,
    eps: float
) -> Callable[[InnerState, None], tuple[InnerState, jnp.ndarray]]:
    """
    Create mode-specific VMC step function.
    
    Closure captures ctx, optimizer, eps for pure JAX execution.
    
    Args:
        ctx: Outer context with mode="effective" or "proxy"
        optimizer: Optax optimizer (e.g., AdamW)
        eps: Numerical threshold for stability
        
    Returns:
        Step function: (state, _) → (new_state, total_energy)
    """
    energy_grad_fn = (
        _compute_energy_grad_eff if ctx.mode == "effective" 
        else _compute_energy_grad_proxy
    )
    e_nuc = ctx.e_nuc
    
    def _step(state: InnerState, _unused) -> tuple[InnerState, jnp.ndarray]:
        """Single VMC step: gradient → AdamW update."""
        # Compute energy and variance-reduced gradient
        result = energy_grad_fn(ctx, state.params, eps)
        
        # AdamW parameter update
        import optax
        updates, new_opt = optimizer.update(
            result.grad, state.opt_state, state.params
        )
        new_params = optax.apply_updates(state.params, updates)
        
        new_state = InnerState(
            params=new_params,
            opt_state=new_opt,
            step=state.step + 1
        )
        
        # Add nuclear repulsion to electronic energy
        e_total = result.energy + e_nuc
        return new_state, e_total
    
    return _step


def create_scan_fn(
    ctx: OuterCtx,
    optimizer,
    eps: float
) -> Callable[[InnerState, int], tuple[InnerState, jnp.ndarray]]:
    """
    Create JIT-compiled scan wrapper for multi-step execution.
    
    Uses lax.scan for efficient loop unrolling.
    
    Args:
        ctx: Outer context
        optimizer: Optax optimizer
        eps: Numerical threshold
        
    Returns:
        Scan function: (state, n_steps) → (final_state, energy_history)
    """
    step_fn = create_step_fn(ctx, optimizer, eps)
    
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


__all__ = ["create_step_fn", "create_scan_fn"]
