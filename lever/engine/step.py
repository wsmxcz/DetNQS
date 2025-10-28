# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JIT-compiled VMC optimization step kernels with unified mode abstraction.

Provides ModeKernel factory for ASYMMETRIC, PROXY, and EFFECTIVE modes.
Each mode computes energy E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ and variance-reduced gradient
∇E via VJP with weighted cotangent vectors.

File: lever/engine/step.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import lax
import optax

from ..optimizers.base import Optimizer as LeverOptimizer
from .geometry import prepare_tape, GeometryTape
from ..utils.dtypes import GradResult, InnerState
from ..config import ComputeMode

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..utils.dtypes import OuterCtx, PyTree


# ============================================================================
# Mode Kernel Factory
# ============================================================================

class ModeKernel:
    """Factory for mode-specific energy/gradient kernels."""
    
    @staticmethod
    def make(ctx: OuterCtx, num_eps: float) -> Callable[[PyTree, GeometryTape], GradResult]:
        """
        Create mode-specific kernel accepting pre-built tape.
        
        Args:
            ctx: Outer context with mode, space, SpMV
            num_eps: Numerical stability threshold
            
        Returns:
            Energy/gradient function: (params, tape) → GradResult(grad, energy)
        """
        mode_map = {
            ComputeMode.ASYMMETRIC: _make_asym_kernel,
            ComputeMode.PROXY: _make_proxy_kernel,
            ComputeMode.EFFECTIVE: _make_eff_kernel,
        }
        
        kernel_fn = mode_map.get(ctx.mode)
        if kernel_fn is None:
            raise ValueError(f"Unknown mode: {ctx.mode}")
        
        return kernel_fn(ctx, num_eps)


# ============================================================================
# ASYMMETRIC Mode: Non-variational S-norm
# ============================================================================

def _make_asym_kernel(ctx: OuterCtx, num_eps: float) -> Callable:
    """
    ASYMMETRIC mode: E = ⟨ψ_S|H|ψ⟩ / ||ψ_S||²
    
    S-space only optimization with consistent measure.
    Tape built from S-only logpsi_fn for QGT compatibility.
    """
    n_s, n_c = ctx.space.n_s, ctx.space.n_c
    spmv_fn = ctx.spmv_fn
    e_nuc = ctx.e_nuc
    
    # Extract full-space logpsi for SpMV (requires C-space)
    logpsi_full_fn = (ctx.logpsi_fn[1] if isinstance(ctx.logpsi_fn, tuple) 
                      else ctx.logpsi_fn)
    
    def energy_grad_fn(params: PyTree, tape: GeometryTape) -> GradResult:
        """
        Compute energy/gradient with S-space measure.
        
        Note: tape.log_psi has shape [n_s], tape.weights is S-normalized.
        """
        # Full wavefunction for SpMV: ψ = [ψ_S, ψ_C]
        log_psi_full = logpsi_full_fn(params)
        psi_all = jnp.exp(log_psi_full)
        psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
        
        # SpMV: N = H·ψ
        N = spmv_fn(psi_s, psi_c)
        
        # Energy: E = ⟨ψ_S|N⟩ / ||ψ_S||²
        num = jnp.vdot(psi_s, N.n_ss + N.n_sc)
        denom = jnp.vdot(psi_s, psi_s)
        energy = (num / jnp.maximum(denom, num_eps)).real
        
        # Local energy: E_loc,i = N_i / ψ_S,i (S-space only)
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            (N.n_ss + N.n_sc) / psi_s,
            0.0
        )
        
        # Variance-reduced cotangent: w_i·(E_loc,i - E)
        cot_s = tape.weights * (e_loc_s - energy)
        cot_s = jnp.asarray(cot_s, dtype=tape.log_psi.dtype)
        
        # Gradient via VJP with conj trick: ∇E = Re[∇*⟨·|cot⟩]
        (grad_conj,) = tape.vjp_fn(jnp.conj(cot_s))
        grad = jax.tree.map(jnp.conj, grad_conj)
        
        return GradResult(grad=grad, energy=energy + e_nuc)
    
    return energy_grad_fn


# ============================================================================
# PROXY Mode: Variational Hermitian surrogate
# ============================================================================

def _make_proxy_kernel(ctx: OuterCtx, num_eps: float) -> Callable:
    """
    PROXY mode: E = ⟨ψ|H_proxy|ψ⟩ / ||ψ||²
    
    Full-space variational with Hermitian proxy Hamiltonian.
    """
    n_s = ctx.space.n_s
    spmv_fn = ctx.spmv_fn
    e_nuc = ctx.e_nuc
    
    def energy_grad_fn(params: PyTree, tape: GeometryTape) -> GradResult:
        """Full-space energy/gradient with proxy Hamiltonian."""
        psi_all = jnp.exp(tape.log_psi)
        psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
        
        # SpMV: N = H_proxy·ψ
        N = spmv_fn(psi_s, psi_c)
        
        # Energy: E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / ||ψ||²
        num_s = jnp.vdot(psi_s, N.n_ss + N.n_sc)
        num_c = jnp.vdot(psi_c, N.n_cs + N.n_cc)
        denom = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
        energy = ((num_s + num_c) / jnp.maximum(denom, num_eps)).real
        
        # Local energies in S and C spaces
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
        
        # Cotangent with full-space weights (already normalized)
        cot_full = tape.weights * (e_loc_all - energy)
        cot_full = jnp.asarray(cot_full, dtype=tape.log_psi.dtype)
        
        (grad_conj,) = tape.vjp_fn(jnp.conj(cot_full))
        grad = jax.tree.map(jnp.conj, grad_conj)
        
        return GradResult(grad=grad, energy=energy + e_nuc)
    
    return energy_grad_fn


# ============================================================================
# EFFECTIVE Mode: S-space downfolded Hamiltonian
# ============================================================================

def _make_eff_kernel(ctx: OuterCtx, num_eps: float) -> Callable:
    """
    EFFECTIVE mode: E = ⟨ψ_S|H_eff|ψ_S⟩ / ||ψ_S||²
    
    S-space optimization with downfolded Hamiltonian H_eff.
    """
    spmv_fn = ctx.spmv_fn
    e_nuc = ctx.e_nuc
    
    def energy_grad_fn(params: PyTree, tape: GeometryTape) -> GradResult:
        """S-space energy/gradient with effective Hamiltonian."""
        psi_s = jnp.exp(tape.log_psi)
        
        # SpMV: N = H_eff·ψ_S
        N = spmv_fn(psi_s)
        
        # Energy: E = ⟨ψ_S|N_S⟩ / ||ψ_S||²
        num = jnp.vdot(psi_s, N.n_ss)
        denom = jnp.vdot(psi_s, psi_s)
        energy = (num / jnp.maximum(denom, num_eps)).real
        
        # Local energy
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            N.n_ss / psi_s,
            0.0
        )
        
        # Cotangent with S-space weights (already normalized)
        cot_s = tape.weights * (e_loc_s - energy)
        cot_s = jnp.asarray(cot_s, dtype=tape.log_psi.dtype)
        
        (grad_conj,) = tape.vjp_fn(jnp.conj(cot_s))
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
    Create VMC step with unified tape management.
    
    Single tape creation per step eliminates redundant linearization.
    Tape is reused for both gradient and optimizer (QGT operations).
    
    Args:
        ctx: Outer context
        optimizer: Optax or LEVER optimizer
        num_eps: Numerical threshold
        
    Returns:
        Step function: (state, None) → (new_state, energy)
    """
    is_lever_opt = isinstance(optimizer, LeverOptimizer)
    energy_grad_fn = ModeKernel.make(ctx, num_eps)
    
    # Select appropriate logpsi_fn for tape creation
    def _select_tape_logpsi_fn() -> Callable:
        if ctx.mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC):
            # S-only modes: extract first element if tuple
            return (ctx.logpsi_fn[0] if isinstance(ctx.logpsi_fn, tuple) 
                    else ctx.logpsi_fn)
        else:
            # PROXY mode: full-space
            return ctx.logpsi_fn
    
    tape_logpsi_fn = _select_tape_logpsi_fn()
    
    def _step(state: InnerState, _unused) -> tuple[InnerState, jnp.ndarray]:
        """
        Single optimization step with unified tape creation.
        
        Steps:
          1. Build linearization tape (VJP closure + weights)
          2. Compute energy/gradient via mode kernel
          3. Update parameters with optimizer
        """
        # Phase 1: Single linearization (key optimization)
        tape = prepare_tape(state.params, tape_logpsi_fn, num_eps)
        
        # Phase 2: Energy/gradient computation
        result = energy_grad_fn(state.params, tape)
        
        # Phase 3: Parameter update
        if is_lever_opt:
            # LEVER optimizer: pass tape for QGT
            updates, new_opt = optimizer.update(
                result.grad,
                state.opt_state,
                state.params,
                tape=tape,
                energy=result.energy
            )
        else:
            # Optax optimizer: standard update
            updates, new_opt = optimizer.update(
                result.grad,
                state.opt_state,
                state.params
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
    
    Uses lax.scan for XLA loop unrolling. For CPU with host callbacks,
    Python loop over single-step JIT may offer better performance.
    
    Args:
        ctx: Outer context
        optimizer: Optax or LEVER optimizer
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
    
    return jax.jit(_scan_wrapper, static_argnums=(1,))


__all__ = ["ModeKernel", "create_step_fn", "create_scan_fn"]
