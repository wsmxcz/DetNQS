# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JIT-compiled VMC optimization step kernels.

Implements mode-specific energy/gradient calculations:
  - EFFECTIVE:  E = ⟨ψ_S|H_eff|ψ_S⟩ / ‖ψ_S‖²
  - PROXY:      E = ⟨ψ|H_proxy|ψ⟩ / ‖ψ‖²  
  - ASYMMETRIC: E = ⟨ψ_S|H|ψ⟩ / ‖ψ_S‖²

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
from ..utils.dtypes import GradResult, OptState
from ..config import ComputeMode

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..utils.dtypes import Workspace, PyTree


class ModeKernel:
    """Factory for mode-specific energy/gradient computation kernels."""
  
    @staticmethod
    def make(
        workspace: Workspace,
        num_eps: float
    ) -> Callable[[PyTree, GeometryTape], GradResult]:
        """
        Create kernel for given computation mode.
      
        Args:
            workspace: Precompiled operators and configuration
            num_eps: Numerical stability threshold for division
      
        Returns:
            Pure function: (params, tape) → GradResult
        """
        kernel_factory = {
            ComputeMode.EFFECTIVE: _make_eff_kernel,
            ComputeMode.PROXY: _make_proxy_kernel,
            ComputeMode.ASYMMETRIC: _make_asym_kernel,
        }
      
        factory_fn = kernel_factory.get(workspace.mode)
        if factory_fn is None:
            raise ValueError(f"Unsupported mode: {workspace.mode}")
      
        return factory_fn(workspace, num_eps)


def _make_eff_kernel(workspace: Workspace, num_eps: float) -> Callable:
    """
    S-space energy and gradient with effective Hamiltonian.
  
    Energy:   E = ⟨ψ_S|H_eff|ψ_S⟩ / ‖ψ_S‖²
    Gradient: ∇E = 2 Re[⟨O_k (E_loc - E)⟩]  (variance-reduced form)
    """
    spmv_fn = workspace.spmv_fn
    e_nuc = workspace.e_nuc
  
    def energy_grad_fn(params: PyTree, tape: GeometryTape) -> GradResult:
        psi_s = jnp.exp(tape.log_psi)
      
        # Matrix-vector product: N = H_eff @ ψ_S
        N = spmv_fn(psi_s)
      
        # Energy: E = ⟨ψ_S|N⟩ / ⟨ψ_S|ψ_S⟩
        numerator = jnp.vdot(psi_s, N.n_ss)
        denominator = jnp.vdot(psi_s, psi_s)
        energy = (numerator / jnp.maximum(denominator, num_eps)).real
      
        # Local energy: E_loc[i] = N[i] / ψ_S[i]
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            N.n_ss / psi_s,
            0.0
        )
      
        # Variance-reduced cotangent: cot = w * (E_loc - E)
        cotangent = tape.weights * (e_loc_s - energy)
        cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
      
        # Gradient via VJP with conjugation trick: ∇E = conj(VJP[conj(cot)])
        (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
        gradient = jax.tree.map(jnp.conj, grad_conj)
      
        return GradResult(grad=gradient, energy=energy + e_nuc)
  
    return energy_grad_fn


def _make_proxy_kernel(workspace: Workspace, num_eps: float) -> Callable:
    """
    Full T-space energy and gradient with proxy Hamiltonian.
  
    Energy: E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / (‖ψ_S‖² + ‖ψ_C‖²)
    """
    n_s = workspace.space.n_s
    spmv_fn = workspace.spmv_fn
    e_nuc = workspace.e_nuc
  
    def energy_grad_fn(params: PyTree, tape: GeometryTape) -> GradResult:
        psi_all = jnp.exp(tape.log_psi)
        psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
      
        # Block SpMV: N = H_proxy @ [ψ_S; ψ_C]
        N = spmv_fn(psi_s, psi_c)
      
        # Energy: E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / ‖ψ‖²
        num_s = jnp.vdot(psi_s, N.n_ss + N.n_sc)
        num_c = jnp.vdot(psi_c, N.n_cs + N.n_cc)
        denom = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
        energy = ((num_s + num_c) / jnp.maximum(denom, num_eps)).real
      
        # Local energies for S and C spaces
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
      
        # Full-space cotangent
        cotangent = tape.weights * (e_loc_all - energy)
        cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
      
        (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
        gradient = jax.tree.map(jnp.conj, grad_conj)
      
        return GradResult(grad=gradient, energy=energy + e_nuc)
  
    return energy_grad_fn


def _make_asym_kernel(workspace: Workspace, num_eps: float) -> Callable:
    """Asymmetric kernel with external full-space evaluator"""
    n_s, n_c = workspace.space.n_s, workspace.space.n_c
    spmv_fn = workspace.spmv_fn
    e_nuc = workspace.e_nuc
    
    def energy_grad_fn(
        params: PyTree,
        tape: GeometryTape,
        logpsi_full_fn: Callable
    ) -> GradResult:
        # Full wavefunction for SpMV (S∪C normalization)
        log_psi_full = logpsi_full_fn(params)
        psi_all = jnp.exp(log_psi_full)
        psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]
        
        # SpMV: N = H @ [ψ_S; ψ_C]
        N = spmv_fn(psi_s, psi_c)
        
        # Energy: E = ⟨ψ_S|N_S⟩ / ⟨ψ_S|ψ_S⟩ (S-space normalization)
        numerator = jnp.vdot(psi_s, N.n_ss + N.n_sc)
        denominator = jnp.vdot(psi_s, psi_s)
        energy = (numerator / jnp.maximum(denominator, num_eps)).real
        
        # Local energy on S-space with tape from S-normalized logψ
        e_loc_s = jnp.where(
            jnp.abs(psi_s) >= num_eps,
            (N.n_ss + N.n_sc) / psi_s,
            0.0
        )
        
        cotangent = tape.weights * (e_loc_s - energy)
        cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)
        
        (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
        gradient = jax.tree.map(jnp.conj, grad_conj)
        
        return GradResult(grad=gradient, energy=energy + e_nuc)
    
    return energy_grad_fn


def create_step_fn(
    workspace: Workspace,
    optimizer,
    num_eps: float
) -> Callable[[OptState, jnp.ndarray, jnp.ndarray], tuple[OptState, jnp.ndarray]]:
    """
    Create JIT-compiled step with features as arguments.
    
    Signature: (state, feat_s, feat_c) → (new_state, energy)
    Features are passed as JIT args to avoid constant folding.
    """
    is_lever_opt = isinstance(optimizer, LeverOptimizer)
    energy_grad_fn = ModeKernel.make(workspace, num_eps)
    logpsi_evals = workspace.log_psi
    
    def _step(
        state: OptState,
        feat_s: jnp.ndarray,
        feat_c: jnp.ndarray
    ) -> tuple[OptState, jnp.ndarray]:
        # Bind features inside JIT (tracers, not Python constants)
        if isinstance(logpsi_evals, tuple):
            eval_s, eval_full = logpsi_evals
            logpsi_opt = lambda p: eval_s(p, feat_s)
            logpsi_full = lambda p: eval_full(p, feat_s, feat_c)
        else:
            eval_full = logpsi_evals
            logpsi_full = lambda p: eval_full(p, feat_s, feat_c)
            logpsi_opt = logpsi_full
        
        # Select logψ for tape construction
        tape_logpsi_fn = (
            logpsi_opt if workspace.mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC)
            else logpsi_full
        )
        
        # Phase 1: Build geometry tape
        tape = prepare_tape(state.params, tape_logpsi_fn, num_eps)
        
        # Phase 2: Energy and gradient
        if workspace.mode == ComputeMode.ASYMMETRIC:
            result = energy_grad_fn(state.params, tape, logpsi_full)
        else:
            result = energy_grad_fn(state.params, tape)
        
        # Phase 3: Parameter update
        if is_lever_opt:
            updates, new_opt_state = optimizer.update(
                result.grad, state.opt_state, state.params,
                tape=tape, energy=result.energy
            )
        else:
            updates, new_opt_state = optimizer.update(
                result.grad, state.opt_state, state.params
            )
        
        new_params = optax.apply_updates(state.params, updates)
        new_state = OptState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1
        )
        
        return new_state, result.energy
    
    return jax.jit(_step, donate_argnums=(0,))


def create_scan_fn(
    workspace: Workspace,
    optimizer,
    num_eps: float
) -> Callable[[OptState, int], tuple[OptState, jnp.ndarray]]:
    """
    Create JIT-compiled multi-step execution via lax.scan.
  
    Args:
        workspace: Precompiled operators
        optimizer: Optax or LEVER optimizer
        num_eps: Numerical stability threshold
  
    Returns:
        Pure function: (state, n_steps) → (final_state, energies)
    """
    step_fn = create_step_fn(workspace, optimizer, num_eps)
  
    def _scan_wrapper(state: OptState, n_steps: int):
        """Execute n_steps optimization iterations."""
        final_state, energies = lax.scan(
            step_fn,
            state,
            None,
            length=n_steps
        )
        return final_state, energies
  
    return jax.jit(_scan_wrapper, static_argnums=(1,))


__all__ = ["ModeKernel", "create_step_fn", "create_scan_fn"]
