# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inner-loop variational optimization with early stopping.

Executes fixed-context gradient descent on neural network parameters
until convergence or max iterations. Uses JIT-compiled step kernel
driven by Python host loop for flexible monitoring.

Algorithm:
  1. Initialize optimizer state
  2. Repeat until convergence or max_steps:
     - Compute ∇E[ψ(θ)] via VQS gradient estimator
     - Update θ ← optimizer(θ, ∇E)
     - Check convergence: |ΔE| < tol for patience steps
  3. Cache final wavefunction ψ(θ*) for outer loop

File: lever/engine/inner.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from ..config import LoopConfig, RuntimeConfig
from ..dtypes import InnerResult, InnerState, OuterCtx, PyTree
from ..utils.convergence import InnerConvergence
from .features import create_psi_cache
from .vqs import create_step_kernel

if TYPE_CHECKING:
    from optax import GradientTransformation


# ============================================================================
# Inner Loop Solver
# ============================================================================

def inner_solve(
    ctx: OuterCtx,
    initial_params: PyTree,
    optimizer: GradientTransformation,
    loop_cfg: LoopConfig,
    runtime_cfg: RuntimeConfig,
    progress_callback: Callable[[int, float], None] | None = None,
) -> InnerResult:
    """
    Execute inner-loop optimization on fixed Hamiltonian context.
    
    Minimizes E[ψ(θ)] = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ using gradient descent with
    optional early stopping based on energy convergence.
    
    Args:
        ctx: Fixed outer context (features, Hamiltonian, space dims)
        initial_params: Starting neural network weights θ₀
        optimizer: Optax optimizer (e.g., Adam, SGD)
        loop_cfg: Convergence criteria (max_inner, tol, patience)
        runtime_cfg: Numerical precision and reporting settings
        progress_callback: Optional host function(step, energy) for monitoring
    
    Returns:
        InnerResult containing:
          - final_params: Optimized θ*
          - energy_trace: E[ψ(θₖ)] history
          - psi_cache: Final wavefunction amplitudes
          - converged: Early stopping flag
          - steps: Total iterations executed
    """
    # Initialize optimizer state
    opt_state = optimizer.init(initial_params)
    state = InnerState(params=initial_params, opt_state=opt_state, step=0)
    
    # JIT-compile single-step kernel: (state, features) → (state', E, log_psi)
    step_kernel = create_step_kernel(ctx, optimizer, runtime_cfg.num_eps)
    
    # Extract convergence parameters
    max_steps = loop_cfg.max_inner
    tol = getattr(loop_cfg, "inner_tol", 0.0)
    patience = getattr(loop_cfg, "inner_patience", 0)
    inner_conv = InnerConvergence(tol=tol, patience=patience)
    
    # Setup progress reporting
    report_interval = getattr(runtime_cfg, "report_interval", 0)
    should_report = (report_interval > 0) and (progress_callback is not None)
    
    # Feature matrices (device arrays, constant during inner loop)
    feat_s = ctx.features_s
    feat_c = ctx.features_c
    
    # Optimization loop state
    energy_trace: list[float] = []
    converged = False
    last_log_psi = None
    
    for k in range(max_steps):
        # Execute one gradient step on device
        state, energy, log_psi = step_kernel(state, feat_s, feat_c)
        
        # Transfer energy to host for convergence check
        energy_host = float(jax.device_get(energy))
        energy_trace.append(energy_host)
        last_log_psi = log_psi
        
        # Check early stopping (host-side logic)
        if inner_conv.enabled:
            done, _ = inner_conv.update(energy_host)
            if done:
                converged = True
                if progress_callback:
                    progress_callback(k + 1, energy_host)
                break
        
        # Periodic progress reporting
        if should_report and ((k + 1) % report_interval == 0):
            progress_callback(k + 1, energy_host)
    
    # Cache final wavefunction for outer loop reuse
    psi_cache = create_psi_cache(last_log_psi, ctx.space.n_s, ctx.space.n_c)
    
    return InnerResult(
        final_params=state.params,
        energy_trace=energy_trace,
        psi_cache=psi_cache,
        converged=converged,
        steps=len(energy_trace),
    )


__all__ = ["inner_solve"]
