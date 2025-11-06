# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-step variational optimization with JIT compilation.

Implements parameter updates for VMC energy minimization. Features are
passed as dynamic tracers to prevent XLA constant folding.

Core algorithm:
  1. Linearize log(ψ) around current parameters
  2. Compute ∇E_loc and covariance via tape
  3. Apply optimizer update (SR/KFAC/Adam/etc.)

File: lever/engine/step.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import optax

from .geometry import prepare_tape
from .gradient import compute_energy_and_grad
from ..dtypes import ComputeMode, InnerState

if TYPE_CHECKING:
    from ..dtypes import OuterCtx


def create_update_step(
    ctx: OuterCtx,
    optimizer: optax.GradientTransformation,
    num_eps: float = 1e-12
) -> Callable[[InnerState, jnp.ndarray, jnp.ndarray], tuple[InnerState, jnp.ndarray]]:
    """
    Factory for JIT-compiled optimization step.
    
    Features passed as runtime tracers prevent XLA from baking constants
    into compiled code, enabling reuse across training iterations.
    
    Args:
        ctx: Outer context with compiled operators (SpMV, log_psi)
        optimizer: Optax-compatible update rule (Adam/SR/KFAC/...)
        num_eps: Numerical stability for finite differences (default: 1e-12)
        
    Returns:
        Pure function: (state, feat_s, feat_c) → (new_state, energy)
        
    Note:
        Natural gradient methods (SR, KFAC) require tape access via
        optimizer.update(..., tape=tape).
    """
    from ..optimizers.base import Optimizer as LeverOptimizer
    
    is_natural_grad = isinstance(optimizer, LeverOptimizer)
    mode = ctx.compute_mode
    
    # Unpack mode-specific evaluators
    if mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC):
        eval_s, eval_full = ctx.log_psi_fn
    else:
        eval_full = ctx.log_psi_fn
    
    def _step(
        state: InnerState,
        feat_s: jnp.ndarray,  # Tracer: S-space features [n_s, d]
        feat_c: jnp.ndarray   # Tracer: C-space features [n_c, d]
    ) -> tuple[InnerState, jnp.ndarray]:
        """
        Single optimization step: params_t → params_{t+1}.
        
        Workflow:
          1. Linearize log(ψ) via finite differences → tape
          2. Compute E_loc and ∇_θ E via covariance formula
          3. Update: θ_{t+1} = θ_t - η · [F^{-1}] · ∇E  (for SR)
                     or     θ_{t+1} = θ_t - η · ∇E       (for Adam)
        
        Args:
            state: Current (params, opt_state, step)
            feat_s/c: Feature matrices (tracers in JIT context)
            
        Returns:
            (new_state, energy): Updated state and current energy
        """
        # Construct parameter-bound evaluator with tracer features
        if mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC):
            logpsi_fn = lambda p: eval_s(p, feat_s)
        else:
            logpsi_fn = lambda p: eval_full(p, feat_s, feat_c)
        
        # Phase 1: Build geometry via linearization
        tape = prepare_tape(state.params, logpsi_fn, num_eps)
        
        # Phase 2: Energy and gradient via covariance formula
        # ∇E = 2 Re[⟨O^† (E_loc - E)⟩], O = ∇log(ψ)
        grad_result = compute_energy_and_grad(
            state.params, tape, ctx, num_eps
        )
        
        # Phase 3: Optimizer update
        if is_natural_grad:
            # Natural gradient: requires Fisher matrix from tape
            # θ_{t+1} = θ_t - η · F^{-1} · ∇E
            updates, new_opt_state = optimizer.update(
                grad_result.grad,
                state.opt_state,
                state.params,
                tape=tape,
                energy=grad_result.energy
            )
        else:
            # Standard gradient descent (Adam/SGD/...)
            # θ_{t+1} = θ_t - η · m_t / (√v_t + ε)  (for Adam)
            updates, new_opt_state = optimizer.update(
                grad_result.grad,
                state.opt_state,
                state.params
            )
        
        new_params = optax.apply_updates(state.params, updates)
        new_state = InnerState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1
        )
        
        return new_state, grad_result.energy
    
    # Donate state for memory efficiency, features are read-only
    return jax.jit(_step, donate_argnums=(0,))


__all__ = ["create_update_step"]
