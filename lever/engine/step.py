# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-step parameter update for variational optimization.

Provides JIT-compiled update step combining energy/gradient computation
with optimizer parameter updates.

File: lever/engine/step.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from .geometry import prepare_tape
from .gradient import compute_energy_and_grad
from ..dtypes import InnerState, ComputeMode

if TYPE_CHECKING:
    from ..dtypes import OuterCtx
    from typing import Callable


def create_update_step(
    ctx: OuterCtx,
    optimizer,
    num_eps: float = 1e-12
) -> Callable[[InnerState], tuple[InnerState, jnp.ndarray]]:
    """
    Create JIT-compiled single optimization step.
    
    Combines three phases into single traced function:
      1. Geometry: Build tape via linearization
      2. Gradient: Compute energy and parameter gradients
      3. Update: Apply optimizer transformation
    
    Args:
        ctx: Outer context with compiled operators
        optimizer: Optax-compatible optimizer
        num_eps: Numerical stability threshold
        
    Returns:
        Pure function: state → (new_state, energy)
    """
    from ..optimizers.base import Optimizer as LeverOptimizer
    is_lever_opt = isinstance(optimizer, LeverOptimizer)
    
    # Select appropriate log_psi function for tape construction
    # All closures already have features captured, only need params
    if ctx.compute_mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC):
        # Use S-space evaluator (tuple format)
        if isinstance(ctx.log_psi_fn, tuple):
            logpsi_for_tape = ctx.log_psi_fn[0]  # S-space closure
        else:
            # Fallback: assume single evaluator, slice output
            logpsi_for_tape = lambda p: ctx.log_psi_fn(p)[:ctx.space.n_s]
    else:
        # PROXY mode: use full evaluator
        if isinstance(ctx.log_psi_fn, tuple):
            logpsi_for_tape = ctx.log_psi_fn[1]  # Full closure
        else:
            logpsi_for_tape = ctx.log_psi_fn
    
    def _step(state: InnerState) -> tuple[InnerState, jnp.ndarray]:
        """Single optimization step: tape → grad → update."""
        
        # Phase 1: Build geometry tape via single linearization
        tape = prepare_tape(state.params, logpsi_for_tape, num_eps)
        
        # Phase 2: Compute energy and gradients
        grad_result = compute_energy_and_grad(
            state.params, tape, ctx, num_eps
        )
        
        # Phase 3: Parameter update
        if is_lever_opt:
            # LEVER optimizers (SR, KFAC) use tape for natural gradient
            updates, new_opt_state = optimizer.update(
                grad_result.grad, state.opt_state, state.params,
                tape=tape, energy=grad_result.energy
            )
        else:
            # Standard Optax optimizers (Adam, SGD)
            updates, new_opt_state = optimizer.update(
                grad_result.grad, state.opt_state, state.params
            )
        
        new_params = optax.apply_updates(state.params, updates)
        new_state = InnerState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1
        )
        
        return new_state, grad_result.energy
    
    return jax.jit(_step, donate_argnums=(0,))


__all__ = ["create_update_step"]
