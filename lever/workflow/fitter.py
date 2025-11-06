# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter optimizer for fixed computational context.

Executes inner loop via gradient descent on variational energy:
  E[θ] = ⟨Ψ(θ)|H|Ψ(θ)⟩ / ⟨Ψ(θ)|Ψ(θ)⟩

File: lever/workflow/fitter.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.lax as lax

from ..dtypes import InnerResult, InnerState
from ..engine.step import create_update_step
from ..utils.features import create_psi_cache
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..config import LoopConfig
    from ..dtypes import OuterCtx, PyTree


class Fitter:
    """
    Inner loop optimizer for variational energy minimization.
    
    Executes fixed number of gradient descent steps on static Hamiltonian.
    """
    
    def __init__(
        self,
        loop_cfg: LoopConfig,
        report_interval: int,
        num_eps: float = 1e-12
    ) -> None:
        """
        Initialize fitter with optimization parameters.
        
        Args:
            loop_cfg: Iteration settings
            report_interval: Progress logging frequency (0 to disable)
            num_eps: Numerical stability threshold
        """
        self.cfg = loop_cfg
        self.report_interval = report_interval
        self.num_eps = num_eps
        self.logger = get_logger()
    
    def fit(
        self,
        ctx: OuterCtx,
        initial_params: PyTree,
        optimizer
    ) -> InnerResult:
        """
        Execute inner loop optimization for fixed number of steps.
        
        Algorithm: θ ← optimizer.update(∇E[θ], θ) via lax.scan
        
        Args:
            ctx: Compiled computational context
            initial_params: Initial network parameters
            optimizer: Optax-compatible optimizer
            
        Returns:
            InnerResult with optimized parameters and energy trace
        """
        opt_state = optimizer.init(initial_params)
        state = InnerState(
            params=initial_params,
            opt_state=opt_state,
            step=0
        )
        
        update_step = create_update_step(ctx, optimizer, self.num_eps)
        state, energies = self._run_optimization_loop(state, update_step, ctx)
        psi_cache = self._create_cache(ctx, state.params)
        
        return InnerResult(
            final_params=state.params,
            energy_trace=energies.tolist(),
            psi_cache=psi_cache,
            converged=False,
            steps=len(energies)
        )
    
    def _run_optimization_loop(
        self,
        initial_state: InnerState,
        update_step,
        ctx: OuterCtx
    ) -> tuple[InnerState, jnp.ndarray]:
        """
        Execute fixed-step optimization via lax.scan with optional progress.
        
        Returns:
            Final state and energy trajectory
        """
        num_steps = self.cfg.inner_steps
        report_interval = self.report_interval
        should_print = report_interval > 0
        
        feat_s = ctx.features_s
        feat_c = ctx.features_c
        
        def scan_fn(state, step_idx):
            """Single optimization step."""
            new_state, energy = update_step(state, feat_s, feat_c)
            
            if should_print:
                is_report_step = (step_idx + 1) % report_interval == 0
                lax.cond(
                    is_report_step,
                    lambda e, s: jax.debug.print(
                        "  Step {step:4d}/{total:4d} | E = {energy:.10f}",
                        step=s + 1,
                        total=num_steps,
                        energy=e,
                        ordered=True
                    ),
                    lambda e, s: None,
                    energy, step_idx
                )
            
            return new_state, energy
        
        step_indices = jnp.arange(num_steps)
        final_state, energies = lax.scan(scan_fn, initial_state, step_indices)
        
        return final_state, jax.device_get(energies)
    
    def _create_cache(
        self,
        ctx: OuterCtx,
        params: PyTree
    ):
        """
        Evaluate and cache Ψ_S, Ψ_C amplitudes.
        
        Args:
            ctx: Compiled context with log_psi evaluator
            params: Optimized network parameters
            
        Returns:
            PsiCache with S/C space amplitudes
        """
        if isinstance(ctx.log_psi_fn, tuple):
            _, logpsi_full = ctx.log_psi_fn
            log_all = logpsi_full(params, ctx.features_s, ctx.features_c)
        else:
            log_all = ctx.log_psi_fn(params, ctx.features_s, ctx.features_c)
        
        return create_psi_cache(log_all, ctx.space.n_s, ctx.space.n_c)


__all__ = ["Fitter"]
