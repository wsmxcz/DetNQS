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
from ..monitor import get_logger

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
        Run inner-loop optimization with optional early stopping.
        """
        opt_state = optimizer.init(initial_params)
        state = InnerState(
            params=initial_params,
            opt_state=opt_state,
            step=0
        )
        
        update_step = create_update_step(ctx, optimizer, self.num_eps)
        # new: get state, energy trace, number of steps, and convergence flag
        state, energies, steps, converged = self._run_optimization_loop(
            state, update_step, ctx
        )

        # Move data to host and convert to Python types
        energies_host = jax.device_get(energies)
        steps_host = int(jax.device_get(steps))
        converged_host = bool(jax.device_get(converged))

        energies_list = [float(e) for e in energies_host[:steps_host]]
        psi_cache = self._create_cache(ctx, state.params)
        
        return InnerResult(
            final_params=state.params,
            energy_trace=energies_list,
            psi_cache=psi_cache,
            converged=converged_host,
            steps=steps_host
        )
    
    def _run_optimization_loop(
        self,
        initial_state: InnerState,
        update_step,
        ctx: OuterCtx
    ) -> tuple[InnerState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Run optimization with optional early stopping based on energy plateaus.
        
        Returns:
            final_state: InnerState after last step
            energies: jnp.ndarray[max_steps] with recorded energies
            steps: scalar int32, number of executed steps
            converged: scalar bool, True if stopped by early-stopping criterion
        """
        max_steps = self.cfg.max_inner
        tol = getattr(self.cfg, "inner_tol", 0.0)
        patience = getattr(self.cfg, "inner_patience", 0)

        # Early stopping is enabled only if both tol and patience are positive.
        use_early = (tol > 0.0) and (patience > 0)

        report_interval = self.report_interval
        should_print = report_interval > 0

        feat_s = ctx.features_s
        feat_c = ctx.features_c

        # Preallocate energy buffer on device
        energies_init = jnp.empty((max_steps,), dtype=jnp.float64)

        # carry = (state, energies, k, last_energy, streak, converged)
        carry_init = (
            initial_state,
            energies_init,
            jnp.array(0, dtype=jnp.int32),       # current step index
            jnp.inf,                              # last energy (no previous value)
            jnp.array(0, dtype=jnp.int32),       # streak of small |ΔE|
            jnp.array(False)                     # early-stopping flag
        )

        def cond_fun(carry):
            state, energies, k, last_energy, streak, converged = carry
            # Stop if we hit max_steps or early-stopping is triggered.
            not_done = jnp.logical_and(
                k < max_steps,
                jnp.logical_not(converged)
            )
            return not_done

        def body_fun(carry):
            state, energies, k, last_energy, streak, converged = carry

            # One optimization step
            new_state, energy = update_step(state, feat_s, feat_c)
            energies = energies.at[k].set(energy)

            # Compute energy difference to previous step
            delta = jnp.abs(energy - last_energy)

            # Update streak: consecutive steps with |ΔE| < tol
            new_streak = jnp.where(delta < tol, streak + 1, 0)

            # Trigger convergence once streak reaches patience
            new_converged = jnp.logical_or(
                converged,
                jnp.logical_and(use_early, new_streak >= patience)
            )

            # Optional progress logging
            if should_print:
                step_for_print = k + 1
                is_report_step = (step_for_print % report_interval) == 0
                lax.cond(
                    is_report_step,
                    lambda e, s: jax.debug.print(
                        "  Step {step:4d}/{total:4d} | E = {energy:.10f}",
                        step=s,
                        total=max_steps,
                        energy=e,
                        ordered=True
                    ),
                    lambda e, s: None,
                    energy,
                    step_for_print
                )

            return (
                new_state,
                energies,
                k + 1,
                energy,        # update last_energy
                new_streak,
                new_converged
            )

        final_state, energies, final_k, _, _, converged = lax.while_loop(
            cond_fun,
            body_fun,
            carry_init
        )

        # final_k and converged remain on device; conversion is done in fit()
        return final_state, energies, final_k, converged
    
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
