# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-level solver orchestrating outer evolution and inner optimization.

Implements the two-level iterative algorithm:
  - Outer loop: determinant space evolution via selection strategy
  - Inner loop: variational parameter optimization for fixed space

Algorithm flow:
  1. Initialize from HF determinant
  2. Build computational context (Hamiltonian, features, operators)
  3. Optimize wavefunction parameters via gradient descent
  4. Evolve determinant space based on |ψ| amplitudes
  5. Check convergence and repeat

File: lever/solver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp

from . import core
from .config import ComputeMode, ExperimentConfig
from .dtypes import InnerResult, LeverResult, OuterCtx, OuterState
from .engine.context import build_context
from .engine.features import create_psi_cache
from .engine.inner import inner_solve
from .utils.convergence import OuterConvergence

if TYPE_CHECKING:
    from .evolution import EvolutionStrategy
    from .models import WavefunctionModel
    from .utils.monitor import RunContext


# ============================================================================
# History Tracking
# ============================================================================

class _History:
    """
    Flat energy trace with cycle boundaries for post-analysis.
    
    Structure: [E₁₁, E₁₂, ..., E₁ₙ, E₂₁, ..., Eₘₙ]
    Boundaries: [0, n, 2n, ..., m*n] marking cycle starts
    """

    def __init__(self) -> None:
        self.inner_energies: list[float] = []
        self.cycle_bounds: list[int] = [0]

    def append_cycle(self, trace: list[float]) -> None:
        """Append inner optimization trace and update boundary."""
        self.inner_energies.extend(trace)
        self.cycle_bounds.append(len(self.inner_energies))


# ============================================================================
# State Management
# ============================================================================

def _initialize_state(cfg: ExperimentConfig, model: "WavefunctionModel") -> OuterState:
    """Bootstrap solver state from Hartree-Fock determinant."""
    from .interface import load_initial_det

    hf_det = load_initial_det(cfg.system.meta_path)
    return OuterState(
        cycle=0,
        s_dets=hf_det,
        params=model.variables,
        e_ref=None,
    )


def _update_state(
    cfg: ExperimentConfig,
    strategy: "EvolutionStrategy",
    state: OuterState,
    inner: InnerResult,
    ctx: OuterCtx,
    evolve: bool,
) -> OuterState:
    """
    Advance state to next cycle.
    
    Args:
        cfg: Experiment configuration
        strategy: Determinant selection strategy
        state: Current outer state
        inner: Inner optimization result
        ctx: Computational context
        evolve: Whether to evolve determinant space
    
    Returns:
        Updated OuterState with new parameters and optional space evolution
    """
    # Update reference energy for effective mode
    new_e_ref = None
    if cfg.runtime.compute_mode == ComputeMode.EFFECTIVE:
        new_e_ref = inner.energy_trace[-1] - ctx.e_nuc

    if not evolve:
        # Convergence reached: freeze determinant space
        return OuterState(
            cycle=state.cycle + 1,
            s_dets=state.s_dets,
            params=inner.final_params,
            e_ref=new_e_ref,
        )

    # Evolve determinant space via amplitude-based selection
    new_s_dets = strategy.evolve(ctx, inner.psi_cache)
    return OuterState(
        cycle=state.cycle + 1,
        s_dets=new_s_dets,
        params=inner.final_params,
        e_ref=new_e_ref,
    )


# ============================================================================
# Main Solver Loop
# ============================================================================

def outer_evolution(
    cfg: ExperimentConfig,
    model: "WavefunctionModel",
    strategy: "EvolutionStrategy",
    optimizer,
    int_ctx: core.IntCtx,
    monitor: "RunContext | None" = None,
) -> tuple[LeverResult, dict]:
    """
    Execute outer evolution loop with inner optimization.
    
    Algorithm:
      For each cycle until convergence:
        1. Build context: construct H_eff/H_full, feature matrices
        2. Inner solve: optimize ψ parameters via gradient descent
        3. Check convergence: monitor ΔE across cycles
        4. Evolve space: select new determinants based on |ψ| amplitudes
    
    Args:
        cfg: Experiment configuration
        model: Neural wavefunction model
        strategy: Determinant evolution strategy
        optimizer: Optax optimizer instance
        int_ctx: Integral context (FCIDUMP data)
        monitor: Optional UI/logging callback handler
    
    Returns:
        (LeverResult, diagnostics) containing final state and trace data
    """
    loop_cfg = cfg.loop
    state = _initialize_state(cfg, model)
    history = _History()

    outer_conv = OuterConvergence(
        tol=loop_cfg.outer_tol,
        patience=loop_cfg.outer_patience,
    )

    converged = False
    final_ctx: OuterCtx | None = None
    final_psi_cache = None
    diag_cycles: list[dict[str, Any]] = []

    if monitor:
        monitor.on_solver_start(cfg, model)

    for cycle in range(loop_cfg.max_outer):
        cycle_start = time.perf_counter()

        # Build computational context for current determinant space
        ctx, comp_diag = build_context(cfg, state, model, int_ctx)

        if monitor:
            monitor.on_cycle_start(cycle + 1, loop_cfg.max_outer, comp_diag)

        # Inner optimization with optional progress callback
        inner_cb: Callable[[int, float], None] | None = None
        if monitor:
            def inner_cb(step: int, energy: float) -> None:
                monitor.on_inner_progress(step, loop_cfg.max_inner, energy)

        inner = inner_solve(
            ctx,
            state.params,
            optimizer,
            loop_cfg,
            cfg.runtime,
            progress_callback=inner_cb,
        )

        # EFFECTIVE mode: re-evaluate full T-space wavefunction for evolution
        if cfg.runtime.compute_mode == ComputeMode.EFFECTIVE:
            log_psi_full = ctx.log_psi.eval_full(
                inner.final_params, ctx.features_s, ctx.features_c
            )
            psi_cache_full = create_psi_cache(
                log_psi_full, ctx.space.n_s, ctx.space.n_c
            )
            inner = InnerResult(
                final_params=inner.final_params,
                energy_trace=inner.energy_trace,
                psi_cache=psi_cache_full,
                converged=inner.converged,
                steps=inner.steps,
            )

        # Record cycle statistics
        final_energy = inner.energy_trace[-1]
        history.append_cycle(inner.energy_trace)

        final_ctx = ctx
        final_psi_cache = inner.psi_cache

        # Check outer convergence: ΔE < tol for patience cycles
        outer_done, max_delta = outer_conv.update(final_energy)

        # Compute wavefunction norms: ‖ψ_S‖² and ‖ψ_C‖²
        ns = float(jnp.sum(jnp.abs(inner.psi_cache.psi_s) ** 2))
        nc = float(jnp.sum(jnp.abs(inner.psi_cache.psi_c) ** 2))
        cycle_time = time.perf_counter() - cycle_start

        entry = {
            "cycle": cycle + 1,
            "compile": comp_diag,
            "inner": {
                "steps": inner.steps,
                "converged": inner.converged,
                "final_energy": final_energy,
            },
            "wavefunction": {"norm_s": ns, "norm_c": nc},
            "outer": {"converged": outer_done, "delta": max_delta},
            "time_sec": cycle_time,
        }
        diag_cycles.append(entry)

        if monitor:
            monitor.on_cycle_end(entry)

        # Update state and check termination
        if outer_done:
            state = _update_state(cfg, strategy, state, inner, ctx, evolve=False)
            converged = True
            break
        else:
            state = _update_state(cfg, strategy, state, inner, ctx, evolve=True)

            # Ensure device synchronization before next cycle
            try:
                jax.block_until_ready(state.params)
            except Exception:
                pass

    result = LeverResult(
        final_params=state.params,
        final_s_dets=state.s_dets,
        full_energy_history=history.inner_energies,
        cycle_boundaries=history.cycle_bounds,
        total_time=0.0,  # Filled by solve()
        config=cfg,
        final_space=final_ctx.space if final_ctx is not None else None,
        final_psi_cache=final_psi_cache,
    )

    diagnostics = {
        "cycles": diag_cycles,
        "converged": converged,
        "int_ctx": int_ctx,
        "e_nuc": int_ctx.get_e_nuc(),
    }

    return result, diagnostics


def solve(
    cfg: ExperimentConfig,
    model: "WavefunctionModel",
    strategy: "EvolutionStrategy",
    optimizer,
    run_ctx: "RunContext | None" = None,
) -> tuple[LeverResult, dict]:
    """
    High-level solver entry point.
    
    Orchestrates full workflow:
      1. Load integrals from FCIDUMP
      2. Execute outer evolution loop
      3. Collect diagnostics and timing
    
    Args:
        cfg: Experiment configuration
        model: Neural wavefunction model
        strategy: Determinant evolution strategy
        optimizer: Optax optimizer instance
        run_ctx: Optional monitoring context for UI/logging
    
    Returns:
        (LeverResult, diagnostics) containing:
          - Final optimized parameters and determinant space
          - Complete energy trace with cycle boundaries
          - Timing and convergence metadata
    """
    # Prepare integral context with Hamiltonian builder
    int_ctx = core.IntCtx(cfg.system.fcidump_path, cfg.system.n_orb)
    int_ctx.hb_prepare(threshold=1e-15)

    t0 = time.perf_counter()

    result, diagnostics = outer_evolution(
        cfg,
        model,
        strategy,
        optimizer,
        int_ctx,
        monitor=run_ctx,
    )

    elapsed = time.perf_counter() - t0

    # Update result with total runtime
    result = LeverResult(
        final_params=result.final_params,
        final_s_dets=result.final_s_dets,
        full_energy_history=result.full_energy_history,
        cycle_boundaries=result.cycle_boundaries,
        total_time=elapsed,
        config=result.config,
        final_space=result.final_space,
        final_psi_cache=result.final_psi_cache,
    )

    if run_ctx:
        final_delta = (
            diagnostics["cycles"][-1]["outer"]["delta"]
            if diagnostics.get("cycles")
            else 0.0
        )
        run_ctx.on_solver_end(
            diagnostics.get("converged", False),
            elapsed,
            cfg.loop.max_outer,
            float(final_delta),
            diagnostics,
        )

    # Expose integral context for post-analysis
    diagnostics["int_ctx"] = int_ctx
    diagnostics["e_nuc"] = int_ctx.get_e_nuc()

    return result, diagnostics


__all__ = ["outer_evolution", "solve"]
