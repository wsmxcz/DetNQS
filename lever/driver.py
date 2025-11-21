# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER workflow driver with sliding-window convergence detection.

Convergence criterion: ∀i ∈ [k-p, k]: |E_i - E_{i-1}| < τ
where p = patience, τ = tolerance, k = current cycle.

File: lever/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from jax import tree_util

from . import core
from .analysis import VariationalEvaluator
from .config import ComputeMode
from .dtypes import LeverResult, OuterState
from .engine import hamiltonian
from .workflow.compiler import Compiler
from .workflow.fitter import Fitter
from .monitor import get_logger, get_run
from .monitor import storage as monitor_store

if TYPE_CHECKING:
    from .config import LeverConfig
    from .evolution import EvolutionStrategy
    from .models import WavefunctionModel
    from .dtypes import OuterCtx, PsiCache, PyTree


class Driver:
    """LEVER evolutionary workflow orchestrator."""
    
    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        strategy: EvolutionStrategy,
        optimizer
    ):
        self.cfg = config
        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer
        self.logger = get_logger()
        
        # Initialize integral engine
        self.int_ctx = core.IntCtx(
            config.system.fcidump_path,
            config.system.n_orbitals
        )
        self.int_ctx.hb_prepare(threshold=1e-15)
        
        # Build workflow components
        self.compiler = Compiler(
            config=self.cfg,
            model=self.model,
            int_ctx=self.int_ctx
        )
        
        self.fitter = Fitter(
            loop_cfg=self.cfg.loop,
            report_interval=self.cfg.report_interval,
            num_eps=self.cfg.num_eps
        )
        
        self.evaluator = VariationalEvaluator(
            int_ctx=self.int_ctx,
            n_orb=self.cfg.system.n_orbitals,
            e_nuc=self.int_ctx.get_e_nuc()
        )

    def _print_cycle_header(self, cycle: int, max_cycles: int, comp_diag: dict):
        """Log compact header for outer iteration."""
        self.logger.info(self.logger._blue(f"Cycle {cycle}/{max_cycles}"))
        
        if comp_diag['bootstrap_range'] is not None:
            min_val, max_val = comp_diag['bootstrap_range']
            self.logger.info(f"Bootstrap: [{min_val:.6f}, {max_val:.6f}]")
        
        n_s, n_c = comp_diag['n_s'], comp_diag['n_c']
        nnz_ss, nnz_sc = comp_diag['nnz_ss'], comp_diag['nnz_sc']
        density = 100.0 * nnz_ss / (n_s * n_s) if n_s > 0 else 0.0
        
        ham_info = f"{comp_diag['ham_label']}: {nnz_ss} nnz ({density:.1f}%)"
        if nnz_sc is not None:
            ham_info += f", H_SC: {nnz_sc} nnz"
        
        self.logger.info(f"Space: S={n_s}, C={n_c} | {ham_info}")
        
    def _print_cycle_summary(
        self,
        psi_s_norm_sq: float,
        psi_c_norm_sq: float,
        energy: float,
        steps: int,
        time_elapsed: float
    ):
        """Log key diagnostics for outer iteration."""
        self.logger.info(
            f"‖ψ_S‖²={psi_s_norm_sq:.4f}, ‖ψ_C‖²={psi_c_norm_sq:.4f}"
        )
        e_str = self.logger._blue(f"E = {energy:.10f}")
        self.logger.info(
            f"{e_str} | Steps = {steps:4d} | Time = {time_elapsed:.2f}s"
        )

    def _check_convergence(
        self,
        recent_energies: list[float],
        patience: int,
        tol: float
    ) -> tuple[bool, float]:
        """
        Sliding-window convergence test.
        
        Returns:
            (converged, max_delta)
        """
        if len(recent_energies) <= patience:
            return False, float('inf')
        
        deltas = [
            abs(recent_energies[i] - recent_energies[i-1])
            for i in range(-patience, 0)
        ]
        
        max_delta = max(deltas)
        return max_delta < tol, max_delta

    def run(self) -> LeverResult:
        """Execute iterative cycles until convergence."""
        self._print_header()
        
        state = self._initialize_state()
        history = _History()
        t0 = time.time()
        
        max_outer = self.cfg.loop.max_outer
        patience = self.cfg.loop.outer_patience
        tol = self.cfg.loop.outer_tol
        
        recent_energies = []
        converged = False

        final_ctx: OuterCtx | None = None
        final_psi_cache: PsiCache | None = None
        
        for cycle in range(max_outer):
            cycle_start = time.time()
            
            # Compile: generate S/C spaces and Hamiltonian
            ctx, comp_diag = self.compiler.compile(state)
            self._print_cycle_header(cycle + 1, max_outer, comp_diag)
            
            # Fit: inner-loop optimization
            result = self.fitter.fit(ctx, state.params, self.optimizer)
            
            final_ctx = ctx
            final_psi_cache = result.psi_cache

            # Compute wavefunction diagnostics
            psi_s = result.psi_cache.psi_s
            psi_c = result.psi_cache.psi_c
            psi_s_norm_sq = float(np.sum(np.abs(psi_s)**2))
            psi_c_norm_sq = float(np.sum(np.abs(psi_c)**2))
            
            final_energy = result.energy_trace[-1]
            max_inner = len(result.energy_trace)
            cycle_time = time.time() - cycle_start
            
            self._print_cycle_summary(
                psi_s_norm_sq, psi_c_norm_sq,
                final_energy, max_inner, cycle_time
            )
            
            history.append_cycle(result.energy_trace)
            recent_energies.append(final_energy)
            
            # Check convergence
            converged, max_delta = self._check_convergence(
                recent_energies, patience, tol
            )
            
            if converged:
                self.logger.converged(cycle + 1, max_delta)
                state = self._update_state(state, result, ctx, evolve=False)
                break
            
            # Trim history window
            if len(recent_energies) > patience + 1:
                recent_energies = recent_energies[-(patience+1):]
            
            state = self._update_state(state, result, ctx, evolve=True)
            
            if cycle < max_outer - 1:
                self.logger.separator()
        
        if not converged:
            self.logger.max_cycles_reached(max_outer)
        
        return self._finalize_results(
            state, history, time.time() - t0, 
            final_ctx, final_psi_cache
        )
    
    def _initialize_state(self) -> OuterState:
        """Bootstrap from Hartree-Fock determinant."""
        sys = self.cfg.system
        hf_det = np.array([
            [(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]
        ], dtype=np.uint64)
        
        return OuterState(
            cycle=0,
            s_dets=hf_det,
            params=self.model.variables,
            e_ref=None
        )
    
    def _update_state(self, state, result, ctx, evolve: bool) -> OuterState:
        """Update parameters and optionally evolve S-space determinants."""
        new_e_ref = self._compute_e_ref(result, ctx)
        
        if not evolve:
            return OuterState(
                cycle=state.cycle + 1,
                s_dets=state.s_dets,
                params=result.final_params,
                e_ref=new_e_ref
            )
        
        new_s_dets = self.strategy.evolve(ctx, result.psi_cache)
        
        return OuterState(
            cycle=state.cycle + 1,
            s_dets=new_s_dets,
            params=result.final_params,
            e_ref=new_e_ref
        )
    
    def _compute_e_ref(self, result, ctx) -> float | None:
        """Compute reference energy for effective Hamiltonian shift."""
        if self.cfg.compute_mode != ComputeMode.EFFECTIVE:
            return None
        return result.energy_trace[-1] - ctx.e_nuc
    
    def _print_header(self):
        """Log run header with configuration details."""
        self.logger.header("LEVER Initialization")
        
        self.logger.info(
            f"Compute mode        : {self.cfg.compute_mode.value.upper()}"
        )
        self.logger.info(
            f"System              : {self.cfg.system.fcidump_path}"
        )
        self.logger.info(
            f"Screening           : {self.cfg.hamiltonian.screening_mode.value}"
        )
        
        self.logger.info(f"{'='*60}")
    
    def _finalize_results(
        self,
        state: OuterState,
        history: _History,
        total_time: float,
        final_ctx: OuterCtx | None,
        final_psi_cache: PsiCache | None
    ) -> LeverResult:
        """Build final result object and persist artifacts."""
        inner = history.inner_energies
        final_energy = inner[-1] if inner else 0.0
        n_cycles = len(history.cycle_bounds) - 1

        self.logger.final_summary(final_energy, n_cycles, total_time)

        result = LeverResult(
            final_params=state.params,
            final_s_dets=state.s_dets,
            full_energy_history=inner,
            cycle_boundaries=history.cycle_bounds,
            total_time=total_time,
            config=self.cfg,
            final_space=final_ctx.space if final_ctx is not None else None,
            final_psi_cache=final_psi_cache
        )

        # Persist structured artifacts
        run = get_run()
        if run is not None:
            try:
                monitor_store.save_run_artifacts(
                    run.root,
                    result,
                    outer_energies=history.outer_energies,
                    outer_steps=history.outer_steps,
                )
            except Exception as e:
                self.logger.warning(f"Failed to save run artifacts: {e}")

        return result


class _History:
    """Energy trajectory tracker for convergence analysis."""

    def __init__(self):
        self.inner_energies: list[float] = []
        self.cycle_bounds: list[int] = [0]
        self.outer_energies: list[float] = []
        self.outer_steps: list[int] = []

    def append_cycle(self, energy_trace: list[float]):
        """Append inner-loop energies for one outer cycle."""
        if not energy_trace:
            self.cycle_bounds.append(len(self.inner_energies))
            self.outer_energies.append(
                self.inner_energies[-1] if self.inner_energies else 0.0
            )
            self.outer_steps.append(0)
            return

        self.inner_energies.extend(energy_trace)
        self.cycle_bounds.append(len(self.inner_energies))
        self.outer_energies.append(float(energy_trace[-1]))
        self.outer_steps.append(len(energy_trace))


__all__ = ["Driver"]
