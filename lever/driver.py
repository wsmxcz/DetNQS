# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER workflow driver with simplified outer loop convergence.

File: lever/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from . import core
from .analysis import EnergyEvaluator
from .config import ComputeMode, EvalMode
from .dtypes import LeverResult, OuterState
from .engine import hamiltonian
from .utils.logger import get_logger
from .workflow.compiler import Compiler
from .workflow.fitter import Fitter

if TYPE_CHECKING:
    from .config import LeverConfig
    from .evolution import EvolutionStrategy
    from .models import WavefunctionModel


class Driver:
    """
    LEVER workflow orchestrator.
    
    Manages evolutionary cycles with sliding-window convergence detection:
      - Compile → Fit (fixed steps) → Diagnose → Evolve
      - Converge when: all recent outer_patience cycles have ΔE < outer_tol
    """

    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        strategy: EvolutionStrategy,
        optimizer
    ):
        """Initialize driver with workflow components."""
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
        
        # Workflow components
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
        
        self.evaluator = EnergyEvaluator(
            int_ctx=self.int_ctx,
            n_orb=self.cfg.system.n_orbitals,
            e_nuc=self.int_ctx.get_e_nuc()
        )
    
    def run(self) -> LeverResult:
        """
        Execute LEVER workflow with outer-loop sliding window convergence.
        
        Convergence criterion:
          - Track last outer_patience+1 cycle energies
          - Stop if all deltas < outer_tol for outer_patience consecutive cycles
        """
        self._print_header()
        
        state = self._initialize_state()
        history = _History()
        t0 = time.time()
        
        max_outer = self.cfg.loop.max_outer
        patience = self.cfg.loop.outer_patience
        tol = self.cfg.loop.outer_tol
        
        # Sliding window of recent cycle energies
        recent_energies = []
        converged = False
        
        for cycle in range(max_outer):
            cycle_start = time.time()
            
            # Compile → Fit
            ctx = self.compiler.compile(state)
            result = self.fitter.fit(ctx, state.params, self.optimizer)
            
            final_energy = result.energy_trace[-1]
            inner_steps = len(result.energy_trace)
            cycle_time = time.time() - cycle_start
            
            # Update history
            history.append_energies(result.energy_trace)
            recent_energies.append(final_energy)
            
            # Print cycle summary
            print(
                f"Cycle {cycle+1:3d}/{max_outer} | "
                f"E = {final_energy:.10f} | "
                f"Steps = {inner_steps:4d} | "
                f"Time = {cycle_time:.2f}s"
            )
            
            # Diagnose (if configured)
            is_final = (cycle == max_outer - 1)
            if self._should_diagnose(cycle, is_final):
                self._evaluate_diagnostics(ctx, result, history)
            
            # Check convergence: need patience+1 energies to compute patience deltas
            if len(recent_energies) > patience:
                deltas = [
                    abs(recent_energies[i] - recent_energies[i-1])
                    for i in range(-patience, 0)
                ]
                
                if all(d < tol for d in deltas):
                    print(f"Outer loop converged at cycle {cycle+1}")
                    converged = True
                    state = self._update_state(state, result, ctx, evolve=False)
                    break
                
                # Trim window to size patience+1
                recent_energies = recent_energies[-(patience+1):]
            
            # Evolve to next cycle
            state = self._update_state(state, result, ctx, evolve=True)
        
        if not converged:
            print(f"Reached maximum outer cycles ({max_outer})")
        
        return self._finalize_results(state, history, time.time() - t0)
    
    def _initialize_state(self) -> OuterState:
        """Create initial state from Hartree-Fock determinant."""
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
    
    def _should_diagnose(self, cycle: int, is_final: bool) -> bool:
        """Check if diagnostics needed: EVERY or FINAL mode."""
        eval_cfg = self.cfg.evaluation
        
        modes = [
            eval_cfg.var_energy_mode,
            eval_cfg.s_ci_energy_mode,
            eval_cfg.t_ci_energy_mode
        ]
        
        return any(
            mode == EvalMode.EVERY or (mode == EvalMode.FINAL and is_final)
            for mode in modes
        )
    
    def _evaluate_diagnostics(self, ctx, result, history):
        """Compute VAR, S-CI, T-CI diagnostic energies."""
        eval_cfg = self.cfg.evaluation
        
        # E_VAR: Full-space variational energy
        if eval_cfg.var_energy_mode != EvalMode.NEVER:
            t_dets = np.concatenate([ctx.space.s_dets, ctx.space.c_dets])
            ham_full, _ = hamiltonian.get_ham_ss(
                S_dets=t_dets,
                int_ctx=self.int_ctx,
                n_orbitals=self.cfg.system.n_orbitals
            )
            
            e_var = self.evaluator.variational_energy_from_cache(
                result.psi_cache, ham_full, t_dets
            )
            history.append_diagnostic("var", e_var)
            self.logger.diagnostic_energy("VAR", e_var)
        
        # E_S-CI: Lowest eigenvalue of H_SS or H_eff
        if eval_cfg.s_ci_energy_mode != EvalMode.NEVER:
            e_s_ci = self.evaluator.diagonalize(ctx.ham_ss)
            history.append_diagnostic("s_ci", e_s_ci)
            self.logger.diagnostic_energy("S-CI", e_s_ci)
        
        # E_T-CI: Full-space CI energy
        if eval_cfg.t_ci_energy_mode != EvalMode.NEVER:
            t_dets = np.concatenate([ctx.space.s_dets, ctx.space.c_dets])
            ham_full, _ = hamiltonian.get_ham_ss(
                S_dets=t_dets,
                int_ctx=self.int_ctx,
                n_orbitals=self.cfg.system.n_orbitals
            )
            
            e_t_ci = self.evaluator.diagonalize(ham_full)
            history.append_diagnostic("t_ci", e_t_ci)
            self.logger.diagnostic_energy("T-CI", e_t_ci)
    
    def _update_state(self, state, result, ctx, evolve: bool) -> OuterState:
        """Update state for next cycle."""
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
        """Compute reference energy for H_eff construction."""
        if self.cfg.compute_mode != ComputeMode.EFFECTIVE:
            return None
        return result.energy_trace[-1] - ctx.e_nuc
    
    def _print_header(self):
        """Log workflow configuration."""
        print(f"\n{'='*60}")
        print(f"{'LEVER Initialization':^60}")
        print(f"{'='*60}")
        
        config_items = {
            "Compute mode": self.cfg.compute_mode.value.upper(),
            "System": self.cfg.system.fcidump_path,
            "Model": self.model.module.__class__.__name__,
            "Screening": self.cfg.hamiltonian.screening_mode.value,
        }
        
        for key, value in config_items.items():
            print(f"{key:20s}: {value}")
        print(f"{'='*60}\n")
    
    def _finalize_results(
        self,
        state: OuterState,
        history: _History,
        total_time: float
    ) -> LeverResult:
        """Package results with full trajectories and timing."""
        final_energy = history.energies[-1] if history.energies else 0.0
        n_cycles = len(history.cycle_bounds) - 1
        
        print(f"\n{'='*60}")
        print(f"Final Energy: {final_energy:.10f}")
        print(f"Total Cycles: {n_cycles}")
        print(f"Total Time:   {total_time:.2f}s")
        print(f"{'='*60}\n")
        
        return LeverResult(
            final_params=state.params,
            final_s_dets=state.s_dets,
            full_energy_history=history.energies,
            cycle_boundaries=history.cycle_bounds,
            var_energy_history=history.diagnostics.get("var", []),
            s_ci_energy_history=history.diagnostics.get("s_ci", []),
            total_time=total_time,
            config=self.cfg
        )


class _History:
    """Internal history tracker for energy trajectories."""
    
    def __init__(self):
        self.energies: list[float] = []
        self.cycle_bounds: list[int] = [0]
        self.diagnostics: dict[str, list[float]] = {}
    
    def append_energies(self, energy_trace: list[float]):
        """Record inner loop trajectory and mark cycle boundary."""
        self.energies.extend(energy_trace)
        self.cycle_bounds.append(len(self.energies))
    
    def append_diagnostic(self, key: str, value: float):
        """Add cycle-level diagnostic."""
        if key not in self.diagnostics:
            self.diagnostics[key] = []
        self.diagnostics[key].append(value)


__all__ = ["Driver"]
