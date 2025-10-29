# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER workflow orchestrator for Compile → Fit → Diagnose → Evolve cycles.

Manages centralized state transitions with determinant space evolution and
parameter optimization.

File: lever/workflow/controller.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np

from .. import core, evolution, engine
from ..analysis import EvalSuite
from ..config import ComputeMode, EvalMode
from ..utils.dtypes import EvolutionState, Workspace, FitResult
from ..utils.logger import get_logger
from .compiler import Compiler
from .fitter import Fitter

if TYPE_CHECKING:
    from ..config import LeverConfig
    from ..models import WavefunctionModel
    from ..driver import DriverResults


@dataclass
class History:
    """
    Energy trajectory tracker for inner/outer loop convergence.
  
    Stores: E_i for each iteration i, cycle boundaries, and diagnostic energies.
    """
    energies: list[float] = field(default_factory=list)
    cycle_bounds: list[int] = field(default_factory=list)
    diagnostics: dict[str, list[float]] = field(default_factory=dict)
  
    def append(self, energy_trace: list[float]):
        """Record inner loop trajectory E_i and mark cycle boundary."""
        self.energies.extend(energy_trace)
        self.cycle_bounds.append(len(self.energies))
  
    def append_diagnostic(self, key: str, value: float):
        """Add cycle-level diagnostic (VAR, S-CI, T-CI)."""
        if key not in self.diagnostics:
            self.diagnostics[key] = []
        self.diagnostics[key].append(value)
  
    @property
    def cycle_energies(self) -> list[float]:
        """Extract final energy E_final from each outer cycle."""
        if not self.cycle_bounds:
            return []
      
        result = []
        prev = 0
        for bound in self.cycle_bounds:
            if bound > prev:
                result.append(self.energies[bound - 1])
            prev = bound
        return result


class Controller:
    """
    Main workflow controller for LEVER optimization.
  
    Algorithm:
      1. Compile: Build H_eff from (S, C) determinant spaces
      2. Fit: Optimize ψ(θ) via gradient descent
      3. Diagnose: Compute VAR/S-CI/T-CI energies if configured
      4. Evolve: Score C-space and select new S-space for next cycle
  
    Convergence: |E_cycle[n] - E_cycle[n-p]| < tol, patience p.
    """
  
    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        strategy: evolution.EvolutionStrategy,
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
      
        self.eval_suite = EvalSuite(
            int_ctx=self.int_ctx,
            n_orb=self.cfg.system.n_orbitals,
            e_nuc=self.int_ctx.get_e_nuc()
        )
  
    def run(self) -> DriverResults:
        """Execute complete LEVER workflow with state evolution."""
        self._print_header()
      
        state = self._initialize_state()
        history = History()
        t0 = time.time()
      
        final_workspace = None
        final_result = None
      
        for cycle in range(self.cfg.loop.max_cycles):
            workspace, result = self._execute_cycle(state, history)
            final_workspace, final_result = workspace, result
          
            is_planned_final = (cycle == self.cfg.loop.max_cycles - 1)
            converged = self._check_convergence(history, cycle)
            is_actual_final = is_planned_final or converged
          
            # Evaluate diagnostics at final iteration or as configured
            if self._should_diagnose(cycle, is_actual_final):
                self._evaluate_diagnostics(workspace, result, history)
          
            if converged:
                state = self._update_state_inplace(state, result, workspace)
                break
          
            # Evolve space for next cycle
            state = self._evolve_to_next_cycle(state, workspace, result)
      
        return self._finalize_results(state, history, time.time() - t0)
  
    def _initialize_state(self) -> EvolutionState:
        """Create initial state from Hartree-Fock determinant |HF⟩."""
        sys = self.cfg.system
        hf_det = np.array([
            [(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]
        ], dtype=np.uint64)
      
        return EvolutionState(
            s_dets=hf_det,
            params=self.model.variables,
            e_ref=None,
            cycle=0
        )
  
    def _execute_cycle(
        self,
        state: EvolutionState,
        history: History
    ) -> tuple[Workspace, FitResult]:
        """
        Execute single cycle: Compile → Fit → Record.
      
        Returns workspace and fit result without mutating input state.
        """
        t0 = time.time()
        self.logger.cycle_start(state.cycle + 1, self.cfg.loop.max_cycles)
      
        # Compile H_eff from current S-space
        workspace = self.compiler.compile(
            s_dets=state.s_dets,
            params=state.params,
            e_ref=state.e_ref
        )
      
        # Optimize ψ(θ) on H_eff
        result = self.fitter.fit(
            workspace=workspace,
            params=state.params,
            optimizer=self.optimizer
        )
      
        history.append(result.energy_trace)
      
        self.logger.timing(f"Cycle {state.cycle + 1}", time.time() - t0)
      
        return workspace, result
  
    def _should_diagnose(self, cycle: int, is_final: bool) -> bool:
        """
        Check if diagnostics needed: EVERY or FINAL mode.
      
        Args:
            cycle: Current cycle index (0-based)
            is_final: True if last iteration (planned/converged)
        """
        eval_cfg = self.cfg.evaluation
      
        modes = [
            eval_cfg.var_energy_mode,
            eval_cfg.s_ci_energy_mode,
            eval_cfg.t_ci_energy_mode
        ]
      
        for mode in modes:
            if mode == EvalMode.EVERY:
                return True
            if mode == EvalMode.FINAL and is_final:
                return True
      
        return False
  
    def _evaluate_diagnostics(
        self,
        workspace: Workspace,
        result: FitResult,
        history: History
    ):
        """
        Compute diagnostic energies: E_VAR, E_S-CI, E_T-CI.
      
        - VAR: ⟨ψ(θ)|H|ψ(θ)⟩ on full (S∪C) space
        - S-CI: Lowest eigenvalue of H_eff
        - T-CI: Lowest eigenvalue of H on (S∪C)
        """
        eval_cfg = self.cfg.evaluation
      
        # E_VAR: Variational energy ⟨ψ|H|ψ⟩
        if eval_cfg.var_energy_mode != EvalMode.NEVER:
            t_dets = np.concatenate([
                workspace.space.s_dets,
                workspace.space.c_dets
            ])
            ham_full, _ = engine.hamiltonian.get_ham_ss(
                S_dets=t_dets,
                int_ctx=self.int_ctx,
                n_orbitals=self.cfg.system.n_orbitals
            )
          
            e_var = self.eval_suite.var_energy_from_cache(
                result.psi_cache, ham_full, t_dets
            )
            history.append_diagnostic("var", e_var)
            self.logger.diagnostic_energy("VAR", e_var)
      
        # E_S-CI: Lowest eigenvalue of H_eff
        if eval_cfg.s_ci_energy_mode != EvalMode.NEVER:
            e_s_ci = self.eval_suite.diag_ham(workspace.ham_opt)
            history.append_diagnostic("s_ci", e_s_ci)
            self.logger.diagnostic_energy("S-CI", e_s_ci)
      
        # E_T-CI: Full-space CI energy
        if eval_cfg.t_ci_energy_mode != EvalMode.NEVER:
            t_dets = np.concatenate([
                workspace.space.s_dets,
                workspace.space.c_dets
            ])
            ham_full, _ = engine.hamiltonian.get_ham_ss(
                S_dets=t_dets,
                int_ctx=self.int_ctx,
                n_orbitals=self.cfg.system.n_orbitals
            )
          
            e_t_ci = self.eval_suite.diag_ham(ham_full)
            history.append_diagnostic("t_ci", e_t_ci)
            self.logger.diagnostic_energy("T-CI", e_t_ci)
  
    def _check_convergence(self, history: History, cycle: int) -> bool:
        """
        Check outer loop convergence: |E[n] - E[n-p]| < tol.
      
        Requires patience p ≥ 1 cycles before checking.
        """
        if cycle < self.cfg.loop.patience:
            return False
      
        cycle_energies = history.cycle_energies
        if len(cycle_energies) < self.cfg.loop.patience + 1:
            return False
      
        e_recent = cycle_energies[-1]
        e_previous = cycle_energies[-(self.cfg.loop.patience + 1)]
      
        delta = abs(e_recent - e_previous)
      
        if delta < self.cfg.loop.cycle_tol:
            self.logger.outer_loop_converged(cycle + 1, delta)
            return True
      
        return False
  
    def _evolve_to_next_cycle(
        self,
        state: EvolutionState,
        workspace: Workspace,
        result: FitResult
    ) -> EvolutionState:
        """
        Evolve to next cycle: score C-space and select new S-space.
      
        Updates:
          - s_dets: From strategy.evolve (scores C-space contributions)
          - params: From optimization result θ*
          - e_ref: E_ref = E_opt - E_nuc (EFFECTIVE mode only)
          - cycle: Incremented
        """
        new_s_dets = self.strategy.evolve(workspace, result.psi_cache)
        new_e_ref = self._compute_next_e_ref(result, workspace)
      
        return EvolutionState(
            s_dets=new_s_dets,
            params=result.params,
            e_ref=new_e_ref,
            cycle=state.cycle + 1
        )
  
    def _update_state_inplace(
        self,
        state: EvolutionState,
        result: FitResult,
        workspace: Workspace
    ) -> EvolutionState:
        """Update params/e_ref without space evolution (early convergence)."""
        new_e_ref = self._compute_next_e_ref(result, workspace)
      
        return EvolutionState(
            s_dets=state.s_dets,
            params=result.params,
            e_ref=new_e_ref,
            cycle=state.cycle
        )
  
    def _compute_next_e_ref(
        self,
        result: FitResult,
        workspace: Workspace
    ) -> float | None:
        """
        Compute reference energy E_ref for next cycle.
      
        EFFECTIVE mode: E_ref = E_opt - E_nuc
        STANDARD mode: E_ref = None
        """
        if self.cfg.compute_mode != ComputeMode.EFFECTIVE:
            return None
        return result.energy_trace[-1] - workspace.e_nuc
  
    def _print_header(self):
        """Log workflow configuration."""
        self.logger.header("LEVER Initialization")
      
        config_items = {
            "Compute mode": self.cfg.compute_mode.value.upper(),
            "System": self.cfg.system.fcidump_path,
            "Model": self.model.module.__class__.__name__,
            "Screening": self.cfg.hamiltonian.screening_mode.value,
        }
        self.logger.config_info(config_items)
  
    def _finalize_results(
        self,
        state: EvolutionState,
        history: History,
        total_time: float
    ) -> DriverResults:
        """Package results with full energy trajectories and timing."""
        from ..driver import DriverResults
      
        final_energy = history.energies[-1] if history.energies else 0.0
        self.logger.final_summary(
            total_time,
            final_energy,
            state.cycle + 1
        )
      
        return DriverResults(
            config=self.cfg,
            final_vars=state.params,
            full_hist=history.energies,
            cycle_bounds=history.cycle_bounds,
            var_hist=history.diagnostics.get("var", []),
            s_ci_hist=history.diagnostics.get("s_ci", []),
            t_ci_hist=history.diagnostics.get("t_ci", []),
            total_time=total_time
        )


__all__ = ["Controller", "History"]