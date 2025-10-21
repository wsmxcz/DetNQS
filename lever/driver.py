# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER Driver: Orchestrates variational space evolution with dual-loop structure.

Outer loop evolves active space (S-space), inner loop optimizes neural network parameters.

File: lever/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np
import optax

from . import core, engine, evolution
from .config import LeverConfig, ScreenMode
from .models import WavefunctionModel


# --- Result Data Structures ---

@dataclass(frozen=True)
class DriverResults:
    """Immutable container for LEVER driver execution results."""
    config: LeverConfig
    final_variables: Any
    full_history: list[float]      # Energy at each optimization step
    cycle_boundaries: list[int]    # Optimization cycle end indices
    var_energy_history: list[float]  # Variational energy in S+C space
    s_ci_energy_history: list[float] # S-space CI energy E_CI(S)
    t_ci_energy_history: list[float] # T-space CI energy E_CI(S+C)
    total_time: float


# --- Core Driver Implementation ---

class Driver:
    """
    Orchestrates LEVER wavefunction optimization with dual-loop structure.
    
    Outer loop: Active space evolution via determinant selection
    Inner loop: Neural network parameter optimization via VMC
    """
    
    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        strategy: evolution.EvolutionStrategy,
    ):
        self.config = config
        self.model = model
        self.strategy = strategy
        self.variables = model.variables
        
        # Initialize integral context with heat-bath precomputation
        self.int_ctx = core.IntCtx(config.system.fcidump_path, config.system.n_orbitals)
        self.int_ctx.hb_prepare()
        self.e_nuc = self.int_ctx.get_e_nuc()

    def run(self) -> DriverResults:
        """Executes complete LEVER variational evolution workflow."""
        cfg = self.config
        print(f"Starting LEVER Driver: {cfg.system.fcidump_path}")
        print(f"  Model: {self.model.module.__class__.__name__}")
        print(f"  Strategy: {self.strategy.__class__.__name__}")

        # Initialize S-space with Hartree-Fock reference
        s_dets = self._get_initial_determinant()
        history = {
            "full": [], "cycle_boundaries": [0], "var": [], "s_ci": [], "t_ci": []
        }
        start_time = time.time()

        for cycle in range(cfg.optimization.num_cycles):
            print(f"\n--- Cycle {cycle + 1}/{cfg.optimization.num_cycles} ---")

            # 1. Build Hamiltonian with current S-space
            ham_ss, ham_sc, space_rep = self._build_hamiltonian(s_dets)
            print(f"Space sizes | S: {space_rep.size_S}, C: {space_rep.size_C}")
            print(f"NNZ counts  | H_SS: {ham_ss.nnz:,}, H_SC: {ham_sc.nnz:,}")

            # 2. Inner loop VMC optimization
            self.variables, cycle_history = self._run_optimization_cycle(
                ham_ss, ham_sc, space_rep
            )
            history["full"].extend(cycle_history)
            history["cycle_boundaries"].append(len(history["full"]))
            print(f"Final optimization energy: {cycle_history[-1]:.8f} Ha")

            # 3. Post-optimization evaluations (CI, Variational)
            self._perform_evaluations(cycle, ham_ss, space_rep, history)

            # 4. Active space evolution for next cycle
            if cycle < cfg.optimization.num_cycles - 1:
                print("Evolving S-space...")
                evaluator = self._create_evaluator(ham_ss, ham_sc, space_rep)
                s_dets = self.strategy.evolve(evaluator)
                print(f"New S-space size: {len(s_dets)}")

        total_time = time.time() - start_time
        print(f"\n--- LEVER Driver finished in {total_time:.2f}s ---")

        return DriverResults(
            config=self.config,
            final_variables=self.variables,
            full_history=history["full"],
            cycle_boundaries=history["cycle_boundaries"],
            var_energy_history=history["var"],
            s_ci_energy_history=history["s_ci"],
            t_ci_energy_history=history["t_ci"],
            total_time=total_time,
        )

    def _get_initial_determinant(self) -> np.ndarray:
        """Constructs Hartree-Fock reference determinant using bitmask representation."""
        sys = self.config.system
        # Bitmask representation: α = (1<<n_α)-1, β = (1<<n_β)-1
        return np.array(
            [[(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]], dtype=np.uint64
        )

    def _build_hamiltonian(self, s_dets: np.ndarray) -> tuple[Any, Any, Any]:
        """Builds Hamiltonian blocks H_SS and H_SC with screening support."""
        cfg = self.config
        
        if cfg.screening.mode == ScreenMode.DYNAMIC:
            # Dynamic screening requires S-space amplitudes ψ_S for importance sampling
            psi_s = self._compute_s_space_amplitudes(s_dets)
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx, n_orbitals=cfg.system.n_orbitals,
                psi_S=psi_s, use_heatbath=True, eps1=cfg.screening.eps1
            )
        elif cfg.screening.mode == ScreenMode.STATIC:
            # Static screening uses only S-space determinant structure
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx, n_orbitals=cfg.system.n_orbitals,
                use_heatbath=True, eps1=cfg.screening.eps1
            )
        else:
            raise NotImplementedError(f"Screening mode {cfg.screening.mode} not supported.")

    def _run_optimization_cycle(
        self, ham_ss, ham_sc, space_rep
    ) -> tuple[Any, list[float]]:
        """Executes inner optimization loop with AdamW optimizer."""
        cfg = self.config.optimization
        optimizer = optax.adamw(cfg.learning_rate)
        opt_state = optimizer.init(self.variables)

        # JIT-compiled optimization step
        jitted_step = self._create_jitted_step_fn(ham_ss, ham_sc, space_rep, optimizer)

        energy_history = []
        for step in range(cfg.steps_per_cycle):
            self.variables, opt_state, total_energy = jitted_step(self.variables, opt_state)
            energy_history.append(float(total_energy))

            if (step + 1) % cfg.report_interval == 0:
                print(f"  Step {step+1:4d}/{cfg.steps_per_cycle} | E = {total_energy:.8f} Ha")
        
        return self.variables, energy_history

    def _create_jitted_step_fn(self, ham_ss, ham_sc, space_rep, optimizer) -> Callable:
        """Creates JIT-compiled optimization step: evaluate E/∇E + AdamW update."""
        logpsi_fn = self.model.log_psi
        n_orb = self.config.system.n_orbitals
        engine_cfg = self.config.engine

        def step_fn(variables: Any, opt_state: Any) -> tuple[Any, Any, float]:
            # Initialize evaluator with current state
            evaluator = engine.Evaluator(
                params=variables, logpsi_fn=logpsi_fn, ham_ss=ham_ss,
                ham_sc=ham_sc, space=space_rep, n_orbitals=n_orb, config=engine_cfg,
            )
            # Compute energy and gradient
            result = engine.compute_energy_and_gradient(evaluator)
            
            # Apply parameter update
            updates, new_opt_state = optimizer.update(result.gradient, opt_state, variables)
            new_variables = optax.apply_updates(variables, updates)
            
            return new_variables, new_opt_state, result.energy_elec + self.e_nuc

        return jax.jit(step_fn)

    def _perform_evaluations(self, cycle: int, ham_ss, space_rep, history: dict):
        """Computes post-optimization CI and variational energies."""
        from . import analysis 
        
        cfg = self.config
        eval_suite = analysis.EvaluationSuite(self.int_ctx, cfg.system.n_orbitals, self.e_nuc)
        num_cycles = cfg.optimization.num_cycles

        def should_eval(mode: str) -> bool:
            return mode == "every" or (mode == "final" and cycle == num_cycles - 1)

        # S-space CI diagonalization
        if should_eval(cfg.evaluation.s_ci_energy_mode.value):
            e_s_ci = eval_suite.diagonalize_hamiltonian(ham_ss)
            history["s_ci"].append(e_s_ci)
            print(f"  S-space CI energy: {e_s_ci:.8f} Ha")
        
        # T-space (S+C) evaluations
        eval_var = should_eval(cfg.evaluation.var_energy_mode.value)
        eval_t_ci = should_eval(cfg.evaluation.t_ci_energy_mode.value)
        
        if eval_var or eval_t_ci:
            # Construct full T-space Hamiltonian
            t_dets = np.concatenate([space_rep.s_dets, space_rep.c_dets])
            ham_tt, _, _ = engine.hamiltonian.get_ham_proxy(
                S_dets=t_dets, int_ctx=self.int_ctx, n_orbitals=cfg.system.n_orbitals,
                use_heatbath=False  # Full matrix construction
            )
            if eval_var:
                e_var = eval_suite.compute_variational_energy(
                    self.variables, self.model.log_psi, ham_tt, t_dets
                )
                history["var"].append(e_var)
                print(f"  Variational energy (T-space): {e_var:.8f} Ha")
            if eval_t_ci:
                e_t_ci = eval_suite.diagonalize_hamiltonian(ham_tt)
                history["t_ci"].append(e_t_ci)
                print(f"  T-space CI energy: {e_t_ci:.8f} Ha")

    def _compute_s_space_amplitudes(self, s_dets: np.ndarray) -> np.ndarray:
        """Computes L₂-normalized S-space amplitudes for dynamic screening."""
        s_vecs = engine.utils.masks_to_vecs(
            jax.device_put(s_dets), self.config.system.n_orbitals
        )
        log_psi_s = self.model.log_psi(self.variables, s_vecs)
        psi_s = np.abs(np.array(jax.device_get(jax.numpy.exp(log_psi_s))))
        
        norm = np.linalg.norm(psi_s)
        return psi_s / norm if norm > 1e-14 else psi_s

    def _create_evaluator(self, ham_ss, ham_sc, space_rep) -> engine.Evaluator:
        """Initializes cached evaluator for current cycle state."""
        return engine.Evaluator(
            params=self.variables, logpsi_fn=self.model.log_psi, ham_ss=ham_ss,
            ham_sc=ham_sc, space=space_rep, n_orbitals=self.config.system.n_orbitals,
            config=self.config.engine,
        )


__all__ = ["Driver", "DriverResults"]
