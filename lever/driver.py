# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER driver orchestrating dual-loop variational optimization.

Outer loop: Active space evolution (S-space determinant selection)
Inner loop: Neural network parameter optimization via VMC with lax.scan

File: lever/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax

from . import core, engine, evolution
from .config import LeverConfig, ScreenMode
from .models import WavefunctionModel


@dataclass(frozen=True)
class DriverResults:
    """Container for LEVER execution results."""
    config: LeverConfig
    final_variables: Any
    full_history: list[float]           # Per-step optimization energies
    cycle_boundaries: list[int]         # Cycle end indices in full_history
    var_energy_history: list[float]     # E_var(S+C) per cycle
    s_ci_energy_history: list[float]    # E_CI(S) per cycle
    t_ci_energy_history: list[float]    # E_CI(S+C) per cycle
    total_time: float


class Driver:
    """
    LEVER optimization driver with dual-loop structure.
  
    Coordinates active space evolution (outer) and VMC optimization (inner)
    using JIT-compiled lax.scan for efficient gradient descent.
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
        """Execute complete LEVER workflow."""
        cfg = self.config
        print(f"Starting LEVER: {cfg.system.fcidump_path}")
        print(f"  Model: {self.model.module.__class__.__name__}")
        print(f"  Strategy: {self.strategy.__class__.__name__}")

        s_dets = self._init_hf_reference()
        history = {
            "full": [], "cycle_boundaries": [0], 
            "var": [], "s_ci": [], "t_ci": []
        }
        t_start = time.time()

        for cycle in range(cfg.optimization.num_cycles):
            print(f"\n--- Cycle {cycle + 1}/{cfg.optimization.num_cycles} ---")

            # Build Hamiltonian with current S-space
            ham_ss, ham_sc, space_rep = self._build_hamiltonian(s_dets)
            print(f"Space: S={space_rep.size_S}, C={space_rep.size_C}")
            print(f"NNZ: H_SS={ham_ss.nnz:,}, H_SC={ham_sc.nnz:,}")

            # VMC optimization via lax.scan
            self.variables, step_energies = self._optimize_parameters(
                ham_ss, ham_sc, space_rep
            )
            history["full"].extend(step_energies)
            history["cycle_boundaries"].append(len(history["full"]))
            print(f"Final step energy: {step_energies[-1]:.8f} Ha")

            # Post-optimization evaluations
            self._evaluate_energies(cycle, ham_ss, space_rep, history)

            # Evolve S-space for next cycle
            if cycle < cfg.optimization.num_cycles - 1:
                print("Evolving S-space...")
                evaluator = self._create_evaluator(ham_ss, ham_sc, space_rep)
                s_dets = self.strategy.evolve(evaluator)
                print(f"New S-space: {len(s_dets)} determinants")

        total_time = time.time() - t_start
        print(f"\nLEVER completed in {total_time:.2f}s")

        return DriverResults(
            config=cfg, final_variables=self.variables,
            full_history=history["full"],
            cycle_boundaries=history["cycle_boundaries"],
            var_energy_history=history["var"],
            s_ci_energy_history=history["s_ci"],
            t_ci_energy_history=history["t_ci"],
            total_time=total_time,
        )

    def _init_hf_reference(self) -> np.ndarray:
        """Initialize Hartree-Fock determinant via bitmask: α=(1<<n_α)-1, β=(1<<n_β)-1."""
        sys = self.config.system
        return np.array(
            [[(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]], 
            dtype=np.uint64
        )

    def _build_hamiltonian(self, s_dets: np.ndarray) -> tuple[Any, Any, Any]:
        """Construct Hamiltonian blocks H_SS, H_SC with optional screening."""
        cfg = self.config
      
        if cfg.screening.mode == ScreenMode.DYNAMIC:
            # Dynamic screening: importance sampling via ψ_S amplitudes
            psi_s = self._compute_s_amplitudes(s_dets)
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx, 
                n_orbitals=cfg.system.n_orbitals,
                psi_S=psi_s, use_heatbath=True, eps1=cfg.screening.eps1
            )
        elif cfg.screening.mode == ScreenMode.STATIC:
            # Static screening: determinant structure only
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx,
                n_orbitals=cfg.system.n_orbitals,
                use_heatbath=True, eps1=cfg.screening.eps1
            )
        else:
            raise NotImplementedError(f"Unsupported mode: {cfg.screening.mode}")

    def _optimize_parameters(
        self, ham_ss, ham_sc, space_rep
    ) -> tuple[Any, list[float]]:
        """
        Execute VMC optimization via lax.scan.
      
        Wraps entire optimization loop into single XLA WhileOp to minimize:
        - Compilation overhead (no loop unrolling)
        - Host-device synchronization
        - Python dispatch per step
        """
        cfg = self.config.optimization
        optimizer = optax.adamw(cfg.learning_rate)
        opt_state = optimizer.init(self.variables)

        # JIT-compiled scan over all optimization steps
        scan_fn = self._make_scan_optimizer(ham_ss, ham_sc, space_rep, optimizer)
        (final_vars, _), energies = scan_fn(
            self.variables, opt_state, cfg.steps_per_cycle
        )
      
        energy_list = [float(e) for e in energies]
      
        # Print progress (outside scan to avoid host callbacks)
        for step in range(0, cfg.steps_per_cycle, cfg.report_interval):
            if step > 0:
                print(f"  Step {step:4d} | E = {energy_list[step-1]:.8f} Ha")
      
        return final_vars, energy_list

    def _make_scan_optimizer(
        self, ham_ss, ham_sc, space_rep, optimizer
    ) -> Callable:
        """
        Create lax.scan-based optimizer with fixed-shape carry.
      
        Closure captures invariant context (H, space, config) to avoid
        recompilation when only parameters change.
      
        Returns:
            JIT function: (vars, opt_state, n_steps) -> ((final_vars, final_opt), energies)
        """
        logpsi = self.model.log_psi
        n_orb = self.config.system.n_orbitals
        engine_cfg = self.config.engine
        e_nuc = self.e_nuc
      
        def step_fn(carry: tuple[Any, Any], _) -> tuple[tuple[Any, Any], float]:
            """Single VMC step: gradient computation + parameter update."""
            variables, opt_state = carry
          
            # Create evaluator with current parameters
            evaluator = engine.Evaluator(
                params=variables, logpsi_fn=logpsi,
                ham_ss=ham_ss, ham_sc=ham_sc, space=space_rep,
                n_orbitals=n_orb, config=engine_cfg,
            )
          
            # VMC energy and gradient
            result = engine.compute_energy_and_gradient(evaluator)
          
            # AdamW update
            updates, new_opt = optimizer.update(result.gradient, opt_state, variables)
            new_vars = optax.apply_updates(variables, updates)
          
            total_energy = result.energy_elec + e_nuc
            return (new_vars, new_opt), total_energy
      
        def scan_wrapper(init_vars, init_opt, num_steps: int):
            """JIT-compiled full optimization loop as single WhileOp."""
            final_carry, energy_hist = lax.scan(
                f=step_fn,
                init=(init_vars, init_opt),
                xs=None,
                length=num_steps
            )
            return final_carry, energy_hist
      
        # JIT with static num_steps (triggers recompilation if changed)
        return jax.jit(scan_wrapper, static_argnames=['num_steps'])

    def _evaluate_energies(
        self, cycle: int, ham_ss, space_rep, history: dict
    ):
        """Compute post-optimization CI and variational energies."""
        from . import analysis
      
        cfg = self.config
        suite = analysis.EvaluationSuite(
            self.int_ctx, cfg.system.n_orbitals, self.e_nuc
        )
        n_cycles = cfg.optimization.num_cycles

        def should_eval(mode: str) -> bool:
            return mode == "every" or (mode == "final" and cycle == n_cycles - 1)

        # S-space CI: E_CI(S)
        if should_eval(cfg.evaluation.s_ci_energy_mode.value):
            e_s_ci = suite.diagonalize_hamiltonian(ham_ss)
            history["s_ci"].append(e_s_ci)
            print(f"  E_CI(S): {e_s_ci:.8f} Ha")
      
        # T-space (S+C) evaluations
        eval_var = should_eval(cfg.evaluation.var_energy_mode.value)
        eval_t_ci = should_eval(cfg.evaluation.t_ci_energy_mode.value)
      
        if eval_var or eval_t_ci:
            # Construct full T-space Hamiltonian
            t_dets = np.concatenate([space_rep.s_dets, space_rep.c_dets])
            ham_tt, _, _, _ = engine.hamiltonian.get_ham_full(
                S_dets=t_dets, C_dets=t_dets,
                int_ctx=self.int_ctx,
                n_orbitals=cfg.system.n_orbitals,
            )
          
            if eval_var:
                e_var = suite.compute_variational_energy(
                    self.variables, self.model.log_psi, ham_tt, t_dets
                )
                history["var"].append(e_var)
                print(f"  E_var(T): {e_var:.8f} Ha")
          
            if eval_t_ci:
                e_t_ci = suite.diagonalize_hamiltonian(ham_tt)
                history["t_ci"].append(e_t_ci)
                print(f"  E_CI(T): {e_t_ci:.8f} Ha")

    def _compute_s_amplitudes(self, s_dets: np.ndarray) -> np.ndarray:
        """Compute L₂-normalized S-space amplitudes for dynamic screening."""
        s_vecs = engine.utils.masks_to_vecs(
            jax.device_put(s_dets), self.config.system.n_orbitals
        )
        log_psi = self.model.log_psi(self.variables, s_vecs)
        psi = np.abs(np.array(jax.device_get(jnp.exp(log_psi))))
      
        norm = np.linalg.norm(psi)
        return psi / norm if norm > 1e-14 else psi

    def _create_evaluator(self, ham_ss, ham_sc, space_rep) -> engine.Evaluator:
        """Create cached evaluator with current cycle state."""
        return engine.Evaluator(
            params=self.variables, logpsi_fn=self.model.log_psi,
            ham_ss=ham_ss, ham_sc=ham_sc, space=space_rep,
            n_orbitals=self.config.system.n_orbitals,
            config=self.config.engine,
        )


__all__ = ["Driver", "DriverResults"]