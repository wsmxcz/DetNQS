# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER driver orchestrating dual-loop variational optimization.

Implements nested optimization structure:
  Outer loop: S-space evolution (determinant selection)
  Inner loop: VMC parameter optimization via JAX/lax.scan

Supports three computation modes:
  - ASYMMETRIC: Full H_SS + H_SC coupling
  - PROXY: Static/dynamic screened H_SC
  - EFFECTIVE: Schur complement H_eff = H_SS + H_SC·D⁻¹·H_CS

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
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
from .config import LeverConfig, ScreenMode, ComputeMode
from .models import WavefunctionModel


@dataclass(frozen=True)
class DriverResults:
    """Container for LEVER execution results."""
    config: LeverConfig
    final_variables: Any
    full_history: list[float]
    cycle_boundaries: list[int]
    var_energy_history: list[float]
    s_ci_energy_history: list[float]
    t_ci_energy_history: list[float]
    total_time: float


class Driver:
    """
    LEVER optimization driver with dual-loop structure.

    Architecture:
      Outer: Determinant space evolution (heat-bath selection)
      Inner: JAX-accelerated VMC with lax.scan batching
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
        self.int_ctx.hb_prepare(threshold=1e-15)
        self.e_nuc = self.int_ctx.get_e_nuc()
        
        # Reference energy E_ref for EFFECTIVE mode: E_ref = ⟨HF|H|HF⟩ initially
        self.e_ref_elec: float | None = None

    def run(self) -> DriverResults:
        """Execute complete LEVER workflow with dual-loop optimization."""
        cfg = self.config
        mode_name = cfg.compute_mode.value.upper()
        print(f"Starting LEVER ({mode_name}): {cfg.system.fcidump_path}")
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

            # Build Hamiltonian blocks based on compute mode
            build_result = self._build_hamiltonian(s_dets)
            
            if cfg.compute_mode == ComputeMode.EFFECTIVE:
                ham_opt, ham_sc_evo, space_rep = build_result
                print(f"Space: S={space_rep.size_S}, C={space_rep.size_C}")
                print(f"E_ref: {self.e_ref_elec:.8f} Ha (electronic)")
                print(f"NNZ: H_eff={ham_opt.nnz:,}, H_SC={ham_sc_evo.nnz:,}")
            else:
                ham_ss, ham_sc, space_rep = build_result
                ham_opt, ham_sc_evo = ham_ss, ham_sc
                print(f"Space: S={space_rep.size_S}, C={space_rep.size_C}")
                print(f"NNZ: H_SS={ham_ss.nnz:,}, H_SC={ham_sc.nnz:,}")

            # Inner loop: VMC parameter optimization
            self.variables, step_energies = self._optimize_parameters(
                ham_opt, ham_sc_evo, space_rep
            )
            history["full"].extend(step_energies)
            history["cycle_boundaries"].append(len(history["full"]))
            
            # Update E_ref for next cycle (EFFECTIVE mode)
            final_step_energy = step_energies[-1]
            print(f"Final step energy: {final_step_energy:.8f} Ha")
            if cfg.compute_mode == ComputeMode.EFFECTIVE:
                self.e_ref_elec = final_step_energy - self.e_nuc

            # Post-optimization energy evaluations
            self._evaluate_energies(cycle, ham_opt, space_rep, history)

            # Outer loop: S-space evolution for next cycle
            if cycle < cfg.optimization.num_cycles - 1:
                print("Evolving S-space...")
                evaluator = self._create_evaluator(
                    ham_opt, ham_sc_evo, space_rep, for_evolution=True
                )
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
        """
        Initialize Hartree-Fock determinant |HF⟩ = |1...n_α⟩|1...n_β⟩.
        
        For EFFECTIVE mode: E_ref ← ⟨HF|H|HF⟩ (diagonal element).
        """
        sys = self.config.system
        hf_det = np.array(
            [[(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]], 
            dtype=np.uint64
        )
        
        if self.config.compute_mode == ComputeMode.EFFECTIVE:
            hf_diag = core.get_ham_diag(dets=hf_det, int_ctx=self.int_ctx)
            self.e_ref_elec = float(hf_diag[0])
            print(f"Initial E_ref (HF): {self.e_ref_elec:.8f} Ha")
        
        return hf_det

    def _build_hamiltonian(self, s_dets: np.ndarray):
        """
        Construct Hamiltonian blocks based on compute mode.
        
        Returns:
            ASYMMETRIC/PROXY: (H_SS, H_SC, space)
            EFFECTIVE:        (H_eff, H_SC, space)
                               ↑ opt  ↑ evolution
        
        EFFECTIVE mode: H_eff = H_SS + H_SC·diag(D⁻¹)·H_CS
        where D_jj = E_ref - H_CC[j,j] (denominator regularized via Tikhonov).
        """
        mode = self.config.compute_mode
        
        if mode in {ComputeMode.ASYMMETRIC, ComputeMode.PROXY}:
            return self._build_proxy_hamiltonian(s_dets)
        elif mode == ComputeMode.EFFECTIVE:
            return self._build_effective_hamiltonian(s_dets)
        else:
            raise ValueError(f"Unknown compute mode: {mode}")

    def _build_proxy_hamiltonian(self, s_dets: np.ndarray):
        """Build H_SS and H_SC with optional amplitude-based screening."""
        cfg = self.config
        
        if cfg.screening.mode == ScreenMode.NONE:
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx,
                n_orbitals=cfg.system.n_orbitals, mode="none",
            )
        elif cfg.screening.mode == ScreenMode.STATIC:
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx,
                n_orbitals=cfg.system.n_orbitals, mode="static",
                eps1=cfg.screening.eps1,
            )
        elif cfg.screening.mode == ScreenMode.DYNAMIC:
            psi_s = self._compute_s_amplitudes(s_dets)
            return engine.hamiltonian.get_ham_proxy(
                S_dets=s_dets, int_ctx=self.int_ctx,
                n_orbitals=cfg.system.n_orbitals, mode="dynamic",
                psi_S=psi_s, eps1=cfg.screening.eps1,
            )
        else:
            raise NotImplementedError(f"Unsupported screening: {cfg.screening.mode}")

    def _build_effective_hamiltonian(self, s_dets: np.ndarray):
        """
        Build effective Hamiltonian via Schur complement downfolding.
        
        Returns: (H_eff, H_SC, space)
                ↑ for optimization (S-only)
                        ↑ retained for evolution (T-space scoring)
        """
        # Phase 1: Build raw blocks with screening
        ham_ss_raw, ham_sc, space_rep = self._build_proxy_hamiltonian(s_dets)
        
        # Phase 2: Assemble H_eff via downfolding
        ham_eff = engine.hamiltonian.get_ham_eff(
            ham_ss=ham_ss_raw,
            ham_sc=ham_sc,
            h_cc_diag=space_rep.H_diag_C,
            e_ref=self.e_ref_elec,
            epsilon=1e-6
        )
        
        return ham_eff, ham_sc, space_rep

    def _optimize_parameters(
        self, ham_ss, ham_sc, space_rep
    ) -> tuple[Any, list[float]]:
        """
        VMC parameter optimization via JAX lax.scan batching.
        
        FIX: In EFFECTIVE mode, pass ham_sc=None to avoid spurious C-space callbacks.
        Cache features_S for reuse in evolution phase.
        """
        cfg = self.config.optimization
        optimizer = optax.adamw(cfg.learning_rate)
        opt_state = optimizer.init(self.variables)
        # Pre-compute occupancy features (cache for evolution reuse)
        features_S = self._precompute_features(space_rep.s_dets)
        self._cached_features_S = features_S  # NEW: Cache for evolution
        
        # EFFECTIVE mode: null out ham_sc for optimization (S-space only)
        # Retain ham_sc only for evolution phase
        ham_opt = ham_ss
        ham_sc_opt = None if self.config.compute_mode == ComputeMode.EFFECTIVE else ham_sc
        
        features_C = (
            self._precompute_features(space_rep.c_dets)
            if ham_sc_opt is not None  # Only compute if needed
            else None
        )
        # JIT-compiled scan over all optimization steps
        scan_fn = self._make_scan_optimizer(
            ham_opt, ham_sc_opt, space_rep, optimizer, features_S, features_C
        )
        (final_vars, _), energies = scan_fn(
            self.variables, opt_state, cfg.steps_per_cycle
        )
        energy_list = [float(e) for e in energies]
        # Print progress at intervals
        for step in range(0, cfg.steps_per_cycle, cfg.report_interval):
            if step > 0:
                print(f"  Step {step:4d} | E = {energy_list[step-1]:.8f} Ha")
        return final_vars, energy_list

    def _precompute_features(self, dets: np.ndarray) -> jnp.ndarray:
        """
        Convert determinant bitmasks to occupancy vectors.
        
        Example: |1010⟩ → [1,0,1,0] (α) + [spin-down occupation] (β)
        """
        if len(dets) == 0:
            return jnp.empty((0, 2 * self.config.system.n_orbitals), dtype=jnp.float32)
        dets_dev = jax.device_put(dets)
        return engine.utils.masks_to_vecs(dets_dev, self.config.system.n_orbitals)

    def _make_scan_optimizer(
        self, ham_ss, ham_sc, space_rep, optimizer, features_S, features_C
    ) -> Callable:
        """
        Create JIT-compiled lax.scan optimizer with captured context.

        Avoids recompilation by capturing invariant quantities:
          - Hamiltonian matrices
          - Pre-computed features
          - Nuclear repulsion energy
        """
        logpsi = self.model.log_psi
        n_orb = self.config.system.n_orbitals
        config = self.config
        e_nuc = self.e_nuc

        def step_fn(carry: tuple[Any, Any], _) -> tuple[tuple[Any, Any], float]:
            """Single VMC step: gradient computation + AdamW update."""
            variables, opt_state = carry
        
            evaluator = engine.Evaluator(
                params=variables,
                logpsi_fn=logpsi,
                ham_ss=ham_ss,
                ham_sc=ham_sc,
                space=space_rep,
                n_orbitals=n_orb,
                config=config,
                features_S=features_S,
                features_C=features_C,
            )
        
            # Compute ⟨E⟩ and ∇_θ⟨E⟩ via automatic differentiation
            result = engine.compute_energy_and_gradient(evaluator)
        
            # AdamW parameter update: θ ← θ - η·m/(√v + ε)
            updates, new_opt = optimizer.update(result.gradient, opt_state, variables)
            new_vars = optax.apply_updates(variables, updates)
        
            total_energy = result.energy_elec + e_nuc
            return (new_vars, new_opt), total_energy

        def scan_wrapper(init_vars, init_opt, num_steps: int):
            """JIT-compiled full optimization loop."""
            final_carry, energy_hist = lax.scan(
                f=step_fn,
                init=(init_vars, init_opt),
                xs=None,
                length=num_steps
            )
            return final_carry, energy_hist

        return jax.jit(scan_wrapper, static_argnames=['num_steps'])

    def _evaluate_energies(
        self, cycle: int, ham_opt, space_rep, history: dict
    ):
        """
        Post-optimization energy diagnostics.
        
        Computes:
          - E_CI(S): Exact diagonalization of H_SS (or H_eff)
          - E_var(T): Variational energy over full T-space
          - E_CI(T): Exact diagonalization of H_TT
        
        Args:
            ham_opt: H_SS for ASYMMETRIC/PROXY, H_eff for EFFECTIVE
        """
        from . import analysis
        cfg = self.config
        suite = analysis.EvaluationSuite(
            self.int_ctx, cfg.system.n_orbitals, self.e_nuc
        )
        n_cycles = cfg.optimization.num_cycles

        def should_eval(mode: str) -> bool:
            return mode == "every" or (mode == "final" and cycle == n_cycles - 1)

        # S-space CI: diagonalize H_SS (or H_eff in EFFECTIVE mode)
        if should_eval(cfg.evaluation.s_ci_energy_mode.value):
            e_s_ci = suite.diagonalize_hamiltonian(ham_opt)
            history["s_ci"].append(e_s_ci)
            print(f"  E_CI(S): {e_s_ci:.8f} Ha")
    
        # T-space evaluations (not applicable for EFFECTIVE mode)
        if cfg.compute_mode in {ComputeMode.ASYMMETRIC, ComputeMode.PROXY}:
            eval_var = should_eval(cfg.evaluation.var_energy_mode.value)
            eval_t_ci = should_eval(cfg.evaluation.t_ci_energy_mode.value)
        
            if eval_var or eval_t_ci:
                # Build full T-space Hamiltonian: T = S ∪ C
                t_dets = np.concatenate([space_rep.s_dets, space_rep.c_dets])
                ham_tt, _ = engine.hamiltonian.get_ham_ss(
                    S_dets=t_dets,
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
        """
        Compute L2-normalized S-space amplitudes for dynamic screening.
        
        Returns: ψ_S / ||ψ_S||₂
        """
        features_S = self._precompute_features(s_dets)
        log_psi = self.model.log_psi(self.variables, features_S)
        psi = np.abs(np.array(jax.device_get(jnp.exp(log_psi))))
    
        norm = np.linalg.norm(psi)
        return psi / norm if norm > 1e-14 else psi

    def _create_evaluator(
        self, ham_ss, ham_sc, space_rep, for_evolution: bool = False
    ) -> engine.Evaluator:
        """
        Create evaluator with context-aware feature computation.
        
        FIX: Reuse cached features_S to avoid redundant computation within same cycle.
        """
        # FIXED: Reuse cached features from optimization phase
        features_S = getattr(self, '_cached_features_S', None)
        if features_S is None:
            features_S = self._precompute_features(space_rep.s_dets)
        
        # Compute C features for:
        # 1. ASYMMETRIC/PROXY modes (always needed)
        # 2. EFFECTIVE mode during evolution (amplitude scoring)
        if self.config.compute_mode != ComputeMode.EFFECTIVE or for_evolution:
            features_C = self._precompute_features(space_rep.c_dets)
        else:
            features_C = None
        
        return engine.Evaluator(
            params=self.variables,
            logpsi_fn=self.model.log_psi,
            ham_ss=ham_ss,
            ham_sc=ham_sc,
            space=space_rep,
            n_orbitals=self.config.system.n_orbitals,
            config=self.config,
            features_S=features_S,
            features_C=features_C,
            force_full_space=for_evolution,
        )


__all__ = ["Driver", "DriverResults"]
