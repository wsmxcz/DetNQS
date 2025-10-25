# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER driver with OuterCycle/InnerCycle architecture.

Implements dual-loop optimization:
  Outer: Core space evolution (S_k -> S_{k+1})
  Inner: Parameter optimization via JIT scan

Author: Zheng (Alex) Che, wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import analysis, core, engine, evolution
from .config import ComputeMode, LeverConfig, ScreenMode
from .models import WavefunctionModel


@dataclass(frozen=True)
class DriverResults:
    """LEVER execution results with optimization history."""
    config: LeverConfig
    final_vars: Any
    full_hist: list[float]          # Per-step energies
    cycle_bounds: list[int]         # Cycle start indices in full_hist
    var_hist: list[float]           # E_var(T) at cycle end
    s_ci_hist: list[float]          # E_CI(S) at cycle end
    t_ci_hist: list[float]          # E_CI(T) at cycle end
    total_time: float


class Driver:
    """
    LEVER driver executing outer/inner optimization loops.
  
    Outer cycle: Evolution of determinant space S
    Inner cycle: VMC parameter optimization via scan
    """

    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        strategy: evolution.EvolutionStrategy,
    ):
        self.cfg = config
        self.model = model
        self.strategy = strategy
        self.params = model.variables
  
        # Initialize integral context
        self.int_ctx = core.IntCtx(
            config.system.fcidump_path,
            config.system.n_orbitals
        )
        self.int_ctx.hb_prepare(threshold=1e-15)
        self.e_nuc = self.int_ctx.get_e_nuc()
      
        # Reference energy for effective mode: E_ref = E_elec
        self.e_ref: float | None = None

    def run(self) -> DriverResults:
        """Execute complete LEVER workflow."""
        self._print_header()
      
        s_dets = self._initialize_hf_space()
        hist = self._create_history_dict()
        t0 = time.time()

        for cycle in range(self.cfg.optimization.num_cycles):
            self._run_outer_cycle(cycle, s_dets, hist)
          
            # Evolve space for next cycle
            if cycle < self.cfg.optimization.num_cycles - 1:
                s_dets = self._evolve_space(s_dets)

        return self._finalize_results(hist, time.time() - t0)

    # ==================== Initialization ====================

    def _print_header(self) -> None:
        """Print workflow initialization info."""
        cfg = self.cfg
        print(f"LEVER ({cfg.compute_mode.value.upper()}): "
              f"{cfg.system.fcidump_path}")
        print(f"  Model: {self.model.module.__class__.__name__}")

    def _initialize_hf_space(self) -> np.ndarray:
        """Initialize S-space with HF determinant."""
        sys = self.cfg.system
        hf_det = np.array([
            [(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]
        ], dtype=np.uint64)
      
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE:
            hf_diag = core.get_ham_diag(dets=hf_det, int_ctx=self.int_ctx)
            self.e_ref = float(hf_diag[0])
            print(f"Initial E_ref (HF): {self.e_ref:.8f} Ha")
      
        return hf_det

    @staticmethod
    def _create_history_dict() -> dict[str, list]:
        """Create history tracking dictionary."""
        return {
            "full": [],
            "bounds": [0],
            "var": [],
            "s_ci": [],
            "t_ci": []
        }

    # ==================== Outer Cycle ====================

    def _run_outer_cycle(
        self,
        cycle: int,
        s_dets: np.ndarray,
        hist: dict[str, list]
    ) -> None:
        """Execute single outer cycle with space S."""
        print(f"\n{'='*60}")
        print(f"Cycle {cycle + 1}/{self.cfg.optimization.num_cycles}")
        print(f"{'='*60}")

        # Build context with all invariants
        outer_ctx = self._build_outer_context(s_dets)
        self._print_space_info(outer_ctx)

        # Inner loop optimization
        self.params, energies = self._optimize_parameters(outer_ctx)
        self._update_history(hist, energies)
      
        # Update reference energy for effective mode
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE:
            self.e_ref = energies[-1] - self.e_nuc

        # Post-optimization diagnostics
        self._evaluate_energies(cycle, outer_ctx, hist)

    def _build_outer_context(self, s_dets: np.ndarray) -> engine.OuterCtx:
        """
        Build outer cycle context with precomputed invariants.
      
        Steps:
          1. Assemble Hamiltonian blocks (H_SS, H_SC or H_eff)
          2. Precompute determinant features
          3. Create JIT-compiled closures (logpsi, spmv)
        """
        cfg = self.cfg
        mode = cfg.compute_mode
      
        # Phase 1: Build Hamiltonian blocks
        if mode == ComputeMode.EFFECTIVE:
            ham_ss_raw, ham_sc, space = self._build_ham_blocks(s_dets)
            ham_opt = self._assemble_effective_hamiltonian(
                ham_ss_raw, ham_sc, space.h_diag_c
            )
        else:
            ham_opt, ham_sc, space = self._build_ham_blocks(s_dets)
      
        # Phase 2: Precompute features
        feat_s, feat_c = self._precompute_features(space, mode)
      
        # Phase 3: Create evaluation closures
        logpsi_fn = self._create_logpsi_closure(feat_s, feat_c, mode)
        spmv_fn = self._create_spmv_closure(
            ham_opt, ham_sc, space, mode
        )
      
        return engine.OuterCtx(
            space=space,
            feat_s=feat_s,
            feat_c=feat_c,
            ham_opt=ham_opt,
            ham_sc=ham_sc if mode != ComputeMode.EFFECTIVE else None,
            logpsi_fn=logpsi_fn,
            spmv_fn=spmv_fn,
            e_ref=self.e_ref or 0.0,
            e_nuc=self.e_nuc,
            mode=mode.value
        )

    def _build_ham_blocks(self, s_dets: np.ndarray):
        """Build Hamiltonian blocks with optional screening."""
        cfg = self.cfg
        kwargs = {
            "S_dets": s_dets,
            "int_ctx": self.int_ctx,
            "n_orbitals": cfg.system.n_orbitals,
        }
      
        match cfg.screening.mode:
            case ScreenMode.NONE:
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs, mode="none"
                )
            case ScreenMode.STATIC:
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs, mode="static", eps1=cfg.screening.eps1
                )
            case ScreenMode.DYNAMIC:
                psi_s = self._compute_normalized_amplitudes(s_dets)
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs, mode="dynamic",
                    psi_S=psi_s, eps1=cfg.screening.eps1
                )
            case _:
                raise ValueError(
                    f"Unknown screening mode: {cfg.screening.mode}"
                )

    def _assemble_effective_hamiltonian(
        self,
        ham_ss: engine.hamiltonian.COOMatrix,
        ham_sc: engine.hamiltonian.COOMatrix,
        h_cc_diag: np.ndarray
    ) -> engine.hamiltonian.COOMatrix:
        """Assemble H_eff = H_SS + H_SC·D^{-1}·H_CS."""
        return engine.hamiltonian.get_ham_eff(
            ham_ss=ham_ss,
            ham_sc=ham_sc,
            h_cc_diag=h_cc_diag,
            e_ref=self.e_ref,
            epsilon=1e-6
        )

    def _precompute_features(
        self,
        space: engine.hamiltonian.DeterminantSpace,
        mode: ComputeMode
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert determinants to occupancy features."""
        feat_s = self._dets_to_features(space.s_dets)
      
        if mode == ComputeMode.EFFECTIVE and len(space.c_dets) == 0:
            n_orb = self.cfg.system.n_orbitals
            feat_c = jnp.empty((0, 2 * n_orb), dtype=jnp.float32)
        else:
            feat_c = self._dets_to_features(space.c_dets)
      
        return feat_s, feat_c

    def _dets_to_features(self, dets: np.ndarray) -> jnp.ndarray:
        """Convert determinant masks to occupancy vectors."""
        if len(dets) == 0:
            n_orb = self.cfg.system.n_orbitals
            return jnp.empty((0, 2 * n_orb), dtype=jnp.float32)
      
        dets_dev = jax.device_put(dets)
        return engine.utils.masks_to_vecs(dets_dev, self.cfg.system.n_orbitals)

    def _create_logpsi_closure(
        self,
        feat_s: jnp.ndarray,
        feat_c: jnp.ndarray,
        mode: ComputeMode
    ):
        """Create JIT-compiled log-amplitude closure."""
        return engine.evaluator.create_logpsi_fn(
            model_fn=self.model.log_psi,
            feat_s=feat_s,
            feat_c=feat_c,
            mode=mode.value,
            normalize=self.cfg.normalize_wf,
            eps=self.cfg.epsilon
        )

    def _create_spmv_closure(
        self,
        ham_opt,
        ham_sc,
        space: engine.hamiltonian.DeterminantSpace,
        mode: ComputeMode
    ):
        """Create JIT-compiled sparse matrix-vector product closure."""
        if mode == ComputeMode.EFFECTIVE:
            return engine.evaluator.create_spmv_eff(
                ham_eff_rows=ham_opt.rows,
                ham_eff_cols=ham_opt.cols,
                ham_eff_vals=ham_opt.vals,
                n_s=space.n_s
            )
        else:
            return engine.evaluator.create_spmv_proxy(
                ham_ss_rows=ham_opt.rows,
                ham_ss_cols=ham_opt.cols,
                ham_ss_vals=ham_opt.vals,
                ham_sc_rows=ham_sc.rows,
                ham_sc_cols=ham_sc.cols,
                ham_sc_vals=ham_sc.vals,
                h_diag_c=space.h_diag_c,
                n_s=space.n_s,
                n_c=space.n_c
            )

    # ==================== Inner Cycle ====================

    def _optimize_parameters(
        self,
        outer_ctx: engine.OuterCtx
    ) -> tuple[Any, list[float]]:
        """
        VMC parameter optimization via JIT scan.
      
        Uses compiled scan for efficient batch updates:
          params_{t+1} = Adam(params_t, ∇E_loc)
        """
        cfg = self.cfg.optimization
      
        # Initialize optimizer state
        optimizer = optax.adamw(cfg.learning_rate)
        opt_state = optimizer.init(self.params)
      
        inner_state = engine.InnerState(
            params=self.params,
            opt_state=opt_state,
            step=0
        )
      
        # JIT-compiled scan over steps
        scan_fn = engine.step.create_scan_fn(
            outer_ctx, optimizer, self.cfg.epsilon
        )
        final_state, energies = scan_fn(inner_state, cfg.steps_per_cycle)
      
        # Report progress
        self._report_optimization_progress(energies)
      
        return final_state.params, [float(e) for e in energies]

    def _report_optimization_progress(self, energies: jnp.ndarray) -> None:
        """Print optimization step energies."""
        cfg = self.cfg.optimization
        for step in range(cfg.report_interval, cfg.steps_per_cycle,
                         cfg.report_interval):
            print(f"  Step {step:4d} | E = {energies[step-1]:.8f} Ha")

    # ==================== Diagnostics ====================

    def _print_space_info(self, ctx: engine.OuterCtx) -> None:
        """Print current space dimensions and Hamiltonian stats."""
        print(f"Space: S={ctx.space.n_s}, C={ctx.space.n_c}")
      
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE:
            print(f"E_ref: {ctx.e_ref:.8f} Ha")
            print(f"NNZ(H_eff): {ctx.ham_opt.nnz:,}")
        else:
            print(f"NNZ(H_SS): {ctx.ham_opt.nnz:,}, "
                  f"NNZ(H_SC): {ctx.ham_sc.nnz:,}")

    def _evaluate_energies(
        self,
        cycle: int,
        ctx: engine.OuterCtx,
        hist: dict[str, list]
    ) -> None:
        """
        Post-optimization energy evaluations.
      
        Computes CI/variational energies based on config:
          - E_CI(S): Diagonalize H_eff or H_SS
          - E_var(T): Variational energy over full space
          - E_CI(T): Diagonalize H_TT
        """
        cfg = self.cfg
        suite = analysis.EvalSuite(
            self.int_ctx, cfg.system.n_orbitals, self.e_nuc
        )
      
        n_cycles = cfg.optimization.num_cycles
        is_final = (cycle == n_cycles - 1)

        # S-space CI energy
        if self._should_evaluate(cfg.evaluation.s_ci_energy_mode, is_final):
            e_s_ci = suite.diag_ham(ctx.ham_opt)
            hist["s_ci"].append(e_s_ci)
            print(f"  E_CI(S): {e_s_ci:.8f} Ha")

        # T-space evaluations (skip for effective mode)
        if cfg.compute_mode == ComputeMode.EFFECTIVE:
            return

        eval_var = self._should_evaluate(
            cfg.evaluation.var_energy_mode, is_final
        )
        eval_t_ci = self._should_evaluate(
            cfg.evaluation.t_ci_energy_mode, is_final
        )

        if eval_var or eval_t_ci:
            t_dets, ham_tt = self._build_full_space_hamiltonian(ctx)
          
            if eval_var:
                e_var = suite.var_energy(
                    self.params, self.model.log_psi, ham_tt, t_dets
                )
                hist["var"].append(e_var)
                print(f"  E_var(T): {e_var:.8f} Ha")
          
            if eval_t_ci:
                e_t_ci = suite.diag_ham(ham_tt)
                hist["t_ci"].append(e_t_ci)
                print(f"  E_CI(T): {e_t_ci:.8f} Ha")

    @staticmethod
    def _should_evaluate(mode_str: str, is_final: bool) -> bool:
        """Check if evaluation should run this cycle."""
        return mode_str == "every" or (mode_str == "final" and is_final)

    def _build_full_space_hamiltonian(
        self,
        ctx: engine.OuterCtx
    ) -> tuple[np.ndarray, engine.hamiltonian.COOMatrix]:
        """Build H_TT over full determinant space T = S ∪ C."""
        t_dets = np.concatenate([ctx.space.s_dets, ctx.space.c_dets])
        ham_tt, _ = engine.hamiltonian.get_ham_ss(
            S_dets=t_dets,
            int_ctx=self.int_ctx,
            n_orbitals=self.cfg.system.n_orbitals
        )
        return t_dets, ham_tt

    # ==================== Space Evolution ====================

    def _evolve_space(self, s_dets: np.ndarray) -> np.ndarray:
        """Evolve S-space using configured strategy."""
        print("\nEvolving S-space...")
        # Build temporary context for evolution
        outer_ctx = self._build_outer_context(s_dets)
        new_s_dets = self.strategy.evolve(outer_ctx, self.params)
        print(f"New S-space: {len(new_s_dets)} determinants")
        return new_s_dets

    # ==================== Utilities ====================

    def _compute_normalized_amplitudes(
        self,
        s_dets: np.ndarray
    ) -> np.ndarray:
        """Compute L2-normalized S-space amplitudes."""
        feat_s = self._dets_to_features(s_dets)
        log_psi = self.model.log_psi(self.params, feat_s)
        psi = np.abs(np.array(jax.device_get(jnp.exp(log_psi))))
      
        norm = np.linalg.norm(psi)
        return psi / norm if norm > 1e-14 else psi

    @staticmethod
    def _update_history(hist: dict[str, list], energies: list[float]) -> None:
        """Update energy history with new cycle results."""
        hist["full"].extend(energies)
        hist["bounds"].append(len(hist["full"]))

    def _finalize_results(
        self,
        hist: dict[str, list],
        total_time: float
    ) -> DriverResults:
        """Package final results with timing info."""
        print(f"\n{'='*60}")
        print(f"LEVER completed in {total_time:.2f}s")
        print(f"{'='*60}")
      
        return DriverResults(
            config=self.cfg,
            final_vars=self.params,
            full_hist=hist["full"],
            cycle_bounds=hist["bounds"],
            var_hist=hist["var"],
            s_ci_hist=hist["s_ci"],
            t_ci_hist=hist["t_ci"],
            total_time=total_time
        )


__all__ = ["Driver", "DriverResults"]