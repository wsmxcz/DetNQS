# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER driver with OuterCycle/InnerCycle architecture.

Implements dual-loop optimization:
  Outer: Core space evolution (S_k -> S_{k+1})
  Inner: Parameter optimization via JIT scan
  
File: lever/driver.py
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
from .config import ComputeMode, LeverConfig, ScreenMode, EvalMode
from .models import WavefunctionModel
from .dtypes import PyTree, PsiCache, HamOp
from .logger import get_logger


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
        self.logger = get_logger()
        
        # Track shape for recompilation detection
        self._last_shape = None
        
        # Initialize integral context
        self.int_ctx = core.IntCtx(
            config.system.fcidump_path,
            config.system.n_orbitals
        )
        self.int_ctx.hb_prepare(threshold=1e-15)
        self.e_nuc = self.int_ctx.get_e_nuc()
        self.e_ref: float | None = None
        
    def run(self) -> DriverResults:
        """Execute complete LEVER workflow."""
        self._print_header()
        
        s_dets = self._initialize_hf_space()
        hist = self._create_history_dict()
        t0 = time.time()
        psi_cache = None
        outer_ctx = None
        
        for cycle in range(self.cfg.optimization.num_cycles):
            psi_cache, outer_ctx = self._run_outer_cycle(cycle, s_dets, hist)
            
            if cycle < self.cfg.optimization.num_cycles - 1:
                s_dets = self._evolve_space(outer_ctx, psi_cache)
        
        return self._finalize_results(hist, time.time() - t0)

    # ==================== Initialization ====================

    def _print_header(self) -> None:
        """Print workflow initialization header."""
        self.logger.header("LEVER Initialization")
        
        config_items = {
            "Compute mode": self.cfg.compute_mode.value.upper(),
            "System": self.cfg.system.fcidump_path,
            "Model": self.model.module.__class__.__name__,
            "Screening": self.cfg.screening.mode.value,
        }
        self.logger.config_info(config_items)

    def _print_space_info(self, ctx: engine.OuterCtx) -> None:
        """Log space info with recompilation detection."""
        n_s, n_c = ctx.space.n_s, ctx.space.n_c
        
        # Check for shape change
        current_shape = (n_s, n_c)
        if self._last_shape is not None and self._last_shape != current_shape:
            self.logger.recompilation_warning(self._last_shape, current_shape)
        self._last_shape = current_shape
        
        # Log space dimensions
        self.logger.space_dimensions(n_s, n_c)
        
        # Log Hamiltonian sparsity
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE:
            self.logger.hamiltonian_sparsity(n_s, ctx.ham_opt.nnz)
        else:
            self.logger.hamiltonian_sparsity(n_s, ctx.ham_opt.nnz, ctx.ham_sc.nnz)

    def _initialize_hf_space(self) -> np.ndarray:
        """Initialize S-space with HF determinant."""
        sys = self.cfg.system
        hf_det = np.array([
            [(1 << sys.n_alpha) - 1, (1 << sys.n_beta) - 1]
        ], dtype=np.uint64)
      
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE:
            hf_diag = core.get_ham_diag(dets=hf_det, int_ctx=self.int_ctx)
            self.e_ref = float(hf_diag[0])
      
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
    ) -> tuple[PsiCache, engine.OuterCtx]:
        """Execute outer cycle with timing."""
        t0 = time.time()
        
        self.logger.cycle_start(cycle + 1, self.cfg.optimization.num_cycles)
        
        # Build context
        outer_ctx = self._build_outer_context(s_dets)
        self._print_space_info(outer_ctx)
        
        # Optimize
        self.params, energies, psi_cache = self._optimize_parameters(outer_ctx)
        self._update_history(hist, energies)
        
        # Update reference
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE:
            self.e_ref = energies[-1] - self.e_nuc
        
        # Diagnostics
        self._evaluate_energies(cycle, outer_ctx, hist, psi_cache)
        
        self.logger.timing(f"Cycle {cycle+1}", time.time() - t0)
        
        return psi_cache, outer_ctx

    def _build_outer_context(self, s_dets: np.ndarray) -> engine.OuterCtx:
        """
        Build outer cycle context with precomputed invariants.
        
        EFFECTIVE mode creates two logpsi closures:
        - logpsi_s_fn: S-only for optimization (hot path)
        - logpsi_full_fn: S∪C for evolution/cache (cold path)
        """
        cfg = self.cfg
        mode = cfg.compute_mode
        
        # Phase 1: Dynamic screening bootstrap (optional)
        psi_s_est = None
        if cfg.screening.mode == ScreenMode.DYNAMIC:
            psi_s_est = self._compute_normalized_amplitudes(s_dets)
            self.logger.bootstrap_amplitudes(psi_s_est.min(), psi_s_est.max())
        
        # Phase 2: Build Hamiltonian blocks
        ham_ss_raw, ham_sc, space = self._build_ham_blocks(s_dets, psi_s_est)
        
        # Phase 3: Mode-specific Hamiltonian preparation
        if mode == ComputeMode.EFFECTIVE:
            ham_opt = self._assemble_effective_hamiltonian(
                ham_ss_raw, ham_sc, space.h_diag_c
            )
            self.logger.hamiltonian_assembled("H_eff", ham_opt.nnz)
        else:
            ham_opt = ham_ss_raw
        
        # Phase 4: Precompute features
        feat_s = self._dets_to_features(space.s_dets)
        feat_c = (
            self._dets_to_features(space.c_dets)
            if mode != ComputeMode.EFFECTIVE or len(space.c_dets) > 0
            else jnp.empty((0, 2 * cfg.system.n_orbitals), dtype=jnp.float32)
        )
        
        # Phase 5: Create evaluation closures
        if mode == ComputeMode.EFFECTIVE:
            logpsi_s_fn = engine.evaluator.create_logpsi_fn(
                model_fn=self.model.log_psi,
                feat_s=feat_s,
                feat_c=jnp.empty((0, feat_s.shape[1]), dtype=feat_s.dtype),
                mode=mode,
                normalize=cfg.normalize_wf,
                eps=cfg.num_eps,
                device_complex=cfg.precision.jax_complex
            )
            logpsi_full_fn = engine.evaluator.create_logpsi_fn(
                model_fn=self.model.log_psi,
                feat_s=feat_s,
                feat_c=feat_c,
                mode=mode,
                normalize=cfg.normalize_wf,
                eps=cfg.num_eps,
                device_complex=cfg.precision.jax_complex
            )
            logpsi_fn = (logpsi_s_fn, logpsi_full_fn)
        else:
            logpsi_fn = engine.evaluator.create_logpsi_fn(
                model_fn=self.model.log_psi,
                feat_s=feat_s,
                feat_c=feat_c,
                mode=mode,
                normalize=cfg.normalize_wf,
                eps=cfg.num_eps,
                device_complex=cfg.precision.jax_complex
            )
        
        spmv_fn = self._create_spmv_closure(ham_opt, ham_sc, space, mode)
        
        return engine.OuterCtx(
            space=space,
            feat_s=feat_s,
            feat_c=feat_c,
            ham_opt=ham_opt,
            ham_sc=ham_sc,
            logpsi_fn=logpsi_fn,
            spmv_fn=spmv_fn,
            e_ref=self.e_ref or 0.0,
            e_nuc=self.e_nuc,
            mode=mode
        )

    def _build_ham_blocks(
        self, 
        s_dets: np.ndarray,
        psi_s_est: np.ndarray | None = None
    ):
        """Build Hamiltonian blocks with screening from unified config."""
        cfg = self.cfg
        kwargs = {
            "S_dets": s_dets,
            "int_ctx": self.int_ctx,
            "n_orbitals": cfg.system.n_orbitals,
        }
        
        match cfg.screening.mode:
            case ScreenMode.NONE:
                return engine.hamiltonian.get_ham_proxy(**kwargs, mode="none")
            case ScreenMode.STATIC:
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs, mode="static", 
                    screen_eps=cfg.screening.screen_eps,
                    diag_shift=cfg.screening.diag_shift
                )
            case ScreenMode.DYNAMIC:
                if psi_s_est is None:
                    raise ValueError("Dynamic mode requires psi_s_est")
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs, mode="dynamic",
                    psi_S=psi_s_est, 
                    screen_eps=cfg.screening.screen_eps,
                    diag_shift=cfg.screening.diag_shift
                )
            case _:
                raise ValueError(f"Unknown screening mode: {cfg.screening.mode}")

    def _assemble_effective_hamiltonian(self, ham_ss, ham_sc, h_cc_diag):
        """Assemble H_eff = H_SS + H_SC·D^{-1}·H_CS."""
        result = engine.hamiltonian.get_ham_eff(
            ham_ss=ham_ss,
            ham_sc=ham_sc,
            h_cc_diag=h_cc_diag,
            e_ref=self.e_ref,
            num_eps=1e-4
        )
        return result

    def _create_psi_cache(self, ctx: engine.OuterCtx, params) -> PsiCache:
        """
        Create wavefunction cache with minimal host transfer.
        
        For EFFECTIVE mode, uses logpsi_full_fn to compute S∪C amplitudes.
        """
        # Extract appropriate logpsi function
        if ctx.mode == ComputeMode.EFFECTIVE:
            _, logpsi_full_fn = ctx.logpsi_fn
            log_all = logpsi_full_fn(params)
        else:
            log_all = ctx.logpsi_fn(params)
        
        psi_all = jnp.exp(log_all)
        
        # Compute norms on device, transfer only scalars
        psi_s_norm_sq = jnp.sum(jnp.abs(psi_all[:ctx.space.n_s])**2)
        psi_c_norm_sq = jnp.sum(jnp.abs(psi_all[ctx.space.n_s:])**2)
        norms = np.array([psi_s_norm_sq, psi_c_norm_sq])
        
        self.logger.wavefunction_cache(norms[0], norms[1])
        
        return PsiCache(
            log_all=log_all,
            psi_all=psi_all,
            n_s=ctx.space.n_s,
            n_c=ctx.space.n_c
        )

    def _dets_to_features(self, dets: np.ndarray) -> jnp.ndarray:
        """Convert determinant masks to occupancy vectors."""
        if len(dets) == 0:
            n_orb = self.cfg.system.n_orbitals
            return jnp.empty((0, 2 * n_orb), dtype=jnp.float32)
      
        dets_dev = jax.device_put(dets)
        return engine.utils.masks_to_vecs(dets_dev, self.cfg.system.n_orbitals)

    def _create_spmv_closure(self, ham_opt, ham_sc, space, mode):
        """Create SpMV closure with unified precision config."""
        if mode == ComputeMode.EFFECTIVE:
            return engine.evaluator.create_spmv_eff(
                ham_eff_rows=ham_opt.rows,
                ham_eff_cols=ham_opt.cols,
                ham_eff_vals=ham_opt.vals,
                n_s=space.n_s,
                precision_config=self.cfg.precision
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
                n_c=space.n_c,
                precision_config=self.cfg.precision
            )

    # ==================== Inner Cycle ====================

    def _optimize_parameters(
        self,
        outer_ctx: engine.OuterCtx
    ) -> tuple[Any, list[float], PsiCache]:
        """VMC optimization with minimal host transfers."""
        cfg = self.cfg.optimization
        
        # Create optimizer
        if cfg.weight_decay > 0:
            optimizer = optax.adamw(
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        else:
            optimizer = optax.adam(learning_rate=cfg.learning_rate)
        
        opt_state = optimizer.init(self.params)
        step_fn = engine.step.create_step_fn(outer_ctx, optimizer, self.cfg.num_eps)
        step_fn_jit = jax.jit(step_fn, donate_argnums=(0,))
        
        state = engine.InnerState(params=self.params, opt_state=opt_state, step=0)
        energies_host = []
        
        self.logger.optimization_header()
        
        for step in range(cfg.steps_per_cycle):
            state, energy = step_fn_jit(state, None)
            energies_host.append(float(energy))
            
            if (step + 1) % cfg.report_interval == 0:
                self.logger.optimization_step(
                    step + 1, 
                    energies_host[-1],
                    cfg.steps_per_cycle
                )
        
        psi_cache = self._create_psi_cache(outer_ctx, state.params)
        return state.params, energies_host, psi_cache

    # ==================== Diagnostics ====================

    def _evaluate_energies(self, cycle, ctx, hist, psi_cache):
        """Post-optimization energy evaluations using cached amplitudes."""
        cfg = self.cfg
        suite = analysis.EvalSuite(self.int_ctx, cfg.system.n_orbitals, self.e_nuc)
        
        n_cycles = cfg.optimization.num_cycles
        is_final = (cycle == n_cycles - 1)
        
        energies_dict = {}
        
        # S-space CI energy
        if self._should_evaluate(cfg.evaluation.s_ci_energy_mode, is_final):
            e_s_ci = suite.diag_ham(ctx.ham_opt)
            hist["s_ci"].append(e_s_ci)
            energies_dict["E_CI(S)"] = e_s_ci
        
        # Skip T-space for effective mode
        if cfg.compute_mode == ComputeMode.EFFECTIVE:
            if energies_dict:
                self.logger.energy_table(energies_dict)
            return
        
        # T-space evaluations
        eval_var = self._should_evaluate(cfg.evaluation.var_energy_mode, is_final)
        eval_t_ci = self._should_evaluate(cfg.evaluation.t_ci_energy_mode, is_final)
        
        if eval_var or eval_t_ci:
            t_dets, ham_tt = self._build_full_space_hamiltonian(ctx)
            
            if eval_var:
                e_var = suite.var_energy_from_cache(psi_cache, ham_tt, t_dets)
                hist["var"].append(e_var)
                energies_dict["E_var(T)"] = e_var
            
            if eval_t_ci:
                e_t_ci = suite.diag_ham(ham_tt)
                hist["t_ci"].append(e_t_ci)
                energies_dict["E_CI(T)"] = e_t_ci
        
        if energies_dict:
            self.logger.energy_table(energies_dict)

    @staticmethod
    def _should_evaluate(mode: EvalMode, is_final: bool) -> bool:
        """Check if evaluation should run."""
        return mode is EvalMode.EVERY or (mode is EvalMode.FINAL and is_final)

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
    
    def _evolve_space(
        self,
        outer_ctx: engine.OuterCtx,
        psi_cache: PsiCache
    ) -> np.ndarray:
        """Evolve S-space using cached amplitudes (silent operation)."""
        return self.strategy.evolve(outer_ctx, psi_cache)

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
        final_energy = hist["full"][-1] if hist["full"] else 0.0
        self.logger.final_summary(
            total_time,
            final_energy,
            self.cfg.optimization.num_cycles
        )
      
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
