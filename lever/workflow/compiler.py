# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Workspace compiler: OuterState → OuterCtx transformation.

Five-phase compilation pipeline:
  1. Bootstrap: Amplitude estimation for dynamic screening
  2. Build: Hamiltonian blocks (H_SS, H_SC) via sparse CI
  3. Features: Neural network input conversion
  4. Closures: JIT-compiled operators (log_psi, SpMV)
  5. Package: Assemble immutable OuterCtx

File: lever/workflow/compiler.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import numpy as np

from .. import core, engine
from ..config import ComputeMode, ScreenMode
from ..dtypes import COOMatrix, OuterCtx, SpaceRep
from ..utils.features import compute_normalized_amplitudes, dets_to_features
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..config import LeverConfig
    from ..models import WavefunctionModel
    from ..dtypes import OuterState, PyTree


class Compiler:
    """
    Stateless compiler: OuterState → OuterCtx.
    
    Transforms abstract search space into concrete computational context
    with pre-compiled JAX operators.
    """
    
    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        int_ctx: core.IntCtx
    ):
        """
        Initialize compiler with system configuration.
        
        Args:
            config: LEVER configuration
            model: Wavefunction neural network
            int_ctx: Integral provider for Hamiltonian construction
        """
        self.cfg = config
        self.model = model
        self.int_ctx = int_ctx
        self.logger = get_logger()
        self._last_shape: tuple[int, int] | None = None
    
    def compile(
        self,
        state: OuterState
    ) -> OuterCtx:
        """
        Compile complete computational context from outer state.
        
        Args:
            state: Current outer loop state with S-space and parameters
        
        Returns:
            Immutable OuterCtx with compiled operators and metadata
        """
        mode = self.cfg.compute_mode
        
        # Phase 1-3: Build Hamiltonian and features
        psi_s_est = self._bootstrap_amplitudes(state)
        ham_ss_raw, ham_sc, space = self._build_hamiltonian(state, psi_s_est)
        feat_s, feat_c = self._prepare_features(space)
        
        # Phase 4: Compile JAX operators
        log_psi_fn = self._compile_log_psi(feat_s, feat_c, mode)
        spmv_fn = self._compile_spmv(ham_ss_raw, ham_sc, space, mode)
        
        # Phase 5: Package context
        self._log_compilation_info(space, ham_ss_raw, ham_sc, mode)
        
        return OuterCtx(
            space=space,
            ham_ss=ham_ss_raw,
            ham_sc=ham_sc,
            e_nuc=self.int_ctx.get_e_nuc(),
            features_s=feat_s,
            features_c=feat_c,
            log_psi_fn=log_psi_fn,
            spmv_fn=spmv_fn,
            compute_mode=mode
        )
    
    def _bootstrap_amplitudes(
        self,
        state: OuterState
    ) -> np.ndarray | None:
        """Compute normalized S-space amplitudes for dynamic screening."""
        if self.cfg.hamiltonian.screening_mode != ScreenMode.DYNAMIC:
            return None
        
        # Extract base log_psi (handle tuple format for EFFECTIVE mode)
        logpsi_fn = (self.model.log_psi[0] if isinstance(self.model.log_psi, tuple)
                    else self.model.log_psi)
        
        psi_s = compute_normalized_amplitudes(
            dets=state.s_dets,
            params=state.params,
            log_psi_fn=logpsi_fn,
            n_orb=self.cfg.system.n_orbitals
        )
        
        self.logger.bootstrap_amplitudes(psi_s.min(), psi_s.max())
        return psi_s
    
    def _build_hamiltonian(
        self,
        state: OuterState,
        psi_s_est: np.ndarray | None
    ) -> tuple[COOMatrix, COOMatrix, SpaceRep]:
        """Build H_SS and H_SC blocks via sparse CI expansion."""
        ham_cfg = self.cfg.hamiltonian
        kwargs = {
            "S_dets": state.s_dets,
            "int_ctx": self.int_ctx,
            "n_orbitals": self.cfg.system.n_orbitals,
        }
        
        match ham_cfg.screening_mode:
            case ScreenMode.NONE:
                ham_ss, ham_sc, space = engine.hamiltonian.get_ham_proxy(
                    **kwargs, mode="none"
                )
            
            case ScreenMode.STATIC:
                ham_ss, ham_sc, space = engine.hamiltonian.get_ham_proxy(
                    **kwargs,
                    mode="static",
                    screen_eps=ham_cfg.screen_eps,
                    diag_shift=ham_cfg.diag_shift
                )
            
            case ScreenMode.DYNAMIC:
                if psi_s_est is None:
                    raise ValueError("Dynamic screening requires bootstrap")
                ham_ss, ham_sc, space = engine.hamiltonian.get_ham_proxy(
                    **kwargs,
                    mode="dynamic",
                    psi_S=psi_s_est,
                    screen_eps=ham_cfg.screen_eps,
                    diag_shift=ham_cfg.diag_shift
                )
        
        # For EFFECTIVE mode, assemble H_eff from H_SS + perturbation
        if self.cfg.compute_mode == ComputeMode.EFFECTIVE and state.e_ref is not None:
            ham_eff = engine.hamiltonian.get_ham_eff(
                ham_ss=ham_ss,
                ham_sc=ham_sc,
                h_cc_diag=space.h_diag_c,
                e_ref=state.e_ref,
                reg_type="sigma",
                num_eps=self.cfg.hamiltonian.reg_eps
            )
            self.logger.hamiltonian_assembled("H_eff", ham_eff.nnz)
            return ham_eff, ham_sc, space
        
        return ham_ss, ham_sc, space
    
    def _prepare_features(self, space: SpaceRep) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert determinants to neural network features."""
        n_orb = self.cfg.system.n_orbitals
        feat_s = dets_to_features(space.s_dets, n_orb)
        feat_c = dets_to_features(space.c_dets, n_orb)
        return feat_s, feat_c
    
    def _compile_log_psi(
        self,
        feat_s: jnp.ndarray,
        feat_c: jnp.ndarray,
        mode: ComputeMode
    ) -> Callable:
        """
        Compile log(ψ) evaluator with feature binding.
        
        Returns closure or tuple of closures depending on mode.
        For EFFECTIVE/ASYMMETRIC: returns (eval_s, eval_full)
        For PROXY: returns eval_full
        """
        from ..engine import operator
        
        return operator.create_logpsi_evals(
            model_fn=self.model.log_psi,
            feat_s=feat_s,
            feat_c=feat_c,
            mode=mode,
            normalize=self.cfg.normalize_wf,
            device_complex=self.cfg.precision.jax_complex,
            chunk_size=self.cfg.loop.chunk_size
        )
    
    def _compile_spmv(
        self,
        ham_opt: COOMatrix,
        ham_sc: COOMatrix,
        space: SpaceRep,
        mode: ComputeMode
    ) -> Callable:
        """
        Compile sparse matrix-vector product H·ψ.
        
        Returns appropriate SpMV function based on compute mode.
        """
        from ..engine import operator
        
        if mode == ComputeMode.EFFECTIVE:
            return operator.create_spmv_eff(
                ham_eff_rows=ham_opt.rows,
                ham_eff_cols=ham_opt.cols,
                ham_eff_vals=ham_opt.vals,
                n_s=space.n_s,
                precision_config=self.cfg.precision
            )
        else:
            return operator.create_spmv_proxy(
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
    
    def _log_compilation_info(
        self,
        space: SpaceRep,
        ham_opt: COOMatrix,
        ham_sc: COOMatrix,
        mode: ComputeMode
    ) -> None:
        """Emit compilation diagnostics and JIT recompilation warnings."""
        n_s, n_c = space.n_s, space.n_c
        current_shape = (n_s, n_c)
        
        # Warn on shape change (triggers JIT recompilation)
        if self._last_shape and self._last_shape != current_shape:
            self.logger.recompilation_warning(self._last_shape, current_shape)
        self._last_shape = current_shape
        
        self.logger.space_dimensions(n_s, n_c)
        
        if mode == ComputeMode.EFFECTIVE:
            self.logger.hamiltonian_sparsity(n_s, ham_opt.nnz)
        else:
            self.logger.hamiltonian_sparsity(n_s, ham_opt.nnz, ham_sc.nnz)


__all__ = ["Compiler"]

