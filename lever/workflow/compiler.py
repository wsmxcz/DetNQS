# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Workspace compilation from S-space determinants.

Five-phase compilation pipeline:
  1. Bootstrap: Amplitude estimation for dynamic screening
  2. Build: Hamiltonian blocks (H_SS, H_SC) via sparse CI  
  3. Assemble: Effective operator H_eff = H_SS + H_SC·D⁻¹·H_CS
  4. Features: Neural network input conversion
  5. Closures: JIT-compiled operators (log_psi, SpMV)

File: lever/workflow/compiler.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from .. import engine
from ..config import ComputeMode, ScreenMode
from ..utils.dtypes import PyTree, Workspace
from ..utils.features import compute_normalized_amplitudes, dets_to_features
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..config import LeverConfig
    from ..core import IntCtx
    from ..models import WavefunctionModel


class Compiler:
    """Workspace compiler for fixed S-space determinant basis."""
    
    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        int_ctx: IntCtx
    ):
        self.cfg = config
        self.model = model
        self.int_ctx = int_ctx
        self.logger = get_logger()
        self._last_shape: tuple[int, int] | None = None
    
    def compile(
        self,
        s_dets: np.ndarray,
        params: PyTree,
        e_ref: float | None = None
    ) -> Workspace:
        """
        Compile complete workspace from S-space determinants.
        
        Args:
            s_dets: Selected space determinants (n_s, n_orb)
            params: Neural network parameters  
            e_ref: Reference energy for H_eff assembly
        
        Returns:
            Workspace with compiled operators and cached data
        """
        mode = self.cfg.compute_mode
        
        # Five-phase compilation pipeline
        psi_s_est = self._bootstrap_amplitudes(s_dets, params)
        ham_ss_raw, ham_sc, space = self._build_hamiltonian(s_dets, psi_s_est)
        ham_opt = self._assemble_operator(ham_ss_raw, ham_sc, space, e_ref)
        feat_s, feat_c = self._prepare_features(space, mode)
        log_psi = self._create_log_psi(feat_s, feat_c, mode)
        spmv_fn = self._create_spmv_fn(ham_opt, ham_sc, space, mode)
        
        self._log_workspace_info(space, ham_opt, ham_sc, mode)
        
        return Workspace(
            space=space,
            feat_s=feat_s,
            feat_c=feat_c,
            ham_opt=ham_opt,
            ham_sc=ham_sc,
            log_psi=log_psi,
            spmv_fn=spmv_fn,
            e_ref=e_ref or 0.0,
            e_nuc=self.int_ctx.get_e_nuc(),
            mode=mode
        )
    
    def _bootstrap_amplitudes(
        self,
        s_dets: np.ndarray,
        params: PyTree
    ) -> np.ndarray | None:
        """Compute normalized S-space amplitudes |ψ_i| for dynamic screening."""
        if self.cfg.hamiltonian.screening_mode != ScreenMode.DYNAMIC:
            return None
        
        # Extract base log_psi (handle EFFECTIVE mode tuple)
        logpsi_fn = (self.model.log_psi[0] if isinstance(self.model.log_psi, tuple) 
                    else self.model.log_psi)
        
        psi_s = compute_normalized_amplitudes(
            dets=s_dets,
            params=params,
            log_psi_fn=logpsi_fn,
            n_orb=self.cfg.system.n_orbitals
        )
        
        self.logger.bootstrap_amplitudes(psi_s.min(), psi_s.max())
        return psi_s
    
    def _build_hamiltonian(
        self,
        s_dets: np.ndarray,
        psi_s_est: np.ndarray | None
    ):
        """Build H_SS and H_SC blocks via sparse CI expansion."""
        ham_cfg = self.cfg.hamiltonian
        kwargs = {
            "S_dets": s_dets,
            "int_ctx": self.int_ctx,
            "n_orbitals": self.cfg.system.n_orbitals,
        }
        
        match ham_cfg.screening_mode:
            case ScreenMode.NONE:
                return engine.hamiltonian.get_ham_proxy(**kwargs, mode="none")
            
            case ScreenMode.STATIC:
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs,
                    mode="static",
                    screen_eps=ham_cfg.screen_eps,
                    diag_shift=ham_cfg.diag_shift
                )
            
            case ScreenMode.DYNAMIC:
                if psi_s_est is None:
                    raise ValueError("Dynamic screening requires bootstrap")
                return engine.hamiltonian.get_ham_proxy(
                    **kwargs,
                    mode="dynamic",
                    psi_S=psi_s_est,
                    screen_eps=ham_cfg.screen_eps,
                    diag_shift=ham_cfg.diag_shift
                )
    
    def _assemble_operator(self, ham_ss, ham_sc, space, e_ref):
        """
        Form optimization operator.
        
        EFFECTIVE: H_eff = H_SS + H_SC·(E_ref·I - H_CC)⁻¹·H_CS
        PROXY/ASYMMETRIC: H_SS
        """
        if self.cfg.compute_mode != ComputeMode.EFFECTIVE:
            return ham_ss
        
        if e_ref is None:
            return ham_ss
        
        ham_eff = engine.hamiltonian.get_ham_eff(
            ham_ss=ham_ss,
            ham_sc=ham_sc,
            h_cc_diag=space.h_diag_c,
            e_ref=e_ref,
            reg_type="sigma",
            num_eps=self.cfg.hamiltonian.reg_eps
        )
        
        self.logger.hamiltonian_assembled("H_eff", ham_eff.nnz)
        return ham_eff
    
    def _prepare_features(self, space, mode) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Convert determinants to neural network features."""
        n_orb = self.cfg.system.n_orbitals
        feat_s = dets_to_features(space.s_dets, n_orb)
        feat_c = dets_to_features(space.c_dets, n_orb)
        return feat_s, feat_c
    
    def _create_log_psi(self, feat_s, feat_c, mode):
        """Compile log(ψ) evaluator without capturing features."""
        return engine.operator.create_logpsi_evals(
            model_fn=self.model.log_psi,
            mode=mode,
            normalize=self.cfg.normalize_wf,
            device_complex=self.cfg.precision.jax_complex
        )
    
    def _create_spmv_fn(self, ham_opt, ham_sc, space, mode):
        """Compile sparse matrix-vector product H·ψ."""
        if mode == ComputeMode.EFFECTIVE:
            return engine.operator.create_spmv_eff(
                ham_eff_rows=ham_opt.rows,
                ham_eff_cols=ham_opt.cols,
                ham_eff_vals=ham_opt.vals,
                n_s=space.n_s,
                precision_config=self.cfg.precision
            )
        else:
            return engine.operator.create_spmv_proxy(
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
    
    def _log_workspace_info(self, space, ham_opt, ham_sc, mode):
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
