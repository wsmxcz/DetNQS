# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compilation pipeline: OuterState → OuterCtx with deterministic spaces.

Constructs numerical context from current outer-loop state, including:
  - Hamiltonian blocks (H_SS, H_SC) with optional screening
  - Feature matrices for S/C spaces
  - JIT-compiled operators (log(ψ) evaluator, SpMV)

File: lever/engine/context.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .. import core
from ..config import ComputeMode, ExperimentConfig, ScreenMode
from ..dtypes import COOMatrix, OuterCtx, OuterState, SpaceRep
from . import hamiltonian, operator
from .features import dets_to_features

if TYPE_CHECKING:
    from typing import Any


# ============================================================================
# Hamiltonian Construction with Screening
# ============================================================================

def _build_hamiltonian(
    cfg: ExperimentConfig,
    state: OuterState,
    int_ctx: core.IntCtx,
    psi_s_est: np.ndarray | None,
) -> tuple[COOMatrix, COOMatrix, SpaceRep]:
    """
    Construct Hamiltonian blocks with mode-specific screening.
    
    Screening modes:
      - NONE:    Full T-space without truncation
      - STATIC:  ε-based threshold on |H_SC|
      - DYNAMIC: Amplitude-weighted threshold on |H_SC·ψ_C|
    
    Args:
        cfg: Experiment configuration
        state: Current outer-loop state with S-space determinants
        int_ctx: Integral context for matrix elements
        psi_s_est: Optional S-space amplitudes for dynamic screening
    
    Returns:
        (ham_opt, ham_sc, space)
        where ham_opt = H_eff (effective) or H_SS (proxy)
    """
    ham_cfg = cfg.hamiltonian
    kwargs = {
        "S_dets": state.s_dets,
        "int_ctx": int_ctx,
        "n_orb": cfg.system.n_orb,
    }
    
    # Select screening strategy
    if ham_cfg.screening_mode == ScreenMode.NONE:
        ham_ss, ham_sc, space = hamiltonian.get_ham_proxy(**kwargs, mode="none")
    
    elif ham_cfg.screening_mode == ScreenMode.STATIC:
        ham_ss, ham_sc, space = hamiltonian.get_ham_proxy(
            **kwargs,
            mode="static",
            screen_eps=ham_cfg.screen_eps,
            diag_shift=ham_cfg.diag_shift,
        )
    
    elif ham_cfg.screening_mode == ScreenMode.DYNAMIC:
        if psi_s_est is None:
            raise ValueError("Dynamic screening requires S-space amplitudes")
        ham_ss, ham_sc, space = hamiltonian.get_ham_proxy(
            **kwargs,
            mode="dynamic",
            psi_S=psi_s_est,
            screen_eps=ham_cfg.screen_eps,
            diag_shift=ham_cfg.diag_shift,
        )
    
    else:
        raise ValueError(f"Unknown screening mode: {ham_cfg.screening_mode}")
    
    # Optionally fold C-space via Löwdin downfolding: H_eff = H_SS - H_SC·(E-H_CC)^-1·H_CS
    if cfg.runtime.compute_mode == ComputeMode.EFFECTIVE and state.e_ref is not None:
        ham_eff = hamiltonian.get_ham_eff(
            ham_ss=ham_ss,
            ham_sc=ham_sc,
            h_cc_diag=space.h_diag_c,
            e_ref=state.e_ref,
            reg_type="sigma",
            num_eps=ham_cfg.reg_eps,
        )
        return ham_eff, ham_sc, space
    
    return ham_ss, ham_sc, space


# ============================================================================
# Dynamic Screening Bootstrap
# ============================================================================

def _maybe_bootstrap_dynamic_amplitudes(
    cfg: ExperimentConfig,
    state: OuterState,
    model: Any,
    feat_s: jnp.ndarray,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """
    Bootstrap S-space amplitudes for dynamic screening.
    
    Evaluates |ψ_S| = |exp(log(ψ))| using current model parameters,
    returns L2-normalized amplitudes on host.
    
    Args:
        cfg: Experiment configuration
        state: Current outer-loop state with model parameters
        model: Neural network with log_psi evaluator
        feat_s: S-space feature matrix [n_s, d_feat]
    
    Returns:
        (psi_s, (min, max)) if dynamic screening enabled, else (None, None)
    """
    if cfg.hamiltonian.screening_mode != ScreenMode.DYNAMIC:
        return None, None
    
    # Extract S-space evaluator (handle tuple wrapper for spin-flip symmetry)
    logpsi_fn = (
        model.log_psi[0] if isinstance(model.log_psi, tuple) else model.log_psi
    )
    
    log_s = logpsi_fn(state.params, feat_s)
    psi = jnp.abs(jnp.exp(log_s))
    
    # Normalize with numerical stability
    norm = jnp.linalg.norm(psi)
    norm_safe = jnp.where(norm > 1e-14, norm, 1.0)
    psi_norm = psi / norm_safe
    
    psi_s = np.array(jax.device_get(psi_norm))
    return psi_s, (float(psi_s.min()), float(psi_s.max()))


# ============================================================================
# Main Compilation Pipeline
# ============================================================================

def build_context(
    cfg: ExperimentConfig,
    state: OuterState,
    model: Any,
    int_ctx: core.IntCtx,
) -> tuple[OuterCtx, dict[str, Any]]:
    """
    Compile pure numerical context from current outer-loop state.
    
    Pipeline stages:
      1. S-space feature extraction
      2. Optional amplitude bootstrap (dynamic screening)
      3. Hamiltonian construction with screening
      4. C-space feature extraction
      5. Operator compilation (log(ψ) + SpMV)
    
    Args:
        cfg: Experiment configuration
        state: Current outer-loop state (determinants, parameters, E_ref)
        model: Neural network with log_psi evaluator
        int_ctx: Integral context for matrix elements
    
    Returns:
        (ctx, diagnostics)
        ctx: Compiled numerical context for inner-loop optimization
        diagnostics: Compilation statistics (space sizes, sparsity, etc.)
    """
    mode = cfg.runtime.compute_mode
    n_orb = cfg.system.n_orb
    
    # Stage 1: S-space features
    feat_s_state = dets_to_features(state.s_dets, n_orb)
    
    # Stage 2: Optional amplitude bootstrap
    psi_s_est, bootstrap_range = _maybe_bootstrap_dynamic_amplitudes(
        cfg, state, model, feat_s_state
    )
    
    # Stage 3: Hamiltonian construction
    ham_ss, ham_sc, space = _build_hamiltonian(cfg, state, int_ctx, psi_s_est)
    
    # Stage 4: C-space features (S-space features reused from Stage 1)
    feat_s = feat_s_state
    feat_c = dets_to_features(space.c_dets, n_orb)
    
    # Stage 5: Operator compilation
    log_psi_eval = operator.create_logpsi_evals(
        model_fn=model.log_psi,
        mode=mode,
        normalize=cfg.runtime.normalize_wf,
        device_complex=cfg.runtime.jax_complex,
        chunk_size=cfg.loop.chunk_size,
        spin_flip_symmetry=cfg.runtime.spin_flip_symmetry,
        feat_s=feat_s,
        feat_c=feat_c,
    )
    
    if mode == ComputeMode.EFFECTIVE:
        spmv_fn = operator.create_spmv_eff(
            ham_eff_rows=ham_ss.rows,
            ham_eff_cols=ham_ss.cols,
            ham_eff_vals=ham_ss.vals,
            n_s=space.n_s,
            precision_config=cfg.runtime,
        )
    else:
        spmv_fn = operator.create_spmv_proxy(
            ham_ss_rows=ham_ss.rows,
            ham_ss_cols=ham_ss.cols,
            ham_ss_vals=ham_ss.vals,
            ham_sc_rows=ham_sc.rows,
            ham_sc_cols=ham_sc.cols,
            ham_sc_vals=ham_sc.vals,
            h_diag_c=space.h_diag_c,
            n_s=space.n_s,
            n_c=space.n_c,
            precision_config=cfg.runtime,
        )
    
    # Assemble context
    ctx = OuterCtx(
        space=space,
        ham_ss=ham_ss,
        ham_sc=ham_sc,
        e_nuc=int_ctx.get_e_nuc(),
        features_s=feat_s,
        features_c=feat_c,
        log_psi=log_psi_eval,
        spmv_fn=spmv_fn,
        compute_mode=mode,
    )
    
    diagnostics = {
        "bootstrap_range": bootstrap_range,
        "n_s": space.n_s,
        "n_c": space.n_c,
        "nnz_ss": ham_ss.nnz,
        "nnz_sc": ham_sc.nnz if mode != ComputeMode.EFFECTIVE else None,
        "ham_label": "H_eff" if mode == ComputeMode.EFFECTIVE else "H_SS",
    }
    
    return ctx, diagnostics


__all__ = ["build_context"]
