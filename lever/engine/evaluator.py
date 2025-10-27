# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Closure factory for batch log(ψ) and SpMV operations.

Creates pure JAX callables with captured constants (COO matrices, shapes)
for efficient JIT compilation and scan loops.

File: lever/engine/evaluator.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..dtypes import Contractions, LogPsiFn, SpMVFn
from ..config import ComputeMode, PrecisionConfig
from . import kernels

if TYPE_CHECKING:
    from ..dtypes import PyTree


# ============================================================================
# Log(ψ) Closure Factory
# ============================================================================

def create_logpsi_fn(
    model_fn: Callable,
    feat_s: jnp.ndarray,
    feat_c: jnp.ndarray,
    mode: ComputeMode,
    normalize: bool,
    eps: float,
    device_complex: jnp.dtype
) -> LogPsiFn:
    """
    Create batch log(ψ) evaluator with mode-specific normalization.
    
    Normalization schemes:
      - Effective: ‖ψ‖² = Σᵢ∈S |ψᵢ|²
      - Proxy:     ‖ψ‖² = Σᵢ∈S |ψᵢ|² + Σⱼ∈C |ψⱼ|²
    
    Args:
        model_fn: Neural network forward pass
        feat_s: S-space feature matrix [n_s, d_feat]
        feat_c: C-space feature matrix [n_c, d_feat]
        mode: ComputeMode.EFFECTIVE or ComputeMode.PROXY
        normalize: Apply log-domain normalization
        eps: Numerical stability (unused, kept for interface compatibility)
        device_complex: Output dtype (jnp.complex64/128)
    
    Returns:
        Pure function: params → log(ψ) [n_s + n_c]
    """
    all_feat = jnp.concatenate([feat_s, feat_c], axis=0)
    n_s = feat_s.shape[0]
    is_effective = (mode is ComputeMode.EFFECTIVE)
    
    def _batch_logpsi(params: PyTree) -> jnp.ndarray:
        log_all = model_fn(params, all_feat)
        log_all = jnp.asarray(log_all, dtype=device_complex)
        
        if not normalize:
            return log_all
        
        # Mode-specific norm: log(‖ψ‖²) = log(Σ |ψᵢ|²)
        if is_effective:
            log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s]))
        else:
            log_s = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s]))
            log_c = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[n_s:]))
            log_norm_sq = jnp.logaddexp(log_s, log_c)
        
        log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
        return log_all - 0.5 * log_norm_sq
    
    return _batch_logpsi


# ============================================================================
# SpMV Closure Factory
# ============================================================================

def create_spmv_eff(
    ham_eff_rows: np.ndarray,
    ham_eff_cols: np.ndarray,
    ham_eff_vals: np.ndarray,
    n_s: int,
    precision_config: PrecisionConfig
) -> SpMVFn:
    """
    Create effective Hamiltonian SpMV: y = H_eff @ ψ_S.
    
    Precision flow: device → host complex128 → device to minimize rounding
    errors in pure_callback boundary crossings.
    
    Args:
        ham_eff_rows/cols/vals: COO format H_eff
        n_s: S-space dimension
        precision_config: Device dtype configuration
    
    Returns:
        Pure function: ψ_S → Contractions(n_ss=H_eff@ψ_S)
    """
    # Pre-convert constants to contiguous host arrays
    rows = np.ascontiguousarray(ham_eff_rows, dtype=np.int64)
    cols = np.ascontiguousarray(ham_eff_cols, dtype=np.int64)
    vals = np.ascontiguousarray(ham_eff_vals, dtype=np.float64)
    
    device_complex = precision_config.jax_complex
    target_np_dtype = precision_config.numpy_complex
    shape_out = jax.ShapeDtypeStruct((n_s,), device_complex)
    
    def _matvec_host(x: np.ndarray, n: int) -> np.ndarray:
        """Host callback: SpMV with explicit dtype control."""
        x_c128 = np.asarray(x, dtype=np.complex128)
        y_c128 = kernels.coo_matvec(rows, cols, vals, x_c128, int(n))
        return y_c128.astype(target_np_dtype, copy=False)
    
    def _spmv_effective(psi_s: jnp.ndarray) -> Contractions:
        n_ss = jax.pure_callback(_matvec_host, shape_out, psi_s, n_s)
        return Contractions(n_ss=n_ss, n_sc=None, n_cs=None, n_cc=None)
    
    return _spmv_effective


def create_spmv_proxy(
    ham_ss_rows: np.ndarray,
    ham_ss_cols: np.ndarray,
    ham_ss_vals: np.ndarray,
    ham_sc_rows: np.ndarray,
    ham_sc_cols: np.ndarray,
    ham_sc_vals: np.ndarray,
    h_diag_c: np.ndarray,
    n_s: int,
    n_c: int,
    precision_config: PrecisionConfig
) -> SpMVFn:
    """
    Create full T-space SpMV with block structure:
    
        ┌─────────┬─────────┐   ┌───┐   ┌────────┐   ┌────────┐
        │  H_SS   │  H_SC   │ @ │ψ_S│ = │H_SS@ψ_S│ + │H_SC@ψ_C│
        ├─────────┼─────────┤   ├───┤   ├────────┤   ├────────┤
        │ H_SC^T  │  H_CC   │   │ψ_C│   │H_CS@ψ_S│   │H_CC@ψ_C│
        └─────────┴─────────┘   └───┘   └────────┘   └────────┘
    
    Single callback computes all four contractions with shared data transfers.
    
    Args:
        ham_ss_*/ham_sc_*: COO format blocks
        h_diag_c: H_CC diagonal elements
        n_s, n_c: Space dimensions
        precision_config: Device dtype configuration
    
    Returns:
        Pure function: (ψ_S, ψ_C) → Contractions(n_ss, n_sc, n_cs, n_cc)
    """
    # Pre-convert constants to contiguous host arrays
    rows_ss = np.ascontiguousarray(ham_ss_rows, dtype=np.int64)
    cols_ss = np.ascontiguousarray(ham_ss_cols, dtype=np.int64)
    vals_ss = np.ascontiguousarray(ham_ss_vals, dtype=np.float64)
    
    rows_sc = np.ascontiguousarray(ham_sc_rows, dtype=np.int64)
    cols_sc = np.ascontiguousarray(ham_sc_cols, dtype=np.int64)
    vals_sc = np.ascontiguousarray(ham_sc_vals, dtype=np.float64)
    
    diag_c = jnp.asarray(h_diag_c)
    device_complex = precision_config.jax_complex
    target_np_dtype = precision_config.numpy_complex
    
    shape_out = (
        jax.ShapeDtypeStruct((n_s,), device_complex),
        jax.ShapeDtypeStruct((n_s,), device_complex),
        jax.ShapeDtypeStruct((n_c,), device_complex),
    )
    
    def _triple_contract_host(xs, xc, ns, nc):
        """
        Host callback: compute three off-diagonal contractions.
        
        Returns: (H_SS@ψ_S, H_SC@ψ_C, H_CS@ψ_S)
        """
        xs_c128 = np.asarray(xs, dtype=np.complex128)
        xc_c128 = np.asarray(xc, dtype=np.complex128)
        
        # H_SS @ ψ_S
        y_ss = kernels.coo_matvec(rows_ss, cols_ss, vals_ss, xs_c128, int(ns))
        
        # H_SC @ ψ_C and H_CS @ ψ_S (exploits H_CS = H_SC^T symmetry)
        y_sc, y_cs = kernels.coo_dual_contract(
            rows_sc, cols_sc, vals_sc, xs_c128, xc_c128, int(ns), int(nc)
        )
        
        # Downcast to device precision
        return (
            y_ss.astype(target_np_dtype, copy=False),
            y_sc.astype(target_np_dtype, copy=False),
            y_cs.astype(target_np_dtype, copy=False)
        )
    
    def _spmv_proxy(psi_s: jnp.ndarray, psi_c: jnp.ndarray) -> Contractions:
        n_ss, n_sc, n_cs = jax.pure_callback(
            _triple_contract_host, shape_out, psi_s, psi_c, n_s, n_c
        )
        n_cc = diag_c * psi_c  # H_CC is diagonal
        return Contractions(n_ss=n_ss, n_sc=n_sc, n_cs=n_cs, n_cc=n_cc)
    
    return _spmv_proxy


__all__ = ["create_logpsi_fn", "create_spmv_eff", "create_spmv_proxy"]
