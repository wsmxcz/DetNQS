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

from .utils import Contractions, LogPsiFn, SpMVFn
from . import kernels

if TYPE_CHECKING:
    from .utils import PyTree


# ============================================================================
# Log(ψ) Closure Factory
# ============================================================================

def create_logpsi_fn(
    model_fn: Callable,
    feat_s: jnp.ndarray,
    feat_c: jnp.ndarray,
    mode: str,
    normalize: bool,
    eps: float
) -> LogPsiFn:
    """
    Create batch log(ψ) evaluator with captured features.
    
    Applies normalization constraint:
      - Effective: ||ψ_S||² = 1
      - Proxy: ||ψ_S||² + ||ψ_C||² = 1
    
    Args:
        model_fn: Neural network mapping (params, features) -> log(ψ)
        feat_s: S-space occupancy vectors [N_S × n_orb]
        feat_c: C-space occupancy vectors [N_C × n_orb]
        mode: "effective" or "proxy"
        normalize: Apply norm constraint
        eps: Stability threshold (kept for API compatibility)
    
    Returns:
        Closure: params -> normalized log(ψ) [N_S+N_C]
    """
    all_feat = jnp.concatenate([feat_s, feat_c], axis=0)
    n_s = feat_s.shape[0]
    is_effective = (mode == "effective")
    
    def _batch_logpsi(params: PyTree) -> jnp.ndarray:
        """Evaluate log(ψ) with optional normalization."""
        log_all = model_fn(params, all_feat)
        
        if not normalize:
            return log_all
        
        # Compute log(||ψ||²) for normalization
        if is_effective:
            # S-space only: log(Σ|ψ_S|²)
            log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s]))
        else:
            # Full space: log(Σ|ψ_S|² + Σ|ψ_C|²)
            log_s_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s]))
            log_c_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[n_s:]))
            log_norm_sq = jnp.logaddexp(log_s_norm_sq, log_c_norm_sq)
        
        # Apply normalization: log(ψ/||ψ||) = log(ψ) - 0.5·log(||ψ||²)
        log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
        return log_all - 0.5 * log_norm_sq
    
    return _batch_logpsi


# ============================================================================
# SpMV Closure Factory
# ============================================================================

def _coo_matvec_cpu(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    psi: np.ndarray,
    n_out: int
) -> np.ndarray:
    """
    CPU SpMV via Numba kernel: y = A @ x (COO format).
    
    Ensures contiguous memory layout for optimal performance.
    """
    psi_c128 = np.ascontiguousarray(psi, dtype=np.complex128)
    rows_i64 = np.ascontiguousarray(rows, dtype=np.int64)
    cols_i64 = np.ascontiguousarray(cols, dtype=np.int64)
    vals_f64 = np.ascontiguousarray(vals, dtype=np.float64)
    
    return kernels.coo_matvec(rows_i64, cols_i64, vals_f64, psi_c128, int(n_out))


def _dual_contract_cpu(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    psi_s: np.ndarray,
    psi_c: np.ndarray,
    n_s: int,
    n_c: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    CPU dual contraction for off-diagonal blocks.
    
    Computes simultaneously:
      y_S = H_SC @ ψ_C  (forward)
      y_C = H_CS @ ψ_S = (H_SC)^T @ ψ_S  (transpose)
    """
    psi_s_c128 = np.ascontiguousarray(psi_s, dtype=np.complex128)
    psi_c_c128 = np.ascontiguousarray(psi_c, dtype=np.complex128)
    rows_i64 = np.ascontiguousarray(rows, dtype=np.int64)
    cols_i64 = np.ascontiguousarray(cols, dtype=np.int64)
    vals_f64 = np.ascontiguousarray(vals, dtype=np.float64)
    
    return kernels.coo_dual_contract(
        rows_i64, cols_i64, vals_f64,
        psi_s_c128, psi_c_c128,
        int(n_s), int(n_c)
    )


def create_spmv_eff(
    ham_eff_rows: np.ndarray,
    ham_eff_cols: np.ndarray,
    ham_eff_vals: np.ndarray,
    n_s: int
) -> SpMVFn:
    """
    Create effective Hamiltonian SpMV operator.
    
    Evaluates: N_SS = H_eff @ ψ_S
    where H_eff = H_SS + H_SC·D^(-1)·H_CS is pre-assembled.
    
    Returns:
        Closure: ψ_S -> Contractions(N_SS, None, None, None)
    """
    rows, cols, vals = ham_eff_rows, ham_eff_cols, ham_eff_vals
    shape_out = jax.ShapeDtypeStruct((n_s,), jnp.complex128)
    
    def _spmv_effective(psi_s: jnp.ndarray) -> Contractions:
        """Apply H_eff to S-space wavefunction."""
        n_ss = jax.pure_callback(
            _coo_matvec_cpu,
            shape_out,
            rows, cols, vals, psi_s, n_s
        )
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
    n_c: int
) -> SpMVFn:
    """
    Create full T-space Hamiltonian SpMV operator.
    
    Evaluates all four blocks:
      N_SS = H_SS @ ψ_S
      N_SC = H_SC @ ψ_C
      N_CS = (H_SC)^T @ ψ_S
      N_CC = diag(H_CC) @ ψ_C
    
    Returns:
        Closure: (ψ_S, ψ_C) -> Contractions(N_SS, N_SC, N_CS, N_CC)
    """
    rows_ss, cols_ss, vals_ss = ham_ss_rows, ham_ss_cols, ham_ss_vals
    rows_sc, cols_sc, vals_sc = ham_sc_rows, ham_sc_cols, ham_sc_vals
    diag_c = jnp.asarray(h_diag_c)
    
    shape_ss = jax.ShapeDtypeStruct((n_s,), jnp.complex128)
    shape_sc = jax.ShapeDtypeStruct((n_s,), jnp.complex128)
    shape_cs = jax.ShapeDtypeStruct((n_c,), jnp.complex128)
    
    def _spmv_proxy(psi_s: jnp.ndarray, psi_c: jnp.ndarray) -> Contractions:
        """Apply full T-space Hamiltonian blocks."""
        # S-space diagonal: N_SS = H_SS @ ψ_S
        n_ss = jax.pure_callback(
            _coo_matvec_cpu,
            shape_ss,
            rows_ss, cols_ss, vals_ss, psi_s, n_s
        )
        
        # Off-diagonal blocks: N_SC and N_CS (computed simultaneously)
        n_sc, n_cs = jax.pure_callback(
            _dual_contract_cpu,
            (shape_sc, shape_cs),
            rows_sc, cols_sc, vals_sc, psi_s, psi_c, n_s, n_c
        )
        
        # C-space diagonal: N_CC = diag(H_CC) @ ψ_C
        n_cc = diag_c * psi_c
        
        return Contractions(n_ss=n_ss, n_sc=n_sc, n_cs=n_cs, n_cc=n_cc)
    
    return _spmv_proxy


__all__ = ["create_logpsi_fn", "create_spmv_eff", "create_spmv_proxy"]
