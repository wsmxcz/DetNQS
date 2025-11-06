# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Closure factory for batch log(ψ) and SpMV operations.

Creates pure JAX callables with captured constants for efficient JIT compilation.
Supports memory-efficient chunked inference for large determinant spaces.

File: lever/engine/operator.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from ..dtypes import Contractions, LogPsiFn, SpMVFn
from ..config import ComputeMode, PrecisionConfig
from . import kernels

if TYPE_CHECKING:
    from ..dtypes import PyTree


# ============================================================================
# Chunked Inference Utilities
# ============================================================================

def _pad_to_fixed_chunks(
    x: jnp.ndarray,
    chunk_size: int
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Pad array to multiple of chunk_size with static shape.
    
    Args:
        x: Input features (n_samples, feat_dim)
        chunk_size: Fixed chunk size for batching
    
    Returns:
        batched: (n_chunks, chunk_size, feat_dim) - padded and reshaped
        mask: (n_chunks, chunk_size) - 1 for real samples, 0 for padding
        n_real: Original sample count before padding
    """
    n_real = x.shape[0]
    remainder = n_real % chunk_size
    n_pad = (chunk_size - remainder) if remainder else 0
    
    if n_pad > 0:
        x = jnp.pad(x, ((0, n_pad), (0, 0)))
    
    n_chunks = x.shape[0] // chunk_size
    x_batched = x.reshape(n_chunks, chunk_size, x.shape[1])
    
    # Build mask: 1 for valid samples, 0 for padding
    mask = jnp.arange(n_chunks * chunk_size) < n_real
    mask = mask.reshape(n_chunks, chunk_size)
    
    return x_batched, mask, n_real


# ============================================================================
# Log(ψ) Closure Factory
# ============================================================================

def create_logpsi_evals(
    model_fn: Callable,
    feat_s: jnp.ndarray,
    feat_c: jnp.ndarray,
    mode: ComputeMode,
    normalize: bool,
    device_complex: jnp.dtype,
    *,
    chunk_size: int | None = None,
) -> Union[Tuple[Callable, Callable], Callable]:
    """
    Create log(ψ) evaluators with features bound as closure captures.
    
    Returns closures that only require params, avoiding JIT constant folding
    of large feature arrays.
    
    Args:
        model_fn: Neural network forward pass (batch-aware)
        feat_s: S-space features (n_s, feat_dim)
        feat_c: C-space features (n_c, feat_dim)
        mode: Compute mode (EFFECTIVE/ASYMMETRIC/PROXY)
        normalize: Apply log-domain normalization
        device_complex: Output dtype (jnp.complex64/128)
        chunk_size: If set, process features in fixed-size batches
    
    Returns:
        For EFFECTIVE/ASYMMETRIC: (eval_s, eval_full) tuple
        For PROXY: eval_full function
        All returned functions have signature: params → log_psi
    """
    
    def _batch_apply(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """Vectorized model application: (batch, feat_dim) → (batch,)"""
        outputs = model_fn(params, features)
        return jnp.asarray(outputs, dtype=device_complex)
    
    def _chunked_apply(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """Memory-efficient chunked inference."""
        n_samples = features.shape[0]
        
        # Fast path: small enough for single batch
        if chunk_size is None or n_samples <= chunk_size:
            return _batch_apply(params, features)
        
        # Prepare fixed-size chunks with padding mask
        feat_batched, mask_batched, n_real = _pad_to_fixed_chunks(
            features, chunk_size
        )
        
        # Generate outputs chunk-by-chunk
        def process_chunk(carry, inputs):
            feat_chunk, _ = inputs
            log_chunk = _batch_apply(params, feat_chunk)
            return carry, log_chunk
        
        _, outputs_batched = lax.scan(
            process_chunk,
            None,
            (feat_batched, mask_batched)
        )
        
        return outputs_batched.reshape(-1)[:n_real]
    
    def _norm_s(log_s: jnp.ndarray) -> jnp.ndarray:
        """S-space normalization: log(ψ) - 0.5 * log(‖ψ_S‖²)"""
        if not normalize:
            return log_s
        log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_s))
        log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
        return log_s - 0.5 * log_norm_sq
    
    def _norm_full(log_all: jnp.ndarray, n_s: int) -> jnp.ndarray:
        """Full T-space normalization: log(ψ) - 0.5 * log(‖ψ_S‖² + ‖ψ_C‖²)"""
        if not normalize:
            return log_all
        log_s = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s]))
        log_c = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[n_s:]))
        log_norm_sq = jnp.logaddexp(log_s, log_c)
        log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
        return log_all - 0.5 * log_norm_sq
    
    # Build internal evaluators (these still take features)
    def _eval_s_internal(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """Evaluate S-space with optional chunking and S-normalization."""
        log_s = _chunked_apply(params, features)
        log_s = jnp.asarray(log_s, dtype=device_complex)
        return _norm_s(log_s)
    
    def _eval_full_internal(params: PyTree, fs: jnp.ndarray, fc: jnp.ndarray) -> jnp.ndarray:
        """Evaluate full T-space with optional chunking and full normalization."""
        all_feat = jnp.concatenate([fs, fc], axis=0)
        log_all = _chunked_apply(params, all_feat)
        log_all = jnp.asarray(log_all, dtype=device_complex)
        return _norm_full(log_all, fs.shape[0])
    
    # Create closures with captured features (avoids JIT constant folding)
    n_s = feat_s.shape[0]
    
    def eval_s_closure(params: PyTree) -> jnp.ndarray:
        """S-space evaluator with captured features."""
        return _eval_s_internal(params, feat_s)
    
    def eval_full_closure(params: PyTree) -> jnp.ndarray:
        """Full T-space evaluator with captured features."""
        return _eval_full_internal(params, feat_s, feat_c)
    
    # Return closures based on mode
    if mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC):
        return (eval_s_closure, eval_full_closure)
    return eval_full_closure

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
    
    Precision flow: device → host complex128 → device to minimize rounding errors.
    
    Args:
        ham_eff_rows/cols/vals: COO format H_eff
        n_s: S-space dimension
        precision_config: Device dtype configuration
    
    Returns:
        Pure function: ψ_S → Contractions(n_ss=H_eff@ψ_S)
    """
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
    
    Single callback computes three contractions with shared data transfers.
    
    Args:
        ham_ss_*/ham_sc_*: COO format blocks
        h_diag_c: H_CC diagonal elements
        n_s, n_c: Space dimensions
        precision_config: Device dtype configuration
    
    Returns:
        Pure function: (ψ_S, ψ_C) → Contractions(n_ss, n_sc, n_cs, n_cc)
    """
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
    
    def _triple_contract_host(xs: np.ndarray, xc: np.ndarray, ns: int, nc: int) -> tuple:
        """Host callback: compute three off-diagonal contractions."""
        xs_c128 = np.asarray(xs, dtype=np.complex128)
        xc_c128 = np.asarray(xc, dtype=np.complex128)
        
        y_ss = kernels.coo_matvec(rows_ss, cols_ss, vals_ss, xs_c128, int(ns))
        y_sc, y_cs = kernels.coo_dual_contract(
            rows_sc, cols_sc, vals_sc, xs_c128, xc_c128, int(ns), int(nc)
        )
        
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


__all__ = ["create_logpsi_evals", "create_spmv_eff", "create_spmv_proxy"]
