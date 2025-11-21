# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Closure factory for batch log(ψ) and SpMV operations.

Creates pure JAX callables for efficient JIT compilation. Features are
passed as runtime parameters to avoid constant folding in XLA.

File: lever/engine/operator.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

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

def _pad_to_chunks(
    x: jnp.ndarray,
    chunk_size: int
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Pad array to multiple of chunk_size for static batching.
  
    Args:
        x: Input features [n_samples, feat_dim]
        chunk_size: Fixed chunk size
  
    Returns:
        batched: [n_chunks, chunk_size, feat_dim]
        mask: [n_chunks, chunk_size] validity mask
        n_real: Original sample count
    """
    n_real = x.shape[0]
    remainder = n_real % chunk_size
    n_pad = (chunk_size - remainder) if remainder else 0
  
    if n_pad > 0:
        x = jnp.pad(x, ((0, n_pad), (0, 0)))
  
    n_chunks = x.shape[0] // chunk_size
    x_batched = x.reshape(n_chunks, chunk_size, x.shape[1])
  
    mask = jnp.arange(n_chunks * chunk_size) < n_real
    mask = mask.reshape(n_chunks, chunk_size)
  
    return x_batched, mask, n_real


# ============================================================================
# Log(ψ) Closure Factory
# ============================================================================

def create_logpsi_evals(
    model_fn: Callable,
    mode: ComputeMode,
    normalize: bool,
    device_complex: jnp.dtype,
    *,
    chunk_size: int | None = None,
) -> tuple[Callable, Callable] | Callable:
    """
    Create log(ψ) evaluators without feature binding.
  
    Normalization schemes:
      - S-space: log(ψ) - 0.5·log(Σᵢ∈S |ψᵢ|²)
      - T-space: log(ψ) - 0.5·log(Σᵢ∈S |ψᵢ|² + Σⱼ∈C |ψⱼ|²)
  
    Args:
        model_fn: Neural network (params, features) → log_psi
        mode: EFFECTIVE/ASYMMETRIC/PROXY
        normalize: Apply log-domain normalization
        device_complex: Output dtype (jnp.complex64/128)
        chunk_size: Optional fixed-size batch processing
  
    Returns:
        EFFECTIVE/ASYMMETRIC: (eval_s, eval_full) tuple
        PROXY: eval_full function
        Signature: (params, features...) → log_psi
    """
  
    def _batch_apply(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized model: [batch, feat_dim] → [batch].

        Wrapped with jax.checkpoint so that intermediate activations
        inside the network are rematerialized during the backward pass
        instead of all being kept in memory at once.
        """
        outputs = model_fn(params, features)
        return jnp.asarray(outputs, dtype=device_complex)

    # Recompute _batch_apply during backward to reduce peak memory usage.
    _batch_apply_checked = jax.checkpoint(_batch_apply)
  
    def _chunked_apply(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """Memory-efficient chunked inference."""
        n_samples = features.shape[0]
      
        if chunk_size is None or n_samples <= chunk_size:
            # For small batches, just use the checkpointed single call.
            return _batch_apply_checked(params, features)
      
        feat_batched, mask_batched, n_real = _pad_to_chunks(features, chunk_size)
      
        def process_chunk(carry, inputs):
            feat_chunk, _ = inputs
            # Each chunk runs through the checkpointed apply function.
            log_chunk = _batch_apply_checked(params, feat_chunk)
            return carry, log_chunk
      
        _, outputs_batched = lax.scan(
            process_chunk, None, (feat_batched, mask_batched)
        )
      
        return outputs_batched.reshape(-1)[:n_real]
  
    def _norm_s(log_s: jnp.ndarray) -> jnp.ndarray:
        """S-space normalization: log(ψ) - 0.5·log(‖ψ_S‖²)"""
        if not normalize:
            return log_s
        log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_s))
        log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
        return log_s - 0.5 * log_norm_sq
  
    def _norm_full(log_all: jnp.ndarray, n_s: int) -> jnp.ndarray:
        """T-space normalization: log(ψ) - 0.5·log(‖ψ_S‖² + ‖ψ_C‖²)"""
        if not normalize:
            return log_all
        log_s = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s]))
        log_c = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[n_s:]))
        log_norm_sq = jnp.logaddexp(log_s, log_c)
        log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
        return log_all - 0.5 * log_norm_sq
  
    def eval_s(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """S-space evaluator with runtime feature binding."""
        log_s = _chunked_apply(params, features)
        log_s = jnp.asarray(log_s, dtype=device_complex)
        return _norm_s(log_s)
  
    def eval_full(
        params: PyTree,
        feat_s: jnp.ndarray,
        feat_c: jnp.ndarray
    ) -> jnp.ndarray:
        """T-space evaluator with runtime feature binding."""
        all_feat = jnp.concatenate([feat_s, feat_c], axis=0)
        log_all = _chunked_apply(params, all_feat)
        log_all = jnp.asarray(log_all, dtype=device_complex)
        return _norm_full(log_all, feat_s.shape[0])
  
    if mode in (ComputeMode.EFFECTIVE, ComputeMode.ASYMMETRIC):
        return (eval_s, eval_full)
    return eval_full


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
  
    Precision flow: device → host complex128 → device.
    Uses pure_callback to isolate COO data from JIT constant folding.
  
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
        """Host SpMV with explicit dtype control."""
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
  
        ┌─────┬─────┐   ┌───┐   ┌────┐   ┌────┐
        │ H_SS│ H_SC│ @ │ψ_S│ = │N_SS│ + │N_SC│
        ├─────┼─────┤   ├───┤   ├────┤   ├────┤
        │H_SC^T│H_CC│   │ψ_C│   │N_CS│   │N_CC│
        └─────┴─────┘   └───┘   └────┘   └────┘
  
    Single callback computes three off-diagonal contractions.
    Exploits H_CS = H_SC^T symmetry. H_CC is diagonal.
  
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
  
    def _triple_contract_host(
        xs: np.ndarray,
        xc: np.ndarray,
        ns: int,
        nc: int
    ) -> tuple:
        """
        Host callback: compute three off-diagonal contractions.
      
        Returns: (H_SS@ψ_S, H_SC@ψ_C, H_CS@ψ_S)
        """
        xs_c128 = np.asarray(xs, dtype=np.complex128)
        xc_c128 = np.asarray(xc, dtype=np.complex128)
      
        y_ss = kernels.coo_matvec(rows_ss, cols_ss, vals_ss, xs_c128, int(ns))
      
        # Exploit H_CS = H_SC^T symmetry
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
        n_cc = diag_c * psi_c
        return Contractions(n_ss=n_ss, n_sc=n_sc, n_cs=n_cs, n_cc=n_cc)
  
    return _spmv_proxy


__all__ = ["create_logpsi_evals", "create_spmv_eff", "create_spmv_proxy"]