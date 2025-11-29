# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Closure factory for batch log(ψ) and SpMV operations.

Creates pure JAX callables with captured constants (COO matrices, shapes)
for efficient JIT compilation and scan loops. Supports spin-flip symmetry
reduction and chunked inference for memory efficiency.

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

from ..dtypes import Contractions, LogPsiEval, SpMVFn, PyTree
from ..config import ComputeMode, RuntimeConfig
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
        x: Input features [n_samples, d_feat]
        chunk_size: Fixed chunk size for batching
    
    Returns:
        batched: Reshaped to [n_chunks, chunk_size, d_feat]
        mask: Validity mask [n_chunks, chunk_size]
        n_real: Original sample count before padding
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
    spin_flip_symmetry: bool = False,
    feat_s: jnp.ndarray | None = None,
    feat_c: jnp.ndarray | None = None,
) -> LogPsiEval:
    """
    Create log(ψ) evaluators with optional spin-flip symmetry reduction.
    
    Spin-flip symmetry exploits Z₂ invariance: ψ(α,β) = ψ(β,α), reducing
    NN evaluations by ~50% via orbit representatives.
    
    Normalization schemes:
      - S-space:  log(ψ) → log(ψ/‖ψ‖) where ‖ψ‖² = Σᵢ∈S |ψᵢ|²
      - T-space:  ‖ψ‖² = Σᵢ∈S |ψᵢ|² + Σⱼ∈C |ψⱼ|²
    
    Args:
        model_fn: Neural network (params, features) → log_psi
        mode: ComputeMode.EFFECTIVE / ASYMMETRIC / PROXY
        normalize: Apply log-domain normalization
        device_complex: Output dtype (jnp.complex64/128)
        chunk_size: Optional fixed-size batch processing
        spin_flip_symmetry: Enable Z₂ spin-flip reduction
        feat_s: S-space features (required when symmetry=True)
        feat_c: C-space features (required when symmetry=True)
    
    Returns:
        LogPsiEval(mode, eval_s, eval_full)
    """

    def _batch_apply(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """Vectorized model application."""
        outputs = model_fn(params, features)
        return jnp.asarray(outputs, dtype=device_complex)

    def _chunked_apply(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
        """
        Memory-efficient chunked inference via lax.scan.
        
        Falls back to direct batch apply if chunk_size is None or
        n_samples ≤ chunk_size.
        """
        n_samples = features.shape[0]
        if chunk_size is None or n_samples <= chunk_size:
            return _batch_apply(params, features)

        feat_batched, mask_batched, n_real = _pad_to_chunks(features, chunk_size)

        def process_chunk(carry, inputs):
            feat_chunk, _ = inputs
            log_chunk = _batch_apply(params, feat_chunk)
            return carry, log_chunk

        _, outputs_batched = lax.scan(
            process_chunk, None, (feat_batched, mask_batched)
        )
        return outputs_batched.reshape(-1)[:n_real]

    def _norm_s(log_s: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize S-space: log(ψ) → log(ψ/‖ψ‖).
        
        Computes log(‖ψ‖²) = logsumexp(2·Re[log(ψ)]) in log-domain
        for numerical stability.
        """
        if not normalize:
            return log_s
        log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_s))
        return log_s - 0.5 * jax.lax.stop_gradient(log_norm_sq)

    def _norm_full(log_all: jnp.ndarray, n_s_count: int) -> jnp.ndarray:
        """
        Normalize full T-space: combines S and C contributions.
        
        log(‖ψ‖²) = logaddexp(log(‖ψ_S‖²), log(‖ψ_C‖²))
        """
        if not normalize:
            return log_all
        log_s = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[:n_s_count]))
        log_c = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all[n_s_count:]))
        log_norm_sq = jnp.logaddexp(log_s, log_c)
        return log_all - 0.5 * jax.lax.stop_gradient(log_norm_sq)

    use_sym = spin_flip_symmetry
    sym_data = None

    if use_sym:
        if feat_s is None or feat_c is None:
            raise ValueError(
                "feat_s and feat_c required when spin_flip_symmetry=True"
            )
        sym_data = _prepare_spinflip_maps(feat_s, feat_c)

    if use_sym:
        assert sym_data is not None
        feat_rep, idx_map_s, idx_map_c = sym_data
        n_s_fixed = idx_map_s.shape[0]

        def eval_s(params: PyTree, _features_s: jnp.ndarray) -> jnp.ndarray:
            """Evaluate S-space log(ψ) using symmetry reduction."""
            log_rep = _chunked_apply(params, feat_rep)
            log_rep = jnp.asarray(log_rep, dtype=device_complex)
            log_s = log_rep[idx_map_s]
            return _norm_s(log_s)

        def eval_full(
            params: PyTree,
            _s: jnp.ndarray,
            _c: jnp.ndarray
        ) -> jnp.ndarray:
            """Evaluate full T-space log(ψ) using symmetry reduction."""
            log_rep = _chunked_apply(params, feat_rep)
            log_rep = jnp.asarray(log_rep, dtype=device_complex)
            log_s = log_rep[idx_map_s]
            log_c = log_rep[idx_map_c]
            log_all = jnp.concatenate([log_s, log_c], axis=0)
            return _norm_full(log_all, n_s_fixed)

    else:
        def eval_s(params: PyTree, features: jnp.ndarray) -> jnp.ndarray:
            """Evaluate S-space log(ψ) with runtime features."""
            log_s = _chunked_apply(params, features)
            log_s = jnp.asarray(log_s, dtype=device_complex)
            return _norm_s(log_s)

        def eval_full(
            params: PyTree,
            feat_s: jnp.ndarray,
            feat_c: jnp.ndarray
        ) -> jnp.ndarray:
            """Evaluate full T-space log(ψ) with runtime features."""
            all_feat = jnp.concatenate([feat_s, feat_c], axis=0)
            log_all = _chunked_apply(params, all_feat)
            log_all = jnp.asarray(log_all, dtype=device_complex)
            return _norm_full(log_all, feat_s.shape[0])

    return LogPsiEval(mode=mode, eval_s=eval_s, eval_full=eval_full)


def _prepare_spinflip_maps(
    feat_s_jax: jnp.ndarray,
    feat_c_jax: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Construct unique Z₂ orbit representatives for spin-flip symmetry.
    
    Maps (α,β) and (β,α) to same representative using canonical ordering:
    key = (min(α,β), max(α,β)). Features are converted to uint64 bitmasks
    for exact comparison.
    
    Args:
        feat_s_jax: S-space features [n_s, 2·n_orb]
        feat_c_jax: C-space features [n_c, 2·n_orb]
    
    Returns:
        feat_rep: Unique representatives [n_unique, 2·n_orb]
        idx_map_s: S-space → representative indices [n_s]
        idx_map_c: C-space → representative indices [n_c]
    """
    fs = np.asarray(jax.device_get(feat_s_jax))
    fc = np.asarray(jax.device_get(feat_c_jax))

    n_s, feat_dim = fs.shape
    n_c = fc.shape[0]
    n_total = n_s + n_c
    half = feat_dim // 2

    orb_idx = np.arange(half, dtype=np.uint64)
    weights = (np.uint64(1) << orb_idx)

    def _to_bits(arr):
        """Convert float features to bitmasks for exact comparison."""
        alpha = arr[:, :half].astype(np.uint64, copy=False)
        beta = arr[:, half:].astype(np.uint64, copy=False)
        a_bits = (alpha * weights).sum(axis=1)
        b_bits = (beta * weights).sum(axis=1)
        return a_bits, b_bits

    as_bits, bs_bits = _to_bits(fs)
    ac_bits, bc_bits = _to_bits(fc)

    alpha_all = np.concatenate([as_bits, ac_bits])
    beta_all = np.concatenate([bs_bits, bc_bits])

    # Canonical form: (min, max) maps both (a,b) and (b,a) to same key
    k1 = np.minimum(alpha_all, beta_all)
    k2 = np.maximum(alpha_all, beta_all)

    keys = np.empty(n_total, dtype=[('k1', np.uint64), ('k2', np.uint64)])
    keys['k1'] = k1
    keys['k2'] = k2

    _, idx_unique, indices = np.unique(
        keys, return_index=True, return_inverse=True
    )

    all_feats = np.concatenate([fs, fc], axis=0)
    feat_rep = all_feats[idx_unique]

    map_s = indices[:n_s]
    map_c = indices[n_s:]

    return (
        jax.device_put(feat_rep),
        jax.device_put(map_s.astype(np.int32)),
        jax.device_put(map_c.astype(np.int32)),
    )


# ============================================================================
# SpMV Closure Factory
# ============================================================================

def create_spmv_eff(
    ham_eff_rows: np.ndarray,
    ham_eff_cols: np.ndarray,
    ham_eff_vals: np.ndarray,
    n_s: int,
    precision_config: RuntimeConfig,
) -> SpMVFn:
    """
    Create effective Hamiltonian SpMV: y = H_eff @ ψ_S.
    
    Precision flow: device → host complex128 → device to minimize rounding
    errors in pure_callback boundary crossings. COO data stays on host.
    
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
    precision_config: RuntimeConfig,
) -> SpMVFn:
    """
    Create full T-space SpMV with block structure:
    
        ┌─────────┬─────────┐   ┌───┐   ┌────────┐   ┌────────┐
        │  H_SS   │  H_SC   │ @ │ψ_S│ = │H_SS@ψ_S│ + │H_SC@ψ_C│
        ├─────────┼─────────┤   ├───┤   ├────────┤   ├────────┤
        │ H_SC^T  │  H_CC   │   │ψ_C│   │H_CS@ψ_S│   │H_CC@ψ_C│
        └─────────┴─────────┘   └───┘   └────────┘   └────────┘
    
    Single callback computes all four contractions. Exploits H_CS = H_SC^T
    symmetry. H_CC is diagonal and computed on host.
    
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

    diag_c = np.ascontiguousarray(h_diag_c, dtype=np.float64)

    device_complex = precision_config.jax_complex
    target_np_dtype = precision_config.numpy_complex

    shape_out = (
        jax.ShapeDtypeStruct((n_s,), device_complex),
        jax.ShapeDtypeStruct((n_s,), device_complex),
        jax.ShapeDtypeStruct((n_c,), device_complex),
        jax.ShapeDtypeStruct((n_c,), device_complex),
    )

    def _triple_contract_host(
        xs: np.ndarray,
        xc: np.ndarray,
        ns: int,
        nc: int,
    ) -> tuple:
        """
        Host callback: compute all four contractions.
        
        Returns: (H_SS@ψ_S, H_SC@ψ_C, H_CS@ψ_S, H_CC@ψ_C)
        """
        xs_c128 = np.asarray(xs, dtype=np.complex128)
        xc_c128 = np.asarray(xc, dtype=np.complex128)

        # H_SS @ ψ_S
        y_ss = kernels.coo_matvec(rows_ss, cols_ss, vals_ss, xs_c128, int(ns))
        
        # H_SC @ ψ_C and H_CS @ ψ_S (exploits H_CS = H_SC^T)
        y_sc, y_cs = kernels.coo_dual_contract(
            rows_sc, cols_sc, vals_sc, xs_c128, xc_c128, int(ns), int(nc)
        )

        # H_CC @ ψ_C (diagonal)
        y_cc = diag_c.astype(np.complex128, copy=False) * xc_c128

        return (
            y_ss.astype(target_np_dtype, copy=False),
            y_sc.astype(target_np_dtype, copy=False),
            y_cs.astype(target_np_dtype, copy=False),
            y_cc.astype(target_np_dtype, copy=False),
        )

    def _spmv_proxy(psi_s: jnp.ndarray, psi_c: jnp.ndarray) -> Contractions:
        n_ss, n_sc, n_cs, n_cc = jax.pure_callback(
            _triple_contract_host,
            shape_out,
            psi_s,
            psi_c,
            n_s,
            n_c,
        )
        return Contractions(n_ss=n_ss, n_sc=n_sc, n_cs=n_cs, n_cc=n_cc)

    return _spmv_proxy


__all__ = ["create_logpsi_evals", "create_spmv_eff", "create_spmv_proxy"]
