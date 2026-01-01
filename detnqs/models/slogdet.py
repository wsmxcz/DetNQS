# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Stable log-determinant computation with FFI acceleration.

Provides:
  - slogdet: Standard N×N determinant with optional FFI kernel
  - slogdet_thouless: Fast k×k padded determinant for Thouless amplitudes

Performance notes:
  - Flattened single-pass gathering preferred over multi-stage indexing
  - FFI kernels support real matrices 2 <= n <= 64 with fused LU decomposition

File: detnqs/models/slogdet.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax._src.numpy.linalg import SlogdetResult as _SlogdetResult
from jax import ffi

from ..utils.det_utils import DetBatch
from .mappers import ThoulessAmplitudes


# ========================= FFI Registration =========================

@lru_cache(maxsize=None)
def _register_ffi_targets() -> None:
    """Register CUDA FFI targets for fused log-determinant kernels."""
    from .. import _detnqs_ffi

    def _reg(name: str, capsule):
        if capsule is not None:
            ffi.register_ffi_target(name, capsule, platform="CUDA")

    _reg("detnqs_fused_logdet_f64_fwd", _detnqs_ffi.fused_logdet_f64_fwd_cuda())
    _reg("detnqs_fused_logdet_f64_bwd", _detnqs_ffi.fused_logdet_f64_bwd_cuda())
    _reg("detnqs_fused_logdet_f32_fwd", _detnqs_ffi.fused_logdet_f32_fwd_cuda())
    _reg("detnqs_fused_logdet_f32_bwd", _detnqs_ffi.fused_logdet_f32_bwd_cuda())


def _layout_row_major(rank: int) -> tuple[int, ...]:
    """Return axis order for row-major layout."""
    return tuple(range(rank))


# ========================= FFI Slogdet Implementation =========================

def _ffi_slogdet_impl(A: jnp.ndarray) -> _SlogdetResult:
    """
    CUDA FFI implementation of slogdet for real matrices.
    
    Uses fused LU decomposition for matrices 2 <= n <= 64.
    
    Args:
        A: Real matrix (..., n, n) with dtype float32 or float64
        
    Returns:
        SlogdetResult(sign, logabsdet)
    """
    _register_ffi_targets()

    if A.dtype not in (jnp.float32, jnp.float64):
        raise TypeError(f"Unsupported dtype {A.dtype}")

    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")

    n = int(A.shape[-1])
    if not (2 <= n <= 64):
        raise ValueError(f"Matrix size {n} outside [2, 64] range")

    batch_shape = A.shape[:-2]
    kernel_name = (
        "detnqs_fused_logdet_f64_fwd" if A.dtype == jnp.float64
        else "detnqs_fused_logdet_f32_fwd"
    )

    sign, logabs = ffi.ffi_call(
        kernel_name,
        [
            jax.ShapeDtypeStruct(batch_shape, A.dtype),
            jax.ShapeDtypeStruct(batch_shape, A.dtype),
        ],
        vmap_method="broadcast_all",
        input_layouts=(_layout_row_major(A.ndim),),
        output_layouts=(
            _layout_row_major(len(batch_shape)),
            _layout_row_major(len(batch_shape)),
        ),
    )(A)

    return _SlogdetResult(sign, logabs)


@jax.custom_vjp
def _fused_logdet(A: jnp.ndarray) -> _SlogdetResult:
    """Wrapper for FFI slogdet with custom gradients."""
    return _ffi_slogdet_impl(A)


def _fused_logdet_fwd(A: jnp.ndarray):
    """Forward pass: compute slogdet and save input."""
    y = _ffi_slogdet_impl(A)
    return y, (A,)


def _fused_logdet_bwd(res, ct):
    """
    Backward pass via FFI kernel.
    
    Gradient of log|det(A)| is A^{-T}.
    """
    (A,) = res

    try:
        cot_sign = ct.sign
        cot_logabs = ct.logabsdet
    except AttributeError:
        cot_sign, cot_logabs = ct

    _register_ffi_targets()

    if A.dtype not in (jnp.float32, jnp.float64):
        return (jnp.zeros_like(A),)

    kernel_name = (
        "detnqs_fused_logdet_f64_bwd" if A.dtype == jnp.float64
        else "detnqs_fused_logdet_f32_bwd"
    )

    grad = ffi.ffi_call(
        kernel_name,
        jax.ShapeDtypeStruct(A.shape, A.dtype),
        vmap_method="broadcast_all",
        input_layouts=(
            _layout_row_major(A.ndim),
            _layout_row_major(A.ndim - 2),
            _layout_row_major(A.ndim - 2),
        ),
        output_layouts=_layout_row_major(A.ndim),
    )(A, cot_sign, cot_logabs)

    return (grad,)


_fused_logdet.defvjp(_fused_logdet_fwd, _fused_logdet_bwd)


# ========================= Public API =========================

def slogdet(A: jnp.ndarray, *, use_fast_kernel: bool = True) -> _SlogdetResult:
    """
    Compute sign and log absolute determinant.
    
    Automatically selects FFI kernel on CUDA for real matrices of size 2-64,
    falls back to JAX otherwise.
    
    Args:
        A: Square matrix (..., n, n)
        use_fast_kernel: Use FFI kernel if applicable
        
    Returns:
        SlogdetResult(sign, logabsdet)
    """
    if not use_fast_kernel:
        return jnp.linalg.slogdet(A)

    is_real_float = A.dtype in (jnp.float32, jnp.float64)
    n = A.shape[-1] if (A.ndim >= 2 and A.shape[-1] == A.shape[-2]) else 0
    use_ffi = is_real_float and (2 <= n <= 64)

    if not use_ffi:
        return jnp.linalg.slogdet(A)

    return jax.lax.platform_dependent(
        A,
        cpu=lambda x: jnp.linalg.slogdet(x),
        cuda=lambda x: _fused_logdet(x),
        default=lambda x: jnp.linalg.slogdet(x),
    )


# ========================= Thouless Amplitude Specialization =========================

def _gather_thouless_block(
    T: jnp.ndarray,
    parts_pos: jnp.ndarray,
    holes_pos: jnp.ndarray,
) -> jnp.ndarray:
    """
    Extract T_{P,H} block via flattened gather.
    
    Build flat indices from particle/hole positions for single-pass gathering,
    avoiding large intermediate arrays from two-stage indexing.
    
    Args:
        T: Thouless matrix (B, N_v, N_o)
        parts_pos: Particle indices (B, k_max), padded with -1
        holes_pos: Hole indices (B, k_max), padded with -1
        
    Returns:
        T_{P,H} block of shape (B, k_max, k_max)
    """
    B, Nv, No = T.shape
    kmax = int(parts_pos.shape[-1])

    pp = jnp.clip(parts_pos, 0, Nv - 1).astype(jnp.int32)
    hp = jnp.clip(holes_pos, 0, No - 1).astype(jnp.int32)

    T_flat = T.reshape((B, Nv * No))

    # Flat index: i * N_o + j
    No_i32 = jnp.int32(No)
    idx = pp[:, :, None] * No_i32 + hp[:, None, :]
    idx_flat = idx.reshape((B, kmax * kmax))

    gathered = jnp.take_along_axis(T_flat, idx_flat, axis=1)
    return gathered.reshape((B, kmax, kmax))


def slogdet_thouless(
    amplitudes: ThoulessAmplitudes,
    batch: DetBatch,
    *,
    kmax: int,
    use_fast_kernel: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Thouless amplitude via padded determinant.
    
    For k-fold excitation (k <= k_max), construct:
        A = diag(T_{P,H}, I_{k_max - k})
    Then det(A) = det(T_{P,H}) yields Thouless amplitude.
    
    For k > k_max, return zero amplitude (sign=0, logabs=-inf).
    
    Args:
        amplitudes: ThoulessAmplitudes with T matrix (B, N_v, N_o)
        batch: DetBatch with k, holes_pos, parts_pos, phase in aux
        kmax: Maximum excitation level for padding
        use_fast_kernel: Use FFI kernel if applicable
        
    Returns:
        (sign, log_abs_amp): Sign and log absolute Thouless amplitude
    """
    if kmax < 0:
        raise ValueError(f"kmax must be non-negative, got {kmax}")

    aux = batch.aux
    k = aux["k"]
    holes_pos = aux["holes_pos"]
    parts_pos = aux["parts_pos"]
    phase = aux["phase"]

    No = int(amplitudes.T.shape[-1])
    if kmax > No:
        raise ValueError(f"kmax={kmax} exceeds N_e={No}")

    B = int(batch.occ.shape[0])

    if kmax == 0:
        sign = phase.astype(jnp.float32)
        logabs = jnp.zeros((B,), dtype=jnp.float32)
        
        over = k > 0
        sign = jnp.where(over, jnp.zeros_like(sign), sign)
        logabs = jnp.where(over, jnp.full_like(logabs, -jnp.inf), logabs)
        return sign, logabs

    T_blk = _gather_thouless_block(amplitudes.T, parts_pos, holes_pos)

    kk = jnp.clip(k.astype(jnp.int32), 0, jnp.int32(kmax))
    ar = jnp.arange(kmax, dtype=jnp.int32)[None, :]
    active = ar < kk[:, None]
    mask2d = active[:, :, None] & active[:, None, :]

    # Padded matrix: A = I + mask * (T_blk - I)
    I = jnp.eye(kmax, dtype=T_blk.dtype)
    m = mask2d.astype(T_blk.dtype)
    A = I + m * (T_blk - I)

    det_sign, det_logabs = slogdet(A, use_fast_kernel=use_fast_kernel)
    det_sign = det_sign * phase.astype(det_sign.dtype)

    over = k > jnp.asarray(kmax, dtype=k.dtype)
    det_sign = jnp.where(over, jnp.zeros_like(det_sign), det_sign)
    det_logabs = jnp.where(over, jnp.full_like(det_logabs, -jnp.inf), det_logabs)

    return det_sign, det_logabs


__all__ = ["slogdet", "slogdet_thouless"]
