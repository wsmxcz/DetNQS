# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stable log-determinant computation with FFI acceleration.

Provides:
  - slogdet: Standard N×N determinant with optional FFI kernel
  - slogdet_thouless: Fast k×k padded determinant for Thouless amplitudes

Performance considerations:
  - JAX gather operations can be expensive; prefer flattened single-pass gathering
  - FFI kernels handle real matrices of size 2 <= n <= 64 with fused LU decomposition

File: lever/models/slogdet.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import ffi

from ..utils.det_utils import DetBatch
from .mappers import ThoulessAmplitudes


# ============================= FFI Registration ===============================

@lru_cache(maxsize=None)
def _register_ffi_targets() -> None:
    """Register CPU and CUDA FFI targets for fused determinant kernels."""
    from .. import _lever_ffi

    def _reg(name: str, capsule, platform: str):
        ffi.register_ffi_target(name, capsule, platform=platform)

    def _reg_cuda(name: str, capsule):
        if capsule is None:
            return
        for plat in ("CUDA", "cuda"):
            try:
                ffi.register_ffi_target(name, capsule, platform=plat)
            except Exception:
                pass

    _reg("lever_fused_logdet_f64_fwd", _lever_ffi.fused_logdet_f64_fwd_cpu(), "cpu")
    _reg("lever_fused_logdet_f64_bwd", _lever_ffi.fused_logdet_f64_bwd_cpu(), "cpu")
    _reg("lever_fused_logdet_f32_fwd", _lever_ffi.fused_logdet_f32_fwd_cpu(), "cpu")
    _reg("lever_fused_logdet_f32_bwd", _lever_ffi.fused_logdet_f32_bwd_cpu(), "cpu")

    _reg_cuda("lever_fused_logdet_f64_fwd", _lever_ffi.fused_logdet_f64_fwd_cuda())
    _reg_cuda("lever_fused_logdet_f64_bwd", _lever_ffi.fused_logdet_f64_bwd_cuda())
    _reg_cuda("lever_fused_logdet_f32_fwd", _lever_ffi.fused_logdet_f32_fwd_cuda())
    _reg_cuda("lever_fused_logdet_f32_bwd", _lever_ffi.fused_logdet_f32_bwd_cuda())


# ========================== FFI Kernel Wrapper ================================

def _flatten_batch(A: jnp.ndarray) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Flatten batch dimensions: (..., n, n) -> (B, n, n) for FFI."""
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Expected (..., n, n), got {A.shape}")

    batch_shape = A.shape[:-2]
    n = A.shape[-1]

    b = 1
    for d in batch_shape:
        if not isinstance(d, int):
            raise ValueError(f"FFI requires static shapes, got {batch_shape}")
        b *= d

    return A.reshape((b, n, n)), batch_shape


@jax.custom_vjp
def _fused_logdet(A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """FFI-accelerated slogdet using fused LU decomposition for 2 <= n <= 64."""
    _register_ffi_targets()

    if A.dtype not in (jnp.float32, jnp.float64):
        raise TypeError(f"Only float32/float64 supported, got {A.dtype}")

    A_flat, batch_shape = _flatten_batch(A)
    b, n, _ = A_flat.shape

    if not (2 <= n <= 64):
        raise ValueError(f"Matrix size must be in [2, 64], got {n}")

    kernel_name = (
        "lever_fused_logdet_f64_fwd" if A.dtype == jnp.float64
        else "lever_fused_logdet_f32_fwd"
    )

    out_spec = [
        jax.ShapeDtypeStruct((b,), A.dtype),
        jax.ShapeDtypeStruct((b,), A.dtype),
    ]

    sign, logabs = ffi.ffi_call(kernel_name, out_spec, vmap_method="broadcast_all")(A_flat)
    return sign.reshape(batch_shape), logabs.reshape(batch_shape)


def _fused_logdet_fwd(A: jnp.ndarray):
    sign, logabs = _fused_logdet.__wrapped__(A)
    return (sign, logabs), (A,)


def _fused_logdet_bwd(res, ct):
    """Backward pass: grad_A = cot_logabs * A^{-T}."""
    (A,) = res
    cot_sign, cot_logabs = ct
    _register_ffi_targets()

    A_flat, _ = _flatten_batch(A)
    b, n, _ = A_flat.shape

    kernel_name = (
        "lever_fused_logdet_f64_bwd" if A.dtype == jnp.float64
        else "lever_fused_logdet_f32_bwd"
    )

    out_spec = jax.ShapeDtypeStruct((b, n, n), A.dtype)
    grad_A = ffi.ffi_call(kernel_name, out_spec)(
        A_flat, cot_sign.reshape((b,)), cot_logabs.reshape((b,))
    )
    return (grad_A.reshape(A.shape),)


_fused_logdet.defvjp(_fused_logdet_fwd, _fused_logdet_bwd)


# ========================== Public Interface ==================================

def slogdet(A: jnp.ndarray, *, use_fast_kernel: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute sign and log|det(A)| with optional FFI acceleration.
  
    Args:
        A: Input matrix of shape (..., n, n)
        use_fast_kernel: Use FFI kernel if applicable (real float, 2 <= n <= 64)
  
    Returns:
        (sign, log_abs_det): Sign and logarithm of absolute determinant
    """
    if use_fast_kernel:
        is_real_float = A.dtype in (jnp.float32, jnp.float64)
        n = A.shape[-1] if A.ndim >= 2 else 0
        if is_real_float and 2 <= n <= 64:
            return _fused_logdet(A)
    return jnp.linalg.slogdet(A)


# ======================= Thouless Mode Specialization =========================

def _gather_thouless_block(
    T: jnp.ndarray,
    parts_pos: jnp.ndarray,
    holes_pos: jnp.ndarray,
) -> jnp.ndarray:
    """
    Extract T_{P,H} block via flattened gather to avoid large intermediates.
  
    Algorithm: Build flattened indices (B, k_max^2) from particle/hole positions,
    then gather in one pass from flattened T matrix.
  
    Args:
        T: Thouless matrix (B, N_v, N_o)
        parts_pos: Particle indices (B, k_max), padded with -1
        holes_pos: Hole indices (B, k_max), padded with -1
  
    Returns:
        T_block: Gathered block (B, k_max, k_max)
    """
    B, Nv, No = T.shape
    kmax = int(parts_pos.shape[-1])

    pp = jnp.clip(parts_pos, 0, Nv - 1).astype(jnp.int32)
    hp = jnp.clip(holes_pos, 0, No - 1).astype(jnp.int32)

    T_flat = T.reshape((B, Nv * No))

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
  
    Algorithm: For excitation level k <= k_max, build padded matrix A:
      A = diag(T_{P,H}, I_{k_max-k})
    Then det(A) = det(T_{P,H}) gives the Thouless amplitude.
  
    Handling k > k_max: Return zero amplitude (sign=0, logabs=-inf).
  
    Args:
        amplitudes: ThoulessAmplitudes with matrix T (B, N_v, N_o)
        batch: DetBatch with auxiliary data (k, holes_pos, parts_pos, phase)
        kmax: Maximum excitation level for padding
        use_fast_kernel: Use FFI kernel if applicable
  
    Returns:
        (sign, log_abs_amp): Sign and log of absolute Thouless amplitude
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
        raise ValueError(f"kmax ({kmax}) cannot exceed N_e ({No})")

    B = int(batch.occ.shape[0])

    if kmax == 0:
        sign_dtype = jnp.float32
        la_dtype = jnp.float32
        s = phase.astype(sign_dtype)
        la = jnp.zeros((B,), dtype=la_dtype)

        over = k > 0
        s = jnp.where(over, jnp.zeros_like(s), s)
        la = jnp.where(over, jnp.full_like(la, -jnp.inf), la)
        return s, la

    T_blk = _gather_thouless_block(amplitudes.T, parts_pos, holes_pos)

    kk = jnp.clip(k.astype(jnp.int32), 0, jnp.int32(kmax))
    ar = jnp.arange(kmax, dtype=jnp.int32)[None, :]
    active = ar < kk[:, None]
    mask2d = active[:, :, None] & active[:, None, :]

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