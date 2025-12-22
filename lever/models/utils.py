# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical utilities for neural quantum states.

Provides:
  - Stable complex arithmetic: logsumexp, logdet, log(cosh)
  - Specialized initializers: Glorot/He with Rayleigh magnitudes
  - Fused FFI kernels: accelerated determinant computation

File: lever/models/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December 2025
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax import ffi


# ============================= Complex Arithmetic =============================

def logsumexp_c(z: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Numerically stable log(sum exp(z)) for complex arrays.
  
    Uses shift-and-log trick: log(sum_i exp(z_i)) = m + log(sum_i exp(z_i - m))
    where m = max(Re(z_i)) to prevent overflow.
  
    Args:
        z: Complex input array
        axis: Reduction axis
      
    Returns:
        Scalar or reduced array of log(sum exp(z))
    """
    m = jnp.max(jnp.real(z), axis=axis, keepdims=True)
    return jnp.squeeze(m, axis=axis) + jnp.log(jnp.sum(jnp.exp(z - m), axis=axis))


def log_cosh(z: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable log(cosh(z)) for complex inputs.
  
    Algorithm: log(cosh(z)) = |Re(z)| + log(1 + exp(-2|Re(z)|)) - log(2)
  
    Args:
        z: Complex input array
      
    Returns:
        log(cosh(z)) with same shape as z
    """
    z_abs = jnp.where(jnp.real(z) < 0, -z, z)
    return z_abs + jnp.log1p(jnp.exp(-2.0 * z_abs)) - jnp.log(2.0)


# ========================== Fused LogDet (FFI) ================================

@lru_cache(maxsize=None)
def _register_ffi_targets() -> None:
    """Register FFI targets for CPU/CUDA with float32/float64 support."""
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

    # CPU targets
    _reg("lever_fused_logdet_f64_fwd", _lever_ffi.fused_logdet_f64_fwd_cpu(), "cpu")
    _reg("lever_fused_logdet_f64_bwd", _lever_ffi.fused_logdet_f64_bwd_cpu(), "cpu")
    _reg("lever_fused_logdet_f32_fwd", _lever_ffi.fused_logdet_f32_fwd_cpu(), "cpu")
    _reg("lever_fused_logdet_f32_bwd", _lever_ffi.fused_logdet_f32_bwd_cpu(), "cpu")

    # CUDA targets
    _reg_cuda("lever_fused_logdet_f64_fwd", _lever_ffi.fused_logdet_f64_fwd_cuda())
    _reg_cuda("lever_fused_logdet_f64_bwd", _lever_ffi.fused_logdet_f64_bwd_cuda())
    _reg_cuda("lever_fused_logdet_f32_fwd", _lever_ffi.fused_logdet_f32_fwd_cuda())
    _reg_cuda("lever_fused_logdet_f32_bwd", _lever_ffi.fused_logdet_f32_bwd_cuda())


def _flatten_batch(A: jnp.ndarray) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Flatten batch dimensions: (..., n, n) -> (B, n, n).
  
    Note: Uses Python int operations to avoid tracing errors.
    """
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"Expected (..., n, n), got {A.shape}")

    batch_shape = A.shape[:-2]
    n = A.shape[-1]
  
    # Compute batch size using Python ints
    b = 1
    for d in batch_shape:
        if not isinstance(d, int):
            raise ValueError(f"FFI requires static shapes, got {batch_shape}")
        b *= d

    return A.reshape((b, n, n)), batch_shape


@jax.custom_vjp
def _fused_logdet(A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """FFI-accelerated slogdet for real float32/float64 with n in [2, 64]."""
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
        jax.ShapeDtypeStruct((b,), A.dtype)
    ]

    sign, logabs = ffi.ffi_call(kernel_name, out_spec, vmap_method="broadcast_all")(A_flat)
    return sign.reshape(batch_shape), logabs.reshape(batch_shape)


def _fused_logdet_fwd(A: jnp.ndarray):
    sign, logabs = _fused_logdet.__wrapped__(A)
    return (sign, logabs), (A,)


def _fused_logdet_bwd(res, ct):
    (A,) = res
    cot_sign, cot_logabs = ct
    _register_ffi_targets()

    A_flat, batch_shape = _flatten_batch(A)
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


def logdet_c(A: jnp.ndarray, use_fast_kernel: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute sign and log|det(A)| with optional FFI acceleration.
  
    FFI kernel activates for real float32/float64 matrices with 2 <= n <= 64.
    Falls back to jnp.linalg.slogdet for other cases.
  
    Args:
        A: Input matrix of shape (..., n, n)
        use_fast_kernel: Enable FFI acceleration if applicable
      
    Returns:
        (sign, log|det(A)|) tuple
    """
    if use_fast_kernel:
        is_real_float = A.dtype in (jnp.float32, jnp.float64)
        n = A.shape[-1] if A.ndim >= 2 else 0
        if is_real_float and 2 <= n <= 64:
            return _fused_logdet(A)

    return jnp.linalg.slogdet(A)


# ============================ Weight Initializers =============================

def _compute_fans(shape: tuple[int, ...]) -> tuple[float, float]:
    """Compute fan_in and fan_out for variance scaling.
  
    For shape (..., receptive_fields, fan_in, fan_out):
        fan_in = receptive_field_size * shape[-2]
        fan_out = receptive_field_size * shape[-1]
  
    Args:
        shape: Weight tensor shape
      
    Returns:
        (fan_in, fan_out) tuple
    """
    if len(shape) < 2:
        return 1.0, 1.0

    receptive_field = jnp.prod(jnp.array(shape[:-2])) if len(shape) > 2 else 1.0
    return float(shape[-2] * receptive_field), float(shape[-1] * receptive_field)


def _complex_rayleigh_init(
    key: jax.Array,
    shape: tuple[int, ...],
    sigma: float,
    dtype: Any = jnp.complex64
) -> jax.Array:
    """Base initializer: magnitude ~ Rayleigh(sigma), phase ~ Uniform(0, 2pi)."""
    k_mag, k_phase = jax.random.split(key)
    magnitude = jax.random.rayleigh(k_mag, sigma, shape=shape)
    phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
    return (magnitude * jnp.exp(1j * phase)).astype(dtype)


def complex_glorot_init() -> Callable:
    """Complex Glorot/Xavier initializer.
  
    Magnitude ~ Rayleigh(sigma) with sigma^2 = 1 / (fan_in + fan_out)
    Phase ~ Uniform(0, 2pi)
  
    Suitable for tanh/sigmoid-like activations.
  
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex64) -> jax.Array:
        fan_in, fan_out = _compute_fans(shape)
        sigma = jnp.sqrt(1.0 / (fan_in + fan_out))
        return _complex_rayleigh_init(key, shape, sigma, dtype)
    return _init


def complex_he_init() -> Callable:
    """Complex He initializer for ReLU-like activations.
  
    Magnitude ~ Rayleigh(sigma) with sigma^2 = 2 / fan_in
    Phase ~ Uniform(0, 2pi)
  
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex64) -> jax.Array:
        fan_in, _ = _compute_fans(shape)
        sigma = jnp.sqrt(1.0 / fan_in)
        return _complex_rayleigh_init(key, shape, sigma, dtype)
    return _init


def c_init(sigma: float) -> Callable:
    """Fixed-scale complex initializer.
  
    Magnitude ~ Rayleigh(sigma), Phase ~ Uniform(0, 2pi)
  
    Args:
        sigma: Rayleigh scale parameter
      
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex128) -> jax.Array:
        return _complex_rayleigh_init(key, shape, sigma, dtype)
    return _init


def c_orthogonal_init(key: jax.Array, shape: tuple[int, ...], dtype: Any) -> jax.Array:
    """Complex orthonormal initializer via QR decomposition.
  
    Generates Q satisfying Q^dagger Q = I for shape (..., m, n) with m >= n.
  
    Args:
        key: PRNG key
        shape: Output shape
        dtype: Complex dtype (complex64 or complex128)
      
    Returns:
        Matrix with orthonormal columns
    """
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    key_re, key_im = jax.random.split(key)

    re = jax.random.normal(key_re, shape, dtype=real_dtype) / jnp.sqrt(2.0)
    im = jax.random.normal(key_im, shape, dtype=real_dtype) / jnp.sqrt(2.0)
    A = jax.lax.complex(re, im)

    Q, _ = jnp.linalg.qr(A)
    return Q


def backflow_orbitals_init(
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    mode: Literal["restricted", "unrestricted", "generalized"],
) -> Callable:
    """Initializer for backflow reference orbitals M.
  
    RHF/UHF: Identity matrix I_{n_orb x n_e}
    GHF: Spin-orbital one-hot encoding with alpha/beta block structure
  
    Args:
        n_orb: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
        mode: Orbital coupling mode
      
    Returns:
        Flax-compatible initializer function
    """
    n_e = n_alpha + n_beta

    def _init_identity(key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        """RHF/UHF: Return identity matrix."""
        del key
        *batch_shape, m, n = shape
        eye = jnp.eye(m, n, dtype=jnp.result_type(dtype, jnp.float32))
        return jnp.broadcast_to(eye, shape).astype(dtype) if batch_shape else eye.astype(dtype)

    def _init_ghf(key: jax.Array, shape: tuple[int, ...], dtype: Any = jnp.float32) -> jax.Array:
        """GHF: One-hot encoding with spin-orbital separation."""
        del key
        *batch_shape, m, n = shape

        if m != 2 * n_orb or n != n_e:
            raise ValueError(f"Expected ({2 * n_orb}, {n_e}), got ({m}, {n})")

        mat = jnp.zeros((m, n), dtype=jnp.result_type(dtype, jnp.float32))
      
        # Alpha block: rows [0:n_alpha], cols [0:n_alpha]
        if n_alpha > 0:
            idx_a = jnp.arange(n_alpha)
            mat = mat.at[idx_a, idx_a].set(1.0)
      
        # Beta block: rows [n_orb:n_orb+n_beta], cols [n_alpha:n_alpha+n_beta]
        if n_beta > 0:
            rows_b = n_orb + jnp.arange(n_beta)
            cols_b = n_alpha + jnp.arange(n_beta)
            mat = mat.at[rows_b, cols_b].set(1.0)

        return jnp.broadcast_to(mat, shape).astype(dtype) if batch_shape else mat.astype(dtype)

    return _init_ghf if mode == "generalized" else _init_identity


__all__ = [
    "logsumexp_c",
    "logdet_c",
    "log_cosh",
    "complex_glorot_init",
    "complex_he_init",
    "c_init",
    "c_orthogonal_init",
    "backflow_orbitals_init",
]