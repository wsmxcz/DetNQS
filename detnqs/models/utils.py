# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Numerical utilities for neural quantum states.

Provides:
  - Stable complex arithmetic: logsumexp, log(cosh)
  - Specialized initializers: Glorot/He with Rayleigh magnitudes
  - Backflow orbital initialization for RHF/UHF/GHF

File: detnqs/models/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December 2025
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp


# ============================= Complex Arithmetic =============================

def logsumexp_c(z: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Numerically stable log(sum exp(z)) for complex arrays.
    
    Algorithm: log(sum_i exp(z_i)) = m + log(sum_i exp(z_i - m))
               where m = max(Re(z_i))
    
    Args:
        z: Complex input array
        axis: Reduction axis
        
    Returns:
        log(sum exp(z)) reduced along axis
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


# ============================ Weight Initializers =============================

def _compute_fans(shape: tuple[int, ...]) -> tuple[float, float]:
    """Compute fan_in and fan_out for variance scaling.
    
    For shape (..., r, n_in, n_out):
        fan_in = r * n_in
        fan_out = r * n_out
    where r is the receptive field size.
    
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
    """Initialize complex weights with Rayleigh-distributed magnitude.
    
    Magnitude ~ Rayleigh(sigma), Phase ~ Uniform(0, 2π)
    
    Args:
        key: PRNG key
        shape: Output shape
        sigma: Rayleigh scale parameter
        dtype: Complex dtype
        
    Returns:
        Complex array with random magnitude and phase
    """
    k_mag, k_phase = jax.random.split(key)
    magnitude = jax.random.rayleigh(k_mag, sigma, shape=shape)
    phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
    return (magnitude * jnp.exp(1j * phase)).astype(dtype)


def complex_glorot_init() -> Callable:
    """Complex Glorot/Xavier initializer.
    
    Variance scaling: sigma^2 = 1 / (fan_in + fan_out)
    
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
    
    Variance scaling: sigma^2 = 2 / fan_in
    
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
    
    Generates Q satisfying Q^† Q = I for shape (..., m, n) with m >= n.
    
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
    
    RHF/UHF: Identity matrix I_{n_orb × n_e}
    GHF: Spin-orbital one-hot encoding [α_block; β_block]
    
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
    "log_cosh",
    "complex_glorot_init",
    "complex_he_init",
    "c_init",
    "c_orthogonal_init",
    "backflow_orbitals_init",
]
