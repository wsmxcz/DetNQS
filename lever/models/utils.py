# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical utilities for neural quantum states.

Provides:
  - Stable complex arithmetic: logsumexp, logdet, log(cosh)
  - Specialized initializers: Glorot/He with Rayleigh magnitudes

File: lever/models/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December 2025
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Literal, Tuple

import jax
import jax.numpy as jnp

def logsumexp_c(z: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Stable log(Σ exp(z)) for complex arrays using max shift trick.
  
    Implements: log(Σ exp(z_i)) = m + log(Σ exp(z_i - m)), where m = max(Re(z_i))
  
    Args:
        z: Complex-valued input array
        axis: Reduction axis
  
    Returns:
        log(Σ exp(z)) along specified axis
    """
    m = jnp.max(jnp.real(z), axis=axis, keepdims=True)
    m_squeezed = jnp.squeeze(m, axis=axis)
    return m_squeezed + jnp.log(jnp.sum(jnp.exp(z - m), axis=axis))

def logdet_c(A: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Stable slogdet wrapper.

    Returns:
        sign:   real (±1) for real A; complex unit-magnitude for complex A
        logabs: real log(|det(A)|)
    """
    sign, logabs = jnp.linalg.slogdet(A)
    return sign, logabs


def log_cosh(z: jnp.ndarray) -> jnp.ndarray:
    """Stable log(cosh(z)) for complex inputs.
  
    Uses identity: log(cosh(z)) = |Re(z)| + log(1 + exp(-2|Re(z)|)) - log(2)
  
    Args:
        z: Complex-valued array
  
    Returns:
        log(cosh(z))
    """
    z_abs_real = jnp.where(jnp.real(z) < 0, -z, z)
    return z_abs_real + jnp.log1p(jnp.exp(-2.0 * z_abs_real)) - jnp.log(2.0)


def _compute_fans(shape: tuple[int, ...]) -> tuple[float, float]:
    """Compute fan-in/fan-out for weight variance scaling.
  
    Args:
        shape: Weight tensor shape
  
    Returns:
        (fan_in, fan_out)
    """
    if len(shape) < 2:
        return 1.0, 1.0

    receptive_field = jnp.prod(jnp.array(shape[:-2])) if len(shape) > 2 else 1.0
    fan_in = float(shape[-2] * receptive_field)
    fan_out = float(shape[-1] * receptive_field)

    return fan_in, fan_out


def complex_glorot_init() -> Callable:
    """Complex Xavier/Glorot initializer.
  
    Magnitude: Rayleigh(σ) with σ^2 = 1/(fan_in + fan_out)
    Phase: U(0, 2π)
  
    Returns:
        Flax-compatible initializer
    """
    def _init(
        key: jax.Array,
        shape: tuple[int, ...],
        dtype=jnp.complex64
    ) -> jax.Array:
        fan_in, fan_out = _compute_fans(shape)
        variance = 2.0 / (fan_in + fan_out)
        sigma = jnp.sqrt(variance / 2.0)
      
        k_amp, k_phase = jax.random.split(key)
        amplitude = jax.random.rayleigh(k_amp, sigma, shape=shape)
        phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
      
        return (amplitude * jnp.exp(1j * phase)).astype(dtype)

    return _init


def complex_he_init() -> Callable:
    """Complex He initializer for ReLU activations.
  
    Magnitude: Rayleigh(σ) with σ^2 = 2/fan_in
    Phase: U(0, 2π)
  
    Returns:
        Flax-compatible initializer
    """
    def _init(
        key: jax.Array,
        shape: tuple[int, ...],
        dtype=jnp.complex64
    ) -> jax.Array:
        fan_in, _ = _compute_fans(shape)
        variance = 2.0 / fan_in
        sigma = jnp.sqrt(variance / 2.0)
      
        k_amp, k_phase = jax.random.split(key)
        amplitude = jax.random.rayleigh(k_amp, sigma, shape=shape)
        phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
      
        return (amplitude * jnp.exp(1j * phase)).astype(dtype)

    return _init


def c_init(sigma: float) -> Callable:
    """Fixed-scale complex initializer.
  
    Args:
        sigma: Rayleigh scale parameter
  
    Returns:
        Flax-compatible initializer
    """
    def _init(
        key: jax.Array,
        shape: tuple[int, ...],
        dtype=jnp.complex128
    ) -> jax.Array:
        k_amp, k_phase = jax.random.split(key)
        amp = jax.random.rayleigh(k_amp, scale=sigma, shape=shape)
        phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
        return (amp * jnp.exp(1j * phase)).astype(dtype)
  
    return _init


def c_orthogonal_init(
    key: jax.Array,
    shape: Tuple[int, ...],
    dtype: Any
) -> jax.Array:
    """Complex orthonormal initializer via QR decomposition.
  
    Generates Q satisfying Q^† Q = I for m ≥ n in shape (..., m, n).
  
    Args:
        key: PRNG key
        shape: Output shape
        dtype: Complex dtype
  
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
    GHF: Spin-orbital one-hot encoding
  
    Args:
        n_orb: Number of spatial orbitals
        n_alpha: Alpha electron count
        n_beta: Beta electron count
        mode: Orbital coupling scheme
  
    Returns:
        Flax-compatible initializer
    """
    n_e = n_alpha + n_beta

    def _init_rhf_uhf(
        key: jax.Array,
        shape: Tuple[int, ...],
        dtype: Any = jnp.float32,
    ) -> jax.Array:
        del key  # Deterministic initialization
        if len(shape) < 2:
            raise ValueError(f"Expected ≥2D shape, got {shape}")

        *batch_shape, m, n = shape
        eye = jnp.eye(m, n, dtype=jnp.result_type(dtype, jnp.float32))

        if batch_shape:
            eye = jnp.broadcast_to(eye, (*batch_shape, m, n))

        return eye.astype(dtype)

    def _init_ghf(
        key: jax.Array,
        shape: Tuple[int, ...],
        dtype: Any = jnp.float32,
    ) -> jax.Array:
        del key  # Deterministic initialization
        if len(shape) < 2:
            raise ValueError(f"Expected ≥2D shape, got {shape}")

        *batch_shape, m, n = shape

        if m != 2 * n_orb:
            raise ValueError(f"Expected m = 2n_orb = {2 * n_orb}, got {m}")
        if n != n_e:
            raise ValueError(f"Expected n = n_e = {n_e}, got {n}")

        base_dtype = jnp.result_type(dtype, jnp.float32)
        mat = jnp.zeros((m, n), dtype=base_dtype)

        if n_alpha > 0:
            rows_a = jnp.arange(n_alpha)
            mat = mat.at[rows_a, rows_a].set(1.0)

        if n_beta > 0:
            rows_b = n_orb + jnp.arange(n_beta)
            cols_b = n_alpha + jnp.arange(n_beta)
            mat = mat.at[rows_b, cols_b].set(1.0)

        if batch_shape:
            mat = jnp.broadcast_to(mat, (*batch_shape, m, n))

        return mat.astype(dtype)

    return _init_ghf if mode == "generalized" else _init_rhf_uhf


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