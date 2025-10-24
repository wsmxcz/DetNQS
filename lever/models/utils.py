# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical utilities for wavefunction models.

Provides stable complex arithmetic operations and specialized initializers
for neural quantum state architectures.

File: lever/models/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2025
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


# ============================================================================
# Stable Complex Arithmetic
# ============================================================================


def logsumexp_c(z: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """
    Numerically stable log-sum-exp for complex arrays.
    
    Computes log(Σ exp(z)) using the shift trick: m + log(Σ exp(z - m))
    where m = max(Re(z)) prevents overflow.
    
    Args:
        z: Complex-valued input
        axis: Reduction axis
    
    Returns:
        log(Σ exp(z)) computed stably
    """
    m = jnp.max(jnp.real(z), axis=axis, keepdims=True)
    m_squeezed = jnp.squeeze(m, axis=axis)
    return m_squeezed + jnp.log(jnp.sum(jnp.exp(z - m), axis=axis))


def logdet_c(A: jnp.ndarray) -> jnp.ndarray:
    """
    Stable log-determinant for complex matrices.
    
    Uses sign-logabs decomposition to handle large/small magnitudes.
    
    Args:
        A: Complex square matrix
    
    Returns:
        log(det(A))
    """
    sign, logabs = jnp.linalg.slogdet(A)
    return logabs + jnp.log(sign.astype(A.dtype))


def log_cosh(z: jnp.ndarray) -> jnp.ndarray:
    """
    Stable log(cosh(z)) for complex inputs.
    
    Uses identity: log(cosh(z)) = |Re(z)| + log(1 + exp(-2|Re(z)|)) - log(2)
    which remains stable for large |Re(z)|.
    
    Args:
        z: Complex-valued array
    
    Returns:
        log(cosh(z)) computed element-wise
    """
    # Exploit cosh symmetry: cosh(z) = cosh(-z)
    z_abs_real = jnp.where(jnp.real(z) < 0, -z, z)
    return z_abs_real + jnp.log1p(jnp.exp(-2.0 * z_abs_real)) - jnp.log(2.0)


# ============================================================================
# Complex Weight Initializers
# ============================================================================


def _compute_fans(shape: tuple[int, ...]) -> tuple[float, float]:
    """
    Compute fan-in/fan-out for variance scaling.
    
    Handles Dense (2D) and Conv (higher-D) weight shapes.
    
    Args:
        shape: Weight tensor shape
    
    Returns:
        (fan_in, fan_out) tuple
    """
    if len(shape) < 2:
        return 1.0, 1.0

    receptive_field = jnp.prod(jnp.array(shape[:-2])) if len(shape) > 2 else 1.0
    fan_in = float(shape[-2] * receptive_field)
    fan_out = float(shape[-1] * receptive_field)

    return fan_in, fan_out


def complex_glorot_init() -> Callable:
    """
    Complex Glorot/Xavier initializer: |W| ~ Rayleigh(σ), arg(W) ~ U(0,2π).
    
    Target variance v = 2/(fan_in + fan_out), achieved via σ² = v/2.
    Maintains same total variance as real Glorot.
    
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex64) -> jax.Array:
        fan_in, fan_out = _compute_fans(shape)
        variance = 2.0 / (fan_in + fan_out)
        sigma = jnp.sqrt(variance / 2.0)  # Rayleigh scale
        
        k_amp, k_phase = jax.random.split(key)
        amplitude = jax.random.rayleigh(k_amp, sigma, shape=shape)
        phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
        
        return (amplitude * jnp.exp(1j * phase)).astype(dtype)

    return _init


def complex_he_init() -> Callable:
    """
    Complex He initializer: |W| ~ Rayleigh(σ), arg(W) ~ U(0,2π).
    
    Target variance v = 2/fan_in for ReLU-like activations.
    Larger scale suitable for post-activation layers.
    
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex64) -> jax.Array:
        fan_in, _ = _compute_fans(shape)
        variance = 2.0 / fan_in
        sigma = jnp.sqrt(variance / 2.0)
        
        k_amp, k_phase = jax.random.split(key)
        amplitude = jax.random.rayleigh(k_amp, sigma, shape=shape)
        phase = jax.random.uniform(k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi)
        
        return (amplitude * jnp.exp(1j * phase)).astype(dtype)

    return _init


def c_init(sigma: float) -> Callable:
    """
    Complex weight initializer with fixed Rayleigh scale.
    
    Generates random complex numbers with:
      - Amplitude ~ Rayleigh(sigma)
      - Phase ~ Uniform(0, 2π)
    
    Args:
        sigma: Rayleigh distribution scale parameter
    
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex128) -> jax.Array:
        k_amp, k_phase = jax.random.split(key)
        amp = jax.random.rayleigh(k_amp, scale=sigma, shape=shape)
        phase = jax.random.uniform(
            k_phase, shape=shape, minval=0.0, maxval=2.0 * jnp.pi
        )
        return (amp * jnp.exp(1j * phase)).astype(dtype)
    
    return _init


def c_orthogonal_init(key: jax.Array, shape: Tuple[int, ...], dtype: Any) -> jax.Array:
    """
    Initialize complex matrix with orthonormal columns via QR decomposition.
    
    For rectangular (m, n) with m ≥ n, produces Q satisfying Q^†Q = I.
    Maintains proper complex Gaussian statistics before orthogonalization.
    
    Args:
        key: PRNG key for random generation
        shape: Output matrix shape (m, n) or (..., m, n) for batched
        dtype: Complex dtype (complex64 or complex128)
    
    Returns:
        Complex matrix with orthonormal columns
    """
    real_dtype = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    key_re, key_im = jax.random.split(key)

    # Generate complex Gaussian: E[|z|²] = 1
    re = jax.random.normal(key_re, shape, dtype=real_dtype) / jnp.sqrt(2.0)
    im = jax.random.normal(key_im, shape, dtype=real_dtype) / jnp.sqrt(2.0)

    A = jax.lax.complex(re, im)

    # QR decomposition for column orthonormality
    Q, _ = jnp.linalg.qr(A)
    return Q


__all__ = [
    "logsumexp_c",
    "logdet_c",
    "log_cosh",
    "complex_glorot_init",
    "complex_he_init",
    "c_init",
    "c_orthogonal_init",
]