# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Numerical utilities for wavefunction models.

Provides numerically stable complex arithmetic operations and initializers
for neural quantum state models.

File: lever/models/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import Callable, Tuple, Any


import jax
import jax.numpy as jnp


def logsumexp_c(z: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """
    Numerically stable log-sum-exp for complex arrays.
    
    Uses the standard trick: log(Σ exp(z)) = m + log(Σ exp(z - m))
    where m = max(Re(z)) to prevent overflow.
    
    Args:
        z: Complex-valued input array
        axis: Reduction axis
        
    Returns:
        log(Σ exp(z)) computed stably
    """
    m = jnp.max(jnp.real(z), axis=axis, keepdims=True)
    m_squeezed = jnp.squeeze(m, axis=axis)
    return m_squeezed + jnp.log(jnp.sum(jnp.exp(z - m), axis=axis))


def logdet_c(A: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable log-determinant for complex matrices.
    
    Computes log(det(A)) via sign-logabs decomposition to handle
    large/small determinant magnitudes.
    
    Args:
        A: Complex square matrix
        
    Returns:
        log(det(A))
    """
    sign, logabs = jnp.linalg.slogdet(A)
    return logabs + jnp.log(sign.astype(A.dtype))


def log_cosh(z: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable log(cosh(z)) for complex inputs.
    
    Uses the identity:
        log(cosh(z)) = |Re(z)| + log(1 + exp(-2|Re(z)|)) - log(2)
    which remains stable for large |Re(z)|.
    
    Args:
        z: Complex-valued array
        
    Returns:
        log(cosh(z)) computed element-wise
    """
    # Exploit cosh(z) = cosh(-z) symmetry
    z_abs_real = jnp.where(jnp.real(z) < 0, -z, z)
    return z_abs_real + jnp.log1p(jnp.exp(-2.0 * z_abs_real)) - jnp.log(2.0)


def c_init(sigma: float) -> Callable:
    """
    Complex weight initializer with Rayleigh amplitude and uniform phase.
    
    Generates random complex numbers with:
      - Amplitude ~ Rayleigh(sigma)
      - Phase ~ Uniform(0, 2π)
    
    Args:
        sigma: Rayleigh distribution scale parameter
        
    Returns:
        Flax-compatible initializer function
    """
    def _init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.complex128):
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
  
    For rectangular (m, n) with m ≥ n, produces columns satisfying Q^†Q = I.
    Maintains proper complex Gaussian statistics before orthogonalization.
  
    Args:
        key: PRNG key for random generation
        shape: Output shape (m, n)
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
  
    # QR decomposition ensures column orthonormality
    Q, _ = jnp.linalg.qr(A)
    return Q

__all__ = ["logsumexp_c", "logdet_c", "log_cosh", "c_init"]
