# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sparse Hamiltonian matrix-vector product (SpMV) kernels via JAX pure_callback.

Implements CPU-based sparse operations for:
  - S-space: H_SS @ psi_S
  - Proxy T-space: H_SS @ psi_S + H_SC @ psi_C, H_CS @ psi_S + diag(H_CC) * psi_C

Uses SciPy CSR/CSC sparse formats inside jax.pure_callback for efficient SpMV
without on-device compilation overhead.

File: lever/operator/kernel.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp

import jax
import jax.numpy as jnp

from .hamiltonian import ProxyHamiltonian, SSHamiltonian


class SSContraction(NamedTuple):
    """Result of H_SS @ psi_S."""
    S: jnp.ndarray


class ProxyContraction(NamedTuple):
    """
    Result of full T-space Hamiltonian action.

    S: H_SS @ psi_S + H_SC @ psi_C
    C: H_CS @ psi_S + diag(H_CC) * psi_C
    """
    S: jnp.ndarray
    C: jnp.ndarray


def build_ss_operator(
    ham: SSHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray], SSContraction]:
    """
    Build S-space operator: H_SS @ psi_S via SciPy CSR matvec.

    Args:
        ham: S-space Hamiltonian
        jax_dtype: JAX array dtype for input/output

    Returns:
        Operator function: psi_S -> SSContraction(S=H_SS @ psi_S)
    """
    H_ss: sp.csr_matrix = ham.ham_ss
    n_s = int(H_ss.shape[0])

    # Match NumPy dtype by name
    dtype_name = jnp.dtype(jax_dtype).name
    dtype_map = {
        'complex128': np.complex128,
        'complex64': np.complex64,
        'float64': np.float64,
        'float32': np.float32,
    }
    np_dtype = dtype_map[dtype_name]

    out_shape = jax.ShapeDtypeStruct((n_s,), jax_dtype)

    def _host_callback(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np_dtype)
        y = H_ss.dot(x_np)
        return np.asarray(y, dtype=np_dtype)

    def operator(psi_s: jnp.ndarray) -> SSContraction:
        y_s = jax.pure_callback(_host_callback, out_shape, psi_s)
        return SSContraction(S=y_s)

    return operator


def build_proxy_operator(
    ham: ProxyHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction]:
    """
    Build full T-space operator via SciPy CSR/CSC matvec.

    Computes:
        y_S = H_SS @ psi_S + H_SC @ psi_C
        y_C = H_CS @ psi_S + diag(H_CC) * psi_C

    Args:
        ham: Proxy Hamiltonian with S and C blocks
        jax_dtype: JAX array dtype for input/output

    Returns:
        Operator function: (psi_S, psi_C) -> ProxyContraction(S=y_S, C=y_C)
    """
    H_ss: sp.csr_matrix = ham.ham_ss
    H_sc: sp.csr_matrix = ham.ham_sc
    H_cs: sp.csc_matrix = ham.ham_cs
    h_diag_c = ham.diagonals.C

    n_s = int(H_ss.shape[0])
    n_c = int(H_sc.shape[1])

    # Match NumPy dtype by name
    dtype_name = jnp.dtype(jax_dtype).name
    dtype_map = {
        'complex128': np.complex128,
        'complex64': np.complex64,
        'float64': np.float64,
        'float32': np.float32,
    }
    np_dtype = dtype_map[dtype_name]

    out_shapes = (
        jax.ShapeDtypeStruct((n_s,), jax_dtype),
        jax.ShapeDtypeStruct((n_c,), jax_dtype),
    )

    def _host_callback(
        xs: np.ndarray, xc: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xs_np = np.asarray(xs, dtype=np_dtype)
        xc_np = np.asarray(xc, dtype=np_dtype)

        y_s = H_ss.dot(xs_np) + H_sc.dot(xc_np)
        y_c = H_cs.dot(xs_np) + np.asarray(h_diag_c, dtype=np_dtype) * xc_np

        return (
            np.asarray(y_s, dtype=np_dtype),
            np.asarray(y_c, dtype=np_dtype),
        )

    def operator(psi_s: jnp.ndarray, psi_c: jnp.ndarray) -> ProxyContraction:
        y_s, y_c = jax.pure_callback(_host_callback, out_shapes, psi_s, psi_c)
        return ProxyContraction(S=y_s, C=y_c)

    return operator


__all__ = [
    "SSContraction",
    "ProxyContraction",
    "build_ss_operator",
    "build_proxy_operator",
]
