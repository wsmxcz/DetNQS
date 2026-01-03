# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Sparse Hamiltonian matrix-vector product (SpMV) kernels via JAX pure_callback.

Implements CPU-based sparse operations for:
  - V-space: H_VV @ psi_V
  - Proxy T-space: H_VV @ psi_V + H_VP @ psi_P, H_PV @ psi_V + diag(H_PP) * psi_P

Uses SciPy CSR/CSC sparse formats inside jax.pure_callback for efficient SpMV
without on-device compilation overhead.

File: detnqs/operator/kernel.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
import scipy.sparse as sp

import jax
import jax.numpy as jnp

from .hamiltonian import ProxyHamiltonian, VVHamiltonian


class VVContraction(NamedTuple):
    """Result of H_VV @ psi_V (V-space Hamiltonian action)."""
    V: jnp.ndarray


class ProxyContraction(NamedTuple):
    """
    Result of full T-space Hamiltonian action.

    V: H_VV @ psi_V + H_VP @ psi_P
    P: H_PV @ psi_V + diag(H_PP) * psi_P
    """
    V: jnp.ndarray
    P: jnp.ndarray


def build_vv_operator(
    ham: VVHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray], VVContraction]:
    """
    Build V-space operator: H_VV @ psi_V via SciPy CSR matvec.

    Args:
        ham: V-space Hamiltonian
        jax_dtype: JAX array dtype for input/output

    Returns:
        Operator function: psi_V -> VVContraction(V=H_VV @ psi_V)
    """
    H_vv: sp.csr_matrix = ham.ham_vv
    n_v = int(H_vv.shape[0])

    dtype_map = {
        'complex128': np.complex128,
        'complex64': np.complex64,
        'float64': np.float64,
        'float32': np.float32,
    }
    np_dtype = dtype_map[jnp.dtype(jax_dtype).name]
    out_shape = jax.ShapeDtypeStruct((n_v,), jax_dtype)

    def _host_callback(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np_dtype)
        y = H_vv.dot(x_np)
        return np.asarray(y, dtype=np_dtype)

    def operator(psi_v: jnp.ndarray) -> VVContraction:
        y_v = jax.pure_callback(_host_callback, out_shape, psi_v)
        return VVContraction(V=y_v)

    return operator


def build_proxy_operator(
    ham: ProxyHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction]:
    """
    Build full T-space operator via SciPy CSR/CSC matvec.

    Computes:
        y_V = H_VV @ psi_V + H_VP @ psi_P
        y_P = H_PV @ psi_V + diag(H_PP) * psi_P

    Args:
        ham: Proxy Hamiltonian with V and P blocks
        jax_dtype: JAX array dtype for input/output

    Returns:
        Operator function: (psi_V, psi_P) -> ProxyContraction(V=y_V, P=y_P)
    """
    H_vv: sp.csr_matrix = ham.ham_vv
    H_vp: sp.csr_matrix = ham.ham_vp
    H_pv: sp.csc_matrix = ham.ham_pv
    h_diag_p = ham.diagonals.P

    n_v = int(H_vv.shape[0])
    n_p = int(H_vp.shape[1])

    dtype_map = {
        'complex128': np.complex128,
        'complex64': np.complex64,
        'float64': np.float64,
        'float32': np.float32,
    }
    np_dtype = dtype_map[jnp.dtype(jax_dtype).name]

    out_shapes = (
        jax.ShapeDtypeStruct((n_v,), jax_dtype),
        jax.ShapeDtypeStruct((n_p,), jax_dtype),
    )

    def _host_callback(
        xv: np.ndarray, xp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xv_np = np.asarray(xv, dtype=np_dtype)
        xp_np = np.asarray(xp, dtype=np_dtype)

        y_v = H_vv.dot(xv_np) + H_vp.dot(xp_np)
        y_p = H_pv.dot(xv_np) + np.asarray(h_diag_p, dtype=np_dtype) * xp_np

        return (
            np.asarray(y_v, dtype=np_dtype),
            np.asarray(y_p, dtype=np_dtype),
        )

    def operator(psi_v: jnp.ndarray, psi_p: jnp.ndarray) -> ProxyContraction:
        y_v, y_p = jax.pure_callback(_host_callback, out_shapes, psi_v, psi_p)
        return ProxyContraction(V=y_v, P=y_p)

    return operator


__all__ = [
    "VVContraction",
    "ProxyContraction",
    "build_vv_operator",
    "build_proxy_operator",
]