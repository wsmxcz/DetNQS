# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
GPU-accelerated sparse matrix operations using JAX BCOO format.

Provides:
  - COO to BCOO conversion with duplicate merging
  - Variational-space SpMV: y_V = H_VV @ ψ_V
  - Block SpMV for proxy mode

File: detnqs/engine/kernel_gpu.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.experimental import sparse as jsp

from .hamiltonian import ProxyHamiltonian, VVHamiltonian
from .kernel import ProxyContraction, VVContraction


def _coo_to_bcoo(
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    vals: jnp.ndarray,
    shape: tuple[int, int],
) -> jsp.BCOO:
    """
    Convert COO triplet to BCOO with duplicate summation.

    Args:
        rows, cols, vals: COO triplet arrays
        shape: Matrix dimensions (n_rows, n_cols)

    Returns:
        Canonical BCOO matrix with merged duplicates
    """
    n_rows, n_cols = shape

    if vals.size == 0 or n_rows == 0 or n_cols == 0:
        data = jnp.zeros((0,), dtype=jnp.float64)
        indices = jnp.zeros((0, 2), dtype=jnp.int32)
        return jsp.BCOO((data, indices), shape=shape)

    data = jnp.asarray(vals, dtype=jnp.float64)
    indices = jnp.stack([
        jnp.asarray(rows, dtype=jnp.int32),
        jnp.asarray(cols, dtype=jnp.int32)
    ], axis=1)

    mat = jsp.BCOO((data, indices), shape=shape,
                   indices_sorted=False, unique_indices=False)
    return mat.sum_duplicates()


def build_vv_operator_gpu(
    ham: VVHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray], VVContraction]:
    """
    Build variational-space operator y_V = H_VV @ ψ_V for GPU.

    Args:
        ham: Variational Hamiltonian with COO blocks
        jax_dtype: Target dtype (complex64/complex128)

    Returns:
        Function mapping ψ_V to VVContraction(V=y_V)
    """
    H_vv = _coo_to_bcoo(
        rows=jnp.asarray(ham.ham_vv.rows),
        cols=jnp.asarray(ham.ham_vv.cols),
        vals=jnp.asarray(ham.ham_vv.vals),
        shape=ham.ham_vv.shape,
    )

    def operator(psi_v: jnp.ndarray) -> VVContraction:
        y_v = H_vv @ psi_v
        return VVContraction(V=y_v.astype(jax_dtype))

    return operator


def build_proxy_operator_gpu(
    ham: ProxyHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction]:
    """
    Build block operator for proxy mode on GPU:

        y_V = H_VV @ ψ_V + H_VP @ ψ_P
        y_P = H_PV @ ψ_V + H_PP_diag @ ψ_P

    where H_PP is diagonal and H_PV = H_VP^T.

    Args:
        ham: Proxy Hamiltonian with block structure
        jax_dtype: Target dtype (complex64/complex128)

    Returns:
        Function mapping (ψ_V, ψ_P) to ProxyContraction(V=y_V, P=y_P)
    """
    H_vv = _coo_to_bcoo(
        rows=jnp.asarray(ham.ham_vv.rows),
        cols=jnp.asarray(ham.ham_vv.cols),
        vals=jnp.asarray(ham.ham_vv.vals),
        shape=ham.ham_vv.shape,
    )

    H_vp = _coo_to_bcoo(
        rows=jnp.asarray(ham.ham_vp.rows),
        cols=jnp.asarray(ham.ham_vp.cols),
        vals=jnp.asarray(ham.ham_vp.vals),
        shape=ham.ham_vp.shape,
    )
    H_pv = H_vp.transpose()  # H_PV = H_VP^T

    h_diag_p = jnp.asarray(ham.diagonals.P, dtype=jnp.float64)
    n_p = h_diag_p.shape[0]

    def operator(psi_v: jnp.ndarray, psi_p: jnp.ndarray) -> ProxyContraction:
        y_v = H_vv @ psi_v
        if n_p > 0:
            y_v += H_vp @ psi_p

        if n_p > 0:
            y_p = H_pv @ psi_v + h_diag_p * psi_p
        else:
            y_p = jnp.zeros_like(psi_p, dtype=jax_dtype)

        return ProxyContraction(
            V=y_v.astype(jax_dtype),
            P=y_p.astype(jax_dtype),
        )

    return operator


__all__ = ["build_vv_operator_gpu", "build_proxy_operator_gpu"]