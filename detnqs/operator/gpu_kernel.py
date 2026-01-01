# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
GPU-accelerated sparse matrix operations using JAX BCOO format.

Provides:
  - COO to BCOO conversion with duplicate merging
  - S-space SpMV: y_S = H_SS @ ψ_S
  - Block SpMV for proxy mode

File: detnqs/engine/kernel_gpu.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.experimental import sparse as jsp

from .hamiltonian import ProxyHamiltonian, SSHamiltonian
from .kernel import ProxyContraction, SSContraction


def _coo_to_bcoo(
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    vals: jnp.ndarray,
    shape: tuple[int, int],
) -> jsp.BCOO:
    """
    Convert COO format to BCOO with duplicate summation.

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


def build_ss_operator_gpu(
    ham: SSHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray], SSContraction]:
    """
    Build S-space operator y_S = H_SS @ ψ_S for GPU.

    Args:
        ham: Effective Hamiltonian with COO blocks
        jax_dtype: Target dtype (complex64/complex128)

    Returns:
        Function mapping ψ_S to SSContraction(S=y_S)
    """
    H_ss = _coo_to_bcoo(
        rows=jnp.asarray(ham.ham_ss.rows),
        cols=jnp.asarray(ham.ham_ss.cols),
        vals=jnp.asarray(ham.ham_ss.vals),
        shape=ham.ham_ss.shape,
    )

    def operator(psi_s: jnp.ndarray) -> SSContraction:
        y_s = H_ss @ psi_s
        return SSContraction(S=y_s.astype(jax_dtype))

    return operator


def build_proxy_operator_gpu(
    ham: ProxyHamiltonian,
    *,
    jax_dtype: jnp.dtype,
) -> Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction]:
    """
    Build block operator for proxy mode on GPU:

        y_S = H_SS @ ψ_S + H_SC @ ψ_C
        y_C = H_CS @ ψ_S + H_CC_d @ ψ_C

    where H_CC is diagonal and H_CS = H_SC^T.

    Args:
        ham: Proxy Hamiltonian with block structure
        jax_dtype: Target dtype (complex64/complex128)

    Returns:
        Function mapping (ψ_S, ψ_C) to ProxyContraction(S=y_S, C=y_C)
    """
    H_ss = _coo_to_bcoo(
        rows=jnp.asarray(ham.ham_ss.rows),
        cols=jnp.asarray(ham.ham_ss.cols),
        vals=jnp.asarray(ham.ham_ss.vals),
        shape=ham.ham_ss.shape,
    )

    H_sc = _coo_to_bcoo(
        rows=jnp.asarray(ham.ham_sc.rows),
        cols=jnp.asarray(ham.ham_sc.cols),
        vals=jnp.asarray(ham.ham_sc.vals),
        shape=ham.ham_sc.shape,
    )
    H_cs = H_sc.transpose()  # H_CS = H_SC^T

    h_diag_c = jnp.asarray(ham.diagonals.C, dtype=jnp.float64)
    n_c = h_diag_c.shape[0]

    def operator(psi_s: jnp.ndarray, psi_c: jnp.ndarray) -> ProxyContraction:
        y_s = H_ss @ psi_s
        if n_c > 0:
            y_s += H_sc @ psi_c

        if n_c > 0:
            y_c = H_cs @ psi_s + h_diag_c * psi_c
        else:
            y_c = jnp.zeros_like(psi_c, dtype=jax_dtype)

        return ProxyContraction(
            S=y_s.astype(jax_dtype),
            C=y_c.astype(jax_dtype),
        )

    return operator


__all__ = ["build_ss_operator_gpu", "build_proxy_operator_gpu"]
