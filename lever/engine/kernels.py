# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-performance Numba kernels for sparse matrix operations.

Implements parallel COO sparse matrix-vector products using thread-local
accumulation to avoid race conditions while maximizing throughput.

File: lever/engine/kernels.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def coo_matvec(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    x: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """
    Parallel sparse matrix-vector product: y = A @ x.

    Algorithm: Thread-local accumulation with reduction:
      y[i] = Σⱼ A[i,j] · x[j]  (parallel over non-zeros)

    Complexity: O(nnz/P) per thread, P = number of threads
    
    Args:
        rows: Row indices (int32/int64)
        cols: Column indices (int32/int64)
        vals: Non-zero values (complex128)
        x: Input vector (complex128)
        n_rows: Output vector dimension
        
    Returns:
        Result vector y (complex128)
        
    Note:
        Input x must be complex128 for Numba dtype inference.
    """
    n_threads = nb.get_num_threads()
    nnz = len(vals)
    chunk_size = (nnz + n_threads - 1) // n_threads

    y_local = np.zeros((n_threads, n_rows), dtype=np.complex128)

    for tid in prange(n_threads):
        start = tid * chunk_size
        end = min(nnz, (tid + 1) * chunk_size)
        for idx in range(start, end):
            y_local[tid, rows[idx]] += vals[idx] * x[cols[idx]]

    return np.sum(y_local, axis=0)


@njit(parallel=True, fastmath=True)
def coo_dual_contract(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    psi_S: np.ndarray,
    psi_C: np.ndarray,
    n_rows: int,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dual Hamiltonian contractions in S/C partitioned space.

    Algorithm: Simultaneous forward/adjoint products:
      y_S[i] = Σⱼ H_SC[i,j] · ψ_C[j]    (forward)
      y_C[j] = Σᵢ H_SC*[i,j] · ψ_S[i]   (adjoint, exploiting Hermiticity)

    Complexity: O(nnz/P) per thread, single matrix traversal

    Args:
        rows: Row indices of H_SC (S-space)
        cols: Column indices of H_SC (C-space)
        vals: Non-zero values (complex128)
        psi_S: S-space vector (complex128)
        psi_C: C-space vector (complex128)
        n_rows: S-space dimension
        n_cols: C-space dimension

    Returns:
        (y_S, y_C): Contracted vectors in S-space and C-space

    Note:
        Inputs must be complex128 for Numba dtype inference.
    """
    n_threads = nb.get_num_threads()
    nnz = len(vals)
    chunk_size = (nnz + n_threads - 1) // n_threads

    s_local = np.zeros((n_threads, n_rows), dtype=np.complex128)
    c_local = np.zeros((n_threads, n_cols), dtype=np.complex128)

    for tid in prange(n_threads):
        start = tid * chunk_size
        end = min(nnz, (tid + 1) * chunk_size)
        for idx in range(start, end):
            i, j, h_ij = rows[idx], cols[idx], vals[idx]
            s_local[tid, i] += h_ij * psi_C[j]
            c_local[tid, j] += np.conj(h_ij) * psi_S[i]

    return np.sum(s_local, axis=0), np.sum(c_local, axis=0)


__all__ = ["coo_matvec", "coo_dual_contract"]
