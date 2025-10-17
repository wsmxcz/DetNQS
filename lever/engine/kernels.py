# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-performance Numba kernels for sparse matrix operations.

Implements parallel COO sparse matrix-vector products using thread-local
accumulation to avoid race conditions while maximizing throughput.

File: lever/engine/kernels.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
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
    
    Uses thread-local accumulation with reduction to avoid race conditions.
    Work is statically distributed across threads for load balancing.
    
    Args:
        rows: Row indices of nonzeros
        cols: Column indices of nonzeros
        vals: Nonzero values
        x: Dense input vector
        n_rows: Matrix row count
        
    Returns:
        Dense output vector y
    """
    n_threads = nb.get_num_threads()
    nnz = len(vals)
    chunk_size = (nnz + n_threads - 1) // n_threads
    
    # Thread-local accumulation buffers
    y_local = np.zeros((n_threads, n_rows), dtype=x.dtype)

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
    Dual Hamiltonian contractions: (H_SC @ psi_C, H_CS @ psi_S).
    
    Computes both products in one pass by exploiting Hermitian symmetry:
    H_CS = H_SC†. Single traversal of nonzeros reduces memory traffic.
    
    Args:
        rows: Row indices (S-space dimension)
        cols: Column indices (C-space dimension)
        vals: H_SC matrix elements
        psi_S: S-space amplitudes
        psi_C: C-space amplitudes
        n_rows: S-space size
        n_cols: C-space size
        
    Returns:
        (H_SC @ psi_C, H_CS @ psi_S)
    """
    n_threads = nb.get_num_threads()
    nnz = len(vals)
    chunk_size = (nnz + n_threads - 1) // n_threads

    # Separate buffers for S and C contractions
    s_local = np.zeros((n_threads, n_rows), dtype=np.complex128)
    c_local = np.zeros((n_threads, n_cols), dtype=np.complex128)

    for tid in prange(n_threads):
        start = tid * chunk_size
        end = min(nnz, (tid + 1) * chunk_size)
        for idx in range(start, end):
            i, j, h_ij = rows[idx], cols[idx], vals[idx]
            # Forward: H_SC @ psi_C
            s_local[tid, i] += h_ij * psi_C[j]
            # Adjoint: H_CS @ psi_S = (H_SC†) @ psi_S
            c_local[tid, j] += np.conjugate(h_ij) * psi_S[i]

    return np.sum(s_local, axis=0), np.sum(c_local, axis=0)


__all__ = ["coo_matvec", "coo_dual_contract"]
