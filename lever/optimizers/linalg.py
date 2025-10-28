# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JIT-compatible linear algebra solvers for optimizer direction computation.

Provides PyTree-aware wrappers around JAX's native solvers with enhanced
diagnostics and numerical stability controls.

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax.numpy as jnp
import jax.scipy.linalg
import jax.scipy.sparse.linalg
from jax.flatten_util import ravel_pytree

if TYPE_CHECKING:
    from ..utils.dtypes import PyTree


def solve_cg(
    matvec_fn: Callable[[PyTree], PyTree],
    b: PyTree,
    x0: PyTree | None = None,
    maxiter: int = 100,
    tol: float = 1e-5
) -> tuple[PyTree, dict]:
    """
    Solve A @ x = b via conjugate gradient method.
    
    Algorithm: Iterative Krylov subspace method for SPD systems.
    Complexity: O(k·n) for k iterations, n unknowns.
    
    Args:
        matvec_fn: Matrix-vector product A @ v (PyTree → PyTree)
        b: Right-hand side (PyTree)
        x0: Initial guess (defaults to zero)
        maxiter: Maximum CG iterations
        tol: Relative residual tolerance ||r||/||b|| < tol
    
    Returns:
        (x, info): Solution and diagnostics with keys:
          - 'niter': Iteration count (upper bound from JAX)
          - 'residual': Final residual norm ||A@x - b||
          - 'converged': Whether tolerance criterion met
    """
    # Flatten PyTree to vector space
    b_flat, unravel_fn = ravel_pytree(b)
    
    x0_flat = jnp.zeros_like(b_flat) if x0 is None else ravel_pytree(x0)[0]
    
    # Wrap matvec for flat arrays
    def flat_matvec(v_flat: jnp.ndarray) -> jnp.ndarray:
        v_tree = unravel_fn(v_flat)
        result_tree = matvec_fn(v_tree)
        return ravel_pytree(result_tree)[0]
    
    # JAX CG solver
    x_flat, info_raw = jax.scipy.sparse.linalg.cg(
        A=flat_matvec,
        b=b_flat,
        x0=x0_flat,
        maxiter=maxiter,
        tol=tol
    )
    
    # Compute diagnostics
    residual_norm = jnp.linalg.norm(flat_matvec(x_flat) - b_flat)
    b_norm = jnp.linalg.norm(b_flat)
    
    info = {
        'niter': maxiter,  # Upper bound (JAX doesn't expose actual count)
        'residual': residual_norm,
        'converged': residual_norm < tol * b_norm
    }
    
    return unravel_fn(x_flat), info


def solve_cholesky(
    A: jnp.ndarray,
    b: PyTree,
    lower: bool = True,
    check_finite: bool = False
) -> PyTree:
    """
    Solve A @ x = b via Cholesky decomposition for SPD matrix A.
    
    Algorithm: Two-stage factorization A = L·L^T, then solves L·y = b
    and L^T·x = y sequentially.
    
    Complexity: O(n³) factorization + O(n²) solve.
    Memory: O(n²) for dense A.
    
    Args:
        A: Symmetric positive definite matrix [n×n]
        b: Right-hand side (PyTree with n total elements)
        lower: Use lower triangular factor (numerically preferred)
        check_finite: Enable finite value checks (debug mode)
    
    Returns:
        Solution x (PyTree matching b structure)
    
    Raises:
        ValueError: If A not positive definite or dimension mismatch
    
    Example:
        >>> A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        >>> b = {'w': jnp.array([1.0, 2.0])}
        >>> x = solve_cholesky(A, b)
    """
    b_flat, unravel_fn = ravel_pytree(b)
    
    if A.shape[0] != b_flat.size:
        raise ValueError(
            f"Dimension mismatch: A is {A.shape[0]}×{A.shape[0]}, "
            f"b has {b_flat.size} elements"
        )
    
    # Cholesky factorization: A = L·L^T
    try:
        L, lower_flag = jax.scipy.linalg.cho_factor(
            A, lower=lower, check_finite=check_finite
        )
    except jnp.linalg.LinAlgError as e:
        raise ValueError(
            "Cholesky failed: A not positive definite. "
            "Try increasing diagonal regularization."
        ) from e
    
    # Solve via triangular substitutions
    x_flat = jax.scipy.linalg.cho_solve((L, lower_flag), b_flat)
    
    return unravel_fn(x_flat)


__all__ = ["solve_cg", "solve_cholesky"]
