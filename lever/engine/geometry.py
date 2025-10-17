# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quantum geometry computations for the LEVER engine.

This module provides pure functions to compute the Quantum Geometric Tensor (QGT)
and related quantities for advanced second-order optimization methods like
Stochastic Reconfiguration (SR) and the Linear Method (LM).

File: lever/engine/geometry.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025 (Refactored)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from .physics import compute_energy_and_gradient, compute_local_energy
from .utils import SOperatorMetadata, SubspaceMetadata

if TYPE_CHECKING:
    from .evaluator import Evaluator
    from .utils import PyTree, MatVecOp


# --- Core Geometry Functions ---

def compute_S_matvec(evaluator: Evaluator) -> tuple[MatVecOp, SOperatorMetadata]:
    """
    Constructs a reusable operator for the Quantum Geometric Tensor (S) action.

    This function returns a JIT-compatible function that computes the matrix-vector
    product S·v for a given vector v (as a PyTree). This is the core component
    for solving the natural gradient linear system S·δθ = -g via iterative methods.

    The action S·v is computed efficiently without materializing the full S matrix,
    using a combination of Jacobian-vector (JVP) and vector-Jacobian (VJP) products.

    Args:
        evaluator: The lazy evaluation context for the current step.

    Returns:
        A tuple containing:
        - S_operator: A function `(v: PyTree) -> PyTree` for the S·v product.
        - metadata: A NamedTuple with diagnostic information.
    """
    params = evaluator.params
    config = evaluator.config

    # Get the unified log-amplitude function for linearization.
    batch_logpsi_fn = evaluator.get_batch_logpsi_fn()

    # Pre-linearize the model once to get efficient JVP and VJP functions.
    _, jvp_fn = jax.linearize(batch_logpsi_fn, params)
    _, vjp_fn = jax.vjp(batch_logpsi_fn, params)

    # Pre-compute probability weights |ψ|² / ||ψ||².
    log_all = evaluator.logpsi_all
    weights_raw = jnp.exp(2.0 * jnp.real(log_all))
    weights = weights_raw / jnp.maximum(jnp.sum(weights_raw), config.epsilon)

    @jax.jit
    def S_operator(direction: PyTree) -> PyTree:
        """Computes S·v via one JVP and one VJP call."""
        # 1. Forward pass: Compute the directional derivative d(v) = (J·v).
        directional_derivs = jvp_fn(direction)

        # 2. Center the derivatives: Δd = d - ⟨d⟩.
        mean_deriv = jnp.sum(weights * directional_derivs)
        centered_derivs = directional_derivs - mean_deriv

        # 3. Backward pass: Project back to parameter space: S·v = Jᵀ·(w * Δd*).
        cotangents = weights * jnp.conj(centered_derivs)
        (S_times_v,) = vjp_fn(cotangents)

        # 4. S is Hermitian, but for real parameter updates we take the real part.
        #    The imaginary part should be zero for real models.
        return jax.tree.map(lambda x: x.real, S_times_v)

    # --- Diagnostics ---
    # Compute diagnostics outside the JIT-compiled path.
    test_deriv = jvp_fn(jax.tree.map(jnp.ones_like, params))
    mean_test = jnp.sum(weights * test_deriv)
    total_norm = float(jnp.sum(weights_raw))

    metadata = SOperatorMetadata(
        total_norm=total_norm,
        mean_deriv_residual=float(jnp.abs(mean_test)),
    )

    return S_operator, metadata


def compute_subspace_matrices(
    evaluator: Evaluator,
    basis: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, SubspaceMetadata]:
    """
    Projects the QGT, Hamiltonian, and gradient onto a low-dimensional subspace.

    This is the core computation for the Linear Method, which solves the optimization
    problem in a subspace spanned by a basis (e.g., gradient, previous updates).

    Args:
        evaluator: The lazy evaluation context.
        basis: A matrix of shape (K, P) where K is the subspace dimension and
               P is the number of flattened parameters. Each row is a basis vector.

    Returns:
        A tuple containing:
        - S_bar: The (K, K) projected QGT matrix.
        - H_bar: The (K, K) projected energy Hessian matrix.
        - g_bar: The (K,) projected gradient vector.
        - metadata: A NamedTuple with conditioning diagnostics for S_bar.
    """
    params = evaluator.params
    config = evaluator.config
    epsilon = config.epsilon

    # --- Prepare necessary quantities ---
    batch_logpsi_fn = evaluator.get_batch_logpsi_fn()
    params_flat, unravel_fn = ravel_pytree(params)

    if basis.shape[1] != params_flat.size:
        raise ValueError(f"Basis dimension mismatch: basis P={basis.shape[1]}, "
                         f"params P={params_flat.size}")

    # Compute weights, local energies, and the reference energy.
    log_all = evaluator.logpsi_all
    weights_raw = jnp.exp(2.0 * jnp.real(log_all))
    weights = weights_raw / jnp.maximum(jnp.sum(weights_raw), epsilon)
    
    eloc = compute_local_energy(evaluator)
    energy_ref = jnp.vdot(weights, jnp.real(eloc))
    eloc_centered = eloc - energy_ref

    # --- Compute directional derivatives for the entire basis ---
    def get_tangent(basis_vec_flat: jnp.ndarray) -> jnp.ndarray:
        """Computes J·v for a single flattened basis vector."""
        basis_vec_pytree = unravel_fn(basis_vec_flat)
        _, tangent = jax.jvp(batch_logpsi_fn, (params,), (basis_vec_pytree,))
        return tangent

    # Vectorize the JVP computation over all K basis vectors.
    derivs = jax.vmap(get_tangent)(basis)  # Shape: (K, N_total)

    # Center the derivatives.
    mean_derivs = jnp.sum(weights[None, :] * derivs, axis=1, keepdims=True)
    centered_derivs = derivs - mean_derivs

    # --- Construct subspace matrices via efficient weighted outer products ---
    sqrt_weights = jnp.sqrt(weights)
    weighted_derivs = sqrt_weights[None, :] * centered_derivs  # Shape: (K, N_total)

    # S̄_kl = ⟨Δd_k* | Δd_l⟩_w
    S_bar = weighted_derivs @ weighted_derivs.T.conj()

    # ḡ_k = ⟨Δd_k* | (E_loc - E)⟩_w
    energy_term = sqrt_weights * eloc_centered
    g_bar = weighted_derivs @ jnp.conj(energy_term)

    # H̄_kl = ⟨Δd_k* | (E_loc - E) | Δd_l⟩_w
    H_bar = (weighted_derivs.conj() * energy_term[None, :]) @ centered_derivs.T

    # --- Hermitization and Regularization ---
    S_bar = 0.5 * (S_bar + S_bar.T.conj())
    H_bar = 0.5 * (H_bar + H_bar.T.conj())
    S_bar_reg = S_bar + epsilon * jnp.eye(S_bar.shape[0], dtype=S_bar.dtype)

    # --- Diagnostics ---
    eigvals_S = jnp.linalg.eigvalsh(S_bar_reg)
    min_eig, max_eig = jnp.min(eigvals_S), jnp.max(eigvals_S)
    cond_num = max_eig / jnp.maximum(min_eig, 1e-16)

    metadata = SubspaceMetadata(
        subspace_dim=basis.shape[0],
        condition_number=float(cond_num),
        min_eigenvalue=float(min_eig),
        jitter=epsilon,
    )
    
    # Return real parts, as is standard for the Linear Method solver.
    return S_bar_reg.real, H_bar.real, g_bar.real, metadata