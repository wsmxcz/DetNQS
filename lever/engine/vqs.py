# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational quantum state optimization kernel for deterministic determinant spaces.

Implements geometry tape construction, QGT operations, and mode-specific energy/gradient
kernels (effective/proxy/asymmetric) with JIT-compiled single-step updates.

Core algorithm:
  1. Linearize log ψ(θ) → Jacobian J = ∂_θ log ψ
  2. Compute energy E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ (mode-dependent)
  3. Variance-reduced gradient: ∇E = 2 Re ⟨O_k (E_loc - E)⟩
  4. Optional natural gradient via QGT: S = ⟨O_k O_l⟩ - ⟨O_k⟩⟨O_l⟩

File: lever/engine/vqs.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
import optax

from ..config import ComputeMode
from ..dtypes import GeometryTape, GradResult, InnerState, OuterCtx
from ..utils.jax_utils import tree_dot, tree_scale, tree_sub, tree_add

if TYPE_CHECKING:
    from ..dtypes import PyTree, LogPsiFn


# ============================================================================
# Geometry Tape: Single Linearization of log ψ
# ============================================================================

def prepare_tape(
    params: PyTree,
    logpsi_fn: LogPsiFn,
    num_eps: float = 1e-12,
) -> GeometryTape:
    """
    Build geometry tape via JVP/VJP linearization for deterministic optimization.

    Captures Jacobian O = ∂_θ log ψ and weighted mean ⟨O⟩ for reuse in both
    gradient and QGT operations.

    Args:
        params: Network parameters θ
        logpsi_fn: Forward pass θ → log ψ [n_samples]
        num_eps: Numerical stability threshold for normalization

    Returns:
        GeometryTape with:
          - jvp_fn: Forward mode J @ v
          - vjp_fn: Reverse mode J† @ u
          - log_psi: log ψ(θ) values
          - weights: Normalized |ψ|² weights
          - centered_mean: ⟨O⟩ with global phase fixed
    """
    logpsi_fn = jax.checkpoint(logpsi_fn)
    log_psi, jvp_fn = jax.linearize(logpsi_fn, params)

    # Transpose JVP to obtain VJP
    vjp_fn = jax.linear_transpose(jvp_fn, params)

    # Normalized weights: w_i = |ψ_i|² / Σ|ψ_j|²
    s = 2.0 * jnp.real(log_psi)
    s = s - jnp.max(s)  # Prevent overflow
    weights = jnp.exp(s)
    weights = weights / jnp.maximum(jnp.sum(weights), num_eps)

    # Mean derivative: ⟨O⟩ = J† @ w
    cotangent = jnp.asarray(weights, dtype=log_psi.dtype)
    mean_T = vjp_fn(cotangent)[0]
    centered_mean = jax.tree.map(jnp.conj, mean_T)

    return GeometryTape(
        jvp_fn=jvp_fn,
        vjp_fn=vjp_fn,
        log_psi=log_psi,
        weights=weights,
        centered_mean=centered_mean,
    )


# ============================================================================
# Quantum Geometric Tensor (QGT) Operations
# ============================================================================

def qgt_matvec(
    tape: GeometryTape,
    v: PyTree,
    diag_shift: float = 1e-4,
) -> PyTree:
    """
    QGT matrix-vector product: (S + λI) @ v.

    Centered covariance S_kl = ⟨O_k O_l⟩ - ⟨O_k⟩⟨O_l⟩ computed via:
      S @ v = J† diag(w) J v - ⟨O⟩ (⟨O⟩ · v)

    Args:
        tape: GeometryTape from prepare_tape
        v: Direction in parameter space
        diag_shift: Diagonal regularization λ

    Returns:
        (S + λI) @ v in same PyTree structure as v
    """
    # Forward: u = J @ v
    u = tape.jvp_fn(v)

    # Backward: J† @ (w ⊙ u)
    weighted_u = tape.weights * u
    result_T = tape.vjp_fn(jnp.conj(weighted_u))[0]
    result = jax.tree.map(jnp.conj, result_T)

    # Centering: subtract ⟨O⟩ (⟨O⟩ · v)
    v_dot_mean = tree_dot(tape.centered_mean, v)
    result = tree_sub(result, tree_scale(tape.centered_mean, v_dot_mean))

    # Regularization: λI @ v
    result = tree_add(result, tree_scale(v, diag_shift))
    return result


def qgt_dense(
    tape: GeometryTape,
    diag_shift: float = 1e-4,
    symmetrize: bool = True,
) -> jnp.ndarray:
    """
    Materialize dense QGT matrix S ∈ ℂ^(n_params × n_params).

    Warning:
        Memory intensive O(n_params²). Prefer qgt_matvec with iterative
        solvers (CG, GMRES) for large models.

    Args:
        tape: GeometryTape from prepare_tape
        diag_shift: Diagonal regularization λ
        symmetrize: Enforce Hermitian symmetry (S + S†)/2

    Returns:
        Dense QGT matrix [n_params, n_params]
    """
    flat_mean, unravel_fn = ravel_pytree(tape.centered_mean)
    n_params = flat_mean.size

    # Compute Jacobian columns via JVP with canonical basis
    identity = jnp.eye(n_params, dtype=flat_mean.dtype)
    O = vmap(lambda e: tape.jvp_fn(unravel_fn(e)), in_axes=0)(identity).T

    # Center: O_k - ⟨O_k⟩
    O_centered = O - jnp.conj(flat_mean)[None, :]

    # Weighted covariance: S = (√w ⊙ O)† @ (√w ⊙ O)
    sqrt_w = jnp.sqrt(tape.weights)[:, None]
    weighted_O = sqrt_w * O_centered
    S = weighted_O.conj().T @ weighted_O

    if symmetrize:
        S = (S + S.conj().T) / 2.0
    S = S + diag_shift * jnp.eye(n_params, dtype=S.dtype)
    return S


# ============================================================================
# Energy and Gradient Kernels (Deterministic, No Sampling)
# ============================================================================

def _compute_energy(
    psi: jnp.ndarray,
    n: jnp.ndarray,
    eps: float,
) -> float:
    """
    Rayleigh quotient: E = ⟨ψ|N⟩ / ⟨ψ|ψ⟩.

    Args:
        psi: Wavefunction vector
        n: Hamiltonian contraction N = H @ ψ
        eps: Denominator protection threshold

    Returns:
        Real-valued energy
    """
    num = jnp.vdot(psi, n)
    den = jnp.vdot(psi, psi)
    return (num / jnp.maximum(den, eps)).real


def _safe_division(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """Element-wise division with zero protection."""
    return jnp.where(
        jnp.abs(denominator) >= eps,
        numerator / denominator,
        0.0,
    )


def _compute_gradient(
    tape: GeometryTape,
    e_loc: jnp.ndarray,
    energy: float,
) -> PyTree:
    """
    Variance-reduced gradient via single VJP:
        ∇E = 2 Re ⟨O_k (E_loc - E)⟩

    Args:
        tape: GeometryTape with VJP closure
        e_loc: Local energies E_loc = N / ψ
        energy: Global energy ⟨E_loc⟩

    Returns:
        Gradient PyTree matching parameter structure
    """
    cotangent = tape.weights * (e_loc - energy)
    cotangent = jnp.asarray(cotangent, dtype=tape.log_psi.dtype)

    (grad_conj,) = tape.vjp_fn(jnp.conj(cotangent))
    return jax.tree.map(jnp.conj, grad_conj)


def _unpack_contractions(result):
    """
    Unpack Hamiltonian contractions from Contractions namedtuple or raw tuple.

    Returns:
        (n_ss, n_sc, n_cs, n_cc) - all may be None
    """
    if hasattr(result, "n_ss"):
        return result.n_ss, result.n_sc, result.n_cs, result.n_cc
    return result


def _effective_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float,
) -> GradResult:
    """
    Effective S-space Hamiltonian kernel:
        E = ⟨ψ_S|H_eff|ψ_S⟩ / ‖ψ_S‖²

    Args:
        params: Network parameters (unused, kept for interface consistency)
        tape: GeometryTape with log ψ and weights
        ctx: OuterCtx with spmv_fn and e_nuc
        num_eps: Numerical stability threshold

    Returns:
        GradResult(grad, energy)
    """
    psi_s = jnp.exp(tape.log_psi)

    result = ctx.spmv_fn(psi_s)
    n_ss = result.n_ss if hasattr(result, "n_ss") else result[0]

    energy = _compute_energy(psi_s, n_ss, num_eps)
    e_loc = _safe_division(n_ss, psi_s, num_eps)
    gradient = _compute_gradient(tape, e_loc, energy)

    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


def _proxy_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float,
) -> GradResult:
    """
    Full T-space proxy Hamiltonian kernel:
        ψ = [ψ_S; ψ_C]
        E = (⟨ψ_S|N_S⟩ + ⟨ψ_C|N_C⟩) / (‖ψ_S‖² + ‖ψ_C‖²)

    Args:
        params: Network parameters (unused)
        tape: GeometryTape with log ψ for full T-space
        ctx: OuterCtx with spmv_fn and space dimensions
        num_eps: Numerical stability threshold

    Returns:
        GradResult(grad, energy)
    """
    n_s = ctx.space.n_s
    psi_all = jnp.exp(tape.log_psi)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]

    result = ctx.spmv_fn(psi_s, psi_c)
    n_ss, n_sc, n_cs, n_cc = _unpack_contractions(result)

    ns_total = n_ss + n_sc
    nc_total = n_cs + n_cc

    num_s = jnp.vdot(psi_s, ns_total)
    num_c = jnp.vdot(psi_c, nc_total)
    den = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
    energy = ((num_s + num_c) / jnp.maximum(den, num_eps)).real

    e_loc_s = _safe_division(ns_total, psi_s, num_eps)
    e_loc_c = _safe_division(nc_total, psi_c, num_eps)
    e_loc_all = jnp.concatenate([e_loc_s, e_loc_c])

    gradient = _compute_gradient(tape, e_loc_all, energy)
    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


def _asymmetric_kernel(
    params: PyTree,
    tape: GeometryTape,
    ctx: OuterCtx,
    num_eps: float,
) -> GradResult:
    """
    Asymmetric contraction kernel:
        E = ⟨ψ_S|H ψ⟩ / ‖ψ_S‖²
    
    Normalization only on S-space, but Hamiltonian acts on full T-space.

    Args:
        params: Network parameters (unused)
        tape: GeometryTape with log ψ for full T-space
        ctx: OuterCtx with spmv_fn and space dimensions
        num_eps: Numerical stability threshold

    Returns:
        GradResult(grad, energy)
    """
    n_s = ctx.space.n_s
    psi_all = jnp.exp(tape.log_psi)
    psi_s, psi_c = psi_all[:n_s], psi_all[n_s:]

    result = ctx.spmv_fn(psi_s, psi_c)
    n_ss, n_sc, _, _ = _unpack_contractions(result)

    ns_total = n_ss + n_sc
    energy = _compute_energy(psi_s, ns_total, num_eps)

    e_loc = _safe_division(ns_total, psi_s, num_eps)
    gradient = _compute_gradient(tape, e_loc, energy)

    return GradResult(grad=gradient, energy=energy + ctx.e_nuc)


# ============================================================================
# JIT-Compiled Single-Step Kernel
# ============================================================================

def create_step_kernel(
    ctx: OuterCtx,
    optimizer: optax.GradientTransformation | object,
    num_eps: float = 1e-12,
) -> Callable[
    [InnerState, jnp.ndarray, jnp.ndarray],
    tuple[InnerState, jnp.ndarray, jnp.ndarray],
]:
    """
    Build JIT-compiled single optimization step with mode-specific energy kernel.

    Workflow:
      1. Linearize log ψ(θ) → geometry tape
      2. Compute energy and gradient (mode-dependent)
      3. Apply optimizer update
      4. Return new state, energy, and log ψ

    Args:
        ctx: OuterCtx with compute_mode, spmv_fn, log_psi evaluator
        optimizer: Optax or LEVER natural gradient optimizer
        num_eps: Numerical stability threshold

    Returns:
        step(state, feat_s, feat_c) → (new_state, energy, log_psi)
    """
    from ..optimizers.base import Optimizer as LeverOptimizer

    # Select kernel based on compute mode
    mode = ctx.compute_mode
    if mode is ComputeMode.EFFECTIVE:
        energy_kernel = _effective_kernel
    elif mode is ComputeMode.PROXY:
        energy_kernel = _proxy_kernel
    elif mode is ComputeMode.ASYMMETRIC:
        energy_kernel = _asymmetric_kernel
    else:
        raise ValueError(f"Unknown compute mode: {mode}")

    is_natural_grad = isinstance(optimizer, LeverOptimizer)
    log_eval = ctx.log_psi

    def _step(
        state: InnerState,
        feat_s: jnp.ndarray,
        feat_c: jnp.ndarray,
    ) -> tuple[InnerState, jnp.ndarray, jnp.ndarray]:
        """Single deterministic update step: θ_t → θ_{t+1}."""
        # Parameter-only view for current determinant space
        logpsi_fn = log_eval.for_tape(feat_s, feat_c)

        # Geometry tape via linearization
        tape = prepare_tape(state.params, logpsi_fn, num_eps)

        # Energy and gradient in selected compute mode
        grad_result = energy_kernel(state.params, tape, ctx, num_eps)

        # Optimizer update (natural gradient may consume tape)
        if is_natural_grad:
            updates, new_opt_state = optimizer.update(
                grad_result.grad,
                state.opt_state,
                state.params,
                tape=tape,
                energy=grad_result.energy,
            )
        else:
            updates, new_opt_state = optimizer.update(
                grad_result.grad,
                state.opt_state,
                state.params,
            )

        new_params = optax.apply_updates(state.params, updates)
        new_state = InnerState(
            params=new_params,
            opt_state=new_opt_state,
            step=state.step + 1,
        )

        # Return log ψ from tape to avoid redundant forward pass
        return new_state, grad_result.energy, tape.log_psi

    # Donate state for better memory reuse
    return jax.jit(_step, donate_argnums=(0,))


__all__ = [
    "prepare_tape",
    "qgt_matvec",
    "qgt_dense",
    "create_step_kernel",
]
