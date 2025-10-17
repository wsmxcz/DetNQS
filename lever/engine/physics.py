# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational energy and gradient computations for quantum many-body systems.

Implements pure functions for calculating observables. Supports multiple
computation modes (ASYMMETRIC, PROXY, FULL) with mode-specific logic
encapsulated in the Evaluator.

File: lever/engine/physics.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .config import EnergyMode, GradMode
from .utils import GradientResult

if TYPE_CHECKING:
    from .evaluator import Evaluator
    from .utils import PyTree


def compute_energy(evaluator: Evaluator) -> jnp.ndarray:
    """
    Compute variational energy via Rayleigh quotient: E = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩.
    
    Mode-specific formulas:
      ASYMMETRIC: E = ⟨ψ_S|Ĥ|ψ⟩ / ||ψ_S||²
      PROXY:      E = (⟨ψ_S|Ĥ|ψ⟩ + ⟨ψ_C|Ĥ_diag|ψ_C⟩) / (||ψ_S||² + ||ψ_C||²)
      FULL:       E = (⟨ψ_S|Ĥ|ψ⟩ + ⟨ψ_C|Ĥ|ψ_C⟩) / (||ψ_S||² + ||ψ_C||²)
    
    Args:
        evaluator: Lazy evaluation context with cached Ĥψ contractions
    
    Returns:
        Scalar energy (real part)
    """
    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions
    config = evaluator.config

    # S-space contribution: ⟨ψ_S| (H_SS + H_SC) |ψ⟩
    num_S = jnp.vdot(ψ_S, N.N_SS + N.N_SC)

    if config.energy_mode == EnergyMode.ASYMMETRIC:
        denom = jnp.vdot(ψ_S, ψ_S)
        return (num_S / jnp.maximum(denom, config.epsilon)).real

    if config.energy_mode in {EnergyMode.PROXY, EnergyMode.FULL}:
        # C-space contribution: ⟨ψ_C| (H_CS + H_CC) |ψ⟩
        # Note: H_CC is diagonal in PROXY, full matrix in FULL
        num_C = jnp.vdot(ψ_C, N.N_CS + N.N_CC)
        denom = jnp.vdot(ψ_S, ψ_S) + jnp.vdot(ψ_C, ψ_C)
        return ((num_S + num_C) / jnp.maximum(denom, config.epsilon)).real

    raise ValueError(f"Unknown energy mode: {config.energy_mode}")


def compute_local_energy(evaluator: Evaluator) -> jnp.ndarray:
    """
    Compute local energy E_loc[i] = (Ĥψ)[i] / ψ[i] for all determinants.
    
    Handles singular points |ψ[i]| < ε via zero assignment to avoid
    numerical instabilities.
    
    Args:
        evaluator: Lazy evaluation context
    
    Returns:
        Concatenated local energies [E_loc,S; E_loc,C] of shape (n_S + n_C,)
    """
    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions
    config = evaluator.config

    # S-space: E_loc,S = (H_SS@ψ_S + H_SC@ψ_C) / ψ_S
    num_S = N.N_SS + N.N_SC
    E_loc_S = jnp.where(
        jnp.abs(ψ_S) >= config.epsilon,
        num_S / ψ_S,
        0.0
    )

    # C-space: E_loc,C = (H_CS@ψ_S + H_CC@ψ_C) / ψ_C
    # Note: H_CC@ψ_C is diagonal approximation in PROXY, exact in FULL
    num_C = N.N_CS + N.N_CC
    E_loc_C = jnp.where(
        jnp.abs(ψ_C) >= config.epsilon,
        num_C / ψ_C,
        0.0
    )

    return jnp.concatenate([E_loc_S, E_loc_C])


def compute_energy_and_gradient(evaluator: Evaluator) -> GradientResult:
    """
    Compute variational gradient ∇_θ E via reverse-mode differentiation.
    
    Algorithm (VJP-based gradient):
      1. Forward: log(ψ(θ)) → E_loc[i]
      2. Weights: w[i] = |ψ[i]|² / Σ_j|ψ[j]|²
      3. Energy:  E = Σ_i w[i]·E_loc[i]
      4. Cotangent: c[i] = w[i]·(E_loc[i] - E)
      5. Backward: ∇_θ E = VJP(log(ψ)*, c*)
    
    Gradient mode:
      ASYMMETRIC: Renormalize weights over S-space only
      PROXY/FULL: Use full-space weights (C-space treated differently internally)
    
    Args:
        evaluator: Lazy evaluation context
    
    Returns:
        GradientResult(gradient=∇_θ E, energy_elec=E)
    """
    config = evaluator.config
    n_S, n_C = evaluator.space.size_S, evaluator.space.size_C

    # Forward pass with VJP setup
    batch_logpsi_fn = evaluator.get_batch_logpsi_fn()
    log_all, vjp_fn = jax.vjp(batch_logpsi_fn, evaluator.params)

    # Compute local energies for all determinants
    E_loc_all = compute_local_energy(evaluator)

    # Probability weights: w[i] = |ψ[i]|² / Σ_j|ψ[j]|²
    weights_raw = jnp.exp(2.0 * jnp.real(log_all))
    weights_full = weights_raw / jnp.maximum(
        jnp.sum(weights_raw), config.epsilon
    )
    E_total = jnp.vdot(weights_full, jnp.real(E_loc_all))

    # Construct cotangent vector based on gradient mode
    if config.grad_mode == GradMode.ASYMMETRIC:
        # Renormalize weights over S-space only
        log_S = log_all[:n_S]
        weights_raw_S = jnp.exp(2.0 * jnp.real(log_S))
        weights_S = weights_raw_S / jnp.maximum(
            jnp.sum(weights_raw_S), config.epsilon
        )
        
        cot_S = weights_S * (E_loc_all[:n_S] - E_total)
        cot_full = jnp.concatenate([
            cot_S,
            jnp.zeros(n_C, dtype=cot_S.dtype)
        ])

    elif config.grad_mode in {GradMode.PROXY, GradMode.FULL}:
        # Use full-space weights
        # Note: PROXY uses diagonal H_CC, FULL uses complete H_CC
        cot_full = weights_full * (E_loc_all - E_total)

    else:
        raise ValueError(f"Unknown gradient mode: {config.grad_mode}")

    # Backward pass: ∇_θ E = conj(VJP(conj(log ψ), conj(cotangent)))
    (grad_conj,) = vjp_fn(jnp.conj(cot_full))
    grad = jax.tree.map(jnp.conj, grad_conj)

    return GradientResult(gradient=grad, energy_elec=E_total)


__all__ = ["compute_energy", "compute_local_energy", "compute_energy_and_gradient"]
