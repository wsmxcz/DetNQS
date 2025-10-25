# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational energy and gradient computations for quantum many-body systems.

Implements pure functions for calculating observables. Supports multiple
computation modes (ASYMMETRIC, PROXY, EFFECTIVE) with mode-specific logic
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
      ASYMMETRIC: E = ⟨ψ_S|(H_SS + H_SC)|ψ⟩ / ||ψ_S||²
                  Full S-C coupling with S-space normalization.
                  Non-Hermitian functional—lacks "force terms" from ∂_θψ_C.
    
      PROXY:      E = ⟨ψ|H̃|ψ⟩ / ||ψ||²
                  where H̃ = [[H_SS, H_SC], [H_CS, H_CC^diag]]
                  Full-space Rayleigh quotient over S∪C with diagonal C approximation.
    
      EFFECTIVE:  E = ⟨ψ_S|H_eff|ψ_S⟩ / ||ψ_S||²
                  Downfolded Hamiltonian via Schur complement (energy-independent).
                  Efficient for small S-space; no runtime S-C coupling.
  
    Args:
        evaluator: Lazy evaluation context with cached Ĥψ contractions
  
    Returns:
        Scalar energy (real part)
    """

    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions
    config = evaluator.config

    # S-space contribution: ⟨ψ_S| (H_SS + H_SC) |ψ⟩
    # Note: In EFFECTIVE mode, H_SS is actually H_eff and N_SC = 0
    num_S = jnp.vdot(ψ_S, N.N_SS + N.N_SC)

    if config.energy_mode in {EnergyMode.ASYMMETRIC, EnergyMode.EFFECTIVE}:
        # Both modes: normalize over S-space only
        # ASYMMETRIC uses H_SS + H_SC; EFFECTIVE uses H_eff (with N_SC=0)
        denom = jnp.vdot(ψ_S, ψ_S)
        return (num_S / jnp.maximum(denom, config.epsilon)).real

    if config.energy_mode == EnergyMode.PROXY:
        # C-space contribution with diagonal approximation: ⟨ψ_C| (H_CS + H_diag) |ψ_C⟩
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
        For EFFECTIVE mode, returns only E_loc,S (C-space not used)
    """
    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions
    config = evaluator.config

    # S-space: E_loc,S = (H_SS@ψ_S + H_SC@ψ_C) / ψ_S
    # Note: In EFFECTIVE mode, N.N_SC = 0
    num_S = N.N_SS + N.N_SC
    E_loc_S = jnp.where(jnp.abs(ψ_S) >= config.epsilon, num_S / ψ_S, 0.0)

    # EFFECTIVE mode: skip C-space computation (not used in energy/gradient)
    if config.grad_mode == GradMode.EFFECTIVE:
        return E_loc_S

    # C-space: E_loc,C = (H_CS@ψ_S + H_diag⊙ψ_C) / ψ_C
    num_C = N.N_CS + N.N_CC
    E_loc_C = jnp.where(jnp.abs(ψ_C) >= config.epsilon, num_C / ψ_C, 0.0)

    return jnp.concatenate([E_loc_S, E_loc_C])


def compute_energy_and_gradient(evaluator: Evaluator) -> GradientResult:
    """
    Compute variational gradient ∇_θ E via reverse-mode differentiation.
    
    Algorithm (VJP-based gradient):
      1. Forward: log(ψ(θ)) → E_loc[i]
      2. Mode-specific weights: w[i] = exp(2·Re[log ψ]) / Σ exp(2·Re[log ψ])
         Normalization domain aligned with energy mode (S-only vs S∪C)
      3. Energy: E computed identically to compute_energy()
      4. Cotangent: c[i] = w[i]·(E_loc[i] - E)
         Weights and E_loc treated as constants (no gradient flow) for VJP
      5. Backward: ∇_θ E = conj(VJP(conj(log ψ), conj(c)))
    
    Gradient modes:
      ASYMMETRIC: S-space normalization (||ψ_S||²)
                  Omits non-Hermitian "force terms" from ∂_θψ_C via H_SC
                  Faster but lacks full variational consistency
      
      EFFECTIVE:  S-space normalization (||ψ_S||²)
                  Uses downfolded H_eff; no runtime S-C coupling
      
      PROXY:      Full-space normalization (||ψ_S||² + ||ψ_C||²)
                  Diagonal H_CC approximation; balanced accuracy/cost
    
    Args:
        evaluator: Lazy evaluation context
    
    Returns:
        GradientResult(gradient=∇_θ E, energy_elec=E)
    """
    config = evaluator.config
    n_S, n_C = evaluator.space.size_S, evaluator.space.size_C
    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions

    # Forward pass with VJP setup
    # vjp_fn computes gradients w.r.t. params only; weights/E_loc are constants
    batch_logpsi_fn = evaluator.get_batch_logpsi_fn()
    log_all, vjp_fn = jax.vjp(batch_logpsi_fn, evaluator.params)

    # Compute local energies for all determinants
    # These serve as constants in the cotangent vector construction
    E_loc_all = compute_local_energy(evaluator)

    # Mode-specific energy and cotangent construction
    if config.grad_mode in {GradMode.ASYMMETRIC, GradMode.EFFECTIVE}:
        # S-space only: weights and energy consistent with compute_energy()
        log_S = log_all[:n_S]
        
        # Weights w_S = exp(2·Re[log ψ_S]) / Σ exp(2·Re[log ψ_S])
        # Treated as constants (no gradient) when constructing cotangent
        weights_raw_S = jnp.exp(2.0 * jnp.real(log_S))
        weights_S = weights_raw_S / jnp.maximum(
            jnp.sum(weights_raw_S), config.epsilon
        )

        # Energy: E = ⟨ψ_S|(H_SS + H_SC)|ψ⟩ / ||ψ_S||²
        # Note: N.N_SC = 0 for EFFECTIVE mode
        num_S = jnp.vdot(ψ_S, N.N_SS + N.N_SC)
        denom_S = jnp.vdot(ψ_S, ψ_S)
        E_mode = (num_S / jnp.maximum(denom_S, config.epsilon)).real

        # Cotangent vector: c_S = w_S·(E_loc,S - E), c_C = 0
        # Weights and (E_loc - E) are constants for VJP backward pass
        cot_S = weights_S * (E_loc_all[:n_S] - E_mode)
        cot_full = jnp.concatenate([cot_S, jnp.zeros(n_C, dtype=cot_S.dtype)])
        E_report = E_mode

    elif config.grad_mode == GradMode.PROXY:
        # Full-space weights and energy (diagonal H_CC approximation)
        
        # Weights w = exp(2·Re[log ψ]) / Σ exp(2·Re[log ψ]) over S∪C
        # Treated as constants in cotangent construction
        weights_raw = jnp.exp(2.0 * jnp.real(log_all))
        weights_full = weights_raw / jnp.maximum(
            jnp.sum(weights_raw), config.epsilon
        )

        # Energy: E = (⟨ψ_S|H|ψ⟩ + ⟨ψ_C|H_diag|ψ_C⟩) / (||ψ_S||² + ||ψ_C||²)
        num_S = jnp.vdot(ψ_S, N.N_SS + N.N_SC)
        num_C = jnp.vdot(ψ_C, N.N_CS + N.N_CC)
        denom = jnp.vdot(ψ_S, ψ_S) + jnp.vdot(ψ_C, ψ_C)
        E_mode = ((num_S + num_C) / jnp.maximum(denom, config.epsilon)).real

        # Cotangent: c = w·(E_loc - E) over full space
        # Weights and (E_loc - E) are constants for VJP
        cot_full = weights_full * (E_loc_all - E_mode)
        E_report = E_mode

    else:
        raise ValueError(f"Unknown gradient mode: {config.grad_mode}")

    # Backward pass: ∇_θ E = conj(VJP(conj(log ψ), conj(cotangent)))
    # VJP only differentiates through log_ψ(θ); cotangent is treated as constant
    # Complex conjugation handles Wirtinger derivatives for real-valued E
    (grad_conj,) = vjp_fn(jnp.conj(cot_full))
    grad = jax.tree.map(jnp.conj, grad_conj)

    return GradientResult(gradient=grad, energy_elec=E_report)


__all__ = ["compute_energy", "compute_local_energy", "compute_energy_and_gradient"]