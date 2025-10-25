# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational energy and gradient computations for LEVER.

Implements three computation modes:
  - ASYMMETRIC: E = ⟨ψ_S|(H_SS + H_SC·ψ_C)|ψ_S⟩ / ||ψ_S||²
  - PROXY:      E = ⟨ψ|H̃|ψ⟩ / ||ψ||² with proxy Hamiltonian
  - EFFECTIVE:  E = ⟨ψ_S|H_eff|ψ_S⟩ / ||ψ_S||² using effective H

Gradient computation via VJP with variance-optimal weighting.

File: lever/engine/physics.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..config import ComputeMode
from .utils import GradientResult

if TYPE_CHECKING:
    from .evaluator import Evaluator
    from .utils import PyTree


def compute_energy(evaluator: Evaluator) -> jnp.ndarray:
    """
    Compute variational energy E = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩.

    Args:
        evaluator: Evaluator instance with wavefunction and contractions.

    Returns:
        Real-valued energy scalar.

    Algorithm:
        - ASYMMETRIC/EFFECTIVE: S-space normalized
        - PROXY: Full-space normalized with proxy Hamiltonian
    """
    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions
    mode = evaluator.config.compute_mode
    eps = evaluator.config.epsilon

    # Numerator: ⟨ψ|Ĥ|ψ⟩
    if mode == ComputeMode.EFFECTIVE:
        num_S = jnp.vdot(ψ_S, N.N_SS)
    else:
        num_S = jnp.vdot(ψ_S, N.N_SS + N.N_SC)

    # Denominator: ⟨ψ|ψ⟩
    if mode in {ComputeMode.ASYMMETRIC, ComputeMode.EFFECTIVE}:
        denom = jnp.vdot(ψ_S, ψ_S)
        return (num_S / jnp.maximum(denom, eps)).real

    if mode == ComputeMode.PROXY:
        num_C = jnp.vdot(ψ_C, N.N_CS + N.N_CC)
        denom = jnp.vdot(ψ_S, ψ_S) + jnp.vdot(ψ_C, ψ_C)
        return ((num_S + num_C) / jnp.maximum(denom, eps)).real

    raise ValueError(f"Unknown compute mode: {mode}")


def compute_local_energy(evaluator: Evaluator) -> jnp.ndarray:
    """
    Compute local energy E_loc[i] = (Ĥψ)[i] / ψ[i].

    Args:
        evaluator: Evaluator instance.

    Returns:
        Local energy array:
          - EFFECTIVE: S-space only (size n_S)
          - Others: Full space [S; C] (size n_S + n_C)
    """
    ψ_S, ψ_C = evaluator.wavefunction
    N = evaluator.contractions
    mode = evaluator.config.compute_mode
    eps = evaluator.config.epsilon

    # S-space local energy
    num_S = N.N_SS if mode == ComputeMode.EFFECTIVE else N.N_SS + N.N_SC
    E_loc_S = jnp.where(jnp.abs(ψ_S) >= eps, num_S / ψ_S, 0.0)

    if mode == ComputeMode.EFFECTIVE:
        return E_loc_S

    # C-space local energy
    num_C = N.N_CS + N.N_CC
    E_loc_C = jnp.where(jnp.abs(ψ_C) >= eps, num_C / ψ_C, 0.0)

    return jnp.concatenate([E_loc_S, E_loc_C])


def compute_energy_and_gradient(evaluator: Evaluator) -> GradientResult:
    """
    ✅ Solution A: Compute energy and gradient with single network forward pass.
    
    Uses variance-reduced VJP where log(ψ) from VJP forward is the single
    source of truth, avoiding redundant network evaluations.
    
    Algorithm:
      1. VJP forward: params → log(ψ) [SINGLE network evaluation]
      2. Derive ψ from log(ψ) directly
      3. Compute contractions from pre-computed ψ
      4. Calculate energy and cotangent vector
      5. VJP backward: cotangent → gradient
    """
    mode = evaluator.config.compute_mode
    eps = evaluator.config.epsilon
    n_S = evaluator.space.size_S

    # ✅ VJP forward pass (SINGLE network evaluation)
    batch_logpsi_fn = evaluator.get_batch_logpsi_fn()
    log_all, vjp_fn = jax.vjp(batch_logpsi_fn, evaluator.params)
    
    # ✅ Derive wavefunction from VJP's log_all (avoid 2nd forward pass)
    ψ_all = jnp.exp(log_all)
    ψ_S = ψ_all[:n_S]
    ψ_C = (ψ_all[n_S:] if mode != ComputeMode.EFFECTIVE 
           else jnp.empty(0, dtype=ψ_all.dtype))
    
    # ✅ Use externally computed ψ (no internal forward pass triggered)
    N = evaluator.compute_contractions_from_psi(ψ_S, ψ_C)

    # Mode-specific energy and gradient computation
    if mode in {ComputeMode.ASYMMETRIC, ComputeMode.EFFECTIVE}:
        # S-space normalization
        weights_S = _compute_normalized_weights(log_all[:n_S], eps)
        
        # Energy: E = ⟨ψ_S|H|ψ_S⟩ / ||ψ_S||²
        num_S = jnp.vdot(ψ_S, N.N_SS if mode == ComputeMode.EFFECTIVE 
                         else N.N_SS + N.N_SC)
        denom_S = jnp.vdot(ψ_S, ψ_S)
        E = (num_S / jnp.maximum(denom_S, eps)).real
        
        # Local energy for S-space
        E_loc_S = jnp.where(
            jnp.abs(ψ_S) >= eps,
            (N.N_SS if mode == ComputeMode.EFFECTIVE else N.N_SS + N.N_SC) / ψ_S,
            0.0
        )
        
        # Cotangent vector: w_i · (E_loc[i] - E)
        cot_S = weights_S * (E_loc_S - E)
        cot_full = (cot_S if mode == ComputeMode.EFFECTIVE 
                    else _pad_c_space(cot_S, evaluator.space.size_C))

    elif mode == ComputeMode.PROXY:
        # Full-space normalization
        weights_full = _compute_normalized_weights(log_all, eps)
        
        # Energy: E = (⟨ψ_S|H|ψ⟩ + ⟨ψ_C|H|ψ⟩) / ||ψ||²
        num_S = jnp.vdot(ψ_S, N.N_SS + N.N_SC)
        num_C = jnp.vdot(ψ_C, N.N_CS + N.N_CC)
        denom = jnp.vdot(ψ_S, ψ_S) + jnp.vdot(ψ_C, ψ_C)
        E = ((num_S + num_C) / jnp.maximum(denom, eps)).real
        
        # Local energy for both spaces
        E_loc_S = jnp.where(jnp.abs(ψ_S) >= eps, (N.N_SS + N.N_SC) / ψ_S, 0.0)
        E_loc_C = jnp.where(jnp.abs(ψ_C) >= eps, (N.N_CS + N.N_CC) / ψ_C, 0.0)
        E_loc_all = jnp.concatenate([E_loc_S, E_loc_C])
        
        # Cotangent vector
        cot_full = weights_full * (E_loc_all - E)

    else:
        raise ValueError(f"Unknown compute mode: {mode}")

    # VJP backward pass with complex conjugation
    (grad_conj,) = vjp_fn(jnp.conj(cot_full))
    grad = jax.tree.map(jnp.conj, grad_conj)

    return GradientResult(gradient=grad, energy_elec=E)


def _compute_normalized_weights(log_psi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Compute normalized probability weights w[i] = |ψ[i]|² / Σ|ψ|².

    Args:
        log_psi: Logarithm of wavefunction values.
        eps: Numerical stability threshold.

    Returns:
        Normalized weights summing to 1.
    """
    weights_raw = jnp.exp(2.0 * jnp.real(log_psi))
    return weights_raw / jnp.maximum(jnp.sum(weights_raw), eps)


def _pad_c_space(array_S: jnp.ndarray, n_C: int) -> jnp.ndarray:
    """
    Pad S-space array with C-space zeros.

    Args:
        array_S: S-space array.
        n_C: C-space dimension.

    Returns:
        Concatenated array [array_S; zeros(n_C)].
    """
    return jnp.concatenate([array_S, jnp.zeros(n_C, dtype=array_S.dtype)])


__all__ = ["compute_energy", "compute_local_energy", "compute_energy_and_gradient"]
