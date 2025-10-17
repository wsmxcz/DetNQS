# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Computation modes and configuration for LEVER variational engine.

Defines algorithmic choices (energy formulas, gradient domains, scoring metrics)
and numerical policies (precision, stability parameters).

File: lever/engine/config.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax.numpy as jnp


# ============================================================================
# Computation Modes
# ============================================================================

class EnergyMode(StrEnum):
    """
    Energy computation formula for Rayleigh quotient E = ⟨ψ|Ĥ|ψ⟩ / ⟨ψ|ψ⟩.
    
    ASYMMETRIC: E = ⟨ψ_S|Ĥ|ψ⟩ / ||ψ_S||²
                Fastest; unstable for small ||ψ_S||²
    
    PROXY:      E = (⟨ψ_S|Ĥ|ψ⟩ + ⟨ψ_C|Ĥ_diag|ψ_C⟩) / (||ψ_S||² + ||ψ_C||²)
                Diagonal H_CC approximation; balanced speed/stability
    
    FULL:       E = (⟨ψ_S|Ĥ|ψ⟩ + ⟨ψ_C|Ĥ|ψ_C⟩) / (||ψ_S||² + ||ψ_C||²)
                Complete H_CC matrix; most accurate, slowest
    """
    ASYMMETRIC = "asymmetric"
    PROXY = "proxy"
    FULL = "full"


class GradMode(StrEnum):
    """
    Gradient computation domain for ∇_θ E.
    
    ASYMMETRIC: Renormalize weights over S-space only
                Ignores C-space gradient contributions
    
    PROXY:      Full (S ∪ C) space with diagonal H_CC approximation
                Recommended for most cases
    
    FULL:       Full (S ∪ C) space with complete H_CC matrix
                Most accurate; computationally expensive
    """
    ASYMMETRIC = "asymmetric"
    PROXY = "proxy"
    FULL = "full"


class ScoreKind(StrEnum):
    """
    Scoring metric for determinant selection in variational space expansion.
    
    AMPLITUDE: Score = |ψ_k|
               Largest wavefunction coefficients
    
    PT1:       Score = |N_k|² / |H_kk - E|
               First-order perturbation theory energy lowering
    
    J_SCORE:   Score = |N_k - E·ψ_k|² / |H_kk - E|
               LEVER specific contribution
    """
    AMPLITUDE = "amplitude"
    PT1 = "PT1"
    J_SCORE = "J"


# ============================================================================
# Engine Configuration
# ============================================================================

@dataclass(frozen=True, slots=True)
class EngineConfig:
    """
    Immutable configuration for variational computation engine.
    
    Attributes:
        compute_dtype: Floating-point precision (complex dtypes derived)
        epsilon: Numerical threshold for division stability
        normalize_wf: Apply joint L2 normalization ||ψ_S||² + ||ψ_C||² = 1
        energy_mode: Formula for energy calculation
        grad_mode: Domain for gradient computation
    """
    # Numerical parameters
    compute_dtype: jnp.dtype = jnp.float32
    epsilon: float = 1e-12

    # Algorithmic modes
    normalize_wf: bool = True
    energy_mode: EnergyMode = EnergyMode.PROXY
    grad_mode: GradMode = GradMode.PROXY


# Default configuration instance
DEFAULT_CONFIG = EngineConfig()


__all__ = [
    "EnergyMode",
    "GradMode",
    "ScoreKind",
    "EngineConfig",
    "DEFAULT_CONFIG",
]
