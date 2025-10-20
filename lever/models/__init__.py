# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wavefunction model components and factory functions.

Provides neural quantum state ansätze: Slater determinants, Pfaffians,
RBMs, Jastrow factors, and their compositions. All models map occupation
basis vectors to complex log-amplitudes log ψ(s).

File: lever/models/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax.numpy as jnp

from .base import WavefunctionModel, make_model
from .product import ProductModel


# --- Model Factories ---


def Slater(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    *,
    seed: int,
    n_dets: int = 1,
    generalized: bool = False,
    restricted: bool = True,
    use_log_coeffs: bool = True,
    param_dtype: Any = jnp.complex64,
    kernel_init: Callable | None = None,
) -> WavefunctionModel:
    """
    Neural Slater determinant ansatz with multi-determinant expansion.
  
    Computes log ψ(s) = log(Σᵢ cᵢ det(Mᵢ[s,:])) where Mᵢ are learnable
    orbital matrices. Supports three formulations:
      - Generalized: Spin-orbital basis M ∈ ℂ^(N×N_orb)
      - Restricted: Shared spatial orbitals M_α = M_β
      - Unrestricted: Separate M_α, M_β matrices
  
    Args:
        n_orbitals: Spatial orbital count
        n_alpha, n_beta: Electron counts per spin
        seed: Parameter initialization seed
        n_dets: CI expansion terms (1 = single determinant)
        generalized: Use spin-orbital formulation
        restricted: Share α/β spatial orbitals (ignored if generalized)
        use_log_coeffs: Learn expansion coefficients (only for n_dets > 1)
        param_dtype: Orbital matrix dtype (complex recommended)
        kernel_init: Weight initializer (default: orthogonal)
      
    Returns:
        Initialized WavefunctionModel
    """
    from flax.linen import initializers
    from .slater import Slater

    kernel_init = kernel_init or initializers.orthogonal()

    ansatz = Slater(
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_dets=n_dets,
        generalized=generalized,
        restricted=restricted,
        use_log_coeffs=use_log_coeffs,
        param_dtype=param_dtype,
        kernel_init=kernel_init,
    )

    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def Pfaffian(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    *,
    seed: int,
    n_terms: int = 1,
    singlet_only: bool = True,
    use_log_coeffs: bool = True,
    param_dtype: Any = jnp.complex64,
    kernel_init: Callable | None = None,
) -> WavefunctionModel:
    """
    Pfaffian/AGP ansatz for systems with pairing correlations.

    Two operational modes:
      - singlet_only=True: AGP form, ψ ∝ det(P) where P_{ij} pairs
        α-spin i with β-spin j
      - singlet_only=False: Full Pfaffian including triplet pairing
  
    Generalizes Slater determinants to capture stronger correlations.

    Args:
        n_orbitals: Spatial orbital count
        n_alpha, n_beta: Electron counts per spin
        seed: Parameter initialization seed
        n_terms: Expansion terms for multi-Pfaffian ansatz
        singlet_only: Use AGP (determinant) form
        use_log_coeffs: Learn expansion coefficients (n_terms > 1)
        param_dtype: Parameter dtype (complex recommended)
        kernel_init: Weight initializer (default: lecun_normal)

    Returns:
        Initialized WavefunctionModel
    """
    from flax.linen import initializers
    from .pfaffian import Pfaffian as PfaffianModule

    kernel_init = kernel_init or initializers.lecun_normal()

    ansatz = PfaffianModule(
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_terms=n_terms,
        singlet_only=singlet_only,
        use_log_coeffs=use_log_coeffs,
        param_dtype=param_dtype,
        kernel_init=kernel_init,
    )

    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def RBM(
    n_orbitals: int,
    *,
    seed: int,
    alpha: float = 1.0,
    use_visible_bias: bool = True,
) -> WavefunctionModel:
    """
    Complex-valued restricted Boltzmann machine.
  
    Maps occupation basis to visible layer (2N_orb units), then to
    hidden layer (α·2N_orb units). Computes ψ via energy function
    partition sum.
  
    Args:
        n_orbitals: Spatial orbital count (visible = 2N_orb)
        seed: Parameter initialization seed
        alpha: Hidden/visible unit ratio
        use_visible_bias: Enable visible bias terms
      
    Returns:
        Initialized WavefunctionModel
    """
    from .rbm import RBM

    ansatz = RBM(
        alpha=alpha,
        use_visible_bias=use_visible_bias,
        param_dtype=jnp.complex64,
    )

    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def RBMModPhase(
    n_orbitals: int,
    *,
    seed: int,
    alpha: float = 1.0,
) -> WavefunctionModel:
    """
    Dual-RBM model with separate modulus and phase networks.
  
    Decomposes ψ = |ψ|e^{iφ} using two real RBMs for improved
    optimization stability.
  
    Args:
        n_orbitals: Spatial orbital count
        seed: Parameter initialization seed
        alpha: Hidden/visible ratio per RBM
      
    Returns:
        Initialized WavefunctionModel
    """
    from .rbm import RBMModPhase

    ansatz = RBMModPhase(alpha=alpha, param_dtype=jnp.float64)
    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def Backflow(
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    *,
    seed: int,
    n_dets: int,
    generalized: bool = False,
    restricted: bool = True,
    hidden_dims: Tuple[int, ...] = (256,),
    param_dtype: Any = jnp.complex64,
) -> WavefunctionModel:
    """
    Hilbert-space backflow with configuration-dependent orbitals.

    Enhances Slater determinants via M_eff(s) = M_base + MLP(s),
    allowing adaptive nodal surfaces for strong correlation.

    Args:
        n_orbitals: Spatial orbital count
        n_alpha, n_beta: Electron counts per spin
        seed: Parameter initialization seed
        n_dets: Multi-determinant expansion terms
        generalized: Use spin-orbital formulation
        restricted: Share α/β spatial orbitals
        hidden_dims: MLP architecture (layer sizes)
        param_dtype: Parameter dtype (complex recommended)

    Returns:
        Initialized WavefunctionModel
    """
    from .backflow import BackflowMLP

    ansatz = BackflowMLP(
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        n_dets=n_dets,
        generalized=generalized,
        restricted=restricted,
        hidden_dims=hidden_dims,
        param_dtype=param_dtype,
    )
  
    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def Jastrow(
    n_orbitals: int,
    *,
    seed: int,
    param_dtype: Any = jnp.float64,
) -> WavefunctionModel:
    """
    Symmetric Jastrow correlation factor in occupation basis.
  
    Pairwise correlation operator: ψ_J = exp(Σ_{i<j} v_{ij} n_i n_j).
    Commonly combined with Slater via Product().
  
    Args:
        n_orbitals: Spatial orbital count
        seed: Parameter initialization seed
        param_dtype: Parameter dtype (typically real)
      
    Returns:
        Initialized WavefunctionModel
    """
    from .jastrow import Jastrow

    ansatz = Jastrow(n_orbitals=n_orbitals, param_dtype=param_dtype)
    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def Product(*models: WavefunctionModel) -> ProductModel:
    """
    Compose wavefunctions via logarithmic product.
  
    Combines models by log ψ_total = Σᵢ log ψᵢ, preserving normalization
    and enabling hybrid ansätze.
  
    Args:
        *models: WavefunctionModel instances
      
    Returns:
        ProductModel (duck-typed WavefunctionModel)
      
    Example:
        >>> slater = Slater(n_orbitals=4, n_alpha=2, n_beta=2, seed=42)
        >>> jastrow = Jastrow(n_orbitals=4, seed=43)
        >>> wf = Product(slater, jastrow)
    """
    return ProductModel(models)


__all__ = [
    "WavefunctionModel",
    "make_model",
    "Slater",
    "Pfaffian",
    "RBM",
    "RBMModPhase",
    "Backflow",
    "Jastrow",
    "Product",
]