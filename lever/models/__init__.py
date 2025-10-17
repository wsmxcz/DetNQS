# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wavefunction model components and factory functions.

Entry points: Slater(), RBM(), RBMModPhase(), Jastrow(), Product()
Each factory returns an initialized WavefunctionModel ready for computation.

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
    Create neural Slater determinant wavefunction model.
    
    Computes log ψ(s) = log(Σᵢ cᵢ det(Mᵢ[s,:])) where Mᵢ are learnable
    orbital matrices and cᵢ are optional expansion coefficients.
    
    Three orbital formulations:
      - Generalized: Single matrix for all electrons (spin-orbital basis)
      - Restricted: Shared spatial orbitals for α/β spins (RHF-like)
      - Unrestricted: Separate α/β orbital matrices (UHF-like)
    
    Args:
        n_orbitals: Number of spatial orbitals
        n_alpha: Alpha-spin electron count
        n_beta: Beta-spin electron count
        seed: RNG seed for parameter initialization
        n_dets: Number of determinants in CI expansion (default: 1)
        generalized: Use spin-orbital formulation
        restricted: Share spatial orbitals for α/β (ignored if generalized)
        use_log_coeffs: Learn expansion coefficients cᵢ (only when n_dets > 1)
        param_dtype: Orbital matrix dtype (complex recommended)
        kernel_init: Weight initializer (default: orthogonal())
        
    Returns:
        Initialized WavefunctionModel with learnable orbital parameters
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
    Creates a neural Pfaffian or AGP wavefunction model.

    This ansatz generalizes the Slater determinant and is particularly
    suited for systems with strong pairing correlations.

    Modes:
      - singlet_only=True: Antisymmetric Geminal Power (AGP). The amplitude
        is given by the determinant of an N_alpha x N_beta pairing matrix.
      - singlet_only=False: General Pfaffian, including both singlet
        (opposite-spin) and triplet (same-spin) pairing terms.

    Args:
        n_orbitals: Number of spatial orbitals.
        n_alpha, n_beta: Electron counts per spin.
        seed: RNG seed for parameter initialization.
        n_terms: Number of terms in the expansion (default: 1).
        singlet_only: If True, use the AGP (determinant) form.
        use_log_coeffs: Learn expansion coefficients for multi-term models.
        param_dtype: Parameter dtype (complex recommended).
        kernel_init: Weight initializer (default: lecun_normal()).

    Returns:
        An initialized WavefunctionModel.
    """
    from flax.linen import initializers
    # Import the new Pfaffian module
    from .pfaffian import Pfaffian as PfaffianModule

    kernel_init = kernel_init or initializers.lecun_normal()

    # Create the Flax Linen module instance
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

    # Wrap it in the WavefunctionModel, which handles JIT, batching, etc.
    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)

def RBM(
    n_orbitals: int,
    *,
    seed: int,
    alpha: float = 1.0,
    use_visible_bias: bool = True,
) -> WavefunctionModel:
    """
    Create complex-valued restricted Boltzmann machine wavefunction.
    
    Maps occupation basis to RBM visible layer (2*n_orbitals units).
    Hidden layer size = alpha * n_visible.
    
    Args:
        n_orbitals: Spatial orbitals (visible units = 2*n_orbitals)
        seed: RNG seed
        alpha: Hidden-to-visible unit ratio
        use_visible_bias: Enable visible layer bias
        
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
    Create twin-RBM model with separate modulus and phase networks.
    
    Uses two real-valued RBMs for improved training stability:
    log|ψ| from first RBM, arg(ψ) from second RBM.
    
    Args:
        n_orbitals: Spatial orbitals
        seed: RNG seed
        alpha: Hidden-to-visible unit ratio per RBM
        
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
    generalized: bool = False,
    restricted: bool = True,
    hidden_dims: Tuple[int, ...] = (256,),
    bf_scale: float = 1.0,
    param_dtype: Any = jnp.complex64,
) -> WavefunctionModel:
    """
    Create Hilbert-space backflow model with MLP-modulated orbitals.

    Enhances a Slater determinant by making the orbital matrix M configuration-
    dependent: M_eff(s) = M_base + MLP(s). This allows the wavefunction's
    nodal surface to dynamically adapt, capturing strong correlation.

    Args:
        n_orbitals: Number of spatial orbitals.
        n_alpha, n_beta: Electron counts per spin.
        seed: RNG seed for parameter initialization.
        generalized: Use a single spin-orbital matrix.
        restricted: Share spatial orbitals for alpha/beta spins.
        hidden_dims: Tuple of hidden layer sizes for the MLP.
        bf_scale: Multiplicative factor for the MLP output, controls backflow strength.
        param_dtype: Dtype for model parameters (complex recommended).

    Returns:
        An initialized WavefunctionModel with backflow correlations.
    """
    from .backflow import BackflowMLP

    ansatz = BackflowMLP(
        n_orbitals=n_orbitals,
        n_alpha=n_alpha,
        n_beta=n_beta,
        generalized=generalized,
        restricted=restricted,
        hidden_dims=hidden_dims,
        bf_scale=bf_scale,
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
    Create Hilbert-space Jastrow correlation factor.
    
    Symmetric pairwise correlation operator in occupation number basis.
    Typically combined with Slater determinants via Product().
    
    Args:
        n_orbitals: Spatial orbitals
        seed: RNG seed
        param_dtype: Parameter dtype (usually real)
        
    Returns:
        Initialized WavefunctionModel
    """
    from .jastrow import Jastrow

    ansatz = Jastrow(n_orbitals=n_orbitals, param_dtype=param_dtype)
    return make_model(module=ansatz, seed=seed, n_orbitals=n_orbitals)


def Product(*models: WavefunctionModel) -> ProductModel:
    """
    Compose multiple wavefunctions via logarithmic product.
    
    Combines models by summing their log-amplitudes:
    log ψ_total = log ψ₁ + log ψ₂ + ... + log ψₙ
    
    Args:
        *models: WavefunctionModel instances to compose
        
    Returns:
        ProductModel (duck-types as WavefunctionModel)
        
    Example:
        >>> slater = Slater(n_orbitals=4, n_alpha=2, n_beta=2, seed=42)
        >>> jastrow = Jastrow(n_orbitals=4, seed=43)
        >>> wf = Product(slater, jastrow)
        >>> log_psi = wf(batch_configs)
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
