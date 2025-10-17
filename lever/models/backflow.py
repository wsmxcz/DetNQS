# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backflow network with split real/imaginary architecture for numerical stability.

Implements configuration-dependent Slater kernel modifications via real-valued
MLP with complex head projection. Uses real activations and proper complex
variance scaling to maintain numerical stability.

File: lever/models/backflow.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2025
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from .utils import logdet_c, c_orthogonal_init


class BackflowMLP(nn.Module):
    """
    Configuration-dependent Slater determinant via real MLP with complex projection.

    Architecture:
    - Real-valued residual network processes occupancy patterns
    - Complex output head via real/imaginary split maintains numerical stability
    - Backflow correction ΔM modifies base orbital matrix M = M₀ + λ·ΔM(s)
  
    The real-valued core avoids complex activation instabilities while deep
    residual structure provides sufficient expressiveness for electron correlations.
  
    Attributes:
        n_orbitals: Number of spatial orbitals
        n_alpha: Number of spin-up electrons
        n_beta: Number of spin-down electrons
        generalized: Single spin-orbital matrix (no spin separation)
        restricted: Shared spatial orbitals for both spins
        hidden_dims: Residual block widths
        bf_scale: Backflow strength λ controlling correction magnitude
        param_dtype: Complex dtype for orbital matrices
        kernel_init: Weight initializer for real MLP
        bias_init: Bias initializer
    """
    n_orbitals: int
    n_alpha: int
    n_beta: int
    generalized: bool = False
    restricted: bool = True
    hidden_dims: Tuple[int, ...] = (256,)
    bf_scale: float = 1.0
    param_dtype: Any = jnp.complex64
    kernel_init: Callable = initializers.glorot_normal()
    bias_init: Callable = initializers.zeros_init()

    def _mlp(self, name: str, out_dim: int, s: jnp.ndarray) -> jnp.ndarray:
        """
        Real-valued residual MLP with complex projection head.
      
        Processing flow:
        1. Real transformations with GELU activations
        2. Skip connections for gradient flow
        3. Split output into real/imaginary channels
        4. Complex assembly with 1/√2 variance scaling
      
        Args:
            name: Parameter namespace prefix
            out_dim: Number of complex outputs
            s: Real input (occupancy vector)
          
        Returns:
            Complex array of shape (..., out_dim)
        """
        real_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
        x = s.astype(real_dtype)
      
        # Residual blocks with pre-activation
        for li, h in enumerate(self.hidden_dims):
            residual = x
          
            # Pre-activation improves gradient flow
            x = nn.gelu(x)
            x = nn.Dense(
                h,
                param_dtype=real_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"{name}_dense_{li}"
            )(x)
          
            # Adapt residual dimension if needed
            if residual.shape[-1] != h:
                residual = nn.Dense(
                    h,
                    param_dtype=real_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    name=f"{name}_skip_{li}"
                )(residual)
          
            x = x + residual
      
        # Complex projection: double width for Re/Im split
        x = nn.gelu(x)
        x = nn.Dense(
            2 * out_dim,
            param_dtype=real_dtype,
            kernel_init=initializers.variance_scaling(
                scale=0.01,  # Small initial corrections
                mode='fan_avg',
                distribution='truncated_normal'
            ),
            bias_init=self.bias_init,
            name=f"{name}_out"
        )(x)
      
        # Maintain unit variance: Var[Re(z)] = Var[Im(z)] = 1/2
        x = x / jnp.sqrt(2.0)
      
        # Assemble complex from real/imaginary components
        re, im = jnp.split(x, 2, axis=-1)
        target_dtype = jnp.real(jnp.zeros(1, dtype=self.param_dtype)).dtype
        return jax.lax.complex(
            re.astype(target_dtype),
            im.astype(target_dtype)
        )

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        """
        Compute log-amplitude via backflow-corrected Slater determinant.
      
        Steps:
        1. Initialize base orbital M with orthonormal columns
        2. Compute ΔM(s) via configuration-dependent MLP
        3. Form M_eff = M + λ·ΔM(s)
        4. Extract occupied rows and evaluate log|det(M_eff[occupied])|
      
        Args:
            s: Binary occupancy vector of shape (2*n_orbitals,)
               Layout: [spin-up₁, ..., spin-upₙ, spin-down₁, ..., spin-downₙ]
          
        Returns:
            Complex log-amplitude log ψ(s)
        """
        x = s
        n_elec = self.n_alpha + self.n_beta

        if self.generalized:
            # Generalized: single (2n × n_e) spin-orbital matrix
            M = self.param(
                "M_gen",
                c_orthogonal_init,
                (2 * self.n_orbitals, n_elec),
                self.param_dtype
            )
          
            delta = self._mlp("BF_gen", (2 * self.n_orbitals) * n_elec, x).reshape(
                (2 * self.n_orbitals, n_elec)
            )
            M_eff = M + self.bf_scale * delta
          
            # Select rows for occupied spin-orbitals
            R = jnp.nonzero(x, size=n_elec)[0]
            A = M_eff[R, :]
            return logdet_c(A)

        if self.restricted:
            # Restricted: shared (n × k) spatial orbitals, k = max(n_α, n_β)
            k = max(self.n_alpha, self.n_beta)
            M = self.param(
                "M_spatial",
                c_orthogonal_init,
                (self.n_orbitals, k),
                self.param_dtype
            )
          
            delta = self._mlp("BF_res", self.n_orbitals * k, x).reshape(
                (self.n_orbitals, k)
            )
            M_eff = M + self.bf_scale * delta
          
            # Split into spin sectors
            alpha = x[: self.n_orbitals]
            beta = x[self.n_orbitals: 2 * self.n_orbitals]
          
            R_a = jnp.nonzero(alpha, size=self.n_alpha)[0]
            R_b = jnp.nonzero(beta, size=self.n_beta)[0]
          
            # Product of spin-up and spin-down determinants
            A_a = M_eff[R_a, : self.n_alpha]
            A_b = M_eff[R_b, : self.n_beta]
            return logdet_c(A_a) + logdet_c(A_b)

        # Unrestricted: separate M_α and M_β matrices
        M_a = self.param(
            "M_alpha",
            c_orthogonal_init,
            (self.n_orbitals, self.n_alpha),
            self.param_dtype
        )
        M_b = self.param(
            "M_beta",
            c_orthogonal_init,
            (self.n_orbitals, self.n_beta),
            self.param_dtype
        )

        # Joint correction for both spins
        out_dim = self.n_orbitals * (self.n_alpha + self.n_beta)
        delta = self._mlp("BF_unres", out_dim, x).reshape(
            (self.n_orbitals, self.n_alpha + self.n_beta)
        )
      
        da = delta[:, : self.n_alpha]
        db = delta[:, self.n_alpha:]
      
        M_a_eff = M_a + self.bf_scale * da
        M_b_eff = M_b + self.bf_scale * db

        # Extract occupied orbitals for each spin
        alpha = x[: self.n_orbitals]
        beta = x[self.n_orbitals: 2 * self.n_orbitals]
      
        R_a = jnp.nonzero(alpha, size=self.n_alpha)[0]
        R_b = jnp.nonzero(beta, size=self.n_beta)[0]
      
        A_a = M_a_eff[R_a, :]
        A_b = M_b_eff[R_b, :]
        return logdet_c(A_a) + logdet_c(A_b)


__all__ = ["BackflowMLP", "logdet_c", "c_orthogonal_init"]