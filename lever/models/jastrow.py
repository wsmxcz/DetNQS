# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Jastrow correlation factor for occupation-based wavefunction reweighting.

Implements diagonal correlation in occupation number basis without altering
nodal surfaces: log J(s) = Σᵢ uᵢsᵢ + ½ Σᵢⱼ sᵢVᵢⱼsⱼ

File: lever/models/jastrow.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers


class Jastrow(nn.Module):
    """
    Occupation-based Jastrow correlation factor.
    
    Reweights determinant amplitudes via diagonal correlation without
    modifying nodal structure. Operates in occupation number basis.
    
    Attributes:
        n_orbitals: Number of spatial orbitals (input dimension: 2*n_orbitals)
        param_dtype: Parameter data type (typically real for positive-definite form)
    """

    n_orbitals: int
    param_dtype: Any = jnp.float64

    # Real-valued correlation factor
    __lever_is_holomorphic__ = False

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log-Jastrow factor for occupation vector.
        
        Args:
            s: Occupation vector of shape (2*n_orbitals,)
            
        Returns:
            Scalar log(J(s)) combining one-body and two-body terms
        """
        n_vis = 2 * self.n_orbitals
        x = s.astype(self.param_dtype)

        # One-body on-site interactions: Σᵢ uᵢsᵢ
        u = self.param("u", initializers.zeros, (n_vis,), self.param_dtype)
        one_body = jnp.dot(u, x)

        # Two-body density-density interactions: ½ Σᵢⱼ sᵢVᵢⱼsⱼ
        V = self.param("V", initializers.zeros, (n_vis, n_vis), self.param_dtype)
        # Enforce symmetry and remove diagonal to prevent double counting
        V_eff = 0.5 * (V + V.T) - jnp.diag(jnp.diag(V))
        two_body = 0.5 * jnp.dot(x, jnp.dot(V_eff, x))

        return two_body


__all__ = ["Jastrow"]
