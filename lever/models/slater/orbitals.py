# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Orbital parametrization modules for Slater determinants.

Provides flexible SPO (Spatial orbital) construction via:
  - Reference orbitals: GHF-like one-hot initialization
  - Additive/Gated updates: direct dphi or multiplicative correction
  - Low-rank/CPD updates: rank-r factorizations for memory efficiency
  - Jastrow factors: determinant-based correlation

Mathematical formulation:
  phi_occ = phi_ref + sum_i dphi_i
  phi_occ *= product_j (1 + gate_j)
  psi = det(phi_occ) * exp(j_logabs) * j_sign

File: lever/models/slater/orbitals.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from ...system import MolecularSystem
from ..utils import backflow_orbitals_init, logdet_c
from .networks import Parametrizer


def _reference_init(system: MolecularSystem, sigma: float):
    """
    Initialize GHF-style reference orbitals with optional Gaussian noise.
  
    Returns initializer function producing one-hot + noise pattern.
    """
    base = backflow_orbitals_init(
        n_orb=system.n_orb,
        n_alpha=system.n_alpha,
        n_beta=system.n_beta,
        mode="generalized",
    )

    def init_fn(key, shape, dtype=jnp.float64):
        phi0 = base(key, shape, dtype=dtype)
        if sigma <= 0.0:
            return phi0
        noise = jax.random.normal(key, shape, jnp.result_type(dtype, jnp.float32))
        return (phi0 + sigma * noise).astype(dtype)

    return init_fn


@struct.dataclass
class OrbitalBundle:
    """
    Container for occupied submatrix and Jastrow correlation factor.
  
    Fields:
      phi_occ:  (..., n_e, n_e) or (..., n_det, n_e, n_e)
      j_sign:   (...) complex or real
      j_logabs: (...) real, log|J|
    """
    phi_occ: jnp.ndarray
    j_sign: jnp.ndarray
    j_logabs: jnp.ndarray


class ReferenceSPO(nn.Module):
    """
    Reference orbital matrix parametrized as full (n_so, n_e) or (n_det, n_so, n_e).
  
    Extracts occupied columns via index array occ_so.
    """
    system: MolecularSystem
    n_det: int = 1
    init_sigma: float = 1e-3
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          occ_so: (..., n_e) integer indices
      
        Returns:
          phi_occ: (..., n_e, n_e) if n_det=1, else (..., n_det, n_e, n_e)
        """
        occ_so = occ_so.astype(jnp.int32)
        n_so, n_e = self.system.n_so, self.system.n_elec
      
        shape = (n_so, n_e) if self.n_det == 1 else (self.n_det, n_so, n_e)
        phi_ref = self.param(
            "phi_ref",
            _reference_init(self.system, self.init_sigma),
            shape,
            self.param_dtype,
        )
      
        if self.n_det == 1:
            return jnp.take(phi_ref, occ_so, axis=0)
        else:
            phi_occ = jnp.take(phi_ref, occ_so, axis=1)
            ndim = phi_occ.ndim
            perm = tuple(range(1, ndim - 2)) + (0, ndim - 2, ndim - 1)
            return jnp.transpose(phi_occ, perm)


class AdditiveUpdate(nn.Module):
    """
    Full additive correction: dphi directly on occupied submatrix.
  
    Parametrizer outputs flattened (n_e * n_e) or (n_det * n_e * n_e).
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    n_det: int = 1
    head: str = "dphi"

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          occ_so: (..., n_e)
      
        Returns:
          dphi_occ: (..., n_e, n_e) or (..., n_det, n_e, n_e)
        """
        n_e = self.system.n_elec
        out_dim = (n_e * n_e) if self.n_det == 1 else (self.n_det * n_e * n_e)
        dphi = self.parametrizer(occ_so, out_dim, head=self.head)
      
        batch_shape = occ_so.shape[:-1]
        if self.n_det == 1:
            return dphi.reshape(batch_shape + (n_e, n_e))
        else:
            return dphi.reshape(batch_shape + (self.n_det, n_e, n_e))


class GatedUpdate(nn.Module):
    """
    Gated multiplicative correction: phi *= (1 + tanh(gate)).
  
    Bounded gate ensures stability.
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    n_det: int = 1
    head: str = "gate"

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          occ_so: (..., n_e)
      
        Returns:
          gate_occ: (..., n_e, n_e) or (..., n_det, n_e, n_e)
        """
        n_e = self.system.n_elec
        out_dim = (n_e * n_e) if self.n_det == 1 else (self.n_det * n_e * n_e)
        raw = self.parametrizer(occ_so, out_dim, head=self.head)
        gate = jnp.tanh(raw)
      
        batch_shape = occ_so.shape[:-1]
        if self.n_det == 1:
            return gate.reshape(batch_shape + (n_e, n_e))
        else:
            return gate.reshape(batch_shape + (self.n_det, n_e, n_e))


class LowRankUpdate(nn.Module):
    """
    Low-rank additive correction: dphi = U @ V^T.
  
    Reduces parameter count from O(n_e^2) to O(n_e * rank).
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    rank: int
    n_det: int = 1
    head: str = "uv"

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          occ_so: (..., n_e)
      
        Returns:
          dphi_occ: (..., n_e, n_e) or (..., n_det, n_e, n_e)
        """
        n_e, r = self.system.n_elec, self.rank
      
        if self.n_det == 1:
            uv = self.parametrizer(occ_so, 2 * n_e * r, head=self.head)
            U = uv[..., : n_e * r].reshape(occ_so.shape[:-1] + (n_e, r))
            V = uv[..., n_e * r :].reshape(occ_so.shape[:-1] + (n_e, r))
            return jnp.einsum("...er,...fr->...ef", U, V)
        else:
            uv = self.parametrizer(occ_so, self.n_det * 2 * n_e * r, head=self.head)
            uv = uv.reshape(occ_so.shape[:-1] + (self.n_det, 2 * n_e * r))
            U = uv[..., : n_e * r].reshape(occ_so.shape[:-1] + (self.n_det, n_e, r))
            V = uv[..., n_e * r :].reshape(occ_so.shape[:-1] + (self.n_det, n_e, r))
            return jnp.einsum("...der,...dfr->...def", U, V)


class CPDUpdate(nn.Module):
    """
    Canonical Polyadic Decomposition: dphi = (A_occ * w) @ B^T.
  
    Shares factor matrices A, B across all occupied orbitals, learning only
    rank-r weights w per configuration. Memory: O(n_so * rank + n_e * rank).
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    rank: int
    n_det: int = 1
    head: str = "w"
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
          occ_so: (..., n_e)
      
        Returns:
          dphi_occ: (..., n_e, n_e) or (..., n_det, n_e, n_e)
        """
        occ_so = occ_so.astype(jnp.int32)
        n_so, n_e, r = self.system.n_so, self.system.n_elec, self.rank
      
        if self.n_det == 1:
            A = self.param("A", nn.initializers.lecun_normal(), (n_so, r), self.param_dtype)
            B = self.param("B", nn.initializers.lecun_normal(), (n_e, r), self.param_dtype)
            w = self.parametrizer(occ_so, r, head=self.head)
          
            A_occ = jnp.take(A, occ_so, axis=0)
            return jnp.einsum("...er,...r,fr->...ef", A_occ, w, B)
        else:
            A = self.param("A", nn.initializers.lecun_normal(), (self.n_det, n_so, r), self.param_dtype)
            B = self.param("B", nn.initializers.lecun_normal(), (self.n_det, n_e, r), self.param_dtype)
            w = self.parametrizer(occ_so, self.n_det * r, head=self.head)
            w = w.reshape(occ_so.shape[:-1] + (self.n_det, r))
          
            A_occ = jnp.take(A, occ_so, axis=1)
            A_occ = jnp.moveaxis(A_occ, 0, -3)
            return jnp.einsum("...der,...dr,dfr->...def", A_occ, w, B)


class JastrowUpdate(nn.Module):
    """
    Determinant-based Jastrow correlation: J = det(I + M).
  
    Returns (sign, logabs) form for numerical stability.
    Parametrizer outputs flattened rank * rank matrix.
    """
    parametrizer: Parametrizer
    rank: int
    head: str = "jastrow"

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
          occ_so: (..., n_e)
      
        Returns:
          j_sign:   (...) complex or real
          j_logabs: (...) real
        """
        r = self.rank
        batch_shape = occ_so.shape[:-1]

        j = self.parametrizer(occ_so, r * r, head=self.head)
        j = j.reshape(batch_shape + (r, r))

        I = jnp.eye(r, dtype=j.dtype)
        j_sign, j_logabs = logdet_c(I + j)
        return j_sign, j_logabs


class SPOMap(nn.Module):
    """
    Complete SPO map: reference + updates + Jastrow.
  
    Composition:
      phi = phi_ref
      phi += sum dphi_i         (additive)
      phi *= prod (1 + gate_j)  (gated)
      psi = det(phi) * J        (Jastrow)
    """
    reference: ReferenceSPO
    additive_updates: Sequence[nn.Module] = ()
    gated_updates: Sequence[nn.Module] = ()
    jastrow: JastrowUpdate | None = None

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> OrbitalBundle:
        """
        Args:
          occ_so: (..., n_e) occupation indices
      
        Returns:
          OrbitalBundle with phi_occ and Jastrow (sign, logabs)
        """
        phi_occ = self.reference(occ_so)

        for upd in self.additive_updates:
            phi_occ = phi_occ + upd(occ_so)

        for gate in self.gated_updates:
            g = gate(occ_so)
            phi_occ = phi_occ * (1.0 + g)

        if self.jastrow is None:
            j_sign = jnp.ones(phi_occ.shape[:-2], dtype=phi_occ.dtype)
            j_logabs = jnp.zeros(phi_occ.shape[:-2], dtype=jnp.real(phi_occ).dtype)
        else:
            j_sign, j_logabs = self.jastrow(occ_so)

        return OrbitalBundle(phi_occ=phi_occ, j_sign=j_sign, j_logabs=j_logabs)


__all__ = [
    "OrbitalBundle",
    "ReferenceSPO",
    "AdditiveUpdate",
    "GatedUpdate",
    "LowRankUpdate",
    "CPDUpdate",
    "JastrowUpdate",
    "SPOMap",
]