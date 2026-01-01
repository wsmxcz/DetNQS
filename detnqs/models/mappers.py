# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Coordinate mappers for Grassmann manifold representation.

Defines different chart choices on Gr(N,M):
  - reference: Fixed HF reference (constant point)
  - thouless: Affine chart via particle-hole amplitudes T_{v,o}
  - full: Stiefel manifold embedding C_{M,N}
  - submatrix: Direct Plücker coordinate fitting A_{N,N}

Each mapper transforms occupation I -> manifold coordinate.

File: detnqs/models/mappers.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..system import MolecularSystem
from .utils import backflow_orbitals_init
from .parametrizers import Parametrizer

MapperType = Literal["reference", "thouless", "full", "submatrix"]


def _gather_occupied_rows(C: jnp.ndarray, occ: jnp.ndarray) -> jnp.ndarray:
    """Extract occupied orbital rows C[occ, :].
    
    Args:
        C: (B, M, N) coefficient matrix
        occ: (B, N) occupied orbital indices
        
    Returns:
        (B, N, N) occupied submatrix
    """
    idx = occ[..., None]  # (B, N, 1)
    idx = jnp.broadcast_to(idx, (C.shape[0], C.shape[2], C.shape[2]))
    return jnp.take_along_axis(C, idx, axis=1)


# ============================================================================
# Reference State
# ============================================================================

class ReferenceCoefficients(nn.Module):
    """Trainable reference orbital coefficients C0.
    
    Initializes with HF-style one-hot + optional noise for symmetry breaking.
    """
    system: MolecularSystem
    init_sigma: float = 1e-3
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, batch_size: int) -> jnp.ndarray:
        """
        Args:
            batch_size: Number of samples
            
        Returns:
            (B, M, N) reference coefficient matrix
        """
        M, N = self.system.n_so, self.system.n_elec
        
        # HF-style initialization with optional noise
        base_init = backflow_orbitals_init(
            n_orb=self.system.n_orb,
            n_alpha=self.system.n_alpha,
            n_beta=self.system.n_beta,
            mode="generalized",
        )

        def init_fn(key, shape, dtype=jnp.float64):
            C0 = base_init(key, shape, dtype=dtype)
            if self.init_sigma > 0.0:
                noise = jax.random.normal(key, shape, jnp.result_type(dtype, jnp.float32))
                C0 = C0 + self.init_sigma * noise
            return C0.astype(dtype)

        C0 = self.param("C0", init_fn, (M, N), self.param_dtype)
        return jnp.broadcast_to(C0[None], (batch_size, M, N))


# ============================================================================
# Coordinate Mappers
# ============================================================================

class ReferenceMapper(nn.Module):
    """No backflow: fixed reference orbitals."""
    system: MolecularSystem
    init_sigma: float = 1e-3
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            occ: (B, N) occupation indices
            
        Returns:
            (B, N, N) reference determinant A0[occ, :]
        """
        C0 = ReferenceCoefficients(
            self.system,
            init_sigma=self.init_sigma,
            param_dtype=self.param_dtype
        )(occ.shape[0])
        return _gather_occupied_rows(C0, occ)


class FullMapper(nn.Module):
    """Full M×N coefficient matrix with neural backflow.
    
    Maps I -> C(I) = C0 + NN(I) via residual connection.
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    init_sigma: float = 1e-3
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            occ: (B, N) occupation indices
            
        Returns:
            (B, N, N) occupied submatrix A(I)
        """
        M, N = self.system.n_so, self.system.n_elec
        
        # Reference + backflow correction
        C0 = ReferenceCoefficients(
            self.system,
            init_sigma=self.init_sigma,
            param_dtype=self.param_dtype
        )(occ.shape[0])
        
        dC = self.parametrizer(occ, M * N, head="full_coeff")
        dC = dC.reshape((occ.shape[0], M, N))
        
        return _gather_occupied_rows(C0 + dC, occ)


class SubmatrixMapper(nn.Module):
    """Direct N×N occupied submatrix prediction.
    
    Maps I -> A(I) = A0(I) + NN(I) with reference anchoring.
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    init_sigma: float = 1e-3
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            occ: (B, N) occupation indices
            
        Returns:
            (B, N, N) occupied submatrix A(I)
        """
        N = self.system.n_elec
        
        # Reference submatrix
        C0 = ReferenceCoefficients(
            self.system,
            init_sigma=self.init_sigma,
            param_dtype=self.param_dtype
        )(occ.shape[0])
        A0 = _gather_occupied_rows(C0, occ)
        
        # Direct submatrix correction
        dA = self.parametrizer(occ, N * N, head="submatrix")
        dA = dA.reshape((occ.shape[0], N, N))
        
        return A0 + dA


@dataclass(frozen=True)
class ThoulessAmplitudes:
    """Container for Thouless particle-hole excitation amplitudes."""
    T: jnp.ndarray  # (B, Nv, No)


class ThoulessMapper(nn.Module):
    """Thouless amplitude mapper: affine chart on Gr(N,M).
    
    Maps I -> T(I) where T_{v,o} are particle-hole excitation amplitudes.
    Provides optimal local parameterization near reference state.
    """
    system: MolecularSystem
    parametrizer: Parametrizer
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ: jnp.ndarray) -> ThoulessAmplitudes:
        """
        Args:
            occ: (B, N) occupation indices
            
        Returns:
            ThoulessAmplitudes with T (B, Nv, No)
        """
        M, N = self.system.n_so, self.system.n_elec
        Nv, No = M - N, N
        
        T = self.parametrizer(occ, Nv * No, head="thouless")
        T = T.reshape((occ.shape[0], Nv, No))
        
        return ThoulessAmplitudes(T=T)


__all__ = [
    "MapperType",
    "ReferenceMapper",
    "FullMapper",
    "SubmatrixMapper",
    "ThoulessMapper",
    "ThoulessAmplitudes",
]
