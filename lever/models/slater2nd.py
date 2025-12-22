# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Second-quantized Slater determinant with neural backflow.

Implements configuration-dependent orbital ansatz via composition:
  - Parametrizer: I -> latent feature
  - Mapper: latent -> manifold coordinate (reference/thouless/full/submatrix)
  - Engine: coordinate -> log|Ψ(I)|

Thouless mode uses fast k×k determinant evaluation.

File: lever/models/slater2nd.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations
from typing import Any

import jax.numpy as jnp
from flax import linen as nn

from ..system import MolecularSystem
from ..utils.det_utils import BatchSpec, DetBatch
from .parametrizers import Parametrizer, MLP
from .mappers import (
    MapperType,
    ReferenceMapper,
    FullMapper,
    SubmatrixMapper,
    ThoulessMapper,
)
from .slogdet import slogdet, slogdet_thouless


class Slater2ndLogAmplitude(nn.Module):
    """Second-quantized Slater determinant with neural backflow.
    
    Computes log-amplitude log|Ψ(I)| = log|det(A(I))| where A(I)
    is predicted by neural network based on occupation I.
    
    Attributes:
        system: Molecular system configuration
        parametrizer: Neural feature extractor (unused for 'reference')
        mapper: Coordinate chart on Grassmann manifold
        kmax: Maximum excitation rank for Thouless mode
        init_sigma: Reference state initialization noise
        use_fast_kernel: Enable FFI determinant acceleration
        param_dtype: Parameter precision
    """
    system: MolecularSystem
    parametrizer: Parametrizer | None = None
    mapper: MapperType = "thouless"
    kmax: int = 16
    init_sigma: float = 1e-3
    use_fast_kernel: bool = True
    param_dtype: Any = jnp.float64

    @property
    def batch_spec(self) -> BatchSpec:
        """Specify auxiliary fields for batch preprocessing."""
        if self.mapper == "thouless":
            return BatchSpec(
                need_k=True,
                hp_kmax=self.kmax,
                need_phase=True,
                need_hp_pos=True,
            )
        return BatchSpec()

    def setup(self):
        """Initialize mapper module based on coordinate choice."""
        mapper_classes = {
            "reference": ReferenceMapper,
            "full": FullMapper,
            "submatrix": SubmatrixMapper,
            "thouless": ThoulessMapper,
        }
        
        if self.mapper not in mapper_classes:
            raise ValueError(f"Unknown mapper: {self.mapper!r}")
        
        cls = mapper_classes[self.mapper]
        kwargs = {
            "system": self.system,
            "param_dtype": self.param_dtype,
        }
        
        # Add parametrizer for backflow modes
        if self.mapper in ("full", "submatrix", "thouless"):
            if self.parametrizer is None:
                raise ValueError(f"Mapper '{self.mapper}' requires parametrizer")
            kwargs["parametrizer"] = self.parametrizer
        
        # Add initialization noise for reference-based modes
        if self.mapper in ("reference", "full", "submatrix"):
            kwargs["init_sigma"] = self.init_sigma
        
        self.mapper_module = cls(**kwargs)

    def __call__(self, batch: DetBatch) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute log-amplitude for batch of determinants.
        
        Args:
            batch: DetBatch with occ (B, N) and optional aux fields
            
        Returns:
            (sign, logabs): Sign in {-1, 0, 1} and log|Ψ|, each (B,)
        """
        occ = batch.occ
        
        # Map occupation to manifold coordinate
        coordinate = self.mapper_module(occ)
        
        # Route to appropriate slogdet implementation
        if self.mapper == "thouless":
            return slogdet_thouless(
                coordinate,
                batch,
                kmax=self.kmax,
                use_fast_kernel=self.use_fast_kernel
            )
        
        # Standard N×N determinant for other modes
        return slogdet(coordinate, use_fast_kernel=self.use_fast_kernel)


def make_slater2nd(
    system: MolecularSystem,
    *,
    parametrizer: Parametrizer | None = None,
    mapper: MapperType = "thouless",
    kmax: int = 16,
    init_sigma: float = 1e-3,
    use_fast_kernel: bool = True,
    param_dtype: Any = jnp.float64,
) -> Slater2ndLogAmplitude:
    """Factory function for Slater2nd model.
    
    Args:
        system: Molecular system
        parametrizer: Neural encoder, defaults to 2-layer MLP(128)
        mapper: Coordinate chart type
        kmax: Maximum excitation rank for Thouless
        init_sigma: Reference initialization noise
        use_fast_kernel: Enable FFI determinant kernel
        param_dtype: Parameter precision
        
    Returns:
        Configured Slater2ndLogAmplitude module
    """
    # Default parametrizer for backflow modes
    if mapper in ("full", "submatrix", "thouless") and parametrizer is None:
        parametrizer = MLP(
            n_so=system.n_so,
            dim=128,
            depth=2,
            pool="sum",
            out_scale=1e-2,
            param_dtype=param_dtype,
        )

    return Slater2ndLogAmplitude(
        system=system,
        parametrizer=parametrizer,
        mapper=mapper,
        kmax=kmax,
        init_sigma=init_sigma,
        use_fast_kernel=use_fast_kernel,
        param_dtype=param_dtype,
    )


__all__ = ["Slater2ndLogAmplitude", "make_slater2nd"]
