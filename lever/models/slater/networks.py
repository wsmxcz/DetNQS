# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parametrizers: occupation-conditioned neural networks with multi-head outputs.

Base protocol: (occ_so, out_dim, head=...) → R^{out_dim}
Supports shared trunk with named output heads to avoid parameter collisions.

File: lever/models/slater/networks.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from .encoders import EmbeddingPoolEncoder, TransformerEncoder, Pool, Activation


def _small_init(scale: float) -> Any:
    """Truncated normal initializer for near-zero output initialization."""
    return initializers.variance_scaling(
        scale=scale, mode="fan_avg", distribution="truncated_normal"
    )


class Parametrizer(nn.Module):
    """
    Base parametrizer with multi-head output projection.
    
    Subclasses implement encode(occ_so) to produce latent representations.
    """

    def encode(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError  # pragma: no cover

    @nn.compact
    def __call__(
        self, occ_so: jnp.ndarray, out_dim: int, *, head: str = "default"
    ) -> jnp.ndarray:
        """
        Forward pass: encode occupation and project to output dimension.
        
        Args:
            occ_so: Spin-orbital occupation indices
            out_dim: Output dimension
            head: Named head to avoid parameter collision
            
        Returns:
            Output tensor of shape (out_dim,)
        """
        latent = self.encode(occ_so)
        out_scale = getattr(self, "out_scale", 1e-2)
        param_dtype = getattr(self, "param_dtype", jnp.float64)
        
        return nn.Dense(
            out_dim,
            kernel_init=_small_init(out_scale),
            bias_init=nn.initializers.zeros,
            param_dtype=param_dtype,
            name=f"head_{head}",
        )(latent)


class MLP(Parametrizer):
    """Embedding-pool encoder with feed-forward trunk."""

    n_so: int
    dim: int = 128
    depth: int = 2
    pool: Pool = "sum"
    activation: Activation = "gelu"
    out_scale: float = 1e-2
    param_dtype: Any = jnp.float64

    @nn.compact
    def encode(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        x = EmbeddingPoolEncoder(
            n_so=self.n_so,
            dim=self.dim,
            pool=self.pool,
            param_dtype=self.param_dtype,
        )(occ_so)

        act_fn = nn.gelu if self.activation == "gelu" else nn.relu
        for _ in range(self.depth):
            x = nn.Dense(self.dim, param_dtype=self.param_dtype)(x)
            x = act_fn(x)
        return x


class ResMLP(Parametrizer):
    """Embedding-pool encoder with pre-LayerNorm residual MLP blocks."""

    n_so: int
    dim: int = 64
    depth: int = 2
    pool: Pool = "sum"
    activation: Activation = "gelu"
    out_scale: float = 1e-2
    param_dtype: Any = jnp.float64

    @nn.compact
    def encode(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        x = EmbeddingPoolEncoder(
            n_so=self.n_so,
            dim=self.dim,
            pool=self.pool,
            param_dtype=self.param_dtype,
        )(occ_so)

        act_fn = nn.gelu if self.activation == "gelu" else nn.relu
        for _ in range(self.depth):
            h = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.Dense(self.dim, param_dtype=self.param_dtype)(h)
            h = act_fn(h)
            h = nn.Dense(self.dim, param_dtype=self.param_dtype)(h)
            x = x + h

        return nn.LayerNorm(param_dtype=self.param_dtype)(x)


class Transformer(Parametrizer):
    """Set transformer encoder with global token pooling."""

    n_so: int
    dim: int = 64
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    activation: Activation = "gelu"
    out_scale: float = 1e-2
    param_dtype: Any = jnp.float64

    @nn.compact
    def encode(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        return TransformerEncoder(
            n_so=self.n_so,
            dim=self.dim,
            depth=self.depth,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
            activation=self.activation,
            param_dtype=self.param_dtype,
        )(occ_so)


__all__ = [
    "Parametrizer",
    "MLP",
    "ResMLP",
    "Transformer",
]
