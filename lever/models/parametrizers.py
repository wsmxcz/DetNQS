# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Neural parametrizers for determinant coefficients.

Provides encoders and parametrizers:
  - EmbeddingPoolEncoder: Permutation-invariant embedding + pooling
  - TransformerEncoder: Set transformer with global token
  - Parametrizer: Base class mapping occ -> latent -> heads
  - MLP, Transformer: Concrete parametrizer implementations

File: lever/models/parametrizers.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

# Type aliases
Pool = Literal["sum", "mean"]
Activation = Literal["gelu", "relu", "tanh"]
EmbedBackend = Literal["gather", "matmul"]


# ============================================================================
# Encoders
# ============================================================================

def _embed_tokens(occ: jnp.ndarray, E: jnp.ndarray, backend: EmbedBackend) -> jnp.ndarray:
    """
    Embed occupation indices via gather or one-hot matmul.
    
    Args:
        occ: (..., n_e) occupation indices
        E: (n_so, dim) embedding matrix
        backend: 'gather' (fast) or 'matmul' (XLA-friendly)
        
    Returns:
        (..., n_e, dim) embedded tokens
    """
    if backend == "matmul":
        oh = jax.nn.one_hot(occ, E.shape[0], dtype=E.dtype)
        return oh @ E
    return jnp.take(E, occ, axis=0)


class EmbeddingPoolEncoder(nn.Module):
    """
    Permutation-invariant encoder: embed + pool.
    
    Maps occ -> sum/mean(embed(occ)) + bias.
    """
    n_so: int
    dim: int
    pool: Pool = "sum"
    backend: EmbedBackend = "gather"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, occ: jnp.ndarray) -> jnp.ndarray:
        E = self.param("E", initializers.glorot_uniform(), 
                      (self.n_so, self.dim), self.param_dtype)
        b = self.param("b", nn.initializers.zeros, 
                      (self.dim,), self.param_dtype)

        tokens = _embed_tokens(occ, E, self.backend)  # (..., n_e, dim)
        x = tokens.sum(axis=-2) + b
        
        if self.pool == "mean":
            x = x / jnp.maximum(1.0, float(occ.shape[-1]))
        return x


class TransformerEncoder(nn.Module):
    """
    Set transformer with global token aggregation.
    
    Uses multi-head attention + FFN with residual connections.
    """
    n_so: int
    dim: int = 64
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    activation: Activation = "gelu"
    backend: EmbedBackend = "gather"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, occ: jnp.ndarray) -> jnp.ndarray:
        batch_shape = occ.shape[:-1]

        # Embed occupation indices
        E = self.param("E", initializers.glorot_uniform(),
                      (self.n_so, self.dim), self.param_dtype)
        tokens = _embed_tokens(occ, E, self.backend)  # (..., n_e, dim)

        # Prepend global aggregation token
        g = self.param("g", nn.initializers.zeros, (self.dim,), self.param_dtype)
        g = jnp.broadcast_to(g, batch_shape + (1, self.dim))
        x = jnp.concatenate([g, tokens], axis=-2)  # (..., n_e+1, dim)

        # Activation function
        act = {"gelu": nn.gelu, "relu": nn.relu, "tanh": jnp.tanh}[self.activation]

        # Transformer blocks
        for _ in range(self.depth):
            # Self-attention
            h = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.dim,
                out_features=self.dim,
                param_dtype=self.param_dtype,
            )(h, h)
            x = x + h

            # FFN
            h = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.Dense(self.mlp_ratio * self.dim, param_dtype=self.param_dtype)(h)
            h = act(h)
            h = nn.Dense(self.dim, param_dtype=self.param_dtype)(h)
            x = x + h

        # Extract global token
        return nn.LayerNorm(param_dtype=self.param_dtype)(x[..., 0, :])


# ============================================================================
# Parametrizers
# ============================================================================

def _small_init(scale: float) -> Any:
    """Near-zero variance scaling for stable training start."""
    return initializers.variance_scaling(
        scale=scale, mode="fan_avg", distribution="truncated_normal"
    )


class Parametrizer(nn.Module):
    """
    Base class: encode(occ) -> latent, then project to named heads.
    
    Subclasses implement encode() method.
    """
    out_scale: float = 1e-2
    param_dtype: Any = jnp.float64

    def encode(self, occ: jnp.ndarray) -> jnp.ndarray:
        """Map occ to latent vector. Override in subclass."""
        raise NotImplementedError

    @nn.compact
    def __call__(self, occ: jnp.ndarray, out_dim: int, *, head: str) -> jnp.ndarray:
        """Project latent to output via named head."""
        latent = self.encode(occ)
        return nn.Dense(
            out_dim,
            kernel_init=_small_init(self.out_scale),
            bias_init=nn.initializers.zeros,
            param_dtype=self.param_dtype,
            name=f"head_{head}",
        )(latent)


class MLP(Parametrizer):
    """
    MLP parametrizer: embed + pool + feedforward layers.
    """
    n_so: int = 0
    dim: int = 128
    depth: int = 2
    pool: Pool = "sum"
    activation: Activation = "gelu"

    @nn.compact
    def encode(self, occ: jnp.ndarray) -> jnp.ndarray:
        x = EmbeddingPoolEncoder(
            n_so=self.n_so,
            dim=self.dim,
            pool=self.pool,
            param_dtype=self.param_dtype,
        )(occ)

        act = {"gelu": nn.gelu, "relu": nn.relu, "tanh": jnp.tanh}[self.activation]
        for _ in range(self.depth):
            x = nn.Dense(self.dim, param_dtype=self.param_dtype)(x)
            x = act(x)
        return x


class Transformer(Parametrizer):
    """
    Transformer parametrizer: global token aggregation.
    """
    n_so: int = 0
    dim: int = 64
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    activation: Activation = "gelu"

    @nn.compact
    def encode(self, occ: jnp.ndarray) -> jnp.ndarray:
        return TransformerEncoder(
            n_so=self.n_so,
            dim=self.dim,
            depth=self.depth,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
            activation=self.activation,
            param_dtype=self.param_dtype,
        )(occ)


__all__ = [
    "Pool", "Activation", "EmbedBackend",
    "EmbeddingPoolEncoder", "TransformerEncoder",
    "Parametrizer", "MLP", "Transformer",
]
