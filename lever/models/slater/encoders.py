# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Permutation-invariant encoders for occupied spin-orbital indices.

Provides:
  - EmbeddingPoolEncoder: sum/mean pooling over token embeddings
  - TransformerEncoder: set transformer with global token aggregation

Backend options:
  - 'gather': jnp.take (may suffer from scatter-add contention on small n_so)
  - 'matmul': one_hot @ E (GEMM-based gradients, often faster for n_so < 256)

File: lever/models/slater/encoders.py
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


def _embed_tokens(
    occ_so: jnp.ndarray,
    E: jnp.ndarray,
    backend: EmbedBackend,
) -> jnp.ndarray:
    """
    Embed occupied spin-orbital indices via gather or matmul path.
  
    Args:
        occ_so: (..., n_elec) int32 indices
        E: (n_so, dim) embedding matrix
        backend: 'gather' or 'matmul'
  
    Returns:
        (..., n_elec, dim) embedded tokens
    """
    if backend == "matmul":
        # one_hot @ E: GEMM-based gradients
        oh = jax.nn.one_hot(occ_so, E.shape[0], dtype=E.dtype)
        return oh @ E
    else:
        # jnp.take: scatter-add gradients (may contend on small n_so)
        return jnp.take(E, occ_so, axis=0)


class EmbeddingPoolEncoder(nn.Module):
    """
    Permutation-invariant encoder via embedding + pooling.
  
    Maps (..., n_elec) occupied indices → (..., dim) aggregated vector.
    """
    n_so: int
    dim: int
    pool: Pool = "sum"
    backend: EmbedBackend = "gather"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """Assumes occ_so is int32; output shape (..., dim)."""
        E = self.param(
            "embedding",
            initializers.glorot_uniform(),
            (self.n_so, self.dim),
            self.param_dtype,
        )
        b = self.param("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
      
        tokens = _embed_tokens(occ_so, E, self.backend)  # (..., n_elec, dim)
        x = jnp.sum(tokens, axis=-2) + b
      
        if self.pool == "mean":
            n_elec = occ_so.shape[-1]
            x = x / jnp.maximum(1.0, float(n_elec))
      
        return x


class TransformerEncoder(nn.Module):
    """
    Set transformer with global token for permutation-invariant encoding.
  
    Architecture:
      1. Embed occupied indices: (..., n_elec, dim)
      2. Prepend learnable global token: (..., n_elec+1, dim)
      3. Self-attention blocks with pre-LayerNorm
      4. Extract global token as output: (..., dim)
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
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """Assumes occ_so is int32; output shape (..., dim)."""
        batch_shape = occ_so.shape[:-1]
      
        # Embed occupied indices: (..., n_elec, dim)
        E = self.param(
            "embedding",
            initializers.glorot_uniform(),
            (self.n_so, self.dim),
            self.param_dtype,
        )
        tokens = _embed_tokens(occ_so, E, self.backend)
      
        # Prepend global token: (..., n_elec+1, dim)
        global_token = self.param(
            "global_token",
            nn.initializers.zeros,
            (self.dim,),
            self.param_dtype,
        )
        global_token = jnp.broadcast_to(global_token, batch_shape + (1, self.dim))
        x = jnp.concatenate([global_token, tokens], axis=-2)
      
        # Transformer blocks: pre-LN + residual
        act_fn = nn.gelu if self.activation == "gelu" else nn.relu
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
          
            # MLP
            h = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.Dense(self.mlp_ratio * self.dim, param_dtype=self.param_dtype)(h)
            h = act_fn(h)
            h = nn.Dense(self.dim, param_dtype=self.param_dtype)(h)
            x = x + h
      
        # Extract global token: (..., dim)
        return nn.LayerNorm(param_dtype=self.param_dtype)(x[..., 0, :])


__all__ = ["EmbeddingPoolEncoder", "TransformerEncoder", "Pool", "Activation", "EmbedBackend"]