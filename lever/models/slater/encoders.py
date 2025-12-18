# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Permutation-invariant encoders for occupied spin-orbital indices.

Provides embedding-pool and transformer encoders that map occ_so → latent vector.
No positional encodings used to preserve permutation symmetry.

File: lever/models/slater/encoders.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

Pool = Literal["sum", "mean"]
Activation = Literal["gelu", "relu", "tanh"]


class EmbeddingPoolEncoder(nn.Module):
    """
    Batch-native permutation-invariant encoder.
    
    Input: (..., n_elec) occupied spin-orbital indices
    Output: (..., dim) aggregated embedding
    """
    n_so: int
    dim: int
    pool: Pool = "sum"
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            occ_so: shape (..., n_elec) - arbitrary batch dimensions
        
        Returns:
            shape (..., dim)
        """
        occ_so = occ_so.astype(jnp.int32)
        
        E = self.param(
            "embedding",
            initializers.glorot_uniform(),
            (self.n_so, self.dim),
            self.param_dtype,
        )
        b = self.param("bias", nn.initializers.zeros, (self.dim,), self.param_dtype)
        
        # Batch gather: E[occ_so] -> (..., n_elec, dim)
        tokens = jnp.take(E, occ_so, axis=0)
        
        # Pool over electron dimension
        x = jnp.sum(tokens, axis=-2) + b
        
        if self.pool == "mean":
            n_elec = occ_so.shape[-1]
            x = x / jnp.maximum(1.0, float(n_elec))
        
        return x


class TransformerEncoder(nn.Module):
    """Batch-native set transformer with global token."""
    n_so: int
    dim: int = 64
    depth: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4
    activation: Activation = "gelu"
    param_dtype: Any = jnp.float64

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            occ_so: shape (..., n_elec)
        
        Returns:
            shape (..., dim) - global token after attention
        """
        occ_so = occ_so.astype(jnp.int32)
        batch_shape = occ_so.shape[:-1]
        n_elec = occ_so.shape[-1]
        
        # Embed: (..., n_elec, dim)
        E = self.param(
            "embedding",
            initializers.glorot_uniform(),
            (self.n_so, self.dim),
            self.param_dtype,
        )
        tokens = jnp.take(E, occ_so, axis=0)
        
        # Prepend global token: (..., 1, dim)
        global_token = self.param(
            "global_token",
            nn.initializers.zeros,
            (self.dim,),
            self.param_dtype,
        )
        global_token = jnp.broadcast_to(
            global_token, batch_shape + (1, self.dim)
        )
        x = jnp.concatenate([global_token, tokens], axis=-2)  # (..., n_elec+1, dim)
        
        # Transformer blocks
        act_fn = nn.gelu if self.activation == "gelu" else nn.relu
        for _ in range(self.depth):
            # Self-attention with pre-LN
            h = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.dim,
                out_features=self.dim,
                param_dtype=self.param_dtype,
            )(h, h)
            x = x + h
            
            # MLP with pre-LN
            h = nn.LayerNorm(param_dtype=self.param_dtype)(x)
            h = nn.Dense(self.mlp_ratio * self.dim, param_dtype=self.param_dtype)(h)
            h = act_fn(h)
            h = nn.Dense(self.dim, param_dtype=self.param_dtype)(h)
            x = x + h
        
        # Return global token: (..., dim)
        return nn.LayerNorm(param_dtype=self.param_dtype)(x[..., 0, :])


__all__ = ["EmbeddingPoolEncoder", "TransformerEncoder", "Pool", "Activation"]