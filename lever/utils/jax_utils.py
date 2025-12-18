# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JAX utilities for PyTree operations and memory-efficient chunking.

Provides:
  - Tree algebra: dot products, norms, element-wise operations
  - Memory-efficient batching: chunked forward/backward passes

File: lever/utils/jax_utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

PyTree = Any


def tree_dot(a: PyTree, b: PyTree) -> jnp.ndarray:
    """
    Compute inner product <a, b> = sum_i conj(a_i) * b_i.
    
    Returns complex scalar for complex inputs, real for real inputs.
    """
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return sum(jnp.sum(jnp.conj(x) * y) for x, y in zip(leaves_a, leaves_b))


def tree_norm(tree: PyTree) -> jnp.ndarray:
    """Compute L2 norm: ||tree|| = sqrt(<tree, tree>)."""
    return jnp.sqrt(tree_dot(tree, tree).real)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    """Element-wise addition: a + b."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    """Element-wise subtraction: a - b."""
    return jax.tree.map(lambda x, y: x - y, a, b)


def tree_scale(scalar: float | jnp.ndarray, tree: PyTree) -> PyTree:
    """Scalar multiplication: scalar * tree."""
    return jax.tree.map(lambda x: scalar * x, tree)


def tree_conj(tree: PyTree) -> PyTree:
    """Element-wise complex conjugation."""
    return jax.tree.map(jnp.conj, tree)


def apply_chunked(
    apply_fn: Callable[[PyTree, jnp.ndarray], PyTree],
    params: PyTree,
    inputs: jnp.ndarray,
    chunk_size: int,
) -> PyTree:
    """
    Apply function in chunks to reduce memory usage.
    
    Splits input batch into chunks of size chunk_size, applies function
    to each chunk via lax.scan, then concatenates results. Pads last
    chunk by repeating final sample if needed.
    
    Args:
        apply_fn: Function (params, inputs) -> outputs (PyTree)
        params: Model parameters
        inputs: Input array, shape (n_samples, ...)
        chunk_size: Maximum samples per chunk
    
    Returns:
        PyTree with leading dimension n_samples
    """
    n_samples = inputs.shape[0]
    if n_samples <= chunk_size:
        return apply_fn(params, inputs)
    
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size
    pad_len = n_padded - n_samples
    input_shape = inputs.shape[1:]
    
    if pad_len > 0:
        padding = jnp.broadcast_to(inputs[-1:], (pad_len, *input_shape))
        inputs = jnp.concatenate([inputs, padding], axis=0)
    
    inputs = inputs.reshape(n_chunks, chunk_size, *input_shape)
    
    def scan_body(carry, x_chunk):
        y_chunk = apply_fn(params, x_chunk)
        return carry, y_chunk
    
    _, outputs = lax.scan(scan_body, None, inputs)
    
    def merge_and_trim(leaf):
        leaf = leaf.reshape((n_padded,) + leaf.shape[2:])
        return leaf[:n_samples]
    
    return jax.tree_util.tree_map(merge_and_trim, outputs)


def vjp_chunked(
    apply_fn: Callable[[PyTree, jnp.ndarray], PyTree],
    params: PyTree,
    inputs: jnp.ndarray,
    cotangents: PyTree,
    chunk_size: int,
) -> PyTree:
    """
    Compute VJP in chunks to reduce memory usage.
    
    Splits forward and backward passes into chunks, accumulating
    gradients via lax.scan. Input padding replicates last sample;
    cotangent padding uses zeros.
    
    Args:
        apply_fn: Function (params, inputs) -> outputs (PyTree)
        params: Model parameters
        inputs: Input array, shape (n_samples, ...)
        cotangents: Output gradients (PyTree with leading dim n_samples)
        chunk_size: Maximum samples per chunk
    
    Returns:
        Gradient PyTree with same structure as params
    """
    n_samples = inputs.shape[0]
    if n_samples <= chunk_size:
        _, vjp_fn = jax.vjp(lambda p: apply_fn(p, inputs), params)
        (grads,) = vjp_fn(cotangents)
        return grads
    
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size
    pad_len = n_padded - n_samples
    input_shape = inputs.shape[1:]
    
    if pad_len > 0:
        in_pad = jnp.broadcast_to(inputs[-1:], (pad_len, *input_shape))
        inputs = jnp.concatenate([inputs, in_pad], axis=0)
    
    inputs = inputs.reshape(n_chunks, chunk_size, *input_shape)
    
    def pad_and_reshape_cot(leaf):
        if pad_len > 0:
            zeros = jnp.zeros((pad_len,) + leaf.shape[1:], dtype=leaf.dtype)
            leaf = jnp.concatenate([leaf, zeros], axis=0)
        return leaf.reshape((n_chunks, chunk_size) + leaf.shape[1:])
    
    cotangents = jax.tree_util.tree_map(pad_and_reshape_cot, cotangents)
    grad_acc = jax.tree_util.tree_map(jnp.zeros_like, params)
    
    def scan_body(grad_acc, scan_in):
        x_chunk, cot_chunk = scan_in
        _, vjp_fn = jax.vjp(lambda p: apply_fn(p, x_chunk), params)
        (grad_chunk,) = vjp_fn(cot_chunk)
        grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad_acc, grad_chunk)
        return grad_acc, None
    
    grads, _ = lax.scan(scan_body, grad_acc, (inputs, cotangents))
    return grads


__all__ = [
    "PyTree",
    "tree_dot",
    "tree_norm",
    "tree_add",
    "tree_sub",
    "tree_scale",
    "tree_conj",
    "apply_chunked",
    "vjp_chunked",
]
