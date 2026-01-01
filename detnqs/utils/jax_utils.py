# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
JAX utilities for PyTree operations and memory-efficient chunking.

Provides:
  - Tree algebra: dot products, norms, element-wise ops
  - Memory-efficient batching: chunked forward/backward passes

File: detnqs/utils/jax_utils.py
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
    """Inner product: <a|b> = sum_i conj(a_i) * b_i."""
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return sum(jnp.sum(jnp.conj(x) * y) for x, y in zip(leaves_a, leaves_b))


def tree_norm(tree: PyTree) -> jnp.ndarray:
    """L2 norm: ||tree|| = sqrt(<tree|tree>)."""
    return jnp.sqrt(tree_dot(tree, tree).real)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    """Element-wise addition."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    """Element-wise subtraction."""
    return jax.tree.map(lambda x, y: x - y, a, b)


def tree_scale(scalar: float | jnp.ndarray, tree: PyTree) -> PyTree:
    """Scalar multiplication."""
    return jax.tree.map(lambda x: scalar * x, tree)


def tree_conj(tree: PyTree) -> PyTree:
    """Element-wise conjugation."""
    return jax.tree.map(jnp.conj, tree)


def _leading_dim(tree: PyTree) -> int:
    """Extract static leading dimension from first leaf for JIT compatibility."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        raise ValueError("Empty PyTree")
    n = leaves[0].shape[0]
    if not isinstance(n, int):
        raise ValueError("Chunking requires static leading dimension")
    return n

def _repack_like(reference: PyTree, tree: PyTree) -> PyTree:
    """Repack `tree` leaves into the same pytree structure as `reference`."""
    leaves = jax.tree_util.tree_leaves(tree)
    return jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(reference), leaves)

def apply_chunked(
    apply_fn: Callable[[PyTree, PyTree], PyTree],
    params: PyTree,
    inputs: PyTree,
    chunk_size: int,
) -> PyTree:
    """
    Apply function in chunks over batch dimension to reduce memory.
  
    Algorithm: Pad inputs (replicate last) to multiple of chunk_size,
    reshape to (n_chunks, chunk_size, ...), scan over chunks, trim output.
  
    Args:
        apply_fn: f(params, inputs) -> outputs
        params: Model parameters
        inputs: Batched inputs (leading dim N)
        chunk_size: Chunk size
      
    Returns:
        Concatenated outputs for all N samples
    """
    n_samples = _leading_dim(inputs)
    if n_samples <= chunk_size:
        return apply_fn(params, inputs)

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size
    pad_len = n_padded - n_samples

    def pad_last(x):
        if pad_len <= 0:
            return x
        pad = jnp.broadcast_to(x[-1:], (pad_len,) + x.shape[1:])
        return jnp.concatenate([x, pad], axis=0)

    inputs_pad = jax.tree_util.tree_map(pad_last, inputs)

    def reshape_in(x):
        return x.reshape((n_chunks, chunk_size) + x.shape[1:])

    inputs_2d = jax.tree_util.tree_map(reshape_in, inputs_pad)

    def scan_body(carry, x_chunk):
        y_chunk = apply_fn(params, x_chunk)
        return carry, y_chunk

    _, outputs = lax.scan(scan_body, None, inputs_2d)

    def merge_and_trim(leaf):
        leaf = leaf.reshape((n_padded,) + leaf.shape[2:])
        return leaf[:n_samples]

    return jax.tree_util.tree_map(merge_and_trim, outputs)


def vjp_chunked(
    apply_fn: Callable[[PyTree, PyTree], PyTree],
    params: PyTree,
    inputs: PyTree,
    cotangents: PyTree,
    chunk_size: int,
) -> PyTree:
    """
    Vector-Jacobian product in chunks to reduce memory.
  
    Algorithm: Pad inputs (replicate last) and cotangents (zeros),
    reshape to (n_chunks, chunk_size, ...), accumulate grad_params
    via scan: grad = sum_i vjp_i(cotangent_i).
  
    Args:
        apply_fn: f(params, inputs) -> outputs
        params: Model parameters
        inputs: Batched inputs (leading dim N)
        cotangents: Gradient w.r.t. outputs
        chunk_size: Chunk size
      
    Returns:
        Gradient w.r.t. params
    """
    n_samples = _leading_dim(inputs)
    if n_samples <= chunk_size:
        out, vjp_fn = jax.vjp(lambda p: apply_fn(p, inputs), params)
        cot_like = _repack_like(out, cotangents)
        (grads,) = vjp_fn(cot_like)
        return grads

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    n_padded = n_chunks * chunk_size
    pad_len = n_padded - n_samples

    def pad_last(x):
        if pad_len <= 0:
            return x
        pad = jnp.broadcast_to(x[-1:], (pad_len,) + x.shape[1:])
        return jnp.concatenate([x, pad], axis=0)

    def pad_zeros(x):
        if pad_len <= 0:
            return x
        zeros = jnp.zeros((pad_len,) + x.shape[1:], dtype=x.dtype)
        return jnp.concatenate([x, zeros], axis=0)

    inputs_pad = jax.tree_util.tree_map(pad_last, inputs)
    cot_pad = jax.tree_util.tree_map(pad_zeros, cotangents)

    def reshape_2d(x):
        return x.reshape((n_chunks, chunk_size) + x.shape[1:])

    inputs_2d = jax.tree_util.tree_map(reshape_2d, inputs_pad)
    cot_2d = jax.tree_util.tree_map(reshape_2d, cot_pad)

    grad_acc = jax.tree_util.tree_map(jnp.zeros_like, params)

    def scan_body(grad_acc, scan_in):
        x_chunk, cot_chunk = scan_in
        out, vjp_fn = jax.vjp(lambda p: apply_fn(p, x_chunk), params)
        cot_like = _repack_like(out, cot_chunk)    # <-- critical
        (g_chunk,) = vjp_fn(cot_like)
        grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad_acc, g_chunk)
        return grad_acc, None

    grads, _ = lax.scan(scan_body, grad_acc, (inputs_2d, cot_2d))
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