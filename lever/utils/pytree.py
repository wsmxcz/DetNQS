# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PyTree arithmetic operations for parameter space manipulation.

Provides element-wise operations on PyTree structures, fully JIT-compatible
with JAX's tree mapping infrastructure.

File: lever/utils/pytree.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .dtypes import PyTree


def tree_dot(a: PyTree, b: PyTree) -> jnp.ndarray:
    """
    Compute inner product: ⟨a, b⟩ = Σᵢ conj(aᵢ)·bᵢ.
    
    Flattens both trees and computes conjugate-linear inner product.
    Handles complex parameters correctly via conjugation.
    
    Args:
        a: First PyTree (e.g., gradient)
        b: Second PyTree (e.g., search direction)
        
    Returns:
        Scalar product (complex if inputs are complex)
        
    Example:
        >>> p = {'w': jnp.array([1+1j, 2]), 'b': jnp.array([3])}
        >>> g = {'w': jnp.array([0.1, 0.2]), 'b': jnp.array([0.3])}
        >>> tree_dot(g, p)  # 0.1·(1-1j) + 0.2·2 + 0.3·3
    """
    products = jax.tree.map(lambda x, y: jnp.sum(jnp.conj(x) * y), a, b)
    leaves = jax.tree.leaves(products)
    
    if not leaves:
        return jnp.array(0.0)
    
    # Stack and sum in single JAX op for better JIT efficiency
    return jnp.sum(jnp.stack(leaves))


def tree_scale(tree: PyTree, scalar: float | jnp.ndarray) -> PyTree:
    """
    Scalar multiplication: α·tree.
    
    Args:
        tree: PyTree structure
        scalar: Multiplicative factor
        
    Returns:
        Scaled PyTree with preserved structure
    """
    return jax.tree.map(lambda x: scalar * x, tree)


def tree_add(a: PyTree, b: PyTree) -> PyTree:
    """
    Element-wise addition: a + b.
    
    Args:
        a, b: PyTrees with matching structure
        
    Returns:
        Sum PyTree
    """
    return jax.tree.map(lambda x, y: x + y, a, b)


def tree_sub(a: PyTree, b: PyTree) -> PyTree:
    """
    Element-wise subtraction: a - b.
    
    Args:
        a, b: PyTrees with matching structure
        
    Returns:
        Difference PyTree
    """
    return jax.tree.map(lambda x, y: x - y, a, b)


def tree_norm(tree: PyTree) -> jnp.ndarray:
    """
    Compute L2 norm: ||tree|| = √⟨tree, tree⟩.
    
    Args:
        tree: PyTree structure
        
    Returns:
        Scalar norm (real-valued)
    """
    return jnp.sqrt(jnp.real(tree_dot(tree, tree)))


__all__ = [
    "tree_dot",
    "tree_scale",
    "tree_add",
    "tree_sub",
    "tree_norm",
]
