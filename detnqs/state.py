# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
State management for NQS-enhanced selected CI optimization.

Core components:
  - Immutable State dataclass bridging CPU determinant space to device computation
  - Batch preparation: CPU uint64 dets -> device DetBatch
  - Forward/backward passes with optional memory-efficient chunking
  - Streaming inference: forward_dets() for block-wise H2D transfer

File: detnqs/state.py
Author: Zheng (Alex) Che, wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from .space import DetSpace
from .system import MolecularSystem
from .utils.det_utils import DetBatch, get_batch_spec, prepare_batch
from .utils.jax_utils import PyTree, apply_chunked, vjp_chunked

ApplyFn = Callable[[PyTree, DetBatch], tuple[jnp.ndarray, jnp.ndarray]]
VjpFn = Callable[[tuple[jnp.ndarray, jnp.ndarray]], PyTree]


@functools.partial(jax.jit, static_argnums=(0,))
def _apply_model(model: Any, params: PyTree, batch: DetBatch):
    """JITed model.apply with model treated as static."""
    batch = batch.replace(occ=batch.occ.astype(jnp.int32))
    return model.apply({"params": params}, batch)


def _pad_dets_last(dets: np.ndarray, target: int) -> tuple[np.ndarray, int]:
    """
    Pad determinants to fixed size by repeating last row (static shape for XLA).

    Returns:
        (padded_dets, n_orig): Padded array and original count
    """
    dets = np.asarray(dets, dtype=np.uint64)
    n = int(dets.shape[0])
    if n == 0:
        return dets, 0
    if n >= target:
        return dets[:target], target
    pad = np.repeat(dets[-1:], target - n, axis=0)
    return np.concatenate([dets, pad], axis=0), n


@struct.dataclass(eq=False)
class State:
    """
    Immutable state container for inner optimization loop.

    Attributes:
        system: Molecular system reference (static)
        model: Neural network model (static, defines batch_spec)
        params: Network parameters (PyTree, updated per inner step)
        batch: Device DetBatch (updated per outer loop)
    """

    system: MolecularSystem = struct.field(pytree_node=False)
    model: Any = struct.field(pytree_node=False)
    params: PyTree
    batch: DetBatch

    @classmethod
    def init(
        cls,
        system: MolecularSystem,
        detspace: DetSpace,
        model: Any,
        *,
        key: jax.Array | None = None,
    ) -> "State":
        """
        Initialize state from molecular system and determinant space.

        Workflow:
          1. Single-sample batch for shape inference
          2. Initialize model parameters
          3. Build full device batch from detspace.T_dets
        """
        T = np.asarray(detspace.T_dets, dtype=np.uint64)
        if T.shape[0] == 0:
            raise ValueError("Empty determinant space")

        spec = get_batch_spec(model)
        key = jax.random.PRNGKey(42) if key is None else key

        # Single-sample batch for param initialization
        b1 = prepare_batch(
            T[:1],
            n_orb=system.n_orb,
            n_alpha=system.n_alpha,
            n_beta=system.n_beta,
            ref=system.hf_determinant(),
            spec=spec,
        )

        variables = model.init(key, jax.device_put(b1))
        params = variables["params"]

        tmp = cls(system=system, model=model, params=params, batch=b1)
        return tmp.update_space(detspace)

    def update_space(self, detspace: DetSpace, *, device_space: str = "T") -> "State":
        """
        Rebuild device batch from new determinant space (outer-loop transition).

        CPU->device transfer: dets -> prepare_batch -> device_put

        Args:
            detspace: DetSpace containing V_dets and T_dets
            device_space: "V" for variational set only; "T" for full target set V_k ∪ P_k
        """
        if device_space not in ("V", "T"):
            raise ValueError(f"device_space must be 'V' or 'T', got {device_space}")

        dets = detspace.V_dets if device_space == "V" else detspace.T_dets
        dets = np.asarray(dets, dtype=np.uint64)
        if dets.shape[0] == 0:
            empty = DetBatch(occ=np.zeros((0, self.system.n_elec), dtype=np.int32), aux={})
            return self.replace(batch=jax.device_put(empty))

        spec = get_batch_spec(self.model)
        cpu_batch = prepare_batch(
            dets,
            n_orb=self.system.n_orb,
            n_alpha=self.system.n_alpha,
            n_beta=self.system.n_beta,
            ref=self.system.hf_determinant(),
            spec=spec,
        )
        return self.replace(batch=jax.device_put(cpu_batch))

    def _apply(self, p: PyTree, b: DetBatch) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Apply model: (params, batch) -> (sign, logabs)."""
        return _apply_model(self.model, p, b)

    @property
    def n_det(self) -> int:
        """Number of determinants in current batch."""
        return int(self.batch.occ.shape[0])

    @property
    def psi_dtype(self) -> jnp.dtype:
        """Infer wavefunction dtype from single forward pass."""
        b1 = self.batch[:1]
        s, la = self._apply(self.params, b1)
        return s.dtype if jnp.issubdtype(s.dtype, jnp.complexfloating) else jnp.real(la).dtype

    def forward_dets(
        self,
        dets: np.ndarray,
        *,
        block_size: int,
        chunk_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass on arbitrary CPU determinants via H2D streaming.

        Uses fixed block_size padding to maintain static shapes (avoid recompilation).
        Memory flow: CPU dets -> pad -> H2D -> model -> D2H -> trim

        Args:
            dets: CPU determinants [N, 2], dtype uint64
            block_size: Fixed batch size for padding (static XLA shape)
            chunk_size: Optional chunking within block to reduce peak VRAM

        Returns:
            (sign, logabs): Model outputs on original N samples
        """
        bs = int(block_size)
        if bs <= 0:
            raise ValueError("block_size must be positive")

        dets_pad, n_orig = _pad_dets_last(dets, bs)
        if n_orig == 0:
            s1, la1 = self._apply(self.params, self.batch[:1])
            return (
                np.zeros((0,), dtype=np.asarray(s1).dtype),
                np.zeros((0,), dtype=np.asarray(jnp.real(la1)).dtype),
            )

        spec = get_batch_spec(self.model)
        cpu_batch = prepare_batch(
            dets_pad,
            n_orb=self.system.n_orb,
            n_alpha=self.system.n_alpha,
            n_beta=self.system.n_beta,
            ref=self.system.hf_determinant(),
            spec=spec,
        )
        b_dev = jax.device_put(cpu_batch)

        if chunk_size is None:
            sign, logabs = self._apply(self.params, b_dev)
        else:
            def _fn(p, b):
                return _apply_model(self.model, p, b)

            sign, logabs = apply_chunked(_fn, self.params, b_dev, int(chunk_size))

        # device_get blocks until ready; frees transient device buffers
        sign_h, logabs_h = jax.device_get((sign, logabs))
        return np.asarray(sign_h)[:n_orig], np.asarray(logabs_h)[:n_orig]

    def forward(
        self,
        indices: jnp.ndarray | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass on device batch: params -> (sign, logabs).

        Args:
            indices: Optional subset of batch indices
            chunk_size: Process in chunks if set to reduce activation peak

        Returns:
            (sign, logabs): Model outputs
        """
        b = self.batch if indices is None else self.batch[indices]
        n = int(b.occ.shape[0])

        if n == 0:
            s1, la1 = self._apply(self.params, self.batch[:1])
            return (
                jnp.zeros((0,), dtype=s1.dtype),
                jnp.zeros((0,), dtype=jnp.real(la1).dtype),
            )

        if chunk_size is None:
            return self._apply(self.params, b)

        return apply_chunked(self._apply, self.params, b, int(chunk_size))

    def value_and_vjp(
        self,
        indices: jnp.ndarray | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], VjpFn]:
        """
        Compute forward outputs and VJP function for backpropagation.

        Returns:
            ((sign, logabs), vjp_fn): Primal outputs and gradient pullback function
        """
        b = self.batch if indices is None else self.batch[indices]
        n = int(b.occ.shape[0])

        if n == 0:
            empty_grad = jax.tree_util.tree_map(jnp.zeros_like, self.params)
            s1, la1 = self._apply(self.params, self.batch[:1])
            outs0 = (
                jnp.zeros((0,), dtype=s1.dtype),
                jnp.zeros((0,), dtype=jnp.real(la1).dtype),
            )
            return outs0, (lambda _: empty_grad)

        if chunk_size is None:
            outs, pullback = jax.vjp(lambda p: self._apply(p, b), self.params)

            def vjp_fn(cot: tuple[jnp.ndarray, jnp.ndarray]) -> PyTree:
                cot_s, cot_la = cot
                cot_s = cot_s.astype(outs[0].dtype)
                cot_la = cot_la.astype(jnp.real(outs[1]).dtype)

                # Repack cotangents to match primal output pytree structure
                cot_like = jax.tree_util.tree_unflatten(
                    jax.tree_util.tree_structure(outs),
                    [cot_s, cot_la],
                )

                (grads,) = pullback(cot_like)
                return jax.tree_util.tree_map(
                    lambda g, p: (g.astype(p.dtype) if g is not None else jnp.zeros_like(p)),
                    grads,
                    self.params,
                )

            return outs, vjp_fn

        outs = self.forward(indices, chunk_size=chunk_size)

        def vjp_fn(cot: tuple[jnp.ndarray, jnp.ndarray]) -> PyTree:
            return vjp_chunked(self._apply, self.params, b, cot, int(chunk_size))

        return outs, vjp_fn

    def apply_gradients(
        self,
        gradients: PyTree,
        opt_state: Any,
        optimizer: optax.GradientTransformation,
    ) -> tuple["State", Any]:
        """
        Apply gradients via Optax optimizer.

        Returns:
            (new_state, new_opt_state): Updated state and optimizer state
        """
        updates, new_opt_state = optimizer.update(gradients, opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(params=new_params), new_opt_state


__all__ = ["State"]