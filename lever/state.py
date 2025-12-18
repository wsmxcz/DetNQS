# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic variational state for sampling-free selected CI NQS.

Implements batch-native neural log(psi) on fixed determinant spaces with
VJP-based gradients. Features chunked forward/backward passes and
functional outer-loop updates.

Wavefunction parametrization: psi = sign * exp(logabs)
  - sign: real (±1) or complex (unit magnitude phase)
  - logabs: real log-amplitude

File: lever/state.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct

from .space import DetSpace
from .system import MolecularSystem
from .utils.jax_utils import PyTree, apply_chunked, vjp_chunked
from .utils.occupations import bitstrings_to_indices


# Type aliases
ApplyFn = Callable[[PyTree, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]
VjpFn = Callable[[tuple[jnp.ndarray, jnp.ndarray]], PyTree]


@struct.dataclass(eq=False)
class DeterministicState:
    """
    Variational quantum state on selected CI space with batch-native neural log(psi).
    
    Manages network parameters and determinant space for sampling-free
    wavefunction evaluation. Supports arbitrary batch dimensions for
    efficient GPU utilization.
    
    Attributes:
        system: Molecular system (integrals, electron counts)
        detspace: Current determinant space (T_dets managed by Driver)
        apply_fn: Batch-native model (params, occ_batch) -> (sign, logabs)
        params: Network parameter PyTree
        _T_dets: Device-resident bitstrings [n_T, 2]
        _occ_so: Precomputed spin-orbital indices [n_T, n_elec] (uint8)
        sign_dtype: Output dtype for sign component
        logabs_dtype: Output dtype for logabs component
        wf_is_complex: True if wavefunction has complex phase
    """
    
    system: MolecularSystem = struct.field(pytree_node=False)
    detspace: DetSpace = struct.field(pytree_node=False)
    apply_fn: ApplyFn = struct.field(pytree_node=False)
    params: PyTree
    _T_dets: jnp.ndarray = struct.field(pytree_node=True, default=None)
    _occ_so: jnp.ndarray = struct.field(pytree_node=True, default=None)
    
    sign_dtype: Any = struct.field(pytree_node=False, default=jnp.float32)
    logabs_dtype: Any = struct.field(pytree_node=False, default=jnp.float32)
    wf_is_complex: bool = struct.field(pytree_node=False, default=False)
    
    @classmethod
    def from_model(
        cls,
        system: MolecularSystem,
        detspace: DetSpace,
        model,
        params: PyTree,
    ) -> DeterministicState:
        """
        Construct state from model and parameters.
        
        Precomputes occupation indices and probes output dtypes.
        """
        T_dets = np.asarray(detspace.T_dets, dtype=np.uint64)
        occ_so_np = bitstrings_to_indices(
            T_dets,
            n_orb=system.n_orb,
            n_alpha=system.n_alpha,
            n_beta=system.n_beta,
            dtype=np.uint8,
        )
        
        def apply_fn(p: PyTree, occ_batch: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            occ_batch = occ_batch.astype(jnp.int32)
            sign, logabs = model.apply({"params": p}, occ_batch)
            return sign, logabs
        
        # Probe output dtypes
        sample_occ = jnp.asarray(occ_so_np[:1], dtype=jnp.int32)
        sign0, logabs0 = apply_fn(params, sample_occ)
        wf_is_complex = jnp.issubdtype(sign0.dtype, jnp.complexfloating)
        
        return cls(
            system=system,
            detspace=detspace,
            apply_fn=apply_fn,
            params=params,
            _T_dets=jax.device_put(jnp.asarray(T_dets, dtype=jnp.uint64)),
            _occ_so=jax.device_put(jnp.asarray(occ_so_np, dtype=jnp.uint8)),
            sign_dtype=sign0.dtype,
            logabs_dtype=jnp.real(logabs0).dtype,
            wf_is_complex=bool(wf_is_complex),
        )
    
    @classmethod
    def init_from_model(
        cls,
        system: MolecularSystem,
        detspace: DetSpace,
        model,
        key: jax.Array | None = None,
    ) -> DeterministicState:
        """
        Construct state with automatic parameter initialization.
        
        Initializes model with batch=1 occupation vector to match
        training shape: (1, n_elec).
        """
        T_dets = np.asarray(detspace.T_dets, dtype=np.uint64)
        if T_dets.shape[0] == 0:
            raise ValueError("Empty determinant space")
        
        occ_so_np = bitstrings_to_indices(
            T_dets[:1],
            n_orb=system.n_orb,
            n_alpha=system.n_alpha,
            n_beta=system.n_beta,
            dtype=np.uint8,
        )
        
        sample_occ = jnp.asarray(occ_so_np, dtype=jnp.int32)
        key = jax.random.PRNGKey(42) if key is None else key
        variables = model.init(key, sample_occ)
        
        return cls.from_model(system, detspace, model, variables["params"])
    
    def update_space(self, new_detspace: DetSpace) -> DeterministicState:
        """
        Create new state with updated determinant space.
        
        Recomputes occupation indices while preserving parameters.
        Called during outer-loop space evolution.
        """
        T_dets_np = np.asarray(new_detspace.T_dets, dtype=np.uint64)
        occ_so_np = bitstrings_to_indices(
            T_dets_np,
            n_orb=self.system.n_orb,
            n_alpha=self.system.n_alpha,
            n_beta=self.system.n_beta,
            dtype=np.uint8,
        )
        
        return self.replace(
            detspace=new_detspace,
            _T_dets=jax.device_put(jnp.asarray(T_dets_np, dtype=jnp.uint64)),
            _occ_so=jax.device_put(jnp.asarray(occ_so_np, dtype=jnp.uint8)),
        )
    
    @property
    def parameters(self) -> PyTree:
        """Current parameter PyTree."""
        return self.params
    
    @property
    def n_det(self) -> int:
        """Total determinant count in current T-space."""
        return self._T_dets.shape[0]
    
    @property
    def param_dtype(self) -> jnp.dtype:
        """Parameter dtype from first leaf."""
        leaves = jax.tree.leaves(self.params)
        return leaves[0].dtype if leaves else jnp.float64
    
    @property
    def psi_dtype(self) -> jnp.dtype:
        """Wavefunction dtype: complex if sign is complex, else real."""
        return self.sign_dtype if self.wf_is_complex else self.logabs_dtype
    
    def forward(
        self,
        indices: jnp.ndarray | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Evaluate (sign, logabs) on specified determinants.
        
        Args:
            indices: Determinant indices (None = full T-space)
            chunk_size: Batch size for chunked evaluation
        
        Returns:
            (sign, logabs)
        """
        occ = self._occ_so if indices is None else self._occ_so[indices]
        n_eval = occ.shape[0]
        
        if n_eval == 0:
            sign = jnp.zeros((0,), dtype=self.sign_dtype)
            logabs = jnp.zeros((0,), dtype=self.logabs_dtype)
            return sign, logabs
        
        if chunk_size is None:
            return self.apply_fn(self.params, occ)
        
        return apply_chunked(self.apply_fn, self.params, occ, int(chunk_size))
    
    def __call__(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Evaluate (sign, logabs) on full T-space."""
        return self.forward()
    
    def value_and_vjp(
        self,
        indices: jnp.ndarray | None = None,
        *,
        chunk_size: int | None = None,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], VjpFn]:
        """
        Compute (sign, logabs) and VJP function for gradient calculation.
        
        Args:
            indices: Determinant indices (None = full T-space)
            chunk_size: Batch size for chunked evaluation
        
        Returns:
            (sign, logabs): Forward pass outputs
            vjp_fn: VJP function (cot_sign, cot_logabs) -> grad_params
        """
        occ = self._occ_so if indices is None else self._occ_so[indices]
        n_eval = occ.shape[0]
        
        if n_eval == 0:
            empty_grad = jax.tree_util.tree_map(jnp.zeros_like, self.params)
            sign = jnp.zeros((0,), dtype=self.sign_dtype)
            logabs = jnp.zeros((0,), dtype=self.logabs_dtype)
            return (sign, logabs), lambda _: empty_grad
        
        if chunk_size is None:
            outs, pullback = jax.vjp(lambda p: self.apply_fn(p, occ), self.params)
            
            def vjp_fn(cot: tuple[jnp.ndarray, jnp.ndarray]) -> PyTree:
                cot_sign, cot_logabs = cot
                cot_sign = cot_sign.astype(outs[0].dtype)
                cot_logabs = cot_logabs.astype(jnp.real(outs[1]).dtype)
                (grads,) = pullback((cot_sign, cot_logabs))
                return jax.tree_util.tree_map(
                    lambda g, p: g.astype(p.dtype) if g is not None else jnp.zeros_like(p),
                    grads,
                    self.params,
                )
            
            return outs, vjp_fn
        
        outs = self.forward(indices, chunk_size=chunk_size)
        
        def vjp_fn(cot: tuple[jnp.ndarray, jnp.ndarray]) -> PyTree:
            return vjp_chunked(self.apply_fn, self.params, occ, cot, int(chunk_size))
        
        return outs, vjp_fn
    
    def apply_gradients(
        self,
        gradients: PyTree,
        opt_state: Any,
        optimizer: optax.GradientTransformation,
    ) -> tuple[DeterministicState, Any]:
        """
        Apply optimizer update to parameters.
        
        Args:
            gradients: Parameter gradients from VJP
            opt_state: Current optimizer state
            optimizer: Optax optimizer instance
        
        Returns:
            new_state: State with updated parameters
            new_opt_state: Updated optimizer state
        """
        updates, new_opt_state = optimizer.update(gradients, opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(params=new_params), new_opt_state


__all__ = ["DeterministicState"]
