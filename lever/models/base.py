# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified wavefunction model abstraction with automatic batching and JIT compilation.

Core design:
  - Wraps Flax Linen modules for dual-mode execution (stateless/stateful)
  - Automatic holomorphic vs. non-holomorphic complex differentiation
  - JIT-compiled vectorized inference and gradient computation

File: lever/models/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization as flax_ser
from flax.core.frozen_dict import FrozenDict
from jax import tree_util

if TYPE_CHECKING:
    from typing_extensions import Self

# --- Type Aliases ---

PyTree = Any
Features = jnp.ndarray  # (batch, n_features)
LogPsi = jnp.ndarray    # (batch,) complex-valued
Variables = PyTree      # Flax variable tree
FeatureFn = Callable[[jnp.ndarray, int], jnp.ndarray]


# --- Core Model Wrapper ---

@dataclass
class WavefunctionModel:
    """
    Unified wavefunction model with automatic batching and compilation.
    
    Supports dual calling modes:
      - Stateless: log_psi(params, x)  # explicit parameters
      - Stateful:  model(x)            # uses internal .variables
    
    Automatic features:
      - JIT compilation with shape inference
      - Complex differentiation (holomorphic/non-holomorphic)
      - Batched execution via vmap
    """
    
    module: nn.Module
    force_holomorphic: Optional[bool] = None
    
    # Internal state (populated by init/create)
    variables: Optional[Variables] = field(default=None, init=False, repr=False)
    
    # Compilation cache
    _apply_jit: Optional[Callable] = field(default=None, init=False, repr=False)
    _ders_jit: Optional[Callable] = field(default=None, init=False, repr=False)
    _is_holo: Optional[bool] = field(default=None, init=False, repr=False)
    
    # --- Initialization ---
    
    def init(self, rng: jax.Array, dummy_input: jnp.ndarray) -> Variables:
        """
        Initialize parameters and compile executors.
        
        Args:
            rng: PRNG key
            dummy_input: Single sample for shape inference (1D)
        
        Returns:
            Initialized variable tree
        """
        # Normalize input shape
        if dummy_input.ndim == 2 and dummy_input.shape[0] == 1:
            dummy_input = dummy_input[0]
        
        variables = self.module.init({"params": rng}, dummy_input)
        self._is_holo = self._detect_holomorphic(variables)
        self._compile(variables, dummy_input)
        
        return variables
    
    def _compile(self, variables: Variables, dummy_input: jnp.ndarray) -> None:
        """Compile JIT executors for batched inference and gradients."""
        module = self.module
        is_holo = self._is_holo
        
        def _single_apply(v: Variables, x: jnp.ndarray) -> jnp.ndarray:
            """Apply model to single sample, ensuring complex scalar output."""
            y = module.apply(v, x)
            y_scalar = y if y.ndim == 0 else jnp.sum(y)
            return jnp.asarray(y_scalar, dtype=jnp.complex128)
        
        def _single_val_and_ders(v: Variables, x: jnp.ndarray) -> Tuple[jnp.ndarray, PyTree]:
            """Compute value and complex derivatives for single sample."""
            f = lambda p: _single_apply(p, x)
            
            if is_holo:
                # Wirtinger derivative for holomorphic parameters
                return jax.value_and_grad(f, holomorphic=True)(v)
            
            # Non-holomorphic: separate real/imag gradients
            val = f(v)
            ders_re = jax.grad(lambda p: jnp.real(f(p)))(v)
            ders_im = jax.grad(lambda p: jnp.imag(f(p)))(v)
            ders = tree_util.tree_map(lambda a, b: a + 1j * b, ders_re, ders_im)
            return val, ders
        
        # Batch and compile
        self._apply_jit = jax.jit(jax.vmap(_single_apply, in_axes=(None, 0)))
        self._ders_jit = jax.jit(jax.vmap(_single_val_and_ders, in_axes=(None, 0)))
    
    # --- Inference API ---
    
    def log_psi(self, variables: Variables, inputs: Features) -> LogPsi:
        """Stateless batched inference with explicit parameters."""
        if self._apply_jit is None:
            raise RuntimeError("Model not compiled. Call .init() first.")
        return self._apply_jit(variables, inputs)
    
    def log_psi_and_ders(
        self, variables: Variables, inputs: Features
    ) -> Tuple[LogPsi, PyTree]:
        """
        Stateless batched inference with per-sample gradients.
        
        Returns:
            (log_psi, gradients): log_psi shape (batch,), gradients with batch axis
        """
        if self._ders_jit is None:
            raise RuntimeError("Model not compiled. Call .init() first.")
        return self._ders_jit(variables, inputs)
    
    def __call__(self, inputs: Features) -> LogPsi:
        """Stateful inference using internal .variables."""
        if self.variables is None:
            raise RuntimeError(
                "Stateful call requires initialized .variables. "
                "Use .init() or .create()."
            )
        return self.log_psi(self.variables, inputs)
    
    def apply(self, inputs: Features) -> LogPsi:
        """Alias for __call__ (Flax convention)."""
        return self(inputs)
    
    # --- Properties ---
    
    @property
    def is_holo(self) -> bool:
        """Whether model uses holomorphic (complex) parameters."""
        if self._is_holo is None:
            raise RuntimeError("Model not initialized.")
        return self._is_holo
    
    def _detect_holomorphic(self, variables: Variables) -> bool:
        """
        Detect holomorphic mode via:
          1. Explicit force_holomorphic flag
          2. Module attribute __lever_is_holomorphic__
          3. Parameter dtype inspection
        """
        if self.force_holomorphic is not None:
            return bool(self.force_holomorphic)
        
        if hasattr(self.module, "__lever_is_holomorphic__"):
            return bool(self.module.__lever_is_holomorphic__)
        
        params = variables.get("params", variables) if isinstance(variables, dict) else variables
        leaves = tree_util.tree_leaves(params)
        return bool(leaves) and all(jnp.iscomplexobj(x) for x in leaves)
    
    # --- Factory Methods ---
    
    @classmethod
    def create(
        cls,
        module: nn.Module,
        rng: jax.Array,
        dummy_input: jnp.ndarray,
        force_holomorphic: Optional[bool] = None,
    ) -> Self:
        """
        Convenience constructor with automatic initialization.
        
        Returns stateful model with populated .variables.
        """
        model = cls(module, force_holomorphic=force_holomorphic)
        variables = model.init(rng, dummy_input)
        model.variables = variables
        return model


# --- Factory Functions ---

def _infer_dummy_input(
    dummy_input: Optional[jnp.ndarray],
    n_orbitals: Optional[int],
    feature_fn_for_dummy: Optional[FeatureFn],
) -> jnp.ndarray:
    """Generate single-sample dummy input for shape inference."""
    if dummy_input is not None:
        if dummy_input.ndim != 1:
            raise ValueError(f"dummy_input must be 1D, got shape {dummy_input.shape}")
        return dummy_input
    
    if n_orbitals is None:
        raise ValueError("Must provide either 'dummy_input' or 'n_orbitals'.")
    
    if feature_fn_for_dummy is not None:
        dets = jnp.zeros((1, 2), dtype=jnp.uint64)
        feats = feature_fn_for_dummy(dets, int(n_orbitals))
        feats = jnp.asarray(feats)
        if feats.ndim == 2 and feats.shape[0] == 1:
            return feats[0]
        if feats.ndim == 1:
            return feats
        raise ValueError(f"feature_fn_for_dummy returned invalid shape: {feats.shape}")
    
    # Default: occupation number vector
    return jnp.zeros(2 * int(n_orbitals), dtype=jnp.float32)


def make_model(
    module: nn.Module,
    *,
    seed: int,
    dummy_input: Optional[jnp.ndarray] = None,
    n_orbitals: Optional[int] = None,
    feature_fn_for_dummy: Optional[FeatureFn] = None,
    force_holomorphic: Optional[bool] = None,
    precision_config: Optional[Any] = None,  # Add parameter
) -> WavefunctionModel:
    """
    Create and initialize stateful wavefunction model with precision control.
    
    Args:
        module: Flax Linen module
        seed: Random seed for initialization
        dummy_input: Explicit single-sample input, OR
        n_orbitals: Number of orbitals (auto-generates dummy input)
        feature_fn_for_dummy: Optional feature function for dummy generation
        force_holomorphic: Override automatic holomorphic detection
        precision_config: PrecisionConfig for dtype control (optional)
    
    Returns:
        Initialized model ready for stateful calls
    """
    
    din = _infer_dummy_input(dummy_input, n_orbitals, feature_fn_for_dummy)
    
    # Cast dummy input to appropriate dtype
    if precision_config is not None:
        target_dtype = (
            jnp.float64 if precision_config.enable_x64_device 
            else jnp.float32
        )
        din = jnp.asarray(din, dtype=target_dtype)
    
    return WavefunctionModel.create(
        module,
        jax.random.PRNGKey(int(seed)),
        din,
        force_holomorphic=force_holomorphic,
    )


# --- Serialization Utilities ---

def to_state_dict(variables: Variables) -> dict:
    """Convert Flax variables to plain dict for serialization."""
    return flax_ser.to_state_dict(variables)


def _flatten_keys(x: Any, prefix: Tuple[str, ...] = ()) -> set[Tuple[str, ...]]:
    """Recursively collect all string-key paths from nested dict."""
    if isinstance(x, (dict, FrozenDict)):
        return {
            sub_key
            for k, v in x.items() if isinstance(k, str)
            for sub_key in _flatten_keys(v, prefix + (k,))
        }
    return set() if not prefix else {prefix}


def from_state_dict(
    template: Variables,
    state_dict: dict,
    *,
    strict_keys: str = "warn",
) -> Variables:
    """
    Restore variables from state dict with optional key validation.
    
    Args:
        template: Variable structure to fill
        state_dict: Serialized state
        strict_keys: Key checking mode ("warn" | "error" | "ignore")
    
    Returns:
        Restored variables (Flax handles shape/dtype validation)
    """
    if strict_keys not in ("warn", "error", "ignore"):
        raise ValueError(f"strict_keys must be 'warn'/'error'/'ignore', got {strict_keys}")
    
    tpl_keys = _flatten_keys(flax_ser.to_state_dict(template))
    st_keys = _flatten_keys(state_dict)
    extra_keys = st_keys - tpl_keys
    
    if extra_keys:
        msg = f"Unexpected keys in state_dict: {sorted(extra_keys)}"
        if strict_keys == "error":
            raise KeyError(msg)
        if strict_keys == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)
    
    return flax_ser.from_state_dict(template, state_dict)


# --- Public API ---

__all__ = [
    "PyTree",
    "Features",
    "LogPsi",
    "Variables",
    "FeatureFn",
    "WavefunctionModel",
    "make_model",
    "to_state_dict",
    "from_state_dict",
]
