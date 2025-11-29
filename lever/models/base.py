# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified wavefunction model with automatic batching and JIT compilation.

Provides dual-mode execution (stateless/stateful) with automatic complex
differentiation (Wirtinger derivatives for holomorphic parameters).

Core features:
  - Wraps Flax Linen modules for neural quantum states
  - JIT-compiled vectorized inference and per-sample gradients
  - Automatic dtype detection and shape inference

File: lever/models/base.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
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

from ..config import RuntimeConfig

if TYPE_CHECKING:
    from typing_extensions import Self

# ============================================================================
# Type Aliases
# ============================================================================

PyTree = Any
Features = jnp.ndarray  # Shape: (batch, n_features)
LogPsi = jnp.ndarray    # Shape: (batch,), complex-valued log(ψ)
Variables = PyTree      # Flax variable tree

# FeatureFn: (determinants, n_orbitals) → features
FeatureFn = Callable[[jnp.ndarray, int], jnp.ndarray]


# ============================================================================
# Wavefunction Model
# ============================================================================

@dataclass
class WavefunctionModel:
    """
    Unified neural quantum state model with automatic compilation.
    
    Calling modes:
      - Stateless: log_psi(params, x)  # explicit parameters
      - Stateful:  model(x)            # uses internal .variables
    
    Complex differentiation:
      - Holomorphic: Wirtinger ∂/∂z for complex parameters
      - Non-holomorphic: separate ∂/∂(Re z) and ∂/∂(Im z)
    
    Attributes:
        module: Flax Linen module implementing forward pass
        force_holomorphic: Override automatic dtype detection
        variables: Initialized parameter tree (None before .init())
    """
    
    module: nn.Module
    force_holomorphic: Optional[bool] = None
    
    # Internal state
    variables: Optional[Variables] = field(default=None, init=False, repr=False)
    
    # Compilation cache
    _apply_jit: Optional[Callable] = field(default=None, init=False, repr=False)
    _ders_jit: Optional[Callable] = field(default=None, init=False, repr=False)
    _is_holo: Optional[bool] = field(default=None, init=False, repr=False)
    
    def init(self, rng: jax.Array, dummy_input: jnp.ndarray) -> Variables:
        """
        Initialize parameters and compile JIT executors.
        
        Args:
            rng: PRNG key for parameter initialization
            dummy_input: Single sample for shape inference (1D or [1, D])
        
        Returns:
            Initialized variable tree
        """
        # Normalize to 1D for Flax init
        if dummy_input.ndim == 2 and dummy_input.shape[0] == 1:
            dummy_input = dummy_input[0]
        
        variables = self.module.init({"params": rng}, dummy_input)
        self._is_holo = self._detect_holomorphic(variables)
        self._compile(variables, dummy_input)
        
        return variables
    
    def _compile(self, variables: Variables, dummy_input: jnp.ndarray) -> None:
        """
        Compile JIT executors for batched inference and gradients.
        
        Strategy:
          1. Define single-sample functions
          2. Vectorize over batch dimension via vmap
          3. Apply JIT compilation
        """
        module = self.module
        is_holo = self._is_holo

        def _single_apply(v: Variables, x: jnp.ndarray) -> jnp.ndarray:
            """Apply module and reduce to scalar log(ψ)."""
            y = module.apply(v, x)
            y_scalar = y if y.ndim == 0 else jnp.sum(y)
            return jnp.asarray(y_scalar)

        def _single_val_and_ders(
            v: Variables, x: jnp.ndarray
        ) -> Tuple[jnp.ndarray, PyTree]:
            """
            Compute log(ψ) and complex derivatives for single sample.
            
            Returns:
                (value, gradients): both match parameter tree structure
            """
            f = lambda p: _single_apply(p, x)

            if is_holo:
                # Wirtinger derivative: ∂f/∂z* for complex parameters
                return jax.value_and_grad(f, holomorphic=True)(v)

            # Non-holomorphic: ∇f = ∂f/∂(Re z) + i·∂f/∂(Im z)
            val = f(v)
            ders_re = jax.grad(lambda p: jnp.real(f(p)))(v)
            ders_im = jax.grad(lambda p: jnp.imag(f(p)))(v)
            ders = tree_util.tree_map(lambda a, b: a + 1j * b, ders_re, ders_im)
            return val, ders

        # Vectorize then JIT-compile
        self._apply_jit = jax.jit(jax.vmap(_single_apply, in_axes=(None, 0)))
        self._ders_jit = jax.jit(jax.vmap(_single_val_and_ders, in_axes=(None, 0)))
    
    def log_psi(self, variables: Variables, inputs: Features) -> LogPsi:
        """
        Stateless batched inference with explicit parameters.
        
        Args:
            variables: Parameter tree
            inputs: Feature batch [N, D]
        
        Returns:
            log(ψ) values [N]
        """
        if self._apply_jit is None:
            raise RuntimeError("Model not compiled. Call .init() first.")
        return self._apply_jit(variables, inputs)
    
    def log_psi_and_ders(
        self, variables: Variables, inputs: Features
    ) -> Tuple[LogPsi, PyTree]:
        """
        Stateless batched inference with per-sample gradients.
        
        Returns:
            (log_psi, gradients):
              - log_psi: shape [N]
              - gradients: parameter tree with batch axis prepended
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
    
    @property
    def is_holo(self) -> bool:
        """Whether model uses holomorphic (complex) parameters."""
        if self._is_holo is None:
            raise RuntimeError("Model not initialized.")
        return self._is_holo
    
    def _detect_holomorphic(self, variables: Variables) -> bool:
        """
        Detect holomorphic mode via priority chain:
          1. Explicit force_holomorphic flag
          2. Module attribute __lever_is_holomorphic__
          3. Parameter dtype inspection (all complex → holomorphic)
        """
        if self.force_holomorphic is not None:
            return bool(self.force_holomorphic)
        
        if hasattr(self.module, "__lever_is_holomorphic__"):
            return bool(self.module.__lever_is_holomorphic__)
        
        # Inspect parameter dtypes
        params = (
            variables.get("params", variables)
            if isinstance(variables, dict)
            else variables
        )
        leaves = tree_util.tree_leaves(params)
        return bool(leaves) and all(jnp.iscomplexobj(x) for x in leaves)
    
    @property
    def summary(self) -> dict[str, Any]:
        """
        Generate model metadata and statistics.
        
        Returns:
            Dict with keys: name, status, is_holomorphic, n_params,
            param_bytes, dtypes
        """
        info = {
            "name": self.module.__class__.__name__,
            "is_holomorphic": False,
            "n_params": 0,
            "param_bytes": 0,
            "dtypes": set(),
        }

        if self.variables is None:
            info["status"] = "Uninitialized"
            return info
        
        info["status"] = "Initialized"
        info["is_holomorphic"] = self.is_holo

        leaves = jax.tree_util.tree_leaves(self.variables)
        
        for leaf in leaves:
            size = leaf.size
            info["n_params"] += size
            info["param_bytes"] += size * leaf.itemsize
            info["dtypes"].add(str(leaf.dtype))
            
        info["dtypes"] = sorted(list(info["dtypes"]))
        return info
    
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
        
        Returns:
            Stateful model with populated .variables
        """
        model = cls(module, force_holomorphic=force_holomorphic)
        variables = model.init(rng, dummy_input)
        model.variables = variables
        return model


# ============================================================================
# Factory Functions
# ============================================================================

def _infer_dummy_input(
    dummy_input: Optional[jnp.ndarray],
    n_orb: Optional[int],
    feature_fn_for_dummy: Optional[FeatureFn],
) -> jnp.ndarray:
    """
    Generate single-sample dummy input for shape inference.
    
    Priority:
      1. Explicit dummy_input (must be 1D)
      2. feature_fn_for_dummy(zero_dets, n_orb)
      3. Zero occupation vector [2·n_orb]
    """
    if dummy_input is not None:
        if dummy_input.ndim != 1:
            raise ValueError(f"dummy_input must be 1D, got shape {dummy_input.shape}")
        return dummy_input
    
    if n_orb is None:
        raise ValueError("Must provide either 'dummy_input' or 'n_orb'.")
    
    if feature_fn_for_dummy is not None:
        dets = jnp.zeros((1, 2), dtype=jnp.uint64)
        feats = feature_fn_for_dummy(dets, int(n_orb))
        feats = jnp.asarray(feats)
        
        if feats.ndim == 2 and feats.shape[0] == 1:
            return feats[0]
        if feats.ndim == 1:
            return feats
        
        raise ValueError(
            f"feature_fn_for_dummy returned invalid shape: {feats.shape}"
        )
    
    # Default: zero occupation number vector
    return jnp.zeros(2 * int(n_orb), dtype=jnp.float32)


def make_model(
    module: nn.Module,
    *,
    seed: int,
    dummy_input: jnp.ndarray | None = None,
    n_orb: int | None = None,
    feature_fn_for_dummy: FeatureFn | None = None,
    force_holomorphic: bool | None = None,
    precision_config: RuntimeConfig | None = None,
) -> WavefunctionModel:
    """
    Create and initialize a WavefunctionModel.
    
    Dummy input is cast to configured device float dtype to match
    runtime feature precision.
    
    Args:
        module: Flax Linen module
        seed: PRNG seed for initialization
        dummy_input: Explicit 1D sample for shape inference
        n_orb: Number of orbitals (alternative to dummy_input)
        feature_fn_for_dummy: Feature generator for shape inference
        force_holomorphic: Override automatic dtype detection
        precision_config: Runtime precision configuration
    
    Returns:
        Initialized stateful model
    """
    din = _infer_dummy_input(dummy_input, n_orb, feature_fn_for_dummy)

    if precision_config is not None:
        din = jnp.asarray(din, dtype=precision_config.jax_float)
    else:
        din = jnp.asarray(din)

    return WavefunctionModel.create(
        module,
        jax.random.PRNGKey(int(seed)),
        din,
        force_holomorphic=force_holomorphic,
    )


# ============================================================================
# Serialization Utilities
# ============================================================================

def to_state_dict(variables: Variables) -> dict:
    """Convert Flax variables to plain dict for serialization."""
    return flax_ser.to_state_dict(variables)


def _flatten_keys(x: Any, prefix: Tuple[str, ...] = ()) -> set[Tuple[str, ...]]:
    """Recursively collect all string-key paths from nested dict."""
    if isinstance(x, (dict, FrozenDict)):
        return {
            sub_key
            for k, v in x.items()
            if isinstance(k, str)
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
        Restored variables
    
    Raises:
        KeyError: If strict_keys="error" and extra keys found
    """
    if strict_keys not in ("warn", "error", "ignore"):
        raise ValueError(
            f"strict_keys must be 'warn'/'error'/'ignore', got {strict_keys}"
        )
    
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
