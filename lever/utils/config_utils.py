# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration capture utilities for reproducible experiments.

Provides @capture_config decorator to record constructor arguments
as ObjectSpec blueprints, enabling config.yaml-driven reconstruction.

File: lever/utils/config_utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

from ..config import ObjectSpec


def _to_spec_value(v: Any) -> Any:
    """
    Convert Python objects to YAML-serializable values.
    
    Handles:
      - Primitives: pass through
      - Collections: recursive conversion
      - LEVER objects: extract _lever_spec
      - Unsupported: raise TypeError for early detection
    """
    # Primitives
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v

    # Collections
    if isinstance(v, (list, tuple)):
        return [_to_spec_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_spec_value(val) for k, val in v.items()}

    # LEVER objects with captured spec
    if hasattr(v, "_lever_spec"):
        spec = v._lever_spec
        if isinstance(spec, ObjectSpec):
            return spec.model_dump(mode="python")
        if isinstance(spec, dict):
            return spec

    # NumPy/JAX scalars
    try:
        import numpy as np
        import jax.numpy as jnp
        if isinstance(v, (np.generic, jnp.generic)):
            return v.item()
    except Exception:
        pass

    # Reject unserializable objects
    raise TypeError(
        f"Unsupported type {type(v).__name__} for config serialization: {v!r}. "
        "Use LEVER factories for complex objects."
    )


def _wrap_function(func: Callable) -> Callable:
    """Wrap factory function to capture constructor arguments."""
    sig = inspect.signature(func)
    target = f"{func.__module__}.{func.__name__}"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to signature
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extract params (exclude self/cls and variadic parameters)
        params = {}
        for name, val in bound.arguments.items():
            # Skip self/cls
            if name in ("self", "cls"):
                continue
            
            # Skip *args and **kwargs (VAR_POSITIONAL/VAR_KEYWORD)
            param = sig.parameters.get(name)
            if param and param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            ):
                continue
            
            params[name] = _to_spec_value(val)

        # Call original function
        obj = func(*args, **kwargs)
        spec = ObjectSpec(target=target, params=params)

        # Attach spec to returned object
        try:
            setattr(obj, "_lever_spec", spec)
            return obj
        except (AttributeError, TypeError):
            # Fallback: wrap immutable objects
            return SpecWrapper(obj, spec)

    return wrapper


def _wrap_class(cls: type) -> type:
    """Wrap class __init__ to capture constructor arguments."""
    orig_init = cls.__init__
    sig = inspect.signature(orig_init)
    target = f"{cls.__module__}.{cls.__qualname__}"

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        # Bind arguments
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Extract params (exclude self and variadic parameters)
        params = {}
        for name, val in bound.arguments.items():
            if name == "self":
                continue
            
            # Skip *args and **kwargs (VAR_POSITIONAL/VAR_KEYWORD)
            param = sig.parameters.get(name)
            if param and param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            ):
                continue
            
            params[name] = _to_spec_value(val)

        # Call original init
        orig_init(self, *args, **kwargs)

        # Attach spec to instance
        spec = ObjectSpec(target=target, params=params)
        setattr(self, "_lever_spec", spec)

    cls.__init__ = __init__
    return cls


def capture_config(obj=None):
    """
    Decorator to capture constructor arguments as ObjectSpec.
    
    Usage:
        @capture_config
        def my_factory(...): ...
        
        @capture_config
        class MyClass: ...
    
    Captured objects expose ._lever_spec attribute for serialization.
    """
    if obj is None:
        # Support @capture_config() with parentheses
        return capture_config

    if inspect.isfunction(obj):
        return _wrap_function(obj)
    if inspect.isclass(obj):
        return _wrap_class(obj)
    
    raise TypeError("@capture_config requires function or class")


class SpecWrapper:
    """
    Transparent proxy for immutable objects (e.g., optax schedules).
    
    Provides ._lever_spec while delegating all other access to wrapped object.
    """
    def __init__(self, wrapped: Any, spec: ObjectSpec):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_lever_spec", spec)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self._wrapped)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_wrapped", "_lever_spec"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)


def freeze_object(obj: Any) -> ObjectSpec:
    """
    Extract ObjectSpec from object built via @capture_config.
    
    Raises TypeError if object lacks ._lever_spec attribute.
    """
    if hasattr(obj, "_lever_spec"):
        spec = obj._lever_spec
        if isinstance(spec, ObjectSpec):
            return spec
        if isinstance(spec, dict):
            return ObjectSpec(**spec)
    
    raise TypeError(
        f"Object {obj!r} has no ._lever_spec. "
        "Use @capture_config decorator for reproducible construction."
    )


__all__ = ["capture_config", "freeze_object", "SpecWrapper"]
