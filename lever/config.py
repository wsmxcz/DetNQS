# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified configuration management for LEVER workflows.

File: lever/config.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any
import jax
import numpy as np


class EvalMode(str, Enum):
    """Energy evaluation timing control."""
    NEVER = "never"
    FINAL = "final"
    EVERY = "every"


class ScreenMode(str, Enum):
    """C-space screening strategy."""
    NONE = "none"        # Full enumeration
    STATIC = "static"    # Heat-bath integral screening
    DYNAMIC = "dynamic"  # Amplitude-weighted screening


class ComputeMode(str, Enum):
    """
    Energy/gradient computation algorithms.
    
    ASYMMETRIC: S-norm with full S-C coupling
                E = ⟨ψ_S|Ĥ|ψ⟩ / ||ψ_S||², gradient over S-space
    
    PROXY:      Full-norm with diagonal C approximation
                E = ⟨ψ|Ĥ̃|ψ⟩ / ||ψ||², gradient over S∪C
    
    EFFECTIVE:  S-norm with Schur downfolding (Ĥ_eff = Ĥ_SS + Ĥ_SC·D⁻¹·Ĥ_CS)
                E = ⟨ψ_S|Ĥ_eff|ψ_S⟩ / ||ψ_S||², gradient over S-space
    """
    ASYMMETRIC = "asymmetric"
    PROXY = "proxy"
    EFFECTIVE = "effective"


@dataclass(frozen=True)
class PrecisionConfig:
    """Numerical precision policy with JAX/NumPy dtype management."""
    enable_x64: bool = False
    override_device_complex: type | None = None
    override_device_float: type | None = None
    
    @property
    def jax_float(self) -> Any:
        """JAX float dtype (respects global x64 setting)."""
        from jax import dtypes
        return self.override_device_float or dtypes.canonicalize_dtype(float)
    
    @property
    def jax_complex(self) -> Any:
        """JAX complex dtype (respects global x64 setting)."""
        from jax import dtypes
        return self.override_device_complex or dtypes.canonicalize_dtype(complex)
    
    @property
    def numpy_float(self) -> type:
        """NumPy float dtype matching device precision."""
        return np.dtype(self.jax_float).type
    
    @property
    def numpy_complex(self) -> type:
        """NumPy complex dtype matching device precision."""
        return np.dtype(self.jax_complex).type


@dataclass(frozen=True)
class SystemConfig:
    """Physical system definition from FCIDUMP."""
    fcidump_path: str
    n_orbitals: int
    n_alpha: int
    n_beta: int


@dataclass(frozen=True)
class OptimizationConfig:
    """VMC optimization parameters."""
    seed: int = 42
    num_cycles: int = 10
    s_space_size: int = 200
    steps_per_cycle: int = 500
    report_interval: int = 50


@dataclass(frozen=True)
class EvaluationConfig:
    """Energy computation timing control."""
    var_energy_mode: EvalMode = EvalMode.FINAL
    t_ci_energy_mode: EvalMode = EvalMode.NEVER
    s_ci_energy_mode: EvalMode = EvalMode.FINAL


@dataclass(frozen=True)
class ScreeningConfig:
    """C-space screening configuration."""
    mode: ScreenMode = ScreenMode.DYNAMIC
    screen_eps: float = 1e-6      # Screening threshold
    diag_shift: float = 0.5       # Level shift for Ĥ_CC stabilization


@dataclass(frozen=True)
class LeverConfig:
    """Top-level LEVER workflow configuration."""
    system: SystemConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    screening: ScreeningConfig
    
    num_eps: float = 1e-12           # Numerical zero threshold
    normalize_wf: bool = True         # Enforce ||ψ||² = 1
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    compute_mode: ComputeMode = ComputeMode.PROXY
    
    def __post_init__(self):
        """Apply precision policy during initialization."""
        if self.precision.enable_x64:
            jax.config.update("jax_enable_x64", True)
            logging.info("Precision policy: 64-bit enabled")
