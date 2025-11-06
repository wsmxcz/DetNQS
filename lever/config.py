# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified configuration management for LEVER workflows.

Provides centralized control of system parameters, Hamiltonian construction,
optimization loops, and numerical precision policies.

File: lever/config.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import numpy as np


# ============================================================================
# Enumeration Types
# ============================================================================

class ScreenMode(str, Enum):
    """C-space screening strategy: disabled, static threshold, or dynamic."""
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class ComputeMode(str, Enum):
    """
    Hamiltonian contraction algorithms:
    - ASYMMETRIC: Full T-space with asymmetric gradient
    - PROXY:      T-space with symmetric proxy gradient
    - EFFECTIVE:  S-space only with downfolded H_eff
    """
    ASYMMETRIC = "asymmetric"
    PROXY = "proxy"
    EFFECTIVE = "effective"


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass(frozen=True)
class PrecisionConfig:
    """
    Numerical precision policy with automatic JAX/NumPy dtype resolution.
    
    Attributes:
        enable_x64: Enable float64/complex128 globally
        override_device_complex: Force specific complex dtype (None for auto)
        override_device_float: Force specific float dtype (None for auto)
    """
    enable_x64: bool = False
    override_device_complex: type | None = None
    override_device_float: type | None = None
    
    @property
    def jax_float(self) -> Any:
        """Device float dtype (float32/64 based on x64 setting)."""
        from jax import dtypes
        return self.override_device_float or dtypes.canonicalize_dtype(float)
    
    @property
    def jax_complex(self) -> Any:
        """Device complex dtype (complex64/128 based on x64 setting)."""
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
    """
    Physical system specification from FCIDUMP file.
    
    Attributes:
        fcidump_path: Path to FCIDUMP integral file
        n_orbitals: Number of spatial orbitals
        n_alpha: Number of α-spin electrons
        n_beta: Number of β-spin electrons
    """
    fcidump_path: str
    n_orbitals: int
    n_alpha: int
    n_beta: int


@dataclass(frozen=True)
class HamiltonianConfig:
    """
    Hamiltonian construction and C-space screening parameters.
    
    Controls effective Hamiltonian assembly via Löwdin partitioning:
        H_eff = H_SS - H_SC @ (E - H_CC + δI)^{-1} @ H_CS
    
    Attributes:
        screening_mode: C-space selection strategy
        screen_eps: Coefficient threshold for dynamic screening
        diag_shift: Diagonal shift δ for H_CC regularization
        reg_eps: Numerical stability for matrix inversion
    """
    screening_mode: ScreenMode = ScreenMode.DYNAMIC
    screen_eps: float = 1e-6
    diag_shift: float = 0.5
    reg_eps: float = 1e-4


@dataclass(frozen=True)
class LoopConfig:
    """
    Iteration and convergence control for LEVER loops.
    
    Outer loop: Evolve determinant space
    Inner loop: Optimize parameters in fixed space
    """
    # Outer loop (cycle evolution)
    max_outer: int = 10                # Maximum evolutionary cycles
    outer_tol: float = 1e-5           # Convergence tolerance
    outer_patience: int = 3           # Consecutive converged cycles needed
    
    # Inner loop (fixed-space optimization)
    inner_steps: int = 500            # Fixed optimization steps per cycle
    
    # Optional batch processing
    chunk_size: int | None = None     # Gradient accumulation chunk size


@dataclass(frozen=True)
class LeverConfig:
    """
    Top-level LEVER workflow configuration.
    
    Organizes parameters by functional domain:
      - system: Physical problem specification
      - hamiltonian: Matrix construction rules
      - loop: Optimization iteration control
    
    Attributes:
        compute_mode: Hamiltonian contraction algorithm
        seed: Random number generator seed
        report_interval: Steps between progress logs
        num_eps: Numerical epsilon for stability checks
        normalize_wf: Enable wavefunction normalization
        precision: Dtype policy configuration
    """
    system: SystemConfig
    hamiltonian: HamiltonianConfig
    loop: LoopConfig
    
    compute_mode: ComputeMode = ComputeMode.PROXY
    seed: int = 42
    report_interval: int = 50
    num_eps: float = 1e-12
    normalize_wf: bool = True
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    
    def __post_init__(self) -> None:
        """Apply global precision policy during initialization."""
        if self.precision.enable_x64:
            jax.config.update("jax_enable_x64", True)


__all__ = [
    "ScreenMode",
    "ComputeMode",
    "PrecisionConfig",
    "SystemConfig",
    "HamiltonianConfig",
    "LoopConfig",
    "LeverConfig",
]
