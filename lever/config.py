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

class EvalMode(str, Enum):
    """Energy evaluation timing: never, final cycle only, or every step."""
    NEVER = "never"
    FINAL = "final"
    EVERY = "every"


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
    Optimization loop control and convergence criteria.
    
    Two-level iteration structure:
      Outer: Macro-cycles with S-space evolution
      Inner: Micro-steps with fixed Hamiltonian
    
    Attributes:
        max_cycles: Maximum macro-cycles
        cycle_tol: Energy convergence threshold for cycles
        patience: Early stopping after non-improving cycles
        max_steps: Maximum optimization steps per cycle
        step_tol: Energy convergence threshold for steps
        check_interval: Steps between convergence checks
        s_space_size: Target S-space dimension after evolution
    """
    max_cycles: int = 10
    cycle_tol: float = 1e-5
    patience: int = 3
    max_steps: int = 500
    step_tol: float = 1e-6
    check_interval: int = 50
    s_space_size: int = 200


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Energy diagnostic timing control.
    
    Attributes:
        var_energy_mode: Variational energy E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
        t_ci_energy_mode: Full T-space CI energy
        s_ci_energy_mode: S-space CI energy
    """
    var_energy_mode: EvalMode = EvalMode.FINAL
    t_ci_energy_mode: EvalMode = EvalMode.NEVER
    s_ci_energy_mode: EvalMode = EvalMode.FINAL


@dataclass(frozen=True)
class LeverConfig:
    """
    Top-level LEVER workflow configuration.
    
    Organizes parameters by functional domain:
      - system: Physical problem specification
      - hamiltonian: Matrix construction rules
      - loop: Optimization iteration control
      - evaluation: Energy diagnostic timing
    
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
    evaluation: EvaluationConfig
    
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
    "EvalMode",
    "ScreenMode",
    "ComputeMode",
    "PrecisionConfig",
    "SystemConfig",
    "HamiltonianConfig",
    "LoopConfig",
    "EvaluationConfig",
    "LeverConfig",
]
