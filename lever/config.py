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

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Union
from enum import Enum
import jax

import numpy as np
import yaml


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
    """Global precision policy.

    Default: single precision on device (float32/complex64),
    with an option to enable full double precision.
    """
    enable_x64: bool = False
    override_device_complex: Any | None = None
    override_device_float: Any | None = None

    def apply(self) -> None:
        """Apply global JAX precision settings.

        Must be called once at startup, before creating JAX arrays.
        """
        jax.config.update("jax_enable_x64", self.enable_x64)

    @property
    def jax_float(self):
        """Device-side real dtype (float32 or float64)."""
        from jax import dtypes
        if self.override_device_float is not None:
            return self.override_device_float
        return dtypes.canonicalize_dtype(float)

    @property
    def jax_complex(self):
        """Device-side complex dtype (complex64 or complex128)."""
        from jax import dtypes
        if self.override_device_complex is not None:
            return self.override_device_complex
        return dtypes.canonicalize_dtype(complex)

    @property
    def numpy_float(self):
        """Host real dtype for arrays going through callbacks.

        Note: heavy CPU kernels (Numba) still compute in float64 internally.
        """
        return np.dtype(self.jax_float).type

    @property
    def numpy_complex(self):
        """Host complex dtype for arrays going through callbacks.

        Note: heavy CPU kernels (Numba) still compute in complex128 internally.
        """
        return np.dtype(self.jax_complex).type


@dataclass(frozen=True)
class SystemConfig:
    """
    Physical system specification from an FCIDUMP + optional HDF5 metadata.

    Attributes:
        fcidump_path: Path to FCIDUMP integral file.
        n_orbitals:   Number of spatial orbitals represented in FCIDUMP.
        n_alpha:      Number of α-spin electrons in the FCIDUMP Hamiltonian.
        n_beta:       Number of β-spin electrons in the FCIDUMP Hamiltonian.
        meta_path:    Optional path to a side-car HDF5 file produced by
                      lever.interface.MoleculeBuilder. When present, LEVER
                      components can load additional metadata (geometry,
                      HF reference, benchmarks, etc.).
    """
    fcidump_path: str
    n_orbitals: int
    n_alpha: int
    n_beta: int
    meta_path: str | None = None

    @classmethod
    def from_hdf5(cls, h5_path: str) -> "SystemConfig":
        """
        Construct SystemConfig from a LEVER interface HDF5 file.

        The HDF5 file is expected to follow the schema produced by
        lever.interface.MoleculeBuilder.export().
        """
        # Local import to avoid circular dependency at module import time.
        from .interface.builder import read_from_hdf5

        data = read_from_hdf5(h5_path)
        return cls(
            fcidump_path=data["fcidump_path"],
            n_orbitals=data["n_orbitals"],
            n_alpha=data["n_alpha"],
            n_beta=data["n_beta"],
            meta_path=data.get("meta_path"),
        )


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
    max_outer: int = 20                # Maximum evolutionary cycles
    outer_tol: float = 1e-5            # Convergence tolerance
    outer_patience: int = 5            # Consecutive converged cycles needed
    
    # Inner loop (fixed-space optimization)
    max_inner: int = 400             # Maximum optimization steps per cycle
    inner_tol: float = 1e-8             # |E_k - E_{k-1}| tolerance (<=0 disables)
    inner_patience: int = 200            # Consecutive steps with small delta

    # Optional batch processing
    chunk_size: int | None = None      # Gradient accumulation chunk size


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
    spin_flip_symmetry: bool = False
    seed: int = 42
    report_interval: int = 10
    num_eps: float = 1e-12
    normalize_wf: bool = True
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    
    def __post_init__(self) -> None:
        """Apply global precision policy once when config is created."""
        self.precision.apply()

__all__ = [
    "ScreenMode",
    "ComputeMode",
    "PrecisionConfig",
    "SystemConfig",
    "HamiltonianConfig",
    "LoopConfig",
    "LeverConfig",
]
