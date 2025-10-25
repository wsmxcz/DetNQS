# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified configuration management for LEVER workflows.

File: lever/config.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    Unified computation mode defining energy/gradient algorithms.
    
    ASYMMETRIC: S-norm with full S-C coupling (non-variational)
                E = ⟨ψ_S|Ĥ|ψ⟩ / ||ψ_S||², ∇_θ over S-space only
    
    PROXY:      Full-norm with diagonal C approximation (balanced)
                E = ⟨ψ|Ĥ̃|ψ⟩ / ||ψ||², ∇_θ over S∪C space
    
    EFFECTIVE:  S-norm with Schur downfolding (efficient)
                E = ⟨ψ_S|Ĥ_eff|ψ_S⟩ / ||ψ_S||², ∇_θ over S-space only
    """
    ASYMMETRIC = "asymmetric"
    PROXY = "proxy"
    EFFECTIVE = "effective"


@dataclass(frozen=True)
class SystemConfig:
    """Physical system definition."""
    fcidump_path: str
    n_orbitals: int
    n_alpha: int
    n_beta: int


@dataclass(frozen=True)
class OptimizationConfig:
    """VMC optimization parameters."""
    seed: int = 42
    learning_rate: float = 5e-4
    num_cycles: int = 10
    s_space_size: int = 200
    steps_per_cycle: int = 500
    report_interval: int = 50


@dataclass(frozen=True)
class EvaluationConfig:
    """Energy computation timing."""
    var_energy_mode: EvalMode = EvalMode.FINAL
    t_ci_energy_mode: EvalMode = EvalMode.NEVER
    s_ci_energy_mode: EvalMode = EvalMode.FINAL


@dataclass(frozen=True)
class ScreeningConfig:
    """C-space screening parameters."""
    mode: ScreenMode = ScreenMode.DYNAMIC
    eps1: float = 1e-6


@dataclass(frozen=True)
class LeverConfig:
    """
    Top-level LEVER run configuration.
    
    Numerical parameters:
        epsilon: Stability threshold for division/normalization
        normalize_wf: Apply joint L2 norm ||ψ_S||² + ||ψ_C||² = 1
    
    Compute mode:
        compute_mode: Unified energy/gradient algorithm (ASYMMETRIC/PROXY/EFFECTIVE)
    """
    system: SystemConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    screening: ScreeningConfig
    
    # Numerical parameters (previously in EngineConfig)
    epsilon: float = 1e-12
    normalize_wf: bool = True
    
    # Unified computation mode
    compute_mode: ComputeMode = ComputeMode.PROXY
