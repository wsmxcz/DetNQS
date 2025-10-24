# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized configuration for LEVER workflows.

Uses typed dataclasses to define system, optimization, and evaluation
parameters for clarity, type safety, and reusability.

File: lever/config.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import engine


class EvalMode(str, Enum):
    """Timing control for expensive energy evaluations."""
    NEVER = "never"
    FINAL = "final"
    EVERY = "every"


class ScreenMode(str, Enum):
    """Screening strategy for C-space generation."""
    NONE = "none"        # Full enumeration without heat-bath
    STATIC = "static"    # Heat-bath screening on integrals
    DYNAMIC = "dynamic"  # Amplitude-weighted screening


@dataclass(frozen=True)
class SystemConfig:
    """Physical system definition."""
    fcidump_path: str
    n_orbitals: int
    n_alpha: int
    n_beta: int


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimization process parameters."""
    seed: int = 42
    learning_rate: float = 5e-4
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
    """Connection space screening parameters."""
    mode: ScreenMode = ScreenMode.DYNAMIC
    eps1: float = 1e-6


@dataclass(frozen=True)
class LeverConfig:
    """Top-level LEVER run configuration."""
    system: SystemConfig
    optimization: OptimizationConfig
    evaluation: EvaluationConfig
    screening: ScreeningConfig
    engine: engine.EngineConfig = field(default_factory=engine.EngineConfig)
