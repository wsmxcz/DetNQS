# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER driver with simplified Controller-based architecture.

Provides backward-compatible interface while delegating all work
to the new three-layer workflow system (Compiler → Fitter → Controller).

File: lever/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LeverConfig
from .evolution import EvolutionStrategy
from .models import WavefunctionModel
from .workflow import Controller


@dataclass(frozen=True)
class DriverResults:
    """
    LEVER optimization results with convergence history.
  
    Attributes:
        config: Runtime configuration
        final_vars: Optimized model parameters
        full_hist: Per-step energy trajectory
        cycle_bounds: Cycle boundaries in full_hist
        var_hist: E_var(T) at cycle endpoints
        s_ci_hist: E_CI(S) at cycle endpoints
        t_ci_hist: E_CI(T) at cycle endpoints
        total_time: Wall-clock execution time (s)
    """
    config: LeverConfig
    final_vars: Any
    full_hist: list[float]
    cycle_bounds: list[int]
    var_hist: list[float]
    s_ci_hist: list[float]
    t_ci_hist: list[float]
    total_time: float


class Driver:
    """
    LEVER workflow driver (simplified facade).
  
    Delegates all work to Controller, providing backward compatibility
    with existing user code while enabling the new architecture.
  
    Architecture:
      Driver → Controller → [Compiler, Fitter, Diagnostics]
  
    Usage:
        >>> driver = Driver(config, model, strategy, optimizer)
        >>> results = driver.run()
    """

    def __init__(
        self,
        config: LeverConfig,
        model: WavefunctionModel,
        strategy: EvolutionStrategy,
        optimizer = None
    ):
        """
        Initialize driver with workflow components.
      
        Args:
            config: Complete LEVER configuration
            model: Wavefunction neural network
            strategy: S-space evolution strategy
            optimizer: Optax-compatible optimizer
        """
        self.controller = Controller(
            config=config,
            model=model,
            strategy=strategy,
            optimizer=optimizer
        )
  
    def run(self) -> DriverResults:
        """
        Execute complete LEVER workflow.
      
        Workflow stages:
          1. Compile: Build Workspace from S-space
          2. Fit: Optimize parameters via gradient descent
          3. Diagnose: Evaluate diagnostic energies
          4. Evolve: Select new S-space from scored C-space
      
        Returns:
            DriverResults with final parameters and convergence history
        """
        return self.controller.run()


__all__ = ["Driver", "DriverResults"]