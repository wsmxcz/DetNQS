# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter optimization engine for fixed workspace.

Implements gradient-based variational minimization:
  E[θ] = ⟨Ψ(θ)|H|Ψ(θ)⟩ / ⟨Ψ(θ)|Ψ(θ)⟩

File: lever/workflow/fitter.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import numpy as np

from .. import engine
from ..utils.dtypes import FitResult, OptState, PyTree, Workspace
from ..utils.features import create_psi_cache
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from ..config import LoopConfig


class Fitter:
    """
    Gradient descent optimizer for neural network parameters.
  
    Minimizes variational energy on a compiled workspace with
    fixed Hamiltonian matrix and basis functions.
  
    Convergence: ΔE < ε over check_interval steps
    """
  
    def __init__(
        self,
        loop_cfg: LoopConfig,
        report_interval: int,
        num_eps: float = 1e-12
    ) -> None:
        """
        Initialize optimizer.
      
        Args:
            loop_cfg: Iteration and convergence settings
            report_interval: Logging frequency (steps)
            num_eps: Numerical stability threshold
        """
        self.cfg = loop_cfg
        self.report_interval = report_interval
        self.num_eps = num_eps
        self.logger = get_logger()
  
    def fit(
        self,
        workspace: Workspace,
        params: PyTree,
        optimizer
    ) -> FitResult:
        """
        Run optimization loop until convergence or max_steps.
      
        Algorithm:
          1. Initialize optimizer state
          2. JIT-compile gradient step
          3. Iterate: θ ← optimizer.update(∇E[θ], θ)
          4. Cache final wavefunction amplitudes
      
        Args:
            workspace: Compiled Hamiltonian + features
            params: Initial network parameters
            optimizer: Optax gradient transformer
      
        Returns:
            Optimization result with energy trace and cached Ψ
        """
        # Initialize optimization state
        opt_state = optimizer.init(params)
        state = OptState(params=params, opt_state=opt_state, step=0)
      
        # Compile single gradient step: (state, _) → (new_state, energy)
        step_fn = engine.step.create_step_fn(
            workspace, optimizer, self.num_eps
        )
        step_fn_jit = jax.jit(step_fn, donate_argnums=(0,))
      
        # Gradient descent loop
        energies: list[float] = []
        self.logger.optimization_header()
      
        for step in range(self.cfg.max_steps):
            state, energy = step_fn_jit(state, None)
            energies.append(float(energy))
          
            if (step + 1) % self.report_interval == 0:
                self.logger.optimization_step(
                    step + 1,
                    energies[-1],
                    self.cfg.max_steps
                )
          
            if self._check_convergence(energies, step):
                converged = True
                break
        else:
            converged = False
      
        # Build amplitude cache for final parameters
        psi_cache = self._create_cache(workspace, state.params)
      
        return FitResult(
            params=state.params,
            energy_trace=energies,
            psi_cache=psi_cache,
            converged=converged,
            steps=len(energies)
        )

    def _check_convergence(
        self,
        energies: list[float],
        step: int
    ) -> bool:
        """
        Test convergence: |E[t] - E[t-w]| < tol.
      
        Args:
            energies: Energy history
            step: Current iteration (0-indexed)
      
        Returns:
            True if converged within tolerance window
        """
        if step < self.cfg.check_interval:
            return False
      
        if (step + 1) % self.cfg.check_interval != 0:
            return False
      
        window = self.cfg.check_interval
        e_recent = energies[-1]
        e_previous = energies[-window] if len(energies) >= window else energies[0]
      
        delta = abs(e_recent - e_previous)
      
        if delta < self.cfg.step_tol:
            self.logger.inner_loop_converged(step + 1, delta)
            return True
      
        return False
  
    def _create_cache(
        self,
        workspace: Workspace,
        params: PyTree
    ):
        """
        Evaluate and cache Ψ_S, Ψ_C amplitudes.
      
        For dual-closure workspaces, uses full-space evaluator.
        Reports L² norms for validation.
      
        Args:
            workspace: Compiled workspace with basis
            params: Optimized network parameters
      
        Returns:
            Amplitude cache with S/C partitions
        """
        # Select appropriate evaluator (handles dual-closure case)
        if isinstance(workspace.log_psi, tuple):
            _, logpsi_full = workspace.log_psi
            log_all = logpsi_full(params)
        else:
            log_all = workspace.log_psi(params)
      
        psi_cache = create_psi_cache(
            log_all,
            workspace.space.n_s,
            workspace.space.n_c
        )
      
        # Compute and log partition norms
        norm_s = float(np.sum(np.abs(np.array(psi_cache.psi_s))**2))
        norm_c = float(np.sum(np.abs(np.array(psi_cache.psi_c))**2))
        self.logger.wavefunction_cache(norm_s, norm_c)
      
        return psi_cache


__all__ = ["Fitter"]