# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multiplicative composition of wavefunction models.

Implements product model: ψ_total = ∏ᵢ ψᵢ via log-space summation.
Delegates parameter management to component models.

File: lever/models/product.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Sequence
    from .base import Features, LogPsi, WavefunctionModel


class ProductModel:
    """
    Compositional product of wavefunction models.
  
    Computes ψ = ∏ᵢ ψᵢ via log(ψ) = Σᵢ log(ψᵢ) for numerical stability.
    Not a WavefunctionModel subclass - orchestrates existing models.
  
    Args:
        models: Pre-initialized component models
      
    Raises:
        ValueError: If models sequence is empty
        RuntimeError: If any model is not initialized
    """
  
    __slots__ = ("models",)
  
    def __init__(self, models: Sequence[WavefunctionModel]) -> None:
        if not models:
            raise ValueError("ProductModel requires at least one component model")
        self.models = tuple(models)
        self._validate_initialization()
  
    def _validate_initialization(self) -> None:
        """Verify all component models are initialized."""
        for idx, model in enumerate(self.models):
            if model.variables is None:
                raise RuntimeError(
                    f"Component model {idx} is uninitialized. "
                    "Initialize all models before creating ProductModel."
                )
  
    @property
    def variables(self) -> tuple:
        """Component model parameters as tuple."""
        return tuple(m.variables for m in self.models)
  
    @variables.setter
    def variables(self, new_vars: tuple) -> None:
        """Update component parameters."""
        if len(new_vars) != len(self.models):
            raise ValueError(
                f"Expected {len(self.models)} parameter sets, got {len(new_vars)}"
            )
        for model, var in zip(self.models, new_vars, strict=True):
            model.variables = var
  
    def log_psi(self, variables: tuple, inputs: Features) -> LogPsi:
        """
        Compute log wavefunction amplitude.
      
        Args:
            variables: Tuple of component parameters
            inputs: Network input features
          
        Returns:
            log(ψ) = Σᵢ log(ψᵢ)
        """
        if len(variables) != len(self.models):
            raise ValueError(
                f"Expected {len(self.models)} parameter sets, got {len(variables)}"
            )
      
        # Sum in log space for numerical stability
        return sum(
            model.log_psi(var, inputs)
            for model, var in zip(self.models, variables, strict=True)
        )
  
    def log_psi_and_ders(
        self, variables: tuple, inputs: Features
    ) -> tuple[LogPsi, tuple]:
        """
        Compute log amplitude and parameter gradients.
      
        Args:
            variables: Tuple of component parameters
            inputs: Network input features
          
        Returns:
            (log_psi, gradients): Total log amplitude and per-component gradients
        """
        log_psis = []
        all_ders = []
      
        for model, var in zip(self.models, variables, strict=True):
            log_psi_i, ders_i = model.log_psi_and_ders(var, inputs)
            log_psis.append(log_psi_i)
            all_ders.append(ders_i)
      
        return sum(log_psis), tuple(all_ders)
  
    def __call__(self, inputs: Features) -> LogPsi:
        """Stateful evaluation using stored parameters."""
        return self.log_psi(self.variables, inputs)
  
    @property
    def is_holo(self) -> bool:
        """Holomorphicity requires all components to be holomorphic."""
        return all(m.is_holo for m in self.models)


__all__ = ["ProductModel"]