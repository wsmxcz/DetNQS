# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Variational space evolution framework for LEVER.

Provides modular components for dynamically evolving the core S-space:
  - Scorer: Computes importance scores for determinants
  - Selector: Chooses new determinant set based on scores  
  - EvolutionStrategy: Orchestrates scorer-selector workflow

Example:
    >>> from lever import evolution
    >>> 
    >>> # Select top 200 determinants by amplitude
    >>> strategy = evolution.BasicStrategy(
    ...     scorer=evolution.scores.AmplitudeScorer(),
    ...     selector=evolution.selectors.TopKSelector(k=200)
    ... )
    >>> 
    >>> # Apply after parameter optimization
    >>> new_s_dets = strategy.evolve(final_evaluator)

File: lever/evolution/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from . import scores, selectors, strategies
from .base import EvolutionStrategy, Scorer, Selector
from .strategies import BasicStrategy, MassLockingStrategy, TwoStageStrategy, CumulativeMassStrategy

__all__ = [
    # Core abstractions
    "EvolutionStrategy",
    "Scorer", 
    "Selector",
    # Concrete strategies
    "BasicStrategy",
    "MassLockingStrategy", 
    "CumulativeMassStrategy",
    "TwoStageStrategy",
    # Submodules
    "scores",
    "selectors",
    "strategies",
]
