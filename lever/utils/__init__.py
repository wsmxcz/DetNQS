# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER utility module: core data structures and computational primitives.

Exports:
  - Type aliases: PyTree, SpMVFn, LogPsiFn, optimizer states
  - Physical types: HamOp, SpaceRep, PsiCache
  - Result containers: Contractions, StepResult, GradResult, ScoreResult
  - Workflow state: Workspace, FitResult, EvolutionState
  - Helper functions: feature engineering, logging, PyTree ops, space ops

File: lever/utils/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from .dtypes import *
from .features import *
from .logger import *
from .pytree import *
from .space import *

__all__ = [
    # ===== Type Aliases =====
    "PyTree",
    "SpMVFn",
    "LogPsiFn",
    "JVPFn",
    "VJPFn",
    "OptimizerState",
  
    # ===== Physical System =====
    "HamOp",
    "SpaceRep",
    "PsiCache",
  
    # ===== Optimizer State =====
    "OptState",
    "GeometryTape",
  
    # ===== Result Containers =====
    "Contractions",
    "StepResult",
    "GradResult",
    "ScoreResult",
  
    # ===== Workflow State =====
    "Workspace",
    "FitResult",
    "EvolutionState",
  
    # ===== Feature Engineering =====
    "masks_to_vecs",
    "dets_to_features",
    "compute_normalized_amplitudes",
    "create_psi_cache",
  
    # ===== Logging =====
    "LeverLogger",
    "get_logger",
  
    # ===== PyTree Operations =====
    "tree_dot",
    "tree_scale",
    "tree_add",
    "tree_sub",
    "tree_norm",
  
    # ===== Space Manipulation =====
    "remove_overlaps",
    "merge_spaces",
    "get_space_overlap",
    "count_overlaps",
]