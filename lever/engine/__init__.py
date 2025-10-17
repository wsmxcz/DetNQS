# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER Engine: The core computational subsystem for variational Monte Carlo.

This package provides a comprehensive toolkit for performing single-iteration
variational optimization steps. It is designed around a JAX-first philosophy,
ensuring that the entire computational graph is JIT-compatible, while seamlessly
integrating with high-performance CPU kernels for specialized tasks.

The public API exposes key components:
- Configuration objects (`EngineConfig`, `EnergyMode`, etc.) to control behavior.
- Data structures (`HamOp`, `SpaceRep`, `GradientResult`) for data exchange.
- The `Evaluator` class, a lazy-caching context for a single optimization step.
- Pure computation functions (`compute_energy_and_gradient`, `compute_S_matvec`)
  that operate on an `Evaluator` instance.
- A bridge function (`get_ham_ops`) to the C++ backend for Hamiltonian construction.

File: lever/engine/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025 (Refactored)
"""

# --- Configuration and Computation Modes ---
from .config import (
    DEFAULT_CONFIG,
    EnergyMode,
    EngineConfig,
    GradMode,
    ScoreKind
)

# --- Core Data Structures ---
# Note: Other data structures are typically used internally or returned by functions.
from .utils import (
    Contractions,
    GradientResult,
    HamOp,
    MatVecOp,
    PyTree,
    SOperatorMetadata,
    SpaceRep,
    SubspaceMetadata
)

# --- Lazy Evaluation Context ---
from .evaluator import Evaluator

# --- Hamiltonian Construction Bridge ---
from .hamiltonian import get_ham_proxy, get_ham_full, get_ham_ss

# --- Physics Computations ---
from .physics import (
    compute_energy,
    compute_energy_and_gradient,
    compute_local_energy
)

# --- Geometry Computations ---
from .geometry import (
    compute_S_matvec,
    compute_subspace_matrices
)


# Define what symbols are exported when a user does `from lever.engine import *`.
__all__ = [
    # Configuration
    "EngineConfig",
    "DEFAULT_CONFIG",
    "EnergyMode",
    "GradMode",
    "ScoreKind",

    # Data Structures
    "HamOp",
    "SpaceRep",
    "Contractions",
    "GradientResult",
    "SOperatorMetadata",
    "SubspaceMetadata",
    "PyTree",
    "MatVecOp",

    # Core Classes and Functions
    "Evaluator",
    "get_ham_proxy",
    "get_ham_full",
    "get_ham_ss",
    "compute_energy_and_gradient",
    "compute_energy",
    "compute_local_energy",
    "compute_S_matvec",
    "compute_subspace_matrices",
]