# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER Engine: JIT-optimized variational Monte Carlo computational core.

Architecture:
  - OuterCycle: Immutable build artifacts (H, features, closures)
  - InnerCycle: Mutable optimization state (params, optimizer)
  - Factory pattern: Pure JAX closures with captured constants
  - Hybrid execution: CPU SpMV (pure_callback) + GPU autodiff

Key components:
  - Closure factories: create_logpsi_fn, create_spmv_{eff,proxy}
  - Step kernels: create_step_fn, create_scan_fn (lax.scan compatible)
  - Hamiltonian builders: get_ham_{ss,proxy,full,eff}

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from ..utils.dtypes import (
    # Type aliases
    PyTree,
    SpMVFn,
    LogPsiFn,
    
    # Data structures
    HamOp,
    SpaceRep,
    PsiCache,
    OuterCtx,
    InnerState,
    
    # Result containers
    Contractions,
    StepResult,
    GradResult,
)

from ..utils.features import masks_to_vecs

from .evaluator import (
    create_logpsi_fn,
    create_spmv_eff,
    create_spmv_proxy,
)

from .step import (
    ModeKernel,
    create_step_fn,
    create_scan_fn,
)

from .hamiltonian import (
    get_ham_ss,
    get_ham_proxy,
    get_ham_full,
    get_ham_eff,
)

from . import kernels

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Types
    "PyTree", "SpMVFn", "LogPsiFn",
    
    # Data structures
    "HamOp", "SpaceRep", "PsiCache", "OuterCtx", "InnerState",
    "Contractions", "StepResult", "GradResult",
    
    # Utilities
    "masks_to_vecs",
    
    # Closure factories
    "create_logpsi_fn", "create_spmv_eff", "create_spmv_proxy",
    
    # Optimization kernels
    "ModeKernel", "create_step_fn", "create_scan_fn",
    
    # Hamiltonian builders
    "get_ham_ss", "get_ham_proxy", "get_ham_full", "get_ham_eff",
    
    # Low-level primitives
    "kernels",
]
