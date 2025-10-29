# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER Engine: JIT-optimized computational core.

Architecture:
  - OuterCycle: Immutable build artifacts (H, features, closures)
  - InnerCycle: Mutable optimization state (params, optimizer)
  - Factory pattern: Pure JAX closures with captured constants
  - Hybrid execution: CPU SpMV (pure_callback) + GPU autodiff

Key components:
  - Closure factories: create_logpsi_fn, create_spmv_{eff,proxy}
  - Step kernels: create_step_fn, create_scan_fn (lax.scan compatible)
  - Hamiltonian builders: get_ham_{ss,proxy,full,eff}

File: lever/engine/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from .operator import *

from .geometry import *

from .step import *

from .hamiltonian import *

from . import kernels

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    
    # Closure factories
    "create_logpsi_fn", "create_spmv_eff", "create_spmv_proxy",
    
    "prepare_tape", "qgt_matvec", "qgt_dense",
    
    # Optimization kernels
    "ModeKernel", "create_step_fn", "create_scan_fn",
    
    # Hamiltonian builders
    "get_ham_ss", "get_ham_proxy", "get_ham_full", "get_ham_eff",
    
    # Low-level primitives
    "kernels",
]
