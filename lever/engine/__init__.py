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

from .utils import (
    # Type aliases
    PyTree,       # JAX pytree type
    SpMVFn,       # Sparse matrix-vector product: (ψ, params) -> H@ψ
    LogPsiFn,     # Wavefunction evaluator: (config, params) -> log(ψ)
  
    # CPU sparse matrix storage
    HamOp,        # Hamiltonian operator (CSR/COO)
    SpaceRep,     # Hilbert space representation (S/C/T partitions)
  
    # Computational contexts
    OuterCtx,     # Outer loop artifacts (immutable per cycle)
    InnerState,   # Inner loop state (mutable parameters)
  
    # Result containers
    Contractions, # Hamiltonian-wavefunction products (H@ψ, ⟨H⟩, etc.)
    StepResult,   # Single optimization step output
    GradResult,   # Energy + gradient computation result
  
    # Utilities
    masks_to_vecs,  # Bitmask -> occupancy vector converter
)

from .evaluator import (
    create_logpsi_fn,   # Build log(ψ) evaluator with captured features
    create_spmv_eff,    # Build H_eff@ψ_S (EFFECTIVE mode, S-space only)
    create_spmv_proxy,  # Build H@ψ (PROXY mode, full T-space)
)

from .step import (
    create_step_fn,  # Single-step gradient descent: (state, key) -> new_state
    create_scan_fn,  # JIT-compiled scan wrapper for multi-step iteration
)

from .hamiltonian import (
    get_ham_ss,     # H_SS: S-space block only
    get_ham_proxy,  # H_SS + H_SC: Proxy mode with PT2 screening
    get_ham_full,   # H_SS + H_SC + H_CC: Full subspace
    get_ham_eff,    # H_eff = H_SS + H_SC·D⁻¹·H_CS: Schur complement
)

from . import kernels  # Numba-accelerated SpMV primitives

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # Types
    "PyTree", "SpMVFn", "LogPsiFn",
  
    # Data structures
    "HamOp", "SpaceRep", "OuterCtx", "InnerState",
    "Contractions", "StepResult", "GradResult",
  
    # Utilities
    "masks_to_vecs",
  
    # Closure factories
    "create_logpsi_fn", "create_spmv_eff", "create_spmv_proxy",
  
    # Optimization kernels
    "create_step_fn", "create_scan_fn",
  
    # Hamiltonian builders
    "get_ham_ss", "get_ham_proxy", "get_ham_full", "get_ham_eff",
  
    # Low-level primitives
    "kernels",
]