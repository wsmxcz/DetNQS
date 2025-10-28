# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified data structures and type aliases for LEVER quantum engine.

Centralizes container types to avoid circular imports. Core structures:
  - HamOp: Sparse Hamiltonian in COO format (CPU storage)
  - SpaceRep: S/C-space determinants and diagonal H elements
  - PsiCache: Optimized wavefunction amplitudes
  - OuterCtx: Immutable cycle context (Hamiltonians + closures)
  - InnerState: Mutable optimization state (params + optimizer)

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import flax.struct
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from ..config import ComputeMode

# ============================================================================
# Type Aliases
# ============================================================================

PyTree = Any
SpMVFn = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]  # type: ignore # Sparse H·ψ product
LogPsiFn = Callable[[PyTree], jnp.ndarray]                       # params → log|ψ⟩
OptimizerState = Any                                             # Protocol-based
JVPFn = Callable[[PyTree], jnp.ndarray]                          # Jacobian-vector product
VJPFn = Callable[[jnp.ndarray], tuple[PyTree]]                   # Vector-Jacobian product


# ============================================================================
# Sparse Matrix Storage (Host)
# ============================================================================

@flax.struct.dataclass
class HamOp:
    """
    Sparse Hamiltonian block in COO format (CPU storage).
  
    Storage: H[rows[k], cols[k]] = vals[k], k ∈ [0, nnz)
    """
    rows: NDArray[np.int32] = flax.struct.field(pytree_node=False)
    cols: NDArray[np.int32] = flax.struct.field(pytree_node=False)
    vals: NDArray[np.float64] = flax.struct.field(pytree_node=False)
    shape: tuple[int, int] = flax.struct.field(pytree_node=False)

    @property
    def nnz(self) -> int:
        """Number of stored elements."""
        return self.vals.size


@flax.struct.dataclass
class SpaceRep:
    """
    Hilbert space representation (CPU storage).
  
    Stores determinant bitmasks and diagonal Hamiltonian elements:
      - {s,c}_dets: (n_{s,c}, 2) uint64 arrays [α-string, β-string]
      - h_diag_{s,c}: Diagonal elements ⟨det_i|H|det_i⟩
    """
    s_dets: NDArray[np.uint64] = flax.struct.field(pytree_node=False)
    c_dets: NDArray[np.uint64] = flax.struct.field(pytree_node=False)
    h_diag_s: NDArray[np.float64] = flax.struct.field(pytree_node=False)
    h_diag_c: NDArray[np.float64] = flax.struct.field(pytree_node=False)

    @property
    def n_s(self) -> int:
        """S-space dimension."""
        return self.s_dets.shape[0]

    @property
    def n_c(self) -> int:
        """C-space dimension."""
        return self.c_dets.shape[0]


# ============================================================================
# Wavefunction Cache
# ============================================================================

@flax.struct.dataclass
class PsiCache:
    """
    Cached wavefunction amplitudes after optimization.
  
    Computed once per outer cycle to avoid redundant forward passes.
    Provides split views into S- and C-space components.
    """
    log_all: jnp.ndarray          # log(ψ) for full S ∪ C space
    psi_all: jnp.ndarray          # ψ = exp(log_all)
    n_s: int = flax.struct.field(pytree_node=False)
    n_c: int = flax.struct.field(pytree_node=False)
  
    @property
    def psi_s(self) -> jnp.ndarray:
        """S-space amplitudes ψ_S."""
        return self.psi_all[:self.n_s]
  
    @property
    def psi_c(self) -> jnp.ndarray:
        """C-space amplitudes ψ_C."""
        return self.psi_all[self.n_s:]


# ============================================================================
# Outer Cycle Context (Immutable)
# ============================================================================

@flax.struct.dataclass
class OuterCtx:
    """
    Immutable outer cycle context bundling Hamiltonians and closures.
  
    Closures (logpsi_fn, spmv_fn) are marked non-pytree to prevent
    redundant JIT recompilation during optimization loops.
    """
    space: SpaceRep = flax.struct.field(pytree_node=False)
    feat_s: jnp.ndarray
    feat_c: jnp.ndarray
  
    ham_opt: HamOp = flax.struct.field(pytree_node=False)
    ham_sc: HamOp = flax.struct.field(pytree_node=False)
  
    # Non-pytree closures for stable JIT caching
    logpsi_fn: LogPsiFn = flax.struct.field(pytree_node=False)
    spmv_fn: SpMVFn = flax.struct.field(pytree_node=False)
  
    e_ref: float = flax.struct.field(pytree_node=False)
    e_nuc: float = flax.struct.field(pytree_node=False)
    mode: ComputeMode = flax.struct.field(pytree_node=False)


# ============================================================================
# Inner Cycle State (Mutable)
# ============================================================================

@flax.struct.dataclass
class InnerState:
    """
    Inner loop optimization state.
  
    Tracks neural parameters θ, optimizer state, and iteration counter.
    """
    params: PyTree                # Neural network parameters θ
    opt_state: PyTree             # Optimizer internal state
    step: int = 0                 # Iteration counter


# ============================================================================
# Geometry Information
# ============================================================================

@flax.struct.dataclass
class GeometryTape:
    """
    Linearization tape for QGT computation via implicit differentiation.
  
    Captures Jacobian structure at current parameters θ. Enables efficient
    gradient and natural gradient computation without redundant forward passes.
    All closures are pure JAX functions with stable compilation footprint.
  
    Attributes:
        jvp_fn: Forward-mode product v → J @ v
        vjp_fn: Reverse-mode product c → J^T @ c
        log_psi: Wavefunction logarithm log ψ(θ)
        weights: Normalized probability |ψ|²/||ψ||²
        centered_mean: Parameter-wise mean ⟨∂_k log ψ⟩ for centering
    """
    jvp_fn: JVPFn = flax.struct.field(pytree_node=False)
    vjp_fn: VJPFn = flax.struct.field(pytree_node=False)
    log_psi: jnp.ndarray
    weights: jnp.ndarray
    centered_mean: PyTree


# ============================================================================
# Computation Results
# ============================================================================

@flax.struct.dataclass
class Contractions:
    """
    Hamiltonian-wavefunction products for energy/gradient evaluation.
  
    Block structure: n_ij = H_ij·ψ_j for i,j ∈ {S,C}
    Optional blocks depend on ComputeMode setting.
    """
    n_ss: jnp.ndarray                        # H_SS·ψ_S (always present)
    n_sc: jnp.ndarray | None = None          # H_SC·ψ_C (mode-dependent)
    n_cs: jnp.ndarray | None = None          # H_CS·ψ_S
    n_cc: jnp.ndarray | None = None          # H_CC·ψ_C


class StepResult(NamedTuple):
    """Single optimization step output."""
    state: InnerState             # Updated state (params, opt_state, step+1)
    energy: jnp.ndarray           # Scalar energy ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩


class GradResult(NamedTuple):
    """Energy and parameter gradient pair."""
    grad: PyTree                  # Parameter gradient ∇_θ E(θ)
    energy: jnp.ndarray           # Energy E(θ)


class ScoreResult(NamedTuple):
    """Scored determinant container for importance sampling."""
    scores: np.ndarray            # Importance scores per determinant
    dets: np.ndarray              # Determinant bitmasks
    meta: dict                    # Additional metadata


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Type aliases
    "PyTree",
    "SpMVFn",
    "LogPsiFn",
    "JVPFn",
    "VJPFn",
    "OptimizerState",
  
    # Data structures
    "HamOp",
    "SpaceRep",
    "PsiCache",
    "OuterCtx",
    "InnerState",
    "GeometryTape",
  
    # Result containers
    "Contractions",
    "StepResult",
    "GradResult",
    "ScoreResult",
]