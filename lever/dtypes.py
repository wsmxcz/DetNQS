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
  
File: lever/dtypes.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import flax.struct
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .config import ComputeMode

# ============================================================================
# Type Aliases
# ============================================================================

PyTree = Any
SpMVFn = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]  # type: ignore # Sparse H·ψ
LogPsiFn = Callable[[PyTree], jnp.ndarray]  # params → log|ψ⟩

# ============================================================================
# Sparse Matrix Storage (Host)
# ============================================================================

@flax.struct.dataclass
class HamOp:
    """
    Sparse Hamiltonian block in COO format (CPU).
    
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
    Hilbert space representation (CPU).
    
    Determinant bitmasks and diagonal Hamiltonian elements:
      - s_dets/c_dets: (n_s/n_c, 2) uint64 arrays [α-string, β-string]
      - h_diag_s/c: Diagonal ⟨det_i|H|det_i⟩ per determinant
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
# Wavefunction Cache (Post-Optimization)
# ============================================================================

@flax.struct.dataclass
class PsiCache:
    """
    Cached wavefunction amplitudes after optimization.
    
    Computed once per outer cycle to avoid redundant forward passes.
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
# Outer Cycle Context (Immutable Build Artifacts)
# ============================================================================

@flax.struct.dataclass
class OuterCtx:
    """
    Outer loop build artifacts (immutable within cycle).
    
    Bundles Hamiltonian blocks, feature arrays, and compute closures.
    """
    space: SpaceRep = flax.struct.field(pytree_node=False)
    feat_s: jnp.ndarray           # S-space features
    feat_c: jnp.ndarray           # C-space features
    
    ham_opt: HamOp = flax.struct.field(pytree_node=False)  # Optimization block
    ham_sc: HamOp = flax.struct.field(pytree_node=False)   # S-C coupling
    
    logpsi_fn: LogPsiFn           # Neural ansatz: θ → log|ψ⟩
    spmv_fn: SpMVFn               # Sparse matvec: H·ψ
    
    e_ref: float = flax.struct.field(pytree_node=False)    # Reference energy
    e_nuc: float = flax.struct.field(pytree_node=False)    # Nuclear repulsion
    mode: ComputeMode = flax.struct.field(pytree_node=False)


# ============================================================================
# Inner Cycle State (Mutable Optimization Variables)
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
# Computation Results
# ============================================================================

@flax.struct.dataclass
class Contractions:
    """
    Hamiltonian-wavefunction products for energy/gradient.
    
    Block structure: n_ij = H_ij·ψ_j for i,j ∈ {S,C}
    """
    n_ss: jnp.ndarray                        # H_SS·ψ_S (always present)
    n_sc: jnp.ndarray | None = None          # H_SC·ψ_C (mode-dependent)
    n_cs: jnp.ndarray | None = None          # H_CS·ψ_S
    n_cc: jnp.ndarray | None = None          # H_CC·ψ_C


class StepResult(NamedTuple):
    """Single optimization step output."""
    state: InnerState             # Updated state (params, opt_state, step+1)
    energy: jnp.ndarray           # Scalar ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩


class GradResult(NamedTuple):
    """Energy and parameter gradient pair."""
    grad: PyTree                  # ∇_θ E(θ)
    energy: jnp.ndarray           # E(θ)


__all__ = [
    # Type aliases
    "PyTree",
    "SpMVFn",
    "LogPsiFn",
    
    # Data structures
    "HamOp",
    "SpaceRep",
    "PsiCache",
    "OuterCtx",
    "InnerState",
    
    # Result containers
    "Contractions",
    "StepResult",
    "GradResult",
]
