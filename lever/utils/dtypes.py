# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified data structures for LEVER quantum engine.

Defines immutable containers in layered architecture:
- Physical: HamOp, SpaceRep, PsiCache (sparse operators, basis states)
- Optimizer: OptState, GeometryTape (gradient computation context)
- Results: Contractions, StepResult, GradResult (computational outputs)
- Workflow: Workspace, FitResult, EvolutionState (high-level state)

File: lever/utils/dtypes.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
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
SpMVFn = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]  # type: ignore
LogPsiFn = Callable[[PyTree], jnp.ndarray]
OptimizerState = Any
JVPFn = Callable[[PyTree], jnp.ndarray]
VJPFn = Callable[[jnp.ndarray], tuple[PyTree]]


# ============================================================================
# Physical Representation Layer
# ============================================================================

@flax.struct.dataclass
class HamOp:
    """
    Sparse Hamiltonian in COO format.
    
    CPU-resident structure: (rows[i], cols[i], vals[i]) defines H_ij.
    """
    rows: NDArray[np.int32] = flax.struct.field(pytree_node=False)
    cols: NDArray[np.int32] = flax.struct.field(pytree_node=False)
    vals: NDArray[np.float64] = flax.struct.field(pytree_node=False)
    shape: tuple[int, int] = flax.struct.field(pytree_node=False)

    @property
    def nnz(self) -> int:
        """Number of nonzero matrix elements."""
        return self.vals.size


@flax.struct.dataclass
class SpaceRep:
    """
    Hilbert space partition: S-space (search) ∪ C-space (complement).
    
    Stores determinants as uint64 pairs [α-string, β-string] and
    precomputed diagonal Hamiltonian elements.
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


@flax.struct.dataclass
class PsiCache:
    """
    Wavefunction snapshot after optimization convergence.
    
    Stores log-amplitudes and normalized amplitudes for both spaces.
    Partition: psi_all[:n_s] = S, psi_all[n_s:] = C.
    """
    log_all: jnp.ndarray
    psi_all: jnp.ndarray
    n_s: int = flax.struct.field(pytree_node=False)
    n_c: int = flax.struct.field(pytree_node=False)

    @property
    def psi_s(self) -> jnp.ndarray:
        """S-space amplitudes."""
        return self.psi_all[:self.n_s]

    @property
    def psi_c(self) -> jnp.ndarray:
        """C-space amplitudes."""
        return self.psi_all[self.n_s:]


# ============================================================================
# Optimizer Layer
# ============================================================================

@flax.struct.dataclass
class OptState:
    """
    Mutable optimization trajectory state.
    
    Encapsulates neural parameters, optimizer internals (e.g., Adam moments),
    and iteration counter.
    """
    params: PyTree
    opt_state: PyTree
    step: int = 0


@flax.struct.dataclass
class GeometryTape:
    """
    Linearization context for quantum geometric tensor (QGT) computation.
    
    Records JVP/VJP functions from forward pass for efficient natural
    gradient: (F + λI)^{-1} · ∇E, where F_ij = Re⟨∂_i ψ|∂_j ψ⟩.
    """
    jvp_fn: JVPFn = flax.struct.field(pytree_node=False)
    vjp_fn: VJPFn = flax.struct.field(pytree_node=False)
    log_psi: jnp.ndarray
    weights: jnp.ndarray
    centered_mean: PyTree


# ============================================================================
# Computation Results Layer
# ============================================================================

@flax.struct.dataclass
class Contractions:
    """
    Block-wise Hamiltonian-wavefunction products: n_ij = ∑_k H_ik · ψ_k.
    
    Notation: n_ss = H_SS·ψ_S, n_sc = H_SC·ψ_C, etc.
    C-space blocks optional for S-only computations.
    """
    n_ss: jnp.ndarray
    n_sc: jnp.ndarray | None = None
    n_cs: jnp.ndarray | None = None
    n_cc: jnp.ndarray | None = None


class StepResult(NamedTuple):
    """Single optimization step output: updated state and energy."""
    state: OptState
    energy: jnp.ndarray


class GradResult(NamedTuple):
    """Energy functional gradient: (∇_θ E, E(θ))."""
    grad: PyTree
    energy: jnp.ndarray


class ScoreResult(NamedTuple):
    """Scored determinant batch for space evolution."""
    scores: np.ndarray
    dets: np.ndarray
    meta: dict


# ============================================================================
# Workflow Layer
# ============================================================================

@flax.struct.dataclass
class Workspace:
    """
    Immutable compilation artifact for fixed Hilbert space.
    
    Contains JIT-compiled closures (log_psi, spmv_fn) and precomputed
    features/Hamiltonians. Compiled once per cycle; reused across steps
    via functional transformations.
    """
    space: SpaceRep = flax.struct.field(pytree_node=False)
    feat_s: jnp.ndarray
    feat_c: jnp.ndarray
    
    ham_opt: HamOp = flax.struct.field(pytree_node=False)
    ham_sc: HamOp = flax.struct.field(pytree_node=False)
    
    log_psi: Callable = flax.struct.field(pytree_node=False)
    spmv_fn: Callable = flax.struct.field(pytree_node=False)
    
    e_ref: float = flax.struct.field(pytree_node=False)
    e_nuc: float = flax.struct.field(pytree_node=False)
    mode: ComputeMode = flax.struct.field(pytree_node=False)


@dataclass(frozen=True)
class FitResult:
    """
    Complete optimization cycle output.
    
    Records converged parameters, energy trajectory, final wavefunction,
    and convergence diagnostics.
    """
    params: PyTree
    energy_trace: list[float]
    psi_cache: PsiCache
    converged: bool
    steps: int


@dataclass(frozen=True)
class EvolutionState:
    """
    Minimal persistent state for iterative space evolution.
    
    Pure data container; state transitions handled by Controller.
    Tracks current S-space basis, optimized parameters, and cycle metadata.
    """
    s_dets: np.ndarray
    params: PyTree
    e_ref: float | None
    cycle: int


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
    
    # Physical
    "HamOp",
    "SpaceRep",
    "PsiCache",
    
    # Optimizer
    "OptState",
    "GeometryTape",
    
    # Results
    "Contractions",
    "StepResult",
    "GradResult",
    "ScoreResult",
    
    # Workflow
    "Workspace",
    "FitResult",
    "EvolutionState",
]
