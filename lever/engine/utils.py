# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core data structures for outer/inner optimization cycles.

Provides:
  - Sparse Hamiltonian storage (COO format)
  - Hilbert space representations
  - Optimization context and state containers
  - Utility functions for determinant transformations

File: lever/engine/utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import flax.struct
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

# ============================================================================
# Type Aliases
# ============================================================================

PyTree = Any
SpMVFn = Callable[[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]  # type: ignore # Sparse H·ψ closure
LogPsiFn = Callable[[PyTree], jnp.ndarray]  # params → log|ψ⟩

# ============================================================================
# Sparse Matrix Storage (Host)
# ============================================================================

@flax.struct.dataclass
class HamOp:
    """
    Sparse Hamiltonian block in COO format (CPU).
  
    Storage: H[rows[k], cols[k]] = vals[k] for k ∈ [0, nnz)
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
  
    Stores determinant bitmasks and diagonal matrix elements:
      - s_dets/c_dets: (n_s/n_c, 2) uint64 bitmasks [α-string, β-string]
      - h_diag_s/c: Diagonal ⟨det_i|H|det_i⟩ for fast variance reduction
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
# Outer Cycle Context (Immutable Build Artifacts)
# ============================================================================

@flax.struct.dataclass
class OuterCtx:
    """
    Outer loop build artifacts (immutable within single cycle).
  
    Contains space partitioning, pre-computed features, and mode-specific
    Hamiltonian operators. All closures capture constants for pure JAX JIT.
  
    Attributes:
        space: Hilbert space (S/C determinants + diagonals)
        feat_s: S-space occupancy vectors (n_s, 2·n_orb)
        feat_c: C-space occupancy vectors (n_c, 2·n_orb) or empty
        ham_opt: Optimization target (H_eff or H_SS)
        ham_sc: H_SC block for evolution mode (None for effective)
        logpsi_fn: Neural network closure params → log|ψ⟩
        spmv_fn: SpMV closure (ψ_s[, ψ_c]) → (H·ψ)
        e_ref: Reference energy E_0 (for effective mode)
        e_nuc: Nuclear repulsion E_nuc
        mode: Optimization strategy ("effective" | "proxy")
    """
    space: SpaceRep = flax.struct.field(pytree_node=False)
    feat_s: jnp.ndarray
    feat_c: jnp.ndarray
  
    ham_opt: HamOp = flax.struct.field(pytree_node=False)
    ham_sc: HamOp | None = flax.struct.field(pytree_node=False)
  
    logpsi_fn: LogPsiFn
    spmv_fn: SpMVFn
  
    e_ref: float = flax.struct.field(pytree_node=False)
    e_nuc: float = flax.struct.field(pytree_node=False)
    mode: str = flax.struct.field(pytree_node=False)


# ============================================================================
# Inner Cycle State (Mutable Optimization Variables)
# ============================================================================

@flax.struct.dataclass
class InnerState:
    """
    Inner loop optimization state.
  
    Tracks neural network parameters, optimizer state, and step counter.
    """
    params: PyTree
    opt_state: PyTree
    step: int = 0


# ============================================================================
# Computation Results
# ============================================================================

class Contractions(NamedTuple):
    """
    Hamiltonian-wavefunction products for energy/gradient computation.
  
    Components:
      - n_ss: H_SS·ψ_S
      - n_sc: H_SC·ψ_C (None for effective mode)
      - n_cs: H_CS·ψ_S (None for effective mode)
      - n_cc: H_CC·ψ_C (None for effective mode)
    """
    n_ss: jnp.ndarray
    n_sc: jnp.ndarray | None
    n_cs: jnp.ndarray | None
    n_cc: jnp.ndarray | None


class StepResult(NamedTuple):
    """Single optimization step output."""
    state: InnerState
    energy: jnp.ndarray  # Scalar energy ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩


class GradResult(NamedTuple):
    """Energy and parameter gradient."""
    grad: PyTree  # ∇_θ E(θ)
    energy: jnp.ndarray  # E(θ)


# ============================================================================
# Utility Functions
# ============================================================================

def masks_to_vecs(dets: jnp.ndarray, n_orb: int) -> jnp.ndarray:
    """
    Convert determinant bitmasks to occupancy vectors.
  
    Maps bit representations to neural network input features:
      |det⟩ = |i₁...i_α⟩⊗|j₁...j_β⟩ → [n₁^α,...,n_M^α, n₁^β,...,n_M^β]
    where n_k^σ ∈ {0,1} denotes spin-σ occupation of orbital k.
  
    Args:
        dets: (..., 2) uint64 bitmasks [α-string, β-string]
        n_orb: Number of spatial orbitals M
  
    Returns:
        (..., 2·M) occupancy vectors in [0,1]
  
    Example:
        >>> dets = jnp.array([[0b1011, 0b0110]])  # HF state for 4 orbitals
        >>> masks_to_vecs(dets, 4)
        [[1,1,0,1, 0,1,1,0]]  # [α₀,α₁,α₂,α₃, β₀,β₁,β₂,β₃]
    """
    if dets.shape[-1] != 2:
        raise ValueError(f"Expected dets.shape[-1]=2, got {dets.shape}")

    dets = dets.astype(jnp.uint64)
    alpha, beta = dets[..., 0], dets[..., 1]
  
    # Extract bit k from all determinants: (det >> k) & 1
    orb_idx = jnp.arange(n_orb, dtype=jnp.uint64)
    alpha_occ = ((alpha[..., None] >> orb_idx) & 1).astype(jnp.float32)
    beta_occ = ((beta[..., None] >> orb_idx) & 1).astype(jnp.float32)
  
    return jnp.concatenate([alpha_occ, beta_occ], axis=-1)


__all__ = [
    "PyTree",
    "SpMVFn",
    "LogPsiFn",
    "HamOp",
    "SpaceRep",
    "OuterCtx",
    "InnerState",
    "Contractions",
    "StepResult",
    "GradResult",
    "masks_to_vecs",
]