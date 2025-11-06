# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Search direction providers for variational optimization.

Implements three strategies:
  - GradientDirection: Steepest descent δ = -∇E
  - SRDirection: Natural gradient via QGT preconditioning
  - LMDirection: Linear method (Phase 5 placeholder)

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Any

import flax.struct
import jax

from ..engine.geometry import qgt_matvec, qgt_dense
from .base import DirectionState
from .linalg import solve_cg, solve_cholesky

if TYPE_CHECKING:
    from ..dtypes import PyTree, GeometryTape


# ============================================================================
# Direction States
# ============================================================================

@flax.struct.dataclass
class SRState(DirectionState):
    """SR solver state with CG diagnostics."""
    cg_iters: int = 0
    cg_residual: Any = 0.0  # Python float or JAX array


# ============================================================================
# Direction Providers
# ============================================================================

@dataclass
class GradientDirection:
    """
    Steepest descent: δ = -∇E (first-order, ignores geometry).
    """
    
    def __call__(
        self,
        grad: PyTree,
        state: DirectionState,
        *,
        tape: GeometryTape | None = None
    ) -> tuple[PyTree, DirectionState]:
        """Return negative gradient as direction."""
        direction = jax.tree.map(lambda g: -g, grad)
        return direction, state


@dataclass
class SRDirection:
    """
    Natural gradient via stochastic reconfiguration.
    
    Solves (S + λI)@δ = -∇E where S is quantum geometric tensor.
    Preconditions by local manifold geometry for faster convergence.
    
    Backends:
      - 'matvec': CG solver (O(n) memory)
      - 'dense': Cholesky solver (O(n²) memory, faster for small n)
    """
    backend: Literal["matvec", "dense"] = "matvec"
    damping: float = 1e-4        # Regularization λ
    cg_maxiter: int = 100        # CG iteration limit
    cg_tol: float = 1e-5         # CG residual tolerance
    
    def __call__(
        self,
        grad: PyTree,
        state: DirectionState,
        *,
        tape: GeometryTape | None = None
    ) -> tuple[PyTree, SRState]:
        """Solve (S + λI)@δ = -∇E for natural gradient direction."""
        if tape is None:
            raise ValueError("SR requires geometry tape for QGT")
        
        rhs = jax.tree.map(lambda g: -g, grad)
        
        if self.backend == "matvec":
            # Matrix-free CG with QGT matvec
            def qgt_op(v: PyTree) -> PyTree:
                return qgt_matvec(tape, v, diag_shift=self.damping)
            
            direction, info = solve_cg(
                qgt_op, rhs,
                maxiter=self.cg_maxiter,
                tol=self.cg_tol
            )
            new_state = SRState(
                cg_iters=info['niter'],
                cg_residual=info['residual']
            )
        
        elif self.backend == "dense":
            # Direct Cholesky with explicit QGT
            S = qgt_dense(tape, diag_shift=self.damping, symmetrize=True)
            direction = solve_cholesky(S, rhs, lower=True)
            new_state = SRState()
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        return direction, new_state


@dataclass
class LMDirection:
    """
    Linear method placeholder (full implementation in Phase 5).
    
    Currently delegates to SR with enhanced damping. True LM will
    solve eigenvalue problem in tangent subspace.
    """
    damping: float = 1e-3
    
    def __call__(
        self,
        grad: PyTree,
        state: DirectionState,
        *,
        tape: GeometryTape | None = None
    ) -> tuple[PyTree, DirectionState]:
        """Compute LM direction (Phase 5: tangent space eigensolver)."""
        if tape is None:
            raise ValueError("LM requires geometry tape")
        
        # Placeholder: SR-like solver with larger damping
        def qgt_op(v: PyTree) -> PyTree:
            return qgt_matvec(tape, v, diag_shift=self.damping)
        
        rhs = jax.tree.map(lambda g: -g, grad)
        direction, _ = solve_cg(qgt_op, rhs, maxiter=100, tol=1e-5)
        
        return direction, state


__all__ = [
    "SRState",
    "GradientDirection",
    "SRDirection",
    "LMDirection",
]
