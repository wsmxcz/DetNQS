# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hamiltonian operator construction bridging Python frontend and C++ backend.

Implements PROXY (streaming screening) and FULL (exact subspace) workflows
for variational quantum chemistry. Uses sparse COO format for efficiency.

File: lever/engine/hamiltonian.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .. import core
from .utils import HamOp, SpaceRep

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Utility Functions
# ============================================================================

def _extract_diagonal(ham_op: HamOp) -> np.ndarray:
    """
    Extract diagonal H_ii from sparse COO matrix.
  
    Algorithm: Filter rows where i=j, index by row, fill dense array.
  
    Args:
        ham_op: Square sparse Hamiltonian in COO format
  
    Returns:
        H_diag[i] = ⟨i|Ĥ|i⟩
  
    Raises:
        ValueError: If operator is not square
    """
    if ham_op.shape[0] != ham_op.shape[1]:
        raise ValueError("Diagonal extraction requires square operator")
      
    size = ham_op.shape[0]
    H_diag = np.zeros(size, dtype=np.float64)
    diag_mask = (ham_op.rows == ham_op.cols)
    H_diag[ham_op.rows[diag_mask]] = ham_op.vals[diag_mask]
  
    return H_diag


# ============================================================================
# Main API: Proxy Hamiltonian Construction
# ============================================================================

def get_ham_proxy(
    S_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
    psi_S: Optional[NDArray[np.float64]] = None,
    use_heatbath: bool = True,
    eps1: float = 1e-3,
    diag_shift: float = 0.5,
    thresh: float = 1e-15,
) -> tuple[HamOp, HamOp, SpaceRep]:
    """
    Construct Ĥ_SS and Ĥ_SC with streaming C-space discovery.
  
    Screening Modes:
      • Static Heat-Bath (psi_S=None): Select by integral |⟨ij||ab⟩| > eps1
      • Dynamic Amplitude (psi_S given): Select by |H_ij·ψ_i| > eps1 (CIPSI-style)
  
    Algorithm:
      1. Generate singles/doubles from S-space
      2. Apply screening criterion (heat-bath or amplitude-weighted)
      3. Build Ĥ_SS and Ĥ_SC simultaneously in single pass
      4. Compute H_diag_C for discovered C-space
  
    Args:
        S_dets: Reference space determinants, shape (n_S, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
        psi_S: Optional wavefunction for dynamic screening
        use_heatbath: Enable heat-bath screening for doubles
        eps1: Screening threshold (mode-dependent interpretation)
        diag_shift: Level shift for H_diag_C (stabilization)
        thresh: Drop |H_ij| < thresh
  
    Returns:
        (ham_ss, ham_sc, space):
          • ham_ss: ⟨S|Ĥ|S⟩ operator
          • ham_sc: ⟨S|Ĥ|C⟩ coupling operator
          • space: S ∪ C Hilbert space representation with diagonals
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)

    # Select C++ backend based on screening mode
    if psi_S is not None:
        if len(psi_S) != size_S:
            raise ValueError(f"psi_S length {len(psi_S)} ≠ S_dets length {size_S}")
      
        psi = np.ascontiguousarray(psi_S, dtype=np.float64)
        result = core.get_ham_conn_amp(
            dets_S=S,
            psi_S=psi,
            int_ctx=int_ctx,
            n_orbitals=n_orbitals,
            eps1=eps1,
            thresh=thresh,
        )
    else:
        result = core.get_ham_conn(
            dets_S=S,
            int_ctx=int_ctx,
            n_orbitals=n_orbitals,
            use_heatbath=use_heatbath,
            eps1=eps1,
            thresh=thresh,
        )

    # Unpack sparse COO data
    coo_ss, coo_sc = result["SS"], result["SC"]
    C_dets = result["det_C"]
    size_C = result["size_C"]

    # Build sparse operators
    ham_ss = HamOp(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(size_S, size_S),
    )
    ham_sc = HamOp(
        rows=coo_sc["row"],
        cols=coo_sc["col"],
        vals=coo_sc["val"],
        shape=(size_S, size_C),
    )

    # Extract H_diag_S from Ĥ_SS diagonal
    H_diag_S = _extract_diagonal(ham_ss)

    # Compute H_diag_C via direct evaluation
    H_diag_C = (
        core.get_ham_diag(dets=C_dets, int_ctx=int_ctx) + diag_shift
        if size_C > 0
        else np.array([], dtype=np.float64)
    )

    space = SpaceRep(
        s_dets=S,
        c_dets=C_dets,
        H_diag_S=H_diag_S,
        H_diag_C=H_diag_C,
    )

    return ham_ss, ham_sc, space


# ============================================================================
# Main API: Full Hamiltonian Construction
# ============================================================================

def get_ham_full(
    S_dets: NDArray[np.uint64],
    C_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
    thresh: float = 1e-15,
) -> tuple[HamOp, HamOp, HamOp, SpaceRep]:
    """
    Construct complete Hamiltonian blocks for exact S ∪ C subspace.
  
    Algorithm:
      1. Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ in single S-space iteration
      2. Build ⟨C|Ĥ|C⟩ in separate C-space iteration
      3. Extract diagonals from square blocks
  
    Args:
        S_dets: Reference space determinants, shape (n_S, 2)
        C_dets: External space determinants, shape (n_C, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
        thresh: Drop |H_ij| < thresh
  
    Returns:
        (ham_ss, ham_sc, ham_cc, space):
          • ham_ss: ⟨S|Ĥ|S⟩ operator
          • ham_sc: ⟨S|Ĥ|C⟩ operator
          • ham_cc: ⟨C|Ĥ|C⟩ operator
          • space: S ∪ C Hilbert space representation with diagonals
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    C = np.ascontiguousarray(C_dets, dtype=np.uint64)
    size_S, size_C = len(S), len(C)

    # Phase 1-2: Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ jointly
    result_S = core.get_ham_block(
        bra_dets=S,
        ket_dets=C,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        thresh=thresh,
    )
  
    coo_ss, coo_sc = result_S["SS"], result_S["SC"]

    ham_ss = HamOp(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(size_S, size_S),
    )
    ham_sc = HamOp(
        rows=coo_sc["row"],
        cols=coo_sc["col"],
        vals=coo_sc["val"],
        shape=(size_S, size_C),
    )

    # Phase 3: Build ⟨C|Ĥ|C⟩ independently
    result_C = core.get_ham_block(
        bra_dets=C,
        ket_dets=None,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        thresh=thresh,
    )
  
    coo_cc = result_C["SS"]  # ⟨C|Ĥ|C⟩ returned as 'SS'

    ham_cc = HamOp(
        rows=coo_cc["row"],
        cols=coo_cc["col"],
        vals=coo_cc["val"],
        shape=(size_C, size_C),
    )

    # Extract diagonals from constructed blocks
    H_diag_S = _extract_diagonal(ham_ss)
    H_diag_C = _extract_diagonal(ham_cc)

    space = SpaceRep(
        s_dets=S,
        c_dets=C,
        H_diag_S=H_diag_S,
        H_diag_C=H_diag_C,
    )

    return ham_ss, ham_sc, ham_cc, space


__all__ = ["get_ham_proxy", "get_ham_full"]