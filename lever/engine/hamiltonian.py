# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hamiltonian operator construction bridging Python frontend and C++ backend.

Provides two workflows for variational quantum chemistry:
  - PROXY: Streaming heat-bath screening with simultaneous Ĥ_SS and Ĥ_SC construction
  - FULL:  Three-phase exact subspace computation for pre-filtered C' space

File: lever/engine/hamiltonian.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import core
from .utils import HamOp, SpaceRep

if TYPE_CHECKING:
    from numpy.typing import NDArray


# --- Helper Functions ---

def _extract_diagonal(ham_op: HamOp) -> np.ndarray:
    """
    Extract diagonal elements from sparse Hamiltonian operator.
    
    Args:
        ham_op: Sparse Hamiltonian in COO format
    
    Returns:
        Dense diagonal array H_diag[i] = ⟨i|Ĥ|i⟩
    """
    size = ham_op.shape[0]
    H_diag = np.zeros(size, dtype=np.float64)
    diag_mask = (ham_op.rows == ham_op.cols)
    H_diag[ham_op.rows[diag_mask]] = ham_op.vals[diag_mask]
    return H_diag


# --- Main API ---

def get_ham_proxy(
    S_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
    use_heatbath: bool = True,
    eps1: float = 1e-3,
    diag_shift: float = 0.5,
    thresh: float = 1e-15,
) -> tuple[HamOp, HamOp, SpaceRep]:
    """
    Construct Ĥ_SS and Ĥ_SC with heat-bath screening for proxy-Hamiltonian method.
    
    Algorithm:
      1. Generate all single/double excitations from S-space
      2. Apply heat-bath screening: |H_Sα| ≥ eps1 × |H_SS|_max
      3. Build sparse matrices Ĥ_SS (intra-S) and Ĥ_SC (S→C coupling)
      4. Compute diagonal elements for variational energy correction
    
    Args:
        S_dets: Reference space determinants (n_S × n_words uint64)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
        use_heatbath: Enable heat-bath screening for C-space
        eps1: Heat-bath threshold relative to max |H_SS|
        diag_shift: Level-shift on H_diag_CC for stability
        thresh: Absolute cutoff for sparse matrix elements
    
    Returns:
        (ham_ss, ham_sc, space):
          - ham_ss: ⟨S|Ĥ|S⟩ operator
          - ham_sc: ⟨S|Ĥ|C⟩ operator  
          - space: Hilbert space representation with S ∪ C and diagonals
    """
    # Ensure C-contiguous memory layout for C++ backend
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)

    # Single C++ call builds both blocks and discovers C-space
    result = core.get_ham_conns_SSSC(
        dets_S=S,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        use_heatbath=use_heatbath,
        eps1=eps1,
        thresh=thresh,
    )

    # Unpack sparse matrix data
    coo_ss, coo_sc = result["SS"], result["SC"]
    C_dets = result["det_C"]
    size_C = result["size_C"]

    # Construct sparse operators
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

    # Extract diagonal elements: H_diag_S from Ĥ_SS diagonal
    H_diag_S = _extract_diagonal(ham_ss)

    # Compute H_diag_C via dedicated C++ routine
    H_diag_C = core.get_ham_diag(
        dets=C_dets,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
    )

    # Assemble space representation
    space = SpaceRep(
        s_dets=S,
        c_dets=C_dets,
        H_diag_S=H_diag_S,
        H_diag_C=H_diag_C + diag_shift,
    )

    return ham_ss, ham_sc, space


def get_ham_full(
    S_dets: NDArray[np.uint64],
    C_prime_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
    thresh: float = 1e-15,
) -> tuple[HamOp, HamOp, HamOp, SpaceRep]:
    """
    Construct full Hamiltonian blocks Ĥ_SS, Ĥ_SC', Ĥ_C'C' for exact subspace method.
    
    Algorithm (three-phase construction):
      1. Build ⟨S|Ĥ|S⟩   via all-to-all S determinant connections
      2. Build ⟨S|Ĥ|C'⟩  via S-to-C' cross connections
      3. Build ⟨C'|Ĥ|C'⟩ via all-to-all C' determinant connections
      4. Extract diagonals from square blocks for energy shifts
    
    Args:
        S_dets: Reference space determinants (n_S × n_words uint64)
        C_prime_dets: Pre-filtered connected space determinants (n_C' × n_words)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
        thresh: Absolute cutoff for sparse matrix elements
    
    Returns:
        (ham_ss, ham_sc_prime, ham_cc_prime, space):
          - ham_ss: ⟨S|Ĥ|S⟩ operator
          - ham_sc_prime: ⟨S|Ĥ|C'⟩ operator
          - ham_cc_prime: ⟨C'|Ĥ|C'⟩ operator (full matrix, not diagonal approx)
          - space: Hilbert space representation with S ∪ C' and diagonals
    """
    # Ensure C-contiguous memory layout
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    C_prime = np.ascontiguousarray(C_prime_dets, dtype=np.uint64)
    size_S, size_C_prime = len(S), len(C_prime)

    # Phase 1: Build ⟨S|Ĥ|S⟩
    coo_ss = core.get_ham_block(
        bra_dets=S,
        ket_dets=S,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        thresh=thresh,
    )
    ham_ss = HamOp(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(size_S, size_S),
    )

    # Phase 2: Build ⟨S|Ĥ|C'⟩
    coo_sc = core.get_ham_block(
        bra_dets=S,
        ket_dets=C_prime,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        thresh=thresh,
    )
    ham_sc_prime = HamOp(
        rows=coo_sc["row"],
        cols=coo_sc["col"],
        vals=coo_sc["val"],
        shape=(size_S, size_C_prime),
    )

    # Phase 3: Build ⟨C'|Ĥ|C'⟩ (full matrix, exact treatment)
    coo_cc = core.get_ham_block(
        bra_dets=C_prime,
        ket_dets=C_prime,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        thresh=thresh,
    )
    ham_cc_prime = HamOp(
        rows=coo_cc["row"],
        cols=coo_cc["col"],
        vals=coo_cc["val"],
        shape=(size_C_prime, size_C_prime),
    )

    # Extract diagonal elements from square blocks
    H_diag_S = _extract_diagonal(ham_ss)
    H_diag_C_prime = _extract_diagonal(ham_cc_prime)

    # Assemble space representation
    space = SpaceRep(
        s_dets=S,
        c_dets=C_prime,
        H_diag_S=H_diag_S,
        H_diag_C=H_diag_C_prime,
    )

    return ham_ss, ham_sc_prime, ham_cc_prime, space


def get_ham_ss(
    S_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
    thresh: float = 1e-15,
) -> HamOp:
    """
    Construct the Hamiltonian block within a given S-space, <S|Ĥ|S>.

    This function is a high-level wrapper around the C++ core's 
    `get_ham_block` routine. It is suitable for tasks like Full CI (when
    S-space is the complete FCI space) or for performing a standard
    Configuration Interaction (CI) calculation within a selected subspace.

    Args:
        S_dets: An array of determinants defining the S-space, with shape
                (n_S, 2) and dtype uint64.
        int_ctx: The C++ integral context, which holds the molecular integrals.
        n_orbitals: The number of spatial orbitals in the system.
        thresh: An absolute threshold to prune small matrix elements. Any
                |H_ij| < thresh will be discarded.

    Returns:
        A HamOp object representing the sparse <S|Ĥ|S> matrix in COO format.
    """
    # Ensure C-contiguous memory layout for performance in the C++ backend.
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)

    # Call the generic C++ block builder with bra=S and ket=S.
    coo_ss = core.get_ham_conns_SS(
        dets_S=S,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
        thresh=thresh,
    )

    # Construct and return the HamOp data structure.
    ham_ss = HamOp(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(size_S, size_S),
    )
    
    return ham_ss


__all__ = ["get_ham_proxy", "get_ham_full", "get_ham_ss"]
