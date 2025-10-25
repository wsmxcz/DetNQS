# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hamiltonian operator construction bridging Python frontend and C++ backend.

Implements PROXY (streaming screening) and FULL (exact subspace) workflows
for variational quantum chemistry using sparse COO format.

File: lever/engine/hamiltonian.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
    Extract diagonal H_ii from sparse COO matrix with duplicate accumulation.
    
    Returns:
        H_diag[i] = ⟨i|Ĥ|i⟩
    """
    if ham_op.shape[0] != ham_op.shape[1]:
        raise ValueError("Diagonal extraction requires square operator")

    size = ham_op.shape[0]
    H_diag = np.zeros(size, dtype=np.float64)
    
    # Filter diagonal entries
    diag_mask = ham_op.rows == ham_op.cols
    rows_diag = ham_op.rows[diag_mask].astype(np.int64, copy=False)
    vals_diag = ham_op.vals[diag_mask].astype(np.float64, copy=False)
    
    # Accumulate duplicates via in-place addition
    np.add.at(H_diag, rows_diag, vals_diag)
    
    return H_diag


def _to_coo_dict(ham_op: HamOp) -> dict[str, np.ndarray | tuple[int, int]]:
    """Convert HamOp to CooData dict for C++ backend."""
    return {
        "row": ham_op.rows,
        "col": ham_op.cols,
        "val": ham_op.vals,
        "shape": ham_op.shape,
    }


def _from_coo_dict(coo_dict: dict) -> HamOp:
    """Convert C++ CooData dict to HamOp."""
    return HamOp(
        rows=coo_dict["row"],
        cols=coo_dict["col"],
        vals=coo_dict["val"],
        shape=tuple(coo_dict["shape"]),
    )


def _remove_s_from_c(
    S_dets: NDArray[np.uint64], 
    C_dets: NDArray[np.uint64]
) -> NDArray[np.uint64]:
    """
    Remove S-space overlaps from C-space determinants.
    
    Algorithm: Set-based lookup with O(|S| + |C|) complexity.
    
    Returns:
        Filtered C-space determinants preserving original order
    """
    if len(C_dets) == 0:
        return np.empty((0, 2), dtype=np.uint64)
    
    S_set = {(int(d[0]), int(d[1])) for d in S_dets}
    C_filtered = [d for d in C_dets if (int(d[0]), int(d[1])) not in S_set]
    
    return (
        np.array(C_filtered, dtype=np.uint64) 
        if C_filtered 
        else np.empty((0, 2), dtype=np.uint64)
    )


# ============================================================================
# Main API: Proxy Hamiltonian Construction
# ============================================================================

def get_ham_proxy(
    S_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
    mode: str = "none",
    psi_S: NDArray[np.float64] | None = None,
    eps1: float = 1e-6,
    diag_shift: float = 0.0,
) -> tuple[HamOp, HamOp, SpaceRep]:
    """
    Build H_SS and H_SC with unified screening modes.
    
    Screening modes:
      • "none": Full singles/doubles enumeration
      • "static": Heat-bath screening with fixed ε₁ cutoff
      • "dynamic": Amplitude-weighted screening (requires ψ_S)
    
    Algorithm:
      1. Generate C-space via C++ backend with selected screening
      2. Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ in COO format
      3. Extract/compute H_diag for S and C spaces
    
    Args:
        S_dets: Reference determinants, shape (n_S, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
        mode: Screening strategy
        psi_S: S-space wavefunction for dynamic mode
        eps1: Screening threshold
        diag_shift: Level shift for h_diag_c stabilization
    
    Returns:
        (H_SS, H_SC, space): Hamiltonian blocks and Hilbert space representation
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)

    # Dispatch to C++ backend based on screening mode
    if mode == "none":
        C_raw = core.gen_excited_dets(S, n_orbitals)
        C = _remove_s_from_c(S, C_raw)
        result = core.get_ham_block(
            bra_dets=S, 
            ket_dets=C, 
            int_ctx=int_ctx, 
            n_orbitals=n_orbitals
        )
    elif mode == "static":
        result = core.get_ham_conn(
            dets_S=S,
            int_ctx=int_ctx,
            n_orbitals=n_orbitals,
            use_heatbath=True,
            eps1=eps1,
        )
    elif mode == "dynamic":
        if psi_S is None:
            raise ValueError("Dynamic mode requires psi_S amplitudes")
        if len(psi_S) != size_S:
            raise ValueError(f"psi_S size mismatch: {len(psi_S)} ≠ {size_S}")
        
        psi = np.ascontiguousarray(psi_S, dtype=np.float64)
        result = core.get_ham_conn_amp(
            dets_S=S,
            psi_S=psi,
            int_ctx=int_ctx,
            n_orbitals=n_orbitals,
            eps1=eps1,
        )
    else:
        raise ValueError(f"Unknown screening mode: {mode}")

    # Unpack sparse COO data
    coo_ss, coo_sc = result["H_SS"], result["H_SC"]
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

    # Extract/compute diagonal elements
    h_diag_s = _extract_diagonal(ham_ss)
    h_diag_c = (
        core.get_ham_diag(dets=C_dets, int_ctx=int_ctx) + diag_shift
        if size_C > 0
        else np.array([], dtype=np.float64)
    )

    space = SpaceRep(
        s_dets=S,
        c_dets=C_dets,
        h_diag_s=h_diag_s,
        h_diag_c=h_diag_c,
    )

    return ham_ss, ham_sc, space

# ============================================================================
# Main API: Basic Hamiltonian Construction
# ============================================================================

def get_ham_ss(
    S_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
) -> tuple[HamOp, SpaceRep]:
    """
    Build H_SS block without C-space discovery.
    
    Efficient for CI evaluations without SC coupling requirements.
    Uses full singles/doubles enumeration within S-space only.
    
    Algorithm:
      1. Call C++ backend for ⟨S|Ĥ|S⟩ block construction
      2. Extract h_diag_s from diagonal elements
      3. Return space with empty C-space
    
    Args:
        S_dets: Reference determinants, shape (n_S, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
    
    Returns:
        (H_SS, space): Hamiltonian block and S-only space representation
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)
    
    # Build H_SS block via C++ backend
    coo_ss = core.get_ham_ss(
        dets_S=S,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
    )
    
    # Construct sparse operator
    ham_ss = HamOp(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(size_S, size_S),
    )
    
    # Extract diagonal elements
    h_diag_s = _extract_diagonal(ham_ss)
    
    # Create space representation with empty C-space
    space = SpaceRep(
        s_dets=S,
        c_dets=np.empty((0, 2), dtype=np.uint64),
        h_diag_s=h_diag_s,
        h_diag_c=np.array([], dtype=np.float64),
    )
    
    return ham_ss, space

# ============================================================================
# Main API: Full Hamiltonian Construction
# ============================================================================

def get_ham_full(
    S_dets: NDArray[np.uint64],
    C_dets: NDArray[np.uint64],
    int_ctx: core.IntCtx,
    n_orbitals: int,
) -> tuple[HamOp, HamOp, HamOp, SpaceRep]:
    """
    Build complete Hamiltonian blocks for exact S ∪ C subspace.
    
    Algorithm:
      1. Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ in single S-space iteration
      2. Build ⟨C|Ĥ|C⟩ in separate C-space iteration
      3. Extract diagonals from square blocks
    
    Returns:
        (H_SS, H_SC, H_CC, space): Complete Hamiltonian blocks and space
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    C = np.ascontiguousarray(C_dets, dtype=np.uint64)
    size_S, size_C = len(S), len(C)

    # Phase 1: Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩
    result_S = core.get_ham_block(
        bra_dets=S,
        ket_dets=C,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
    )
    
    coo_ss, coo_sc = result_S["H_SS"], result_S["H_SC"]
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

    # Phase 2: Build ⟨C|Ĥ|C⟩
    result_C = core.get_ham_block(
        bra_dets=C,
        ket_dets=None,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
    )
    
    coo_cc = result_C["H_SS"]
    ham_cc = HamOp(
        rows=coo_cc["row"],
        cols=coo_cc["col"],
        vals=coo_cc["val"],
        shape=(size_C, size_C),
    )

    # Extract diagonals
    h_diag_s = _extract_diagonal(ham_ss)
    h_diag_c = _extract_diagonal(ham_cc)

    space = SpaceRep(
        s_dets=S,
        c_dets=C,
        h_diag_s=h_diag_s,
        h_diag_c=h_diag_c,
    )

    return ham_ss, ham_sc, ham_cc, space


# ============================================================================
# Main API: Effective Hamiltonian Assembly
# ============================================================================

def get_ham_eff(
    ham_ss: HamOp,
    ham_sc: HamOp,
    h_cc_diag: NDArray[np.float64],
    e_ref: float,
    reg_type: Literal["linear_shift", "sigma"] = "sigma",
    epsilon: float = 1e-12,
    upper_only: bool = True,
) -> HamOp:
    """
    Assemble effective Hamiltonian via perturbative correction.
    
    Computes: H_eff = H_SS + H_SC · D⁻¹ · H_CS
    where D_jj = E_ref - H_CC[j,j]
    
    Regularization:
      • "linear_shift": 1/(d + ε·sign(d)) - Sharp zero crossing
      • "sigma": d/(d² + ε²) - Smooth Tikhonov-style
    
    Algorithm:
      1. Compute regularized D_jj⁻¹
      2. Column-wise outer products: ΔH = Σⱼ (D_jj⁻¹)·bⱼ⊗bⱼ
      3. Assemble H_eff = H_SS + ΔH with duplicate merging
    
    Args:
        ham_ss: ⟨S|Ĥ|S⟩ block
        ham_sc: ⟨S|Ĥ|C⟩ block
        h_cc_diag: Diagonal elements H_CC[j,j]
        e_ref: Reference energy for denominator
        reg_type: Regularization strategy
        epsilon: Regularization parameter
        upper_only: Compute upper triangle only, then mirror
    
    Returns:
        Assembled H_eff in COO format
    """
    if ham_ss.shape[0] != ham_ss.shape[1]:
        raise ValueError("H_SS must be square")
    if ham_sc.shape[0] != ham_ss.shape[0]:
        raise ValueError("H_SC row dimension mismatch")
    if ham_sc.shape[1] != len(h_cc_diag):
        raise ValueError("H_SC column dimension mismatch with h_cc_diag")
    
    # Convert to C++ interface format
    H_SS_dict = _to_coo_dict(ham_ss)
    H_SC_dict = _to_coo_dict(ham_sc)
    h_cc = np.ascontiguousarray(h_cc_diag, dtype=np.float64)
    
    # Call C++ backend
    result = core.get_ham_eff(
        H_SS=H_SS_dict,
        H_SC=H_SC_dict,
        h_cc_diag=h_cc,
        e_ref=e_ref,
        reg_type=reg_type,
        epsilon=epsilon,
        upper_only=upper_only,
    )
    
    return _from_coo_dict(result)


__all__ = ["get_ham_ss", "get_ham_proxy", "get_ham_full", "get_ham_eff"]
