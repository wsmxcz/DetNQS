# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hamiltonian operator construction bridging Python frontend and C++ backend.

Implements PROXY (streaming screening) and FULL (exact subspace) workflows
for variational quantum chemistry using sparse COO format.

File: lever/engine/hamiltonian.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .. import core
from ..dtypes import HamOp, SpaceRep

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ============================================================================
# Utility Functions
# ============================================================================

def _extract_diagonal(ham_op: HamOp) -> np.ndarray:
    """
    Extract diagonal elements H_ii from sparse COO matrix.
    
    Accumulates duplicates via np.add.at for stability.
    
    Returns:
        H_diag[i] = ⟨i|Ĥ|i⟩
    """
    if ham_op.shape[0] != ham_op.shape[1]:
        raise ValueError("Diagonal extraction requires square operator")

    size = ham_op.shape[0]
    H_diag = np.zeros(size, dtype=np.float64)
    
    diag_mask = ham_op.rows == ham_op.cols
    rows_diag = ham_op.rows[diag_mask].astype(np.int64, copy=False)
    vals_diag = ham_op.vals[diag_mask].astype(np.float64, copy=False)
    
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
    Remove S-space overlaps from C-space determinants via set-based lookup.
    
    Complexity: O(|S| + |C|)
    
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
    screen_eps: float = 1e-6,
    diag_shift: float = 0.5,
) -> tuple[HamOp, HamOp, SpaceRep]:
    """
    Build H_SS and H_SC with unified screening strategies.
    
    Screening modes:
      • "none"    : Full singles/doubles enumeration
      • "static"  : Heat-bath screening with fixed ε₁ cutoff
      • "dynamic" : Amplitude-weighted screening (requires ψ_S)
    
    Workflow:
      1. Generate C-space via selected screening backend
      2. Construct ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ in COO format
      3. Extract diagonal elements for variational solver
    
    Args:
        S_dets: Reference determinants, shape (n_S, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
        mode: Screening strategy in {"none", "static", "dynamic"}
        psi_S: S-space wavefunction for dynamic mode
        screen_eps: Screening threshold ε₁
        diag_shift: Stabilization shift for h_diag_c
    
    Returns:
        (H_SS, H_SC, space): Hamiltonian blocks and space representation
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)

    # Dispatch to screening backend
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
            eps1=screen_eps,
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
            eps1=screen_eps,
        )
    else:
        raise ValueError(f"Unknown screening mode: {mode}")

    coo_ss, coo_sc = result["H_SS"], result["H_SC"]
    C_dets = result["det_C"]
    size_C = result["size_C"]

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
    
    Optimized for CI evaluations requiring only S-space energies.
    Returns empty C-space in SpaceRep for API compatibility.
    
    Args:
        S_dets: Reference determinants, shape (n_S, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
    
    Returns:
        (H_SS, space): Hamiltonian block and S-only space representation
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    size_S = len(S)
    
    coo_ss = core.get_ham_ss(
        dets_S=S,
        int_ctx=int_ctx,
        n_orbitals=n_orbitals,
    )
    
    ham_ss = HamOp(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(size_S, size_S),
    )
    
    h_diag_s = _extract_diagonal(ham_ss)
    
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
    
    Two-phase construction:
      Phase 1: Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ in single S-iteration
      Phase 2: Build ⟨C|Ĥ|C⟩ in separate C-iteration
    
    Args:
        S_dets: Reference determinants, shape (n_S, 2)
        C_dets: External determinants, shape (n_C, 2)
        int_ctx: C++ integral context
        n_orbitals: Number of spatial orbitals
    
    Returns:
        (H_SS, H_SC, H_CC, space): Complete Hamiltonian and space
    """
    S = np.ascontiguousarray(S_dets, dtype=np.uint64)
    C = np.ascontiguousarray(C_dets, dtype=np.uint64)
    size_S, size_C = len(S), len(C)

    # Phase 1: S-space blocks
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

    # Phase 2: C-space block
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
    num_eps: float = 1e-12,
    upper_only: bool = True,
) -> HamOp:
    """
    Assemble effective Hamiltonian via perturbative correction.
    
    Formula: H_eff = H_SS + H_SC · D⁻¹ · H_CS
    where D_jj = E_ref - H_CC[j,j]
    
    Regularization strategies:
      • "linear_shift": d_inv = 1/(d + ε·sign(d))  - Sharp cutoff
      • "sigma":        d_inv = d/(d² + ε²)        - Smooth Tikhonov
    
    Algorithm: Column-wise outer products ΔH = Σⱼ (D_jj⁻¹)·b_j⊗b_j^T
    
    Args:
        ham_ss: ⟨S|Ĥ|S⟩ block
        ham_sc: ⟨S|Ĥ|C⟩ block
        h_cc_diag: Diagonal elements H_CC[j,j]
        e_ref: Reference energy for denominator
        reg_type: Regularization strategy
        num_eps: Regularization parameter ε
        upper_only: Build upper triangle only, exploit symmetry
    
    Returns:
        Assembled H_eff in COO format
    """
    if ham_ss.shape[0] != ham_ss.shape[1]:
        raise ValueError("H_SS must be square")
    if ham_sc.shape[0] != ham_ss.shape[0]:
        raise ValueError("H_SC row dimension mismatch")
    if ham_sc.shape[1] != len(h_cc_diag):
        raise ValueError("H_SC column dimension mismatch with h_cc_diag")
    
    H_SS_dict = _to_coo_dict(ham_ss)
    H_SC_dict = _to_coo_dict(ham_sc)
    h_cc = np.ascontiguousarray(h_cc_diag, dtype=np.float64)
    
    result = core.get_ham_eff(
        H_SS=H_SS_dict,
        H_SC=H_SC_dict,
        h_cc_diag=h_cc,
        e_ref=e_ref,
        reg_type=reg_type,
        epsilon=num_eps,
        upper_only=upper_only,
    )
    
    return _from_coo_dict(result)


__all__ = ["get_ham_ss", "get_ham_proxy", "get_ham_full", "get_ham_eff"]
