# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type stubs for LEVER C++ extension module.

Provides type hints for determinant generation, Hamiltonian construction,
and heat-bath screening routines.
"""

from typing import Literal, Optional, TypedDict

import numpy as np
import numpy.typing as npt

# --- Type Aliases ---

DetArray = npt.NDArray[np.uint64]  # Shape (N, 2): determinant bitstrings
F64Array = npt.NDArray[np.float64]  # Shape (N,): floating point values
U32Array = npt.NDArray[np.uint32]   # Shape (N,): integer indices


class CooData(TypedDict):
    """Sparse matrix in COO format."""
    row: U32Array
    col: U32Array
    val: F64Array
    shape: tuple[int, int]


class HamBlockResult(TypedDict, total=False):
    """Hamiltonian blocks for predefined spaces."""
    H_SS: CooData       # <S|H|S> block (always present)
    H_SC: CooData       # <S|H|C> block (always present)
    det_C: DetArray     # C-space determinants (if ket_dets provided)
    size_C: int         # Number of C-space determinants (if ket_dets provided)


class HamConnResult(TypedDict):
    """Hamiltonian with dynamically discovered C-space."""
    H_SS: CooData       # <S|H|S> block
    H_SC: CooData       # <S|H|C> block
    det_C: DetArray     # Discovered C-space determinants
    size_C: int         # Number of C-space determinants


# --- Integral Context ---

class IntCtx:
    """
    Integral context with optional heat-bath cache.
    
    Loads molecular integrals from FCIDUMP and provides Hamiltonian evaluation.
    Heat-bath table enables efficient importance-based screening.
    """
    
    def __init__(self, fcidump_path: str, num_orb: int) -> None:
        """
        Load integrals from FCIDUMP file.
        
        Args:
            fcidump_path: Path to FCIDUMP file
            num_orb: Number of spatial orbitals
        """
        ...
    
    def get_e_nuc(self) -> float:
        """Get nuclear repulsion energy."""
        ...
    
    def hb_prepare(self, threshold: float = 1e-8) -> None:
        """
        Build heat-bath table cache.
        
        Args:
            threshold: Minimum |integral| value to store (default: MAT_ELEMENT_THRESH)
        """
        ...
    
    def hb_clear(self) -> None:
        """Release heat-bath table memory."""
        ...


# --- Determinant Generation ---

def gen_fci_dets(n_orb: int, n_alpha: int, n_beta: int) -> DetArray:
    """
    Generate full CI space via combinatorial enumeration.
    
    Args:
        n_orb: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
        
    Returns:
        Determinants as (α, β) bitstring pairs, shape (N_FCI, 2)
    """
    ...


def gen_excited_dets(ref_dets: DetArray, n_orb: int) -> DetArray:
    """
    Generate all unique single and double excitations.
    
    Args:
        ref_dets: Reference determinants, shape (N, 2)
        n_orb: Number of spatial orbitals
        
    Returns:
        Canonicalized (sorted, unique) excited determinants
    """
    ...


# --- Hamiltonian Construction ---

def get_ham_diag(dets: DetArray, int_ctx: IntCtx) -> F64Array:
    """
    Compute diagonal Hamiltonian elements ⟨D|Ĥ|D⟩.
    
    Args:
        dets: Determinants, shape (N, 2)
        int_ctx: Integral context
        
    Returns:
        Diagonal elements, shape (N,)
    """
    ...


def get_ham_block(
    bra_dets: DetArray,
    ket_dets: Optional[DetArray],
    int_ctx: IntCtx,
    n_orbitals: int,
) -> HamBlockResult:
    """
    Compute Hamiltonian blocks for predefined spaces.
    
    Full singles/doubles enumeration within S-space.
    If ket_dets provided, computes cross-space SC block.
    
    Args:
        bra_dets: S-space determinants (rows)
        ket_dets: C-space determinants (columns). If None, only <S|H|S> computed
        int_ctx: Integral context
        n_orbitals: Number of spatial orbitals
        
    Returns:
        Dictionary with 'H_SS', 'H_SC', and optional 'det_C', 'size_C'
    """
    ...


def get_ham_conn(
    dets_S: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    use_heatbath: bool = False,
    eps1: float = 1e-6,
) -> HamConnResult:
    """
    Build Hamiltonian with static heat-bath screening.
    
    Discovers C-space determinants using importance-based selection on integrals.
    Singles always enumerated; doubles screened by |⟨ij||ab⟩| ≥ eps1.
    
    Args:
        dets_S: S-space determinants
        int_ctx: Integral context (heat-bath table required if use_heatbath=True)
        n_orbitals: Number of spatial orbitals
        use_heatbath: Enable heat-bath screening for doubles
        eps1: Heat-bath threshold |⟨ij||ab⟩|
    
    Returns:
        Dictionary with 'H_SS', 'H_SC', 'det_C', 'size_C'
    """
    ...


def get_ham_conn_amp(
    dets_S: DetArray,
    psi_S: F64Array,
    int_ctx: IntCtx,
    n_orbitals: int,
    eps1: float = 1e-6,
) -> HamConnResult:
    """
    Build Hamiltonian with dynamic amplitude screening.
    
    Discovers C-space using amplitude-weighted importance.
    Per-row heat-bath cutoff: τᵢ = eps1 / max(|ψₛ[i]|, δ)
    Singles post-filtered by |Hᵢₖ · ψₛ[i]| ≥ eps1
    
    Args:
        dets_S: S-space determinants
        psi_S: Amplitudes for dets_S, shape (N,)
        int_ctx: Integral context (heat-bath table required)
        n_orbitals: Number of spatial orbitals
        eps1: Screening threshold |H_ij · ψ_i|
    
    Returns:
        Dictionary with 'H_SS', 'H_SC', 'det_C', 'size_C'
    """
    ...


def get_ham_eff(
    H_SS: CooData,
    H_SC: CooData,
    h_cc_diag: F64Array,
    e_ref: float,
    reg_type: Literal["linear_shift", "sigma"] = "sigma",
    epsilon: float = 1e-12,
    upper_only: bool = True,
) -> CooData:
    """
    Assemble effective Hamiltonian via perturbative correction.
    
    Computes: H_eff = H_SS + H_SC · D⁻¹ · H_CS
    where D_jj = E_ref - H_CC[j,j]
    
    Regularization strategies:
      - "linear_shift": 1/(d + ε·sign(d)) - Sharp transition at zero
      - "sigma": d/(d² + ε²) - Smooth Tikhonov-style (recommended)
    
    Args:
        H_SS: <S|H|S> block in COO format
        H_SC: <S|H|C> block in COO format
        h_cc_diag: Diagonal elements H_CC[j,j], shape (|C|,)
        e_ref: Reference energy for denominator
        reg_type: Regularization strategy
        epsilon: Regularization parameter
        upper_only: Compute upper triangle only, then mirror
        
    Returns:
        Assembled H_eff in COO format
    """
    ...


__all__ = [
    "DetArray",
    "F64Array",
    "U32Array",
    "CooData",
    "HamBlockResult",
    "HamConnResult",
    "IntCtx",
    "gen_fci_dets",
    "gen_excited_dets",
    "get_ham_diag",
    "get_ham_block",
    "get_ham_conn",
    "get_ham_conn_amp",
    "get_ham_eff",
]
