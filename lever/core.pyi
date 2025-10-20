# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type stubs for LEVER C++ extension module.

Provides type hints for determinant generation, Hamiltonian construction,
and heat-bath screening routines.
"""

from typing import Optional, TypedDict

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


class HamBlockResult(TypedDict):
    """Hamiltonian blocks for predefined spaces."""
    SS: CooData  # <S|H|S> block
    SC: CooData  # <S|H|C> block


class HamConnResult(TypedDict):
    """Hamiltonian with dynamically discovered C-space."""
    SS: CooData      # <S|H|S> block
    SC: CooData      # <S|H|C> block
    det_C: DetArray  # Discovered C-space determinants
    size_C: int      # Number of C-space determinants


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
            threshold: Minimum |integral| value to store
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
    thresh: float = 1e-15,
) -> HamBlockResult:
    """
    Compute Hamiltonian blocks for predefined spaces.
    
    Args:
        bra_dets: S-space determinants (rows)
        ket_dets: C-space determinants (columns). If None, only <S|H|S> computed
        int_ctx: Integral context
        n_orbitals: Number of spatial orbitals
        thresh: Cutoff for small matrix elements
        
    Returns:
        Dictionary with 'SS' and 'SC' COO matrices
    """
    ...


def get_ham_conn(
    dets_S: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    use_heatbath: bool = False,
    eps1: float = 1e-6,
    thresh: float = 1e-15,
) -> HamConnResult:
    """
    Build Hamiltonian with static heat-bath screening.
    
    Discovers C-space determinants using importance-based selection on integrals.
    
    Args:
        dets_S: S-space determinants
        int_ctx: Integral context
        n_orbitals: Number of spatial orbitals
        use_heatbath: Enable heat-bath screening for doubles
        eps1: Heat-bath threshold |<ij||ab>|
        thresh: Drop matrix elements |H_ij| < thresh
    
    Returns:
        Dictionary with SS/SC matrices and discovered C-space
    """
    ...


def get_ham_conn_amp(
    dets_S: DetArray,
    psi_S: F64Array,
    int_ctx: IntCtx,
    n_orbitals: int,
    eps1: float = 1e-6,
    thresh: float = 1e-15,
) -> HamConnResult:
    """
    Build Hamiltonian with dynamic amplitude screening.
    
    Discovers C-space determinants using amplitude-weighted importance.
    Requires heat-bath table.
    
    Args:
        dets_S: S-space determinants
        psi_S: Amplitudes for dets_S
        int_ctx: Integral context (heat-bath table required)
        n_orbitals: Number of spatial orbitals
        eps1: Screening threshold |H_ij * psi_i|
        thresh: Drop matrix elements |H_ij| < thresh
    
    Returns:
        Dictionary with SS/SC matrices and discovered C-space
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
]
