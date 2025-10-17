# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type stubs for C++ bridge module (nanobind extension).

Provides type hints for low-level computational routines including:
- Determinant space generation (Slater determinant enumeration)
- Hamiltonian matrix construction (COO sparse format)
- Heat-bath screening (importance-based determinant selection)

File: lever/core.pyi
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

# --- Type Aliases ---

DetArray = npt.NDArray[np.uint64]  # Determinants: shape (N, 2), bitstring pairs
F64Array = npt.NDArray[np.float64]  # Real values: shape (N,)
U32Array = npt.NDArray[np.uint32]   # Indices: shape (N,)


class CooData(TypedDict):
    """COO sparse matrix components."""
    row: U32Array
    col: U32Array
    val: F64Array


class HamSCResult(TypedDict):
    """<S|H|C> block with generated C-space."""
    conns: CooData
    det_C: DetArray
    size_C: int


class HamSTResult(TypedDict):
    """<S|H|T> unified block where T = S ∪ C."""
    conns: CooData
    det_T: DetArray
    size_S: int
    size_T: int


class HamSSSCResult(TypedDict):
    """Combined <S|H|S> and <S|H|C> blocks."""
    SS: CooData
    SC: CooData
    det_C: DetArray
    size_C: int


# --- C++ Extension Classes ---

class IntCtx:
    """
    Integral context with molecular integrals and optional heat-bath cache.
    
    Manages FCIDUMP-loaded integrals and provides Hamiltonian evaluation.
    Heat-bath table enables efficient importance-based screening.
    """
    
    def __init__(self, fcidump_path: str, num_orb: int) -> None:
        """Load integrals from FCIDUMP file."""
        ...
    
    def get_e_nuc(self) -> float:
        """Return nuclear repulsion energy."""
        ...
    
    def hb_prepare(self, threshold: float = 1e-8) -> None:
        """
        Build heat-bath table cache.
        
        Precomputes sorted excitation importance values for screening.
        
        Args:
            threshold: Minimum |integral| value to store
        """
        ...
    
    def hb_clear(self) -> None:
        """Release heat-bath table memory."""
        ...


# --- C++ Extension Functions ---

def gen_fci_dets(n_orb: int, n_alpha: int, n_beta: int) -> DetArray:
    """
    Generate full FCI space via combinatorial enumeration.
    
    Returns:
        Determinants as (α, β) bitstring pairs, shape (N_FCI, 2)
    """
    ...

# MODIFIED: Updated signature for gen_excited_dets
def gen_excited_dets(
    ref_dets: DetArray,
    n_orb: int,
    int_ctx: IntCtx,
    use_heatbath: bool = False,
    eps1: float = 1e-3,
) -> DetArray:
    """
    Generate all unique single/double excitations from reference determinants.
    
    Args:
        ref_dets: The reference determinants to generate excitations from.
        n_orb: The number of spatial orbitals.
        int_ctx: The integral context, required for heat-bath screening.
        use_heatbath: If True, enables importance-based screening for doubles.
        eps1: The heat-bath selection threshold for doubles.
        
    Returns:
        A canonicalized (sorted, unique) array of excited determinants.
    """
    ...


def get_ham_diag(dets: DetArray, int_ctx: IntCtx, n_orbitals: int) -> F64Array:
    """
    Compute diagonal Hamiltonian elements ⟨D|Ĥ|D⟩.
    
    Evaluates one-body and two-body terms via bitstring manipulation.
    """
    ...

# NEW: Added signature for get_ham_block
def get_ham_block(
    bra_dets: DetArray,
    ket_dets: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    thresh: float = 1e-12,
) -> CooData:
    """
    Compute the Hamiltonian block <bra|H|ket> between two arbitrary sets.
    
    Args:
        bra_dets: Determinants for the rows of the matrix.
        ket_dets: Determinants for the columns of the matrix.
        int_ctx: The integral context.
        n_orbitals: The number of spatial orbitals.
        thresh: A cutoff to prune small matrix elements.
        
    Returns:
        A dictionary representing the COO sparse matrix.
    """
    ...


def get_ham_conns_SS(
    dets_S: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    thresh: float = 1e-12,
) -> CooData:
    """
    Build ⟨S|Ĥ|S⟩ block in COO format via direct excitation enumeration.
    
    Args:
        thresh: Drop matrix elements with |H_ij| < thresh
    """
    ...


def get_ham_conns_SC(
    dets_S: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    use_heatbath: bool = False,
    eps1: float = 1e-3,
    thresh: float = 1e-12,
) -> HamSCResult:
    """
    Build ⟨S|Ĥ|C⟩ block with optional heat-bath screening.
    
    Args:
        use_heatbath: Enable importance-based determinant selection
        eps1: Heat-bath selection threshold for |H_SC|
        thresh: Drop matrix elements with |H_ij| < thresh
    
    Returns:
        Dictionary with COO matrix, generated C-space, and C-space size
    """
    ...


def get_ham_conns_ST(
    dets_S: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    use_heatbath: bool = False,
    eps1: float = 1e-3,
    thresh: float = 1e-12,
) -> HamSTResult:
    """
    Build unified ⟨S|Ĥ|T⟩ block where T = S ∪ C.
    
    Concatenates S and C spaces for single-matrix representation.
    
    Returns:
        Dictionary with COO matrix, T-space determinants, and sizes
    """
    ...


def get_ham_conns_SSSC(
    dets_S: DetArray,
    int_ctx: IntCtx,
    n_orbitals: int,
    use_heatbath: bool = False,
    eps1: float = 1e-3,
    thresh: float = 1e-12,
) -> HamSSSCResult:
    """
    Build ⟨S|Ĥ|S⟩ and ⟨S|Ĥ|C⟩ in single pass for efficiency.
    
    Avoids redundant excitation generation by processing both blocks together.
    
    Returns:
        Dictionary with both COO matrices, C-space determinants, and C-space size
    """
    ...


# Update the __all__ list to include the new function
__all__ = [
    "DetArray",
    "F64Array", 
    "U32Array",
    "CooData",
    "HamSCResult",
    "HamSTResult",
    "HamSSSCResult",
    "IntCtx",
    "gen_fci_dets",
    "gen_excited_dets",
    "get_ham_diag",
    "get_ham_block",      # Added the new function
    "get_ham_conns_SS",
    "get_ham_conns_SC",
    "get_ham_conns_ST",
    "get_ham_conns_SSSC",
]