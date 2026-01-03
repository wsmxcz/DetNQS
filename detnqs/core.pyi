# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Type stubs for detnqs C++ extension module.

Provides type hints for determinant generation, Hamiltonian construction,
and heat-bath screening routines. Implements deterministic configuration
interaction with neural quantum state enhancement.

File: lever/core/detnqs.pyi
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from typing import Literal, Optional, TypedDict

import numpy as np
import numpy.typing as npt

# --- Type Aliases ---

DetArray = npt.NDArray[np.uint64]  # Shape (N, 2): (alpha, beta) bitstrings
F64Array = npt.NDArray[np.float64]  # Floating point arrays
U32Array = npt.NDArray[np.uint32]  # Unsigned integer indices
I32Array = npt.NDArray[np.int32]   # Signed integer indices


class CooData(TypedDict):
    """Sparse matrix in COO (Coordinate) format."""
    row: U32Array       # Row indices
    col: U32Array       # Column indices
    val: F64Array       # Nonzero values
    shape: tuple[int, int]  # Matrix dimensions


class HamBlockResult(TypedDict, total=False):
    """Hamiltonian blocks for predefined subspaces."""
    H_VV: CooData       # <V|H|V> block (always present)
    H_VX: CooData       # <V|H|X> block where X is external space (always present)
    det_X: DetArray     # External space determinants (if ket_dets provided)
    size_X: int         # Number of external determinants (if ket_dets provided)


class HamConnResult(TypedDict):
    """Hamiltonian with dynamically discovered connected space."""
    H_VV: CooData       # <V|H|V> block within variational space
    H_VP: CooData       # <V|H|P> block to perturbative space
    det_P: DetArray     # Perturbative space determinants (P = C \ V)
    size_P: int         # Number of perturbative determinants


class LocalConnRow(TypedDict):
    """Local Hamiltonian connectivity for a single bra determinant."""
    dets: DetArray      # Connected ket determinants, shape (M, 2)
    values: F64Array    # Matrix elements <bra|H|ket>, shape (M,)


class LocalConnBatch(TypedDict):
    """CSR-like local Hamiltonian connectivity for batched evaluation."""
    offsets: I32Array   # Row pointer array, shape (N_bra + 1,)
    dets: DetArray      # Concatenated ket determinants, shape (M, 2)
    values: F64Array    # Matrix elements, shape (M,)


# --- Integral Context ---

class IntCtx:
    """
    Integral context with optional heat-bath importance cache.
  
    Loads molecular integrals from FCIDUMP and provides Hamiltonian
    evaluation routines. Heat-bath table enables efficient screening
    based on integral magnitudes.
    """
  
    def __init__(self, fcidump_path: str, num_orb: int) -> None:
        """
        Initialize integral context from FCIDUMP file.
      
        Args:
            fcidump_path: Path to FCIDUMP file
            num_orb: Number of spatial orbitals
        """
        ...
  
    def get_e_nuc(self) -> float:
        """Nuclear repulsion energy from FCIDUMP."""
        ...
  
    def hb_prepare(self, threshold: float = 1e-8) -> None:
        """
        Build heat-bath importance table for screening.
      
        Stores two-electron integrals above threshold for efficient
        lookup during excitation generation.
      
        Args:
            threshold: Minimum |<ij||ab>| value to cache
        """
        ...
  
    def hb_clear(self) -> None:
        """Release heat-bath table memory."""
        ...


# --- Determinant Generation ---

def gen_fci_dets(n_orb: int, n_alpha: int, n_beta: int) -> DetArray:
    """
    Generate full CI space via combinatorial enumeration.
  
    Produces all possible electron configurations for given particle
    numbers and orbital count.
  
    Args:
        n_orb: Number of spatial orbitals
        n_alpha: Number of alpha electrons
        n_beta: Number of beta electrons
      
    Returns:
        Determinants as (alpha, beta) bitstring pairs, shape (N_FCI, 2)
    """
    ...


def gen_connected_dets(ref_dets: DetArray, n_orb: int) -> DetArray:
    """
    Generate all unique single and double excitations from reference.
  
    Args:
        ref_dets: Reference determinants, shape (N_ref, 2)
        n_orb: Number of spatial orbitals
      
    Returns:
        Canonicalized excited determinants (sorted, unique)
    """
    ...


# --- Hamiltonian Construction ---

def get_ham_diag(dets: DetArray, int_ctx: IntCtx) -> F64Array:
    """
    Compute diagonal Hamiltonian elements <x|H|x>.
  
    Args:
        dets: Determinants, shape (N, 2)
        int_ctx: Integral context
      
    Returns:
        Diagonal matrix elements, shape (N,)
    """
    ...


def get_ham_vv(
    dets_V: DetArray,
    int_ctx: IntCtx,
    n_orb: int,
) -> CooData:
    """
    Compute H_VV block via full singles/doubles enumeration.
  
    Efficient routine for CI calculations that only require the
    intra-space Hamiltonian without external coupling.
  
    Args:
        dets_V: Variational space determinants, shape (N_V, 2)
        int_ctx: Integral context
        n_orb: Number of spatial orbitals
      
    Returns:
        Sparse COO matrix for <V|H|V>
    """
    ...


def get_ham_block(
    bra_dets: DetArray,
    ket_dets: Optional[DetArray],
    int_ctx: IntCtx,
    n_orb: int,
) -> HamBlockResult:
    """
    Compute Hamiltonian blocks for predefined subspaces.
  
    Full S/D enumeration within bra space. If ket_dets provided,
    computes cross-space coupling block.
  
    Args:
        bra_dets: Variational space determinants (row indices)
        ket_dets: External space determinants (column indices).
                  If None, only H_VV is computed.
        int_ctx: Integral context
        n_orb: Number of spatial orbitals
      
    Returns:
        Dictionary with 'H_VV', 'H_VX', and optional 'det_X', 'size_X'
    """
    ...


def get_ham_conn(
    dets_V: DetArray,
    int_ctx: IntCtx,
    n_orb: int,
    use_heatbath: bool = False,
    eps1: float = 1e-6,
) -> HamConnResult:
    """
    Build Hamiltonian with static heat-bath screening.
  
    Discovers perturbative space P = C \ V through importance-based
    selection on integrals. Singles always enumerated; doubles
    screened by |<ij||ab>| >= eps1.
  
    Args:
        dets_V: Variational space V determinants
        int_ctx: Integral context (heat-bath table required if use_heatbath=True)
        n_orb: Number of spatial orbitals
        use_heatbath: Enable heat-bath screening for double excitations
        eps1: Integral threshold |<ij||ab>|
  
    Returns:
        Dictionary with 'H_VV', 'H_VP', 'det_P', 'size_P'
    """
    ...


def get_ham_conn_amp(
    dets_V: DetArray,
    psi_v: F64Array,
    int_ctx: IntCtx,
    n_orb: int,
    eps1: float = 1e-6,
) -> HamConnResult:
    """
    Build Hamiltonian with dynamic amplitude-weighted screening.
  
    Discovers perturbative space using amplitude importance.
    Per-row heat-bath cutoff: tau_i = eps1 / max(|psi_v[i]|, delta)
    Singles post-filtered by |H_ik * psi_v[i]| >= eps1.
  
    Args:
        dets_V: Variational space V determinants
        psi_v: Amplitudes for dets_V, shape (N_V,)
        int_ctx: Integral context (heat-bath table required)
        n_orb: Number of spatial orbitals
        eps1: Product threshold |H_ij * psi_i|
  
    Returns:
        Dictionary with 'H_VV', 'H_VP', 'det_P', 'size_P'
    """
    ...


def get_ham_eff(
    H_VV: CooData,
    H_VP: CooData,
    h_pp_diag: F64Array,
    e_ref: float,
    reg_type: Literal["linear_shift", "sigma"] = "sigma",
    epsilon: float = 1e-12,
    upper_only: bool = True,
) -> CooData:
    """
    Assemble effective Hamiltonian via second-order perturbation theory.
  
    Computes: H_eff = H_VV + H_VP * D^{-1} * H_PV
    where D_jj = E_ref - H_PP[j,j]
  
    Regularization strategies:
      - "linear_shift": 1/(d + epsilon*sign(d)) - Sharp cutoff at zero
      - "sigma": d/(d^2 + epsilon^2) - Smooth Tikhonov (recommended)
  
    Args:
        H_VV: <V|H|V> block in COO format
        H_VP: <V|H|P> block in COO format
        h_pp_diag: Diagonal elements H_PP[j,j], shape (N_P,)
        e_ref: Reference energy for denominator
        reg_type: Regularization strategy
        epsilon: Regularization parameter
        upper_only: Compute upper triangle only, then symmetrize
      
    Returns:
        Assembled H_eff in COO format
    """
    ...


def compute_variational_energy(
    dets: DetArray,
    coeffs: F64Array,
    int_ctx: IntCtx,
    n_orb: int,
    use_heatbath: bool = False,
    eps1: float = 1e-6,
) -> float:
    """
    Compute variational energy <Psi|H|Psi> on fixed basis.
  
    Note: coeffs must be domain-normalized before calling this function.
  
    Args:
        dets: Determinant basis, shape (N, 2)
        coeffs: Wavefunction amplitudes, shape (N,)
        int_ctx: Integral context
        n_orb: Number of spatial orbitals
        use_heatbath: Enable heat-bath screening
        eps1: Screening threshold
      
    Returns:
        Electronic energy (excludes nuclear repulsion)
    """
    ...


def compute_pt2(
    dets_V: DetArray,
    coeffs_V: F64Array,
    int_ctx: IntCtx,
    n_orb: int,
    e_ref: float,
    use_heatbath: bool = False,
    eps1: float = 1e-6,
) -> float:
    """
    Compute Epstein-Nesbet second-order perturbation correction.
  
    Evaluates: Delta_E_PT2 = sum_{j in P} |<j|H|V>|^2 / (E_ref - H_jj)
  
    Note: coeffs_V must be domain-normalized before calling.
          e_ref must be the electronic energy from optimizer.
  
    Args:
        dets_V: Variational space V determinants
        coeffs_V: Wavefunction amplitudes on V, shape (N_V,)
        int_ctx: Integral context
        n_orb: Number of spatial orbitals
        e_ref: Reference electronic energy
        use_heatbath: Enable heat-bath screening
        eps1: Screening threshold
      
    Returns:
        PT2 energy correction
    """
    ...


__all__ = [
    "DetArray",
    "F64Array",
    "U32Array",
    "I32Array",
    "CooData",
    "HamBlockResult",
    "HamConnResult",
    "LocalConnRow",
    "LocalConnBatch",
    "IntCtx",
    "gen_fci_dets",
    "gen_connected_dets",
    "get_ham_diag",
    "get_ham_vv",
    "get_ham_block",
    "get_ham_conn",
    "get_ham_conn_amp",
    "get_ham_eff",
    "compute_variational_energy",
    "compute_pt2",
]