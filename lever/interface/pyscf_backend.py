# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PySCF interface for quantum chemistry calculations.

Provides thin wrappers for molecular integrals, orbital transformations,
and configuration interaction methods.

File: lever/interface/pyscf_backend.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from pyscf import gto, scf, mp, ci, cc, fci, mcscf, lo
from pyscf.lo import edmiston, iao, ibo, orth, vvo
from pyscf.tools import fcidump as fcidump_mod


@dataclass
class ActiveSpaceSpec:
    """Configuration interaction active space specification."""
    n_core: int
    n_active: int
    n_virt: int
    nelecas_alpha: int
    nelecas_beta: int


def run_scf(
    atom: Any,
    basis: str,
    *,
    charge: int = 0,
    spin: int = 0,
    symmetry: bool = False,
    scf_type: str = "auto",
    conv_tol: float = 1e-10,
    conv_tol_grad: float = 1e-8,
) -> tuple[gto.Mole, scf.hf.SCF]:
    """
    Build molecular system and run self-consistent field calculation.
    
    Algorithm: RHF/ROHF/UHF with automatic spin adaptation.
    Fallback to Second-Order SCF (SOSCF) if standard DIIS fails.
    
    Args:
        atom: Molecular geometry specification
        basis: Basis set identifier
        charge: Total molecular charge
        spin: Spin multiplicity (2S)
        symmetry: Enable point group symmetry
        scf_type: Electronic structure method
        conv_*: Convergence thresholds
    
    Returns:
        (mol, mf) Molecular and mean-field objects
    """
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.symmetry = bool(symmetry)
    mol.build()

    method_map = {
        "auto": scf.RHF if mol.spin == 0 else scf.ROHF,
        "rhf": scf.RHF,
        "rohf": scf.ROHF, 
        "uhf": scf.UHF
    }
    
    if scf_type.lower() not in method_map:
        raise ValueError(f"Unsupported SCF type: {scf_type}")
    
    mf = method_map[scf_type.lower()](mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    
    # First attempt: standard DIIS
    mf.kernel()

    # Fallback: Second-Order SCF if DIIS fails
    if not mf.converged:
        print("    ! SCF (DIIS) non-convergence. Attempting Second-Order SCF (SOSCF)...")
        mf = scf.newton(mf)
        mf.kernel()
        
        if not mf.converged:
            print("    ! SOSCF also failed. Proceeding with unconverged orbitals.")

    return mol, mf


def _sorted_eigh(dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize density matrix and sort by descending occupation."""
    dm_sym = 0.5 * (dm + dm.T.conj())
    occ, U = np.linalg.eigh(dm_sym)
    return occ[::-1], U[:, ::-1]


def compute_natural_orbitals(
    mf: scf.hf.SCF,
    kind: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute natural orbitals from correlated density matrices.
    
    Methods:
      - MP2: Second-order Møller-Plesset perturbation theory
      - CISD: Configuration interaction with singles and doubles  
      - CCSD: Coupled cluster with singles and doubles
    
    Args:
        mf: Mean-field object
        kind: Correlation method for density matrix
    
    Returns:
        (C, occ) Orbital coefficients and occupations
    """
    kind = kind.lower()
    
    if kind == "none":
        return np.array(mf.mo_coeff), np.array(mf.mo_occ)

    if kind == "mp2":
        m2 = mp.MP2(mf).run(verbose=0)
        try:
            occ, U = m2.make_natorbs()
        except Exception:
            dm = m2.make_rdm1(ao_repr=False)
            occ, U = _sorted_eigh(dm)
        return mf.mo_coeff @ U, occ

    if kind == "cisd":
        myci = ci.CISD(mf).run(verbose=0)
        dm = myci.make_rdm1()
        occ, U = _sorted_eigh(dm)
        return mf.mo_coeff @ U, occ

    if kind == "ccsd":
        mycc = cc.CCSD(mf).run(verbose=0)
        mycc.solve_lambda()
        from pyscf.cc import ccsd_rdm
        dm = ccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2, ao_repr=False)
        occ, U = _sorted_eigh(dm)
        return mf.mo_coeff @ U, occ

    raise ValueError(f"Unknown natural orbital method: {kind}")


def _localize_block(
    mol: gto.Mole,
    C_sub: np.ndarray,
    method: str,
) -> np.ndarray:
    """Apply orbital localization to subspace."""
    method_map = {
        "boys": lo.boys.Boys,
        "pipek": lo.pipek.PipekMezey, 
        "edmiston": edmiston.ER,
    }
    
    if method in method_map:
        return method_map[method](mol, C_sub).kernel()
    
    if method == "ibo":
        S = mol.intor_symmetric("int1e_ovlp")
        iaos = iao.iao(mol, C_sub)
        iaos = orth.vec_lowdin(iaos, S)
        return ibo.ibo(mol, C_sub, iaos=iaos, s=S)
    
    raise ValueError(f"Unknown localization: {method}")


def localize_orbitals(
    mol: gto.Mole,
    mf: scf.hf.SCF,
    C: np.ndarray,
    *,
    method: str,
    loc_virtual: bool = False,
    active_window: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Localize molecular orbitals using specified method.
    
    Localization schemes:
      - Boys: Maximize orbital dipole moment separation
      - Pipek-Mezey: Maximize atomic population localization
      - Edmiston-Ruedenberg: Maximize self-repulsion energy
      - IBO: Intrinsic bond orbitals from atomic basis
    
    Args:
        mol: Molecular system
        mf: Mean-field reference  
        C: Orbital coefficients
        method: Localization algorithm
        loc_virtual: Apply LIVVO to virtual space
        active_window: Active orbital range for CAS
    
    Returns:
        Localized orbital coefficients
    """
    if method.lower() in ("canonical", ""):
        return C

    nmo = C.shape[1]

    if active_window is None:
        # Full space: occupied/virtual partitioning
        occ_mask = mf.mo_occ > 1e-8
        C_occ = C[:, occ_mask]
        C_vir = C[:, ~occ_mask]

        C_occ_loc = _localize_block(mol, C_occ, method)
        if loc_virtual and C_vir.shape[1] > 0:
            C_vir_loc = vvo.livvo(mol, C_occ_loc, C_vir)
        else:
            C_vir_loc = C_vir

        return np.hstack([C_occ_loc, C_vir_loc])

    # Active space: core|active|external partitioning
    start, stop = active_window
    if not (0 <= start <= stop <= nmo):
        raise ValueError(f"Invalid active space: {active_window}")

    C_core = C[:, :start]
    C_act = C[:, start:stop] 
    C_ext = C[:, stop:]

    C_act_loc = _localize_block(mol, C_act, method)
    if loc_virtual and C_ext.shape[1] > 0:
        C_ext_loc = vvo.livvo(mol, C_act_loc, C_ext)
    else:
        C_ext_loc = C_ext

    return np.hstack([C_core, C_act_loc, C_ext_loc])


def build_casci(
    mf: scf.hf.SCF,
    C: np.ndarray,
    ncas: int,
    nelecas: Tuple[int, int],
    ncore: int,
) -> mcscf.casci.CASCI:
    """Construct CASCI solver with fixed orbital basis."""
    mc = mcscf.CASCI(mf, ncas, nelecas, ncore=ncore)
    mc.mo_coeff = C
    return mc


def write_fcidump_full(
    mol: gto.Mole,
    C: np.ndarray,
    path: str,
    tol: float = 1e-12,
) -> None:
    """Export full molecular Hamiltonian to FCIDUMP format."""
    fcidump_mod.from_mo(mol, str(path), C, tol=tol)


def write_fcidump_cas(
    mc: mcscf.casci.CASCI,
    path: str,
    tol: float = 1e-12,
) -> None:
    """Export active space Hamiltonian to FCIDUMP format."""
    fcidump_mod.from_mcscf(mc, str(path), tol=tol)


def run_benchmarks(
    mol: gto.Mole,
    mf: scf.hf.SCF,
    *,
    kinds: Set[str],
    cas_mc: Optional[mcscf.casci.CASCI] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute benchmark energies for method validation.
    
    Supported methods:
      - HF: Hartree-Fock
      - MP2: Second-order perturbation theory  
      - CISD: Configuration interaction
      - CCSD: Coupled cluster
      - CCSD(T): Perturbative triples correction
      - FCI: Full configuration interaction
      - CASCI: Complete active space CI
    
    Args:
        mol: Molecular system
        mf: Mean-field reference
        kinds: Set of methods to compute
        cas_mc: CASCI object for active space calculations
    
    Returns:
        Dictionary of computed energies
    """
    methods = {k.lower() for k in kinds}
    results: Dict[str, Optional[float]] = {}

    if "hf" in methods:
        results["hf"] = float(mf.e_tot)

    if "mp2" in methods:
        try:
            results["mp2"] = float(mp.MP2(mf).run(verbose=0).e_tot)
        except Exception:
            results["mp2"] = None

    if "cisd" in methods:
        try:
            results["cisd"] = float(ci.CISD(mf).run(verbose=0).e_tot)
        except Exception:
            results["cisd"] = None

    mycc = None
    if "ccsd" in methods:
        try:
            mycc = cc.CCSD(mf).run(verbose=0)
            results["ccsd"] = float(mycc.e_tot)
        except Exception:
            results["ccsd"] = None

    if "ccsd_t" in methods and mycc:
        try:
            e_t = mycc.ccsd_t()
            results["ccsd_t"] = float(mycc.e_tot + e_t)
        except Exception:
            results["ccsd_t"] = None

    if "fci" in methods:
        try:
            cisolver = fci.FCI(mf, singlet=False)
            if mol.spin == 0:
                from pyscf import fci as fci_mod
                cisolver = fci_mod.addons.fix_spin(cisolver, ss=0, shift=0.5)
            e_fci, _ = cisolver.kernel()
            results["fci"] = float(e_fci)
        except Exception:
            results["fci"] = None

    if "casci" in methods and cas_mc:
        try:
            out = cas_mc.kernel()
            results["casci"] = float(out[0] if isinstance(out, (tuple, list)) else out)
        except Exception:
            results["casci"] = None

    return results


__all__ = [
    "ActiveSpaceSpec",
    "run_scf", 
    "compute_natural_orbitals",
    "localize_orbitals",
    "build_casci",
    "write_fcidump_full",
    "write_fcidump_cas", 
    "run_benchmarks"
]
