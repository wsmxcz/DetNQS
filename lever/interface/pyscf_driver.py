# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PySCF backend driver for quantum chemistry calculations.

Provides pure functional interface isolating all PySCF dependencies.
Supports SCF, post-HF correlation methods, natural/localized orbitals,
and FCIDUMP generation.

File: lever/interface/pyscf_driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
from pyscf import cc, ci, fci, gto, lo, mcscf, mp, scf
from pyscf.tools import fcidump

from .metadata import BenchmarkItem, MoleculeInfo, SCFConfig


# ============================================================================
# SCF Calculation
# ============================================================================

@dataclass
class ScfResult:
    """SCF results with wavefunction stability analysis."""
    mol: gto.Mole
    mf: scf.hf.SCF
    converged: bool
    stable_int: Optional[bool]
    stable_ext: Optional[bool]
    s2: float
    n_cycle: int


def run_scf(
    info: MoleculeInfo,
    cfg: SCFConfig,
    log_path: str | Path | None = None,
) -> ScfResult:
    """
    Execute SCF with configurable stability checks.
    
    Method selection: RHF for closed-shell (spin=0), ROHF otherwise.
    Stability analysis checks internal/external wavefunction stability
    via orbital rotation Hessian eigenvalues.
    
    Args:
        info: Molecular geometry and basis set
        cfg: Convergence thresholds and stability settings
        log_path: Optional PySCF verbose output file
    
    Returns:
        SCF results with stability flags
    """
    mol = gto.Mole()
    mol.atom = info.atom
    mol.basis = info.basis
    mol.charge = info.charge
    mol.spin = info.spin
    mol.symmetry = info.symmetry
    mol.unit = info.unit
    
    if log_path is not None:
        mol.output = str(log_path)
    
    mol.build()
    
    # Method dispatch
    if cfg.type == "auto":
        method = scf.RHF if mol.spin == 0 else scf.ROHF
    else:
        method = getattr(scf, cfg.type.upper())
    
    mf = method(mol)
    mf.conv_tol = cfg.tol
    mf.conv_tol_grad = cfg.grad_tol
    mf.max_cycle = cfg.max_cycle
    mf.kernel()
    
    # Extract iteration count
    n_cycle = getattr(mf, "cycles", getattr(mf, "cycle", -1))
    
    # Stability analysis: check orbital rotation Hessian
    s_int, s_ext = None, None
    if cfg.stability != "none" and mf.converged:
        check_int = cfg.stability in ("internal", "both")
        check_ext = cfg.stability in ("external", "both")
        
        try:
            _, _, stable_int, stable_ext = mf.stability(
                internal=check_int,
                external=check_ext,
                verbose=0,
                return_status=True,
            )
            s_int = stable_int if check_int else None
            s_ext = stable_ext if check_ext else None
        except Exception:
            pass  # Best-effort analysis
    
    s2, _ = mf.spin_square()
    
    return ScfResult(mol, mf, mf.converged, s_int, s_ext, s2, n_cycle)


# ============================================================================
# Post-HF Benchmarks
# ============================================================================

def run_benchmarks(
    res: ScfResult,
    methods: Set[str],
) -> Dict[str, BenchmarkItem]:
    """
    Run correlation methods for validation.
    
    Supported: HF, MP2, CISD, CCSD, CCSD(T), FCI.
    All methods run with suppressed verbose output.
    
    Args:
        res: SCF reference calculation
        methods: Set of correlation methods to execute
    
    Returns:
        Dictionary of benchmark results
    """
    out: Dict[str, BenchmarkItem] = {}
    mf = res.mf
    
    old_verbose = mf.verbose
    mf.verbose = 0
    
    try:
        def _run(key: str, fn: callable) -> None:
            if key not in methods:
                return
            try:
                e = fn()
                out[key] = BenchmarkItem(
                    energy=float(e), converged=True, status="ok"
                )
            except Exception as err:
                out[key] = BenchmarkItem(status="failed", comment=str(err))
        
        # Reference energy
        if "hf" in methods:
            out["hf"] = BenchmarkItem(
                energy=float(mf.e_tot),
                converged=res.converged,
                status="ok" if res.converged else "failed",
            )
        
        # Correlation methods
        _run("mp2", lambda: mp.MP2(mf).run().e_tot)
        _run("cisd", lambda: ci.CISD(mf).run().e_tot)
        
        # CCSD with optional (T) correction
        mycc = None
        if "ccsd" in methods or "ccsd_t" in methods:
            try:
                mycc = cc.CCSD(mf).run()
                out["ccsd"] = BenchmarkItem(
                    energy=float(mycc.e_tot),
                    converged=mycc.converged,
                    status="ok" if mycc.converged else "failed",
                )
            except Exception as e:
                out["ccsd"] = BenchmarkItem(status="failed", comment=str(e))
        
        ccsd_item = out.get("ccsd")
        if "ccsd_t" in methods and mycc and ccsd_item and ccsd_item.status == "ok":
            _run("ccsd_t", lambda: mycc.e_tot + mycc.ccsd_t())
        
        # Full CI reference
        if "fci" in methods:
            def _exec_fci():
                solver = fci.FCI(mf)
                if res.mol.spin == 0:
                    solver = fci.addons.fix_spin(solver, ss=0)
                e, _ = solver.kernel()
                return e
            
            _run("fci", _exec_fci)
    
    finally:
        mf.verbose = old_verbose
    
    return out


# ============================================================================
# Natural Orbitals
# ============================================================================

def make_natural(
    res: ScfResult,
    kind: str,
    hf_occ: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute natural orbitals from correlated 1-RDM.
    
    Natural orbitals diagonalize the one-particle density matrix:
        ρ = Σᵢ nᵢ |φᵢ⟩⟨φᵢ|
    where nᵢ are natural occupation numbers (0 ≤ nᵢ ≤ 2).
    
    Args:
        res: SCF reference
        kind: Correlation method ('mp2', 'cisd', 'ccsd', 'none')
        hf_occ: HF occupation numbers for permutation tracking
    
    Returns:
        (Natural orbitals, NO occupations, permuted HF occupations)
    """
    if kind == "none":
        return res.mf.mo_coeff, res.mf.mo_occ, hf_occ
    
    mf = res.mf
    old_verbose = mf.verbose
    mf.verbose = 0
    
    try:
        # Compute correlated 1-RDM
        if kind == "mp2":
            obj = mp.MP2(mf).run()
        elif kind == "cisd":
            obj = ci.CISD(mf).run()
        elif kind == "ccsd":
            obj = cc.CCSD(mf).run()
            obj.solve_lambda()
        else:
            raise ValueError(f"Unknown NO type: {kind}")
        
        dm = obj.make_rdm1()
        nmo = mf.mo_coeff.shape[1]
        
        # Transform to MO basis if needed
        if dm.shape != (nmo, nmo):
            s = mf.get_ovlp()
            dm = mf.mo_coeff.T @ s @ dm @ s @ mf.mo_coeff
        
        # Diagonalize: ρ = U · diag(n) · U†
        occ, u = np.linalg.eigh(dm)
        idx = np.argsort(occ)[::-1]  # Sort by occupation (descending)
        
        return mf.mo_coeff @ u[:, idx], occ[idx], hf_occ[idx]
    
    finally:
        mf.verbose = old_verbose


# ============================================================================
# Active Space Selection
# ============================================================================

def make_active(
    mo_coeff: np.ndarray,
    mo_occ: np.ndarray,
    hf_occ: np.ndarray,
    n_elec: int,
    n_orb: int,
    total_elec: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Partition orbitals into [Core | Active | Virtual] subspaces.
    
    Orbitals are sorted by occupation (descending) before partitioning.
    Core orbitals are assumed doubly occupied.
    
    Args:
        mo_coeff: Molecular orbital coefficients
        mo_occ: Current occupation numbers
        hf_occ: HF occupation numbers
        n_elec: Active space electrons
        n_orb: Active space orbitals
        total_elec: Total system electrons
    
    Returns:
        (Sorted MOs, sorted occupations, sorted HF occupations,
         (n_core, n_active, n_virtual))
    """
    if n_orb <= 0:
        return mo_coeff, mo_occ, hf_occ, (0, mo_coeff.shape[1], 0)
    
    # Validate active space
    if n_elec > 2 * n_orb:
        raise ValueError(
            f"Active space impossible: {n_elec} electrons in {n_orb} orbitals"
        )
    
    n_core_elec = total_elec - n_elec
    if n_core_elec < 0 or n_core_elec % 2 != 0:
        raise ValueError(
            f"Invalid core electrons: total={total_elec}, active={n_elec}"
        )
    
    n_core = n_core_elec // 2
    n_mo = mo_coeff.shape[1]
    n_virt = n_mo - n_core - n_orb
    
    if n_virt < 0:
        raise ValueError(f"Active space exceeds basis size: n_mo={n_mo}")
    
    # Sort by occupation for CAS selection
    idx = np.argsort(mo_occ)[::-1]
    
    return (
        mo_coeff[:, idx],
        mo_occ[idx],
        hf_occ[idx],
        (n_core, n_orb, n_virt),
    )


# ============================================================================
# Orbital Localization
# ============================================================================

def localize_orbs(
    mol: gto.Mole,
    mo_coeff: np.ndarray,
    method: str,
    window: str,
    window_slices: Tuple[slice, slice, slice],
    loc_virtual: bool,
) -> np.ndarray:
    """
    Localize orbitals via unitary transformation.
    
    Methods:
      - IBO: Intrinsic Bonding Orbitals (Knizia, 2013)
      - Boys: Maximize Σᵢ ⟨φᵢ|r²|φᵢ⟩
      - Pipek-Mezey: Maximize Mulliken population localization
      - Edmiston-Ruedenberg: Maximize Coulomb self-repulsion
    
    Args:
        mol: Molecular structure
        mo_coeff: Orbital coefficients [n_ao, n_mo]
        method: Localization method ('ibo', 'boys', 'pipek', 'edmiston', 'none')
        window: Localization window ('full' or 'active')
        window_slices: (core_slice, active_slice, virtual_slice)
        loc_virtual: Whether to localize virtual orbitals
    
    Returns:
        Localized orbital coefficients
    """
    if method == "none":
        return mo_coeff
    
    c_new = mo_coeff.copy()
    s_core, s_act, s_virt = window_slices
    
    def _do_loc(c_sub: np.ndarray) -> np.ndarray:
        if c_sub.shape[1] == 0:
            return c_sub
        
        if method == "ibo":
            return lo.ibo.ibo(mol, c_sub)
        elif method == "boys":
            return lo.boys.Boys(mol, c_sub).kernel()
        elif method == "pipek":
            return lo.pipek.PipekMezey(mol, c_sub).kernel()
        elif method == "edmiston":
            return lo.edmiston.ER(mol, c_sub).kernel()
        return c_sub
    
    # Apply localization to specified windows
    if window == "full":
        c_new[:, s_core] = _do_loc(c_new[:, s_core])
        c_new[:, s_act] = _do_loc(c_new[:, s_act])
    else:  # 'active'
        c_new[:, s_act] = _do_loc(c_new[:, s_act])
    
    if loc_virtual:
        c_new[:, s_virt] = _do_loc(c_new[:, s_virt])
    
    return c_new


# ============================================================================
# FCIDUMP Export
# ============================================================================

def write_fcidump(
    mol: gto.Mole,
    mf: scf.hf.SCF,
    mo_coeff: np.ndarray,
    path: str,
    n_core: int,
    n_act: int,
) -> None:
    """
    Export FCIDUMP file with frozen core support.
    
    Uses CASCI logic to compute active space integrals:
        h_eff = h_core + Σ_core [2(ij|ij) - (ij|ji)]
    
    Args:
        mol: Molecular structure
        mf: Mean-field object
        mo_coeff: Orbital coefficients
        path: Output file path
        n_core: Number of frozen core orbitals
        n_act: Number of active orbitals
    """
    if n_core > 0 or n_act < mo_coeff.shape[1]:
        n_act_elec = mol.nelectron - 2 * n_core
        cas = mcscf.CASCI(mf, n_act, n_act_elec)
        cas.mo_coeff = mo_coeff
        cas.ncore = n_core
        fcidump.from_mcscf(cas, path)
    else:
        fcidump.from_mo(mol, path, mo_coeff)


# ============================================================================
# Utility Functions
# ============================================================================

def occ_to_bitstring(occ: np.ndarray) -> Tuple[int, int]:
    """
    Convert occupation vector to α/β bitstrings.
    
    Occupation thresholds:
      - α-orbital: n > 0.5
      - β-orbital: n > 1.5
    
    Args:
        occ: Occupation number vector [n_orb]
    
    Returns:
        (alpha_bitstring, beta_bitstring)
    
    Example:
        occ = [2.0, 1.0, 0.0] → α=0b011, β=0b001
    """
    alpha = 0
    beta = 0
    for i, n in enumerate(occ):
        if n > 0.5:
            alpha |= (1 << i)
        if n > 1.5:
            beta |= (1 << i)
    return int(alpha), int(beta)


__all__ = [
    "ScfResult",
    "run_scf",
    "run_benchmarks",
    "make_natural",
    "make_active",
    "localize_orbs",
    "write_fcidump",
    "occ_to_bitstring",
]
