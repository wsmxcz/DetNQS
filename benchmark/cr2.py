# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
FCIDUMP generator for Cr2 benchmark (Ahlrichs' SV basis).

Generates two FCIDUMP files reproducing Li et al., PRR 2, 012015 (2020):
  - (48e, 42o): all-electron space
  - (24e, 30o): frozen Mg-core space (12 spatial orbitals frozen)

Reference: Li et al., Phys. Rev. Research 2, 012015 (2020)
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2026
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf, mcscf, symm, lib, fci
from pyscf.tools import fcidump


def load_basis() -> dict:
    """
    Load Ahlrichs' SV basis (BSE: 'Ahlrichs VDZ').
  
    Note: In Basis Set Exchange, this is labeled 'Ahlrichs VDZ'.
    """
    try:
        from pyscf.gto.basis import bse
        return bse.get_basis("Ahlrichs VDZ", elements="Cr")
    except Exception as e:
        raise RuntimeError("Failed to load basis from BSE.") from e


def tag_orb_sym(mol: gto.Mole, mf: scf.hf.SCF, mo: np.ndarray) -> np.ndarray:
    """Attach D2h symmetry labels to orbitals."""
    s = mf.get_ovlp()
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo, s=s, check=True)
    return lib.tag_array(mo, orbsym=np.asarray(orbsym, dtype=int))


def build_cas_orbs(
    mol: gto.Mole, mf: scf.RHF
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Run CASSCF(12e, 12o) with D2h symmetry for Cr2 valence space.
    
    D2h irrep distribution for Cr2 sextuple bond (6g + 6u):
      Ag : 3 (sigma_s, sigma_d, delta_x2-y2)
      B1u: 3 (sigma_s*, sigma_d*, delta_x2-y2*)
      B3u: 1 (pi_x *)
      B2u: 1 (pi_y *)
      B2g: 1 (pi_x)
      B3g: 1 (pi_y)
      B1g: 1 (delta_xy)
      Au : 1 (delta_xy*)
    """
    # mapping for Cr2 Ground State
    cas_irrep = {
        "Ag": 3, "B1u": 3,
        "B2g": 1, "B3g": 1,
        "B2u": 1, "B3u": 1,
        "B1g": 1, "Au": 1,
    }
    cas_list = [cas_irrep.get(name, 0) for name in mol.irrep_name]

    mc = mcscf.CASSCF(mf, ncas=12, nelecas=(6, 6))
    mc.conv_tol = 1e-8
    mc.fcisolver = fci.direct_spin0_symm.FCI(mol)
    mc.fcisolver.wfnsym = symm.irrep_name2id(mol.groupname, "Ag")

    # Disable implicit canonicalization for reproducibility
    mc.natorb = False
    mc.canonicalization = False
    mc.sorting_mo_energy = False

    mo0 = mc.sort_mo_by_irrep(cas_list, mo_coeff=mf.mo_coeff)
    mc.kernel(mo0)

    # Explicit canonicalization: core/virtual via Fock, active via NO
    mo_tag = tag_orb_sym(mol, mf, mc.mo_coeff)
    mo, ci, eps = mc.canonicalize(
        mo_coeff=mo_tag, ci=mc.ci, sort=False, cas_natorb=True
    )
    mo = tag_orb_sym(mol, mf, mo)

    n_core = int(mc.ncore)
    n_cas = int(mc.ncas)
  
    # Sanity check: expected 18 core + 12 active + 12 virtual = 42 total
    assert mo.shape[1] == 42, f"Expected 42 orbitals, got {mo.shape[1]}"
    assert n_core == 18, f"Expected 18 core orbitals, got {n_core}"
    assert n_cas == 12, f"Expected 12 active orbitals, got {n_cas}"

    return mo, ci, np.asarray(eps), n_core, n_cas


def build_frozen_mo(
    mo: np.ndarray, eps: np.ndarray, n_core: int, n_cas: int, n_freeze: int = 12
) -> np.ndarray:
    """
    Build MO for frozen Mg-core (24e, 30o).
  
    Freezes the 12 lowest-energy core orbitals (Mg: 1s/2s/2p/3s).
    Reordered layout: [frozen_core | active_core | active | virtual].
  
    Args:
      mo: full MO matrix (n_ao, n_mo)
      eps: orbital energies
      n_core: inactive core count
      n_cas: active orbital count
      n_freeze: number of core orbitals to freeze (default 12)
  
    Returns:
      mo_reorder: reordered MO matrix
    """
    n_mo = mo.shape[1]
    assert n_freeze <= n_core, f"Cannot freeze {n_freeze} from {n_core} core orbitals"

    core_idx = np.arange(n_core)
    eps_core = eps[core_idx]

    # Stable sort to handle near-degenerate energies consistently
    order = np.argsort(eps_core, kind="mergesort")
    idx_frozen = core_idx[order[:n_freeze]]
    idx_active_core = core_idx[order[n_freeze:]]

    idx_active = np.arange(n_core, n_core + n_cas)
    idx_virtual = np.arange(n_core + n_cas, n_mo)

    # Reorder: frozen | active_core | active | virtual
    mo_reorder = np.hstack([
        mo[:, idx_frozen],
        mo[:, idx_active_core],
        mo[:, idx_active],
        mo[:, idx_virtual]
    ])
  
    return mo_reorder


def make_fcidumps(
    bond_len: float = 1.5,
    out_24e: str = "Cr2_24e30o.FCIDUMP",
    out_48e: str = "Cr2_48e42o.FCIDUMP",
) -> None:
    """
    Generate FCIDUMP files for Cr2 at r=bond_len (Angstrom).
  
    Args:
      bond_len: Cr-Cr bond length in Angstrom
      out_24e: output path for (24e, 30o) frozen-core FCIDUMP
      out_48e: output path for (48e, 42o) all-electron FCIDUMP
    """
    # Build molecule with D2h symmetry
    mol = gto.Mole()
    mol.unit = "Angstrom"
    mol.atom = [("Cr", (0, 0, -bond_len / 2)), ("Cr", (0, 0, bond_len / 2))]
    mol.charge = 0
    mol.spin = 0
    mol.symmetry = True
    mol.symmetry_subgroup = "D2h"
    mol.basis = load_basis()
    mol.build()

    # Run RHF
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()

    # Build CAS(12e, 12o) orbitals with active NO
    mo_cas, ci, eps, n_core, n_cas = build_cas_orbs(mol, mf)

    # === (48e, 42o): all-electron space ===
    mc_48 = mcscf.CASCI(mf, ncas=42, nelecas=(24, 24))
    mc_48.mo_coeff = tag_orb_sym(mol, mf, mo_cas)
    fcidump.from_mcscf(mc_48, out_48e, molpro_orbsym=True)

    # === (24e, 30o): frozen Mg-core space ===
    mo_24 = build_frozen_mo(mo_cas, eps, n_core, n_cas, n_freeze=12)
    mc_24 = mcscf.CASCI(mf, ncas=30, nelecas=(12, 12))
    mc_24.ncore = 12
    mc_24.mo_coeff = tag_orb_sym(mol, mf, mo_24)
    fcidump.from_mcscf(mc_24, out_24e, molpro_orbsym=True)

    # Print verification info
    print(f"RHF energy: {mf.e_tot:.10f}")
    print(f"CAS(12e,12o) frozen cores: {n_core}")
    print(f"Generated: {out_48e} (NORB=42, NELEC=48)")
    print(f"Generated: {out_24e} (NORB=30, NELEC=24)")
    print(f"\nFrozen core orbital energies (12 lowest):")
    frozen_eps = eps[:n_core][np.argsort(eps[:n_core], kind="mergesort")[:12]]
    for i, e in enumerate(frozen_eps, 1):
        print(f"  {i:2d}: {e:12.6f}")


if __name__ == "__main__":
    make_fcidumps(bond_len=1.5)