# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Generate FCIDUMP files with orbital localization and natural orbital transforms.

Supports multiple molecular geometries, orbital localization (Boys, Pipek-Mezey,
Edmiston-Ruedenberg), natural orbitals (MP2, CISD, CCSD), optional LIVVO for
virtual orbitals, and benchmark calculations (MP2, CISD, CCSD, CCSD(T), FCI).

File: tools/FCIDUMP.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025

Examples:
  python FCIDUMP.py h2 --basis sto-3g --output H2.FCIDUMP
  python FCIDUMP.py h2o --basis cc-pvdz --units bohr --output H2O.FCIDUMP
  python FCIDUMP.py hchain --n_atoms 10 --distance 2.0 --basis sto-3g \\
      --localize boys --output H10_2.00_boys.FCIDUMP
  python FCIDUMP.py n2 --basis cc-pvtz --localize no-mp2 \\
      --output N2.FCIDUMP --benchmark
"""

import argparse
import math
import sys
from datetime import datetime

import numpy as np
from pyscf import ao2mo, cc, ci, fci, gto, lo, mp, scf
from pyscf.cc import ccsd_rdm
from pyscf.lo import edmiston, iao, orth, vvo
from pyscf.tools import fcidump


# ============================================================================
# Geometry Utilities
# ============================================================================

def _round_coords(geom: list, decimals: int = 4) -> list:
    """Round coordinates to specified decimals."""
    return [(atom, tuple(round(x, decimals) for x in xyz)) for atom, xyz in geom]


def _rotate_z(xyz: tuple, deg: float) -> tuple:
    """Rotate (x,y,z) around z-axis by degrees."""
    x, y, z = xyz
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (x * c - y * s, x * s + y * c, z)


def _diatomic(a1: str, a2: str, d: float) -> list:
    """Diatomic molecule along z-axis."""
    return [(a1, (0.0, 0.0, 0.0)), (a2, (0.0, 0.0, d))]


# ============================================================================
# Molecular Geometries (in Angstrom by default)
# ============================================================================

def geom_h_chain(n: int, d: float = 1.4) -> list:
    """Linear H_n chain along z-axis."""
    return [("H", (0.0, 0.0, i * d)) for i in range(n)]


def geom_h_ring(n: int, r: float) -> list:
    """Planar H_n ring in xy-plane."""
    theta = 2 * math.pi / n
    return [("H", (r * math.cos(i * theta), r * math.sin(i * theta), 0.0))
            for i in range(n)]


def geom_h2o(oh: float = 1.84345) -> list:
    # """H_2O with HOH angle ~ 104.5 deg."""
    # ang = math.radians(110.6)
    # x = oh * math.sin(ang / 2)
    # y = oh * math.cos(ang / 2)
    # return [("O", (0.0, 0.0, 0.0)), ("H", (x, y, 0.0)), ("H", (-x, y, 0.0))]
    # return [("O", (0.0, 0.0, -0.0090)), ("H", (0.0, 1.515263, -1.058898)), ("H", (0.0, -1.515263, -1.058898))]
    # return [("O", (0.0, 0.0, -0.0135)), ("H", (0.0, 2.2728945, -1.588347)), ("H", (0.0, -2.2728945, -1.588347))]
    # return [("O", (0.0, 0.0, -0.0180)), ("H", (0.0, 3.030526, -2.117796)), ("H", (0.0, -3.030526, -2.117796))]
    # return [("O", (0.0, 0.0, -0.0225)), ("H", (0.0, 3.7881575, -2.647245)), ("H", (0.0, -3.7881575, -2.647245))]
    return [("O", (0.0, 0.0, -0.0270)), ("H", (0.0, 4.545789, -3.176694)), ("H", (0.0, -4.545789, -3.176694))]


def geom_nh3(nh: float = 1.012) -> list:
    """NH_3 pyramid."""
    r_xy = nh / 1.08265
    h = 0.4147 * r_xy
    geom = [("N", (0.0, 0.0, 0.0))]
    for ang in [0, 120, 240]:
        rad = math.radians(ang)
        geom.append(("H", (r_xy * math.cos(rad), r_xy * math.sin(rad), -h)))
    return geom


def geom_ch4(ch: float = 1.09) -> list:
    """Tetrahedral CH_4."""
    dirs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    geom = [("C", (0.0, 0.0, 0.0))]
    for v in dirs:
        norm = math.sqrt(sum(x**2 for x in v))
        geom.append(("H", tuple(ch * x / norm for x in v)))
    return geom


def geom_co2(co: float = 1.16) -> list:
    """Linear CO_2."""
    return [("C", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, co)), ("O", (0.0, 0.0, -co))]


def geom_o3() -> list:
    """Ozone."""
    return [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 1.0885, 0.6697)),
            ("O", (0.0, -1.0885, 0.6697))]


def geom_li2o() -> list:
    """Li_2O."""
    return [("Li", (3.732, 0.25, 0.0)), ("Li", (2.0, 0.25, 0.0)),
            ("O", (2.866, -0.25, 0.0))]


def geom_c2h4(cc: float = 1.339, ang: float = 0.0) -> list:
    """C_2H_4 with CC distance and rotation angle (deg) around CC bond."""
    scale = cc / 1.339
    c1 = (0.0, 0.0, 0.6695 * scale)
    c2 = (0.0, 0.0, -0.6695 * scale)
    h3 = (0.0, 0.9289 * scale, 1.2321 * scale)
    h4 = (0.0, -0.9289 * scale, 1.2321 * scale)
    h5 = (0.0, 0.9289 * scale, -1.2321 * scale)
    h6 = (0.0, -0.9289 * scale, -1.2321 * scale)
    return [("C", c1), ("C", _rotate_z(c2, ang)),
            ("H", h3), ("H", h4),
            ("H", _rotate_z(h5, ang)), ("H", _rotate_z(h6, ang))]


def geom_c2h6(cc: float = 1.536, ang: float = 0.0) -> list:
    """C_2H_6 with CC distance and rotation angle (deg) around CC bond."""
    scale = cc / 1.536
    c1 = (0.0, 0.0, 0.7680 * scale)
    c2 = (0.0, 0.0, -0.7680 * scale)
    h3 = (-1.0192 * scale, 0.0, 1.1573 * scale)
    h4 = (0.5096 * scale, 0.8826 * scale, 1.1573 * scale)
    h5 = (0.5096 * scale, -0.8826 * scale, 1.1573 * scale)
    h6 = (1.0192 * scale, 0.0, -1.1573 * scale)
    h7 = (-0.5096 * scale, -0.8826 * scale, -1.1573 * scale)
    h8 = (-0.5096 * scale, 0.8826 * scale, -1.1573 * scale)
    return [("C", c1), ("C", _rotate_z(c2, ang)),
            ("H", h3), ("H", h4), ("H", h5),
            ("H", _rotate_z(h6, ang)), ("H", _rotate_z(h7, ang)),
            ("H", _rotate_z(h8, ang))]


def geom_c3h8() -> list:
    """Propane (C_3H_8)."""
    return [
        ("C", (0.0000, 0.0000, 0.5862)),
        ("C", (0.0000, 1.2745, -0.2600)),
        ("C", (0.0000, -1.2745, -0.2600)),
        ("H", (0.8717, 0.0000, 1.2403)),
        ("H", (-0.8717, 0.0000, 1.2403)),
        ("H", (0.0000, 2.1655, 0.3650)),
        ("H", (0.0000, -2.1655, 0.3650)),
        ("H", (0.8788, 1.3196, -0.9019)),
        ("H", (-0.8788, 1.3196, -0.9019)),
        ("H", (-0.8788, -1.3196, -0.9019)),
        ("H", (0.8788, -1.3196, -0.9019)),
    ]


def geom_c2h4o() -> list:
    """Ethylene oxide (C_2H_4O)."""
    return [
        ("O", (-0.0007, 0.8141, 0.0)),
        ("C", (0.7509, -0.4065, 0.0)),
        ("C", (-0.7502, -0.4076, 0.0)),
        ("H", (1.2625, -0.6786, 0.9136)),
        ("H", (1.2625, -0.6787, -0.9136)),
        ("H", (-1.2614, -0.6806, -0.9136)),
        ("H", (-1.2614, -0.6805, 0.9136)),
    ]


def geom_c2h4o2() -> list:
    """1,2-Dioxetane (C_2H_4O_2)."""
    return [
        ("C", (-0.9702, 0.0000, 0.0000)),
        ("C", (0.9702, 0.0000, 0.0000)),
        ("O", (0.0000, 1.0129, 0.0000)),
        ("O", (0.0000, -1.0129, 0.0000)),
        ("H", (-1.5867, 0.0000, 0.8958)),
        ("H", (1.5867, 0.0000, 0.8958)),
        ("H", (-1.5867, 0.0000, -0.8958)),
        ("H", (1.5867, 0.0000, -0.8958)),
    ]


# ============================================================================
# Natural Orbital Transforms
# ============================================================================

def _sorted_eigh(dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize 1-RDM and return sorted (occ, U) in descending order."""
    dm_sym = (dm + dm.T.conj()) * 0.5
    occ, U = np.linalg.eigh(dm_sym)
    idx = np.argsort(-occ)
    return occ[idx], U[:, idx]


def get_mp2_no(mf: scf.hf.RHF) -> tuple[np.ndarray, np.ndarray]:
    """Compute MP2 natural orbitals via 1-RDM diagonalization."""
    m2 = mp.MP2(mf).run()
    if hasattr(m2, "make_natorbs"):
        occ, U = m2.make_natorbs()
        return mf.mo_coeff @ U, occ
    dm_mo = m2.make_rdm1(ao_repr=False)
    occ, U = _sorted_eigh(dm_mo)
    return mf.mo_coeff @ U, occ


def get_cisd_no(mf: scf.hf.RHF) -> tuple[np.ndarray, np.ndarray]:
    """Compute CISD natural orbitals via 1-RDM diagonalization."""
    myci = ci.CISD(mf).run()
    dm_mo = myci.make_rdm1()
    occ, U = _sorted_eigh(dm_mo)
    return mf.mo_coeff @ U, occ


def get_ccsd_no(mf: scf.hf.RHF) -> tuple[np.ndarray, np.ndarray]:
    """Compute CCSD natural orbitals via lambda-relaxed 1-RDM."""
    mycc = cc.CCSD(mf).run()
    mycc.solve_lambda()
    dm_mo = ccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2,
                               mycc.l1, mycc.l2, ao_repr=False)
    occ, U = _sorted_eigh(dm_mo)
    return mf.mo_coeff @ U, occ


# ============================================================================
# Benchmark Calculations
# ============================================================================

def run_benchmarks(mol: gto.Mole, mf: scf.hf.SCF, log) -> dict:
    """Run MP2, CISD, CCSD, CCSD(T), FCI benchmarks."""
    results = {}
    log.write("\n" + "=" * 60 + "\n")
    log.write("BENCHMARK CALCULATIONS\n")
    log.write("=" * 60 + "\n\n")

    # MP2
    try:
        log.write("Running MP2...\n")
        log.flush()
        m2 = mp.MP2(mf).run(verbose=0)
        results["mp2"] = m2.e_tot
        log.write(f"  E(MP2)  = {m2.e_tot:.12f} Ha\n")
        log.write(f"  Ecorr   = {m2.e_corr:.12f} Ha\n\n")
    except Exception as e:
        log.write(f"  MP2 failed: {e}\n\n")
        results["mp2"] = None

    # CISD
    try:
        log.write("Running CISD...\n")
        log.flush()
        myci = ci.CISD(mf).run(verbose=0)
        results["cisd"] = myci.e_tot
        log.write(f"  E(CISD) = {myci.e_tot:.12f} Ha\n")
        log.write(f"  Ecorr   = {myci.e_corr:.12f} Ha\n\n")
    except Exception as e:
        log.write(f"  CISD failed: {e}\n\n")
        results["cisd"] = None

    # CCSD
    mycc = None
    try:
        log.write("Running CCSD...\n")
        log.flush()
        mycc = cc.CCSD(mf).run(verbose=0)
        results["ccsd"] = mycc.e_tot
        log.write(f"  E(CCSD) = {mycc.e_tot:.12f} Ha\n")
        log.write(f"  Ecorr   = {mycc.e_corr:.12f} Ha\n\n")
    except Exception as e:
        log.write(f"  CCSD failed: {e}\n\n")
        results["ccsd"] = None

    # CCSD(T)
    if mycc is not None:
        try:
            log.write("Running CCSD(T)...\n")
            log.flush()
            e_t = mycc.ccsd_t()
            results["ccsd_t"] = mycc.e_tot + e_t
            log.write(f"  E(T)       = {e_t:.12f} Ha\n")
            log.write(f"  E(CCSD(T)) = {results['ccsd_t']:.12f} Ha\n")
            log.write(f"  Ecorr      = {results['ccsd_t'] - mf.e_tot:.12f} Ha\n\n")
        except Exception as e:
            log.write(f"  CCSD(T) failed: {e}\n\n")
            results["ccsd_t"] = None
    else:
        log.write("CCSD(T) skipped (CCSD failed)\n\n")
        results["ccsd_t"] = None

    # FCI (only for small systems)
    n_orb = mol.nao_nr()
    if n_orb <= 14:
        try:
            log.write("Running FCI (singlet)...\n")
            cisolver = fci.FCI(mf, singlet=True)
            cisolver = fci.addons.fix_spin(cisolver, ss=0, shift=2.0)
            e_fci, _ = cisolver.kernel()
            results["fci"] = e_fci
            log.write(f"  E(FCI)  = {e_fci:.12f} Ha\n\n")
        except Exception as e:
            log.write(f"  FCI failed: {e}\n\n")
            results["fci"] = None
    else:
        log.write(f"FCI skipped (n_orb={n_orb} > 14)\n\n")
        results["fci"] = None

    log.write("=" * 60 + "\n\n")
    log.flush()
    return results


# ============================================================================
# Orbital Localization
# ============================================================================

def transform_orbitals(mol: gto.Mole, mf: scf.hf.SCF,
                      method: str, loc_vir: bool = False) -> np.ndarray:
    """
    Transform MO coefficients via localization or natural orbitals.

    Args:
        mol: PySCF Mole object
        mf: Converged mean-field object
        method: Transform method (boys/pipek/edmiston/no-mp2/no-cisd/no-ccsd)
        loc_vir: Apply LIVVO to virtual orbitals

    Returns:
        Transformed MO coefficient matrix
    """
    C = mf.mo_coeff
    method = (method or "").lower()

    # Natural orbitals (full space transform)
    if method in ("no-mp2", "mp2_no"):
        C_no, _ = get_mp2_no(mf)
        return C_no
    if method in ("no-cisd", "cisd_no"):
        C_no, _ = get_cisd_no(mf)
        return C_no
    if method in ("no-ccsd", "ccsd_no"):
        C_no, _ = get_ccsd_no(mf)
        return C_no

    # Localization (occupied space only)
    occ_mask = mf.mo_occ > 1e-8
    C_occ, C_vir = C[:, occ_mask], C[:, ~occ_mask]

    def _localize(C_sub: np.ndarray, how: str) -> np.ndarray:
        if how == "boys":
            return lo.boys.Boys(mol, C_sub).kernel()
        elif how == "pipek":
            return lo.pipek.PipekMezey(mol, C_sub).kernel()
        elif how == "edmiston":
            return edmiston.ER(mol, C_sub).kernel()
        else:
            raise ValueError(f"Unknown localization method: {how}")

    if method in ("boys", "pipek", "edmiston"):
        C_occ_loc = _localize(C_occ, method)
        C_vir_loc = vvo.livvo(mol, C_occ_loc, C_vir) if loc_vir else C_vir
        return np.hstack([C_occ_loc, C_vir_loc])

    if method in ("canonical", None, ""):
        return C

    raise ValueError(f"Unknown transform method: {method}")


# ============================================================================
# FCIDUMP Generation
# ============================================================================

def gen_fcidump(geom: list, basis: str = "sto-3g", out: str = "FCIDUMP",
                charge: int = 0, spin: int = 0, unit: str = "angstrom",
                loc: str = None, verbose: bool = False, loc_vir: bool = False,
                benchmark: bool = False) -> None:
    """
    Generate FCIDUMP file from molecular geometry.

    Args:
        geom: Molecular geometry [(atom, (x,y,z)), ...]
        basis: Basis set name
        out: Output FCIDUMP filename
        charge: Molecular charge
        spin: Number of unpaired electrons (2S)
        unit: Coordinate unit ('angstrom' or 'bohr')
        loc: Orbital transform method
        verbose: Verbose PySCF output
        loc_vir: Localize virtual orbitals with LIVVO
        benchmark: Run benchmark calculations
    """
    log_filename = out.replace(".FCIDUMP", "") + ".log"
    if not log_filename.endswith(".log"):
        log_filename = out + ".log"

    with open(log_filename, "w") as log:
        # Header
        log.write("=" * 60 + "\n")
        log.write("FCIDUMP GENERATION LOG\n")
        log.write("=" * 60 + "\n")
        log.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Output: {out}\n\n")

        # Build molecule using PySCF's built-in unit handling
        log.write("=" * 60 + "\n")
        log.write("MOLECULAR GEOMETRY\n")
        log.write("=" * 60 + "\n")
      
        mol = gto.Mole()
        mol.atom = geom
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.unit = unit  # Use PySCF built-in unit handling
        mol.verbose = 4 if verbose else 2
        mol.build()

        log.write(f"Charge: {charge}\n")
        log.write(f"Spin (2S): {spin}\n")
        log.write(f"Basis: {basis}\n")
        log.write(f"Unit: {unit}\n")
        log.write(f"N_electrons: {mol.nelectron}\n")
        log.write(f"N_orbitals: {mol.nao_nr()}\n\n")
        log.write(f"Atomic coordinates ({unit}):\n")
        for atom, coord in geom:
            log.write(f"  {atom:4s}  {coord[0]:12.6f}  {coord[1]:12.6f}  "
                     f"{coord[2]:12.6f}\n")
        log.write("\n")

        # SCF calculation
        log.write("=" * 60 + "\n")
        log.write("SCF CALCULATION\n")
        log.write("=" * 60 + "\n")
        mf = scf.ROHF(mol) if spin > 0 else scf.RHF(mol)
        mf.verbose = 4 if verbose else 0
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-8
        log.write(f"Method: {'ROHF' if spin > 0 else 'RHF'}\n")
        log.write(f"Conv_tol: {mf.conv_tol}\n")
        log.write(f"Conv_tol_grad: {mf.conv_tol_grad}\n\n")
        log.flush()

        mf.run()

        log.write(f"E(SCF) = {mf.e_tot:.12f} Ha\n")
        log.write(f"Converged: {mf.converged}\n\n")

        # Orbital transformation
        if loc:
            log.write("=" * 60 + "\n")
            log.write("ORBITAL TRANSFORMATION\n")
            log.write("=" * 60 + "\n")
            log.write(f"Method: {loc}\n")
            log.write(f"Localize virtuals (LIVVO): {loc_vir}\n\n")
            log.flush()

            mf.mo_coeff = transform_orbitals(mol, mf, loc, loc_vir)
            log.write("Transformation completed\n\n")

        # Benchmark calculations
        if benchmark:
            bench_results = run_benchmarks(mol, mf, log)

            log.write("=" * 60 + "\n")
            log.write("ENERGY SUMMARY\n")
            log.write("=" * 60 + "\n")
            log.write(f"{'Method':<12s} {'Energy (Ha)':<20s} "
                     f"{'Corr Energy (Ha)':<20s}\n")
            log.write("-" * 60 + "\n")
            log.write(f"{'HF':<12s} {mf.e_tot:< 20.12f} {'---':<20s}\n")

            for method in ["mp2", "cisd", "ccsd", "ccsd_t", "fci"]:
                method_name = "CCSD(T)" if method == "ccsd_t" else method.upper()
                if bench_results.get(method) is not None:
                    e_tot = bench_results[method]
                    e_corr = e_tot - mf.e_tot
                    log.write(f"{method_name:<12s} {e_tot:< 20.12f} "
                             f"{e_corr:< 20.12f}\n")
                else:
                    log.write(f"{method_name:<12s} {'N/A':<20s} {'N/A':<20s}\n")
            log.write("=" * 60 + "\n\n")

        # Write FCIDUMP
        log.write("=" * 60 + "\n")
        log.write("FCIDUMP GENERATION\n")
        log.write("=" * 60 + "\n")
        log.write(f"Writing to: {out}\n")
        log.write(f"Tolerance: 1e-15\n\n")
        log.flush()

        fcidump.from_scf(mf, out, tol=1e-15)

        log.write("FCIDUMP file generated successfully\n\n")
        log.write("=" * 60 + "\n")
        log.write("END OF LOG\n")
        log.write("=" * 60 + "\n")

    # Console output
    print(f"\nGenerated: {out}")
    print(f"Log file: {log_filename}")
    print(f"  Basis: {basis}")
    print(f"  Unit: {unit}")
    print(f"  E(SCF) = {mf.e_tot:.12f}")

    if benchmark and 'bench_results' in locals():
        print("\n  Benchmark energies:")
        for key, label in [("mp2", "MP2"), ("cisd", "CISD"),
                          ("ccsd", "CCSD"), ("ccsd_t", "CCSD(T)"),
                          ("fci", "FCI")]:
            if key in bench_results and bench_results[key]:
                print(f"    E({label}) = {bench_results[key]:.12f}")


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Generate FCIDUMP with orbital transforms",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("molecule", help="Molecule type")
    p.add_argument("--basis", default="sto-3g", help="Basis set")
    p.add_argument("--output", default="FCIDUMP", help="Output filename")
    p.add_argument("--charge", type=int, default=0, help="Molecular charge")
    p.add_argument("--spin", type=int, default=0, help="Unpaired electrons (2S)")
    p.add_argument("--units", choices=["angstrom", "bohr"], default="angstrom",
                   help="Coordinate units (default: angstrom)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument("--benchmark", action="store_true",
                   help="Run benchmark calculations")

    # Geometry parameters
    p.add_argument("--distance", type=float, help="Bond distance")
    p.add_argument("--n_atoms", type=int, help="Number of atoms (chain/ring)")
    p.add_argument("--radius", type=float, help="Ring radius")
    p.add_argument("--cc_dist", type=float, help="C-C bond distance")
    p.add_argument("--cc_angle", type=float, default=0.0,
                   help="C-C rotation angle (degrees)")

    # Orbital transforms
    p.add_argument("--localize",
                   choices=["canonical", "boys", "pipek", "edmiston",
                           "no-mp2", "no-cisd", "no-ccsd",
                           "mp2_no", "cisd_no", "ccsd_no"],
                   help="Orbital transform method")
    p.add_argument("--localize_virtual", action="store_true",
                   help="Apply LIVVO to virtual orbitals")

    return p.parse_args()


def main():
    args = parse_args()
    mol = args.molecule.lower()

    # Geometry dispatch with default parameters
    geom_funcs = {
        "h2": lambda: _diatomic("H", "H", args.distance or 0.74),
        "b2": lambda: _diatomic("B", "B", args.distance or 2.0),
        "c2": lambda: _diatomic("C", "C", args.distance or 1.2425),
        "n2": lambda: _diatomic("N", "N", args.distance or 1.098),
        "o2": lambda: _diatomic("O", "O", args.distance or 1.208),
        "o3": geom_o3,
        "licl": lambda: _diatomic("Li", "Cl", args.distance or 2.02),
        "li2o": geom_li2o,
        "hchain": lambda: geom_h_chain(args.n_atoms or 4, args.distance or 1.0),
        "hring": lambda: geom_h_ring(args.n_atoms or 4, args.radius or 1.0),
        "h6_chain": lambda: geom_h_chain(6, args.distance or 1.0),
        "h2o": lambda: geom_h2o(args.distance or 0.9572),
        "water": lambda: geom_h2o(args.distance or 0.9572),
        "nh3": lambda: geom_nh3(args.distance or 1.012),
        "ammonia": lambda: geom_nh3(args.distance or 1.012),
        "ch4": lambda: geom_ch4(args.distance or 1.09),
        "methane": lambda: geom_ch4(args.distance or 1.09),
        "co2": lambda: geom_co2(args.distance or 1.16),
        "c2h4": lambda: geom_c2h4(args.cc_dist or 1.339, args.cc_angle),
        "ethylene": lambda: geom_c2h4(args.cc_dist or 1.339, args.cc_angle),
        "c2h6": lambda: geom_c2h6(args.cc_dist or 1.536, args.cc_angle),
        "ethane": lambda: geom_c2h6(args.cc_dist or 1.536, args.cc_angle),
        "c3h8": geom_c3h8,
        "propane": geom_c3h8,
        "c2h4o": geom_c2h4o,
        "ethylene_oxide": geom_c2h4o,
        "c2h4o2": geom_c2h4o2,
        "dioxetane": geom_c2h4o2,
    }

    if mol not in geom_funcs:
        print(f"Error: Unsupported molecule '{mol}'")
        print(f"Supported: {', '.join(sorted(geom_funcs.keys()))}")
        sys.exit(1)

    # Generate geometry and round coordinates
    geom = geom_funcs[mol]()
    geom = _round_coords(geom, decimals=6)

    # Generate FCIDUMP
    gen_fcidump(
        geom=geom,
        basis=args.basis,
        out=args.output,
        charge=args.charge,
        spin=args.spin,
        unit=args.units,  # Pass unit directly to PySCF
        loc=args.localize,
        verbose=args.verbose,
        loc_vir=args.localize_virtual,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
