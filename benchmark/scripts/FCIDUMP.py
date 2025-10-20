# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate FCIDUMP files with orbital localization and natural orbital transforms.

Supports multiple molecular geometries, orbital localization (Boys, Pipek-Mezey,
Edmiston-Ruedenberg, IBO), natural orbitals (MP2, CISD, CCSD), optional LIVVO
for virtual orbitals, and benchmark calculations (MP2, CISD, CCSD, CCSD(T), FCI).

File: tools/FCIDUMP.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2025

Examples:
  python FCIDUMP.py h2 --basis sto-3g --output H2.FCIDUMP
  python FCIDUMP.py h2o --basis cc-pvdz --output H2O.FCIDUMP
  python FCIDUMP.py hchain --n_atoms 10 --distance 2.0 --basis sto-3g \\
      --localize boys --output H10_2.00_boys.FCIDUMP
  python FCIDUMP.py n2 --basis cc-pvtz --localize no-mp2 \\
      --output N2.FCIDUMP --benchmark
  python FCIDUMP.py c2h6 --basis sto-3g --cc_angle 30 \\
      --output C2H6_30.FCIDUMP --benchmark
"""

import argparse
import math
import sys
from datetime import datetime
from typing import Callable

import numpy as np
from pyscf import ao2mo, cc, ci, fci, gto, lo, mp, scf
from pyscf.cc import ccsd_rdm
from pyscf.lo import edmiston, iao, ibo, orth, vvo
from pyscf.tools import fcidump


# ============================================================================
# Geometry Utilities
# ============================================================================

def _round(geom: list, decimals: int = 4) -> list:
    """Round coordinates to specified decimals."""
    return [(atom, tuple(round(x, decimals) for x in xyz)) for atom, xyz in geom]


def _diatomic(a1: str, a2: str, d: float) -> list:
    """Diatomic molecule along z-axis."""
    return _round([(a1, (0, 0, 0)), (a2, (0, 0, d))])


def _rotate_z(x: float, y: float, z: float, deg: float) -> tuple:
    """Rotate (x,y,z) around z-axis by deg degrees."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return (x * c - y * s, x * s + y * c, z)


# ============================================================================
# Molecular Geometries
# ============================================================================

def geom_h_chain(n: int, d: float = 1.4) -> list:
    """Linear H chain along z-axis."""
    return _round([("H", (0, 0, i * d)) for i in range(n)])


def geom_h_ring(n: int, r: float) -> list:
    """Planar H ring in xy-plane."""
    theta = 2 * math.pi / n
    return _round([("H", (r * math.cos(i * theta), r * math.sin(i * theta), 0))
                   for i in range(n)])


def geom_h2o(oh: float = 0.9572) -> list:
    """H₂O with ∠HOH ≈ 104.5°."""
    ang = math.radians(104.5)
    x = oh * math.sin(ang / 2)
    y = oh * math.cos(ang / 2)
    return _round([("O", (0, 0, 0)), ("H", (x, y, 0)), ("H", (-x, y, 0))])


def geom_nh3(nh: float = 1.012) -> list:
    """NH₃ pyramid."""
    r_xy = nh / 1.08265
    h = 0.4147 * r_xy
    geom = [("N", (0, 0, 0))]
    for ang in [0, 120, 240]:
        rad = math.radians(ang)
        geom.append(("H", (r_xy * math.cos(rad), r_xy * math.sin(rad), -h)))
    return _round(geom)


def geom_ch4(ch: float = 1.09) -> list:
    """Tetrahedral CH₄."""
    dirs = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    geom = [("C", (0, 0, 0))]
    for v in dirs:
        norm = math.sqrt(sum(x**2 for x in v))
        geom.append(("H", tuple(ch * x / norm for x in v)))
    return _round(geom)


def geom_co2(co: float = 1.16) -> list:
    """Linear CO₂."""
    return _round([("C", (0, 0, 0)), ("O", (0, 0, co)), ("O", (0, 0, -co))])


def geom_o3() -> list:
    """Ozone."""
    return _round([("O", (0, 0, 0)), ("O", (0, 1.0885, 0.6697)),
                   ("O", (0, -1.0885, 0.6697))])


def geom_li2o() -> list:
    """Li₂O."""
    return _round([("Li", (3.732, 0.25, 0)), ("Li", (2.0, 0.25, 0)),
                   ("O", (2.866, -0.25, 0))])


def geom_c2h4(cc: float = 1.339, ang: float = 0) -> list:
    """
    C₂H₄ with CC distance and rotation angle around CC bond.
  
    Args:
        cc: C-C distance (Å), equilibrium ≈ 1.339
        ang: Rotation angle (deg) of second CH₂ group
    """
    scale = cc / 1.339
    c1 = (0, 0, 0.6695 * scale)
    c2 = (0, 0, -0.6695 * scale)
    h3 = (0, 0.9289 * scale, 1.2321 * scale)
    h4 = (0, -0.9289 * scale, 1.2321 * scale)
    h5 = (0, 0.9289 * scale, -1.2321 * scale)
    h6 = (0, -0.9289 * scale, -1.2321 * scale)

    c2_rot = _rotate_z(*c2, ang)
    h5_rot = _rotate_z(*h5, ang)
    h6_rot = _rotate_z(*h6, ang)

    return _round([("C", c1), ("C", c2_rot), ("H", h3), ("H", h4),
                   ("H", h5_rot), ("H", h6_rot)])


def geom_c2h6(cc: float = 1.536, ang: float = 0) -> list:
    """
    C₂H₆ with CC distance and rotation angle around CC bond.
  
    Args:
        cc: C-C distance (Å), equilibrium ≈ 1.536
        ang: Rotation angle (deg) of second CH₃ group (staggered at 0°)
    """
    scale = cc / 1.536
    c1 = (0, 0, 0.7680 * scale)
    c2 = (0, 0, -0.7680 * scale)
    h3 = (-1.0192 * scale, 0, 1.1573 * scale)
    h4 = (0.5096 * scale, 0.8826 * scale, 1.1573 * scale)
    h5 = (0.5096 * scale, -0.8826 * scale, 1.1573 * scale)
    h6 = (1.0192 * scale, 0, -1.1573 * scale)
    h7 = (-0.5096 * scale, -0.8826 * scale, -1.1573 * scale)
    h8 = (-0.5096 * scale, 0.8826 * scale, -1.1573 * scale)

    c2_rot = _rotate_z(*c2, ang)
    h6_rot = _rotate_z(*h6, ang)
    h7_rot = _rotate_z(*h7, ang)
    h8_rot = _rotate_z(*h8, ang)

    return _round([("C", c1), ("C", c2_rot), ("H", h3), ("H", h4), ("H", h5),
                   ("H", h6_rot), ("H", h7_rot), ("H", h8_rot)])


# ============================================================================
# Natural Orbital Transforms
# ============================================================================

def _sorted_eigh(dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize 1-RDM, return sorted eigenvalues and eigenvectors."""
    dm_sym = (dm + dm.T.conj()) * 0.5
    occ, U = np.linalg.eigh(dm_sym)
    idx = np.argsort(-occ)
    return occ[idx], U[:, idx]


def get_mp2_no(mf: scf.hf.RHF) -> tuple[np.ndarray, np.ndarray]:
    """MP2 natural orbitals via 1-RDM diagonalization."""
    m2 = mp.MP2(mf).run()
    if hasattr(m2, "make_natorbs"):
        occ, U = m2.make_natorbs()
        return mf.mo_coeff @ U, occ
    dm_mo = m2.make_rdm1(ao_repr=False)
    occ, U = _sorted_eigh(dm_mo)
    return mf.mo_coeff @ U, occ


def get_cisd_no(mf: scf.hf.RHF) -> tuple[np.ndarray, np.ndarray]:
    """CISD natural orbitals via 1-RDM diagonalization."""
    myci = ci.CISD(mf).run()
    dm_mo = myci.make_rdm1()
    occ, U = _sorted_eigh(dm_mo)
    return mf.mo_coeff @ U, occ


def get_ccsd_no(mf: scf.hf.RHF) -> tuple[np.ndarray, np.ndarray]:
    """CCSD natural orbitals via lambda-relaxed 1-RDM."""
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

    # CCSD and CCSD(T)
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

    if mycc is not None:
        try:
            log.write("Running CCSD(T)...\n")
            log.flush()
            e_t = mycc.ccsd_t()
            results["ccsd_t"] = mycc.e_tot + e_t
            log.write(f"  E(T)    = {e_t:.12f} Ha\n")
            log.write(f"  E(CCSD(T)) = {results['ccsd_t']:.12f} Ha\n")
            log.write(f"  Ecorr   = {results['ccsd_t'] - mf.e_tot:.12f} Ha\n\n")
        except Exception as e:
            log.write(f"  CCSD(T) failed: {e}\n\n")
            results["ccsd_t"] = None
    else:
        log.write("CCSD(T) skipped (CCSD failed)\n\n")
        results["ccsd_t"] = None

    # FCI (only for small systems)
    n_orb = mol.nao_nr()
    if n_orb <= 12:
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
        log.write(f"FCI skipped (n_orb={n_orb} > 12)\n\n")
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
        mf: Mean-field object
        method: Transform method (boys/pipek/edmiston/ibo/no-mp2/no-cisd/no-ccsd)
        loc_vir: Apply LIVVO to virtual orbitals
    """
    C = mf.mo_coeff
    method = (method or "").lower()

    # Natural orbitals (full space)
    if method in ("no-mp2", "mp2_no"):
        C_no, _ = get_mp2_no(mf)
        return C_no
    if method in ("no-cisd", "cisd_no"):
        C_no, _ = get_cisd_no(mf)
        return C_no
    if method in ("no-ccsd", "ccsd_no"):
        C_no, _ = get_ccsd_no(mf)
        return C_no

    # Localization (occupied only)
    occ_mask = mf.mo_occ > 1e-8
    C_occ, C_vir = C[:, occ_mask], C[:, ~occ_mask]

    def _localize(C_sub: np.ndarray, how: str) -> np.ndarray:
        if how == "boys":
            return lo.boys.Boys(mol, C_sub).kernel()
        elif how == "pipek":
            return lo.pipek.PipekMezey(mol, C_sub).kernel()
        elif how == "edmiston":
            return edmiston.ER(mol, C_sub).kernel()
        elif how == "ibo":
            S = mol.intor_symmetric("int1e_ovlp")
            iaos = iao.iao(mol, C_sub)
            iaos = orth.vec_lowdin(iaos, S)
            return ibo.ibo(mol, C_sub, iaos=iaos, s=S)
        else:
            raise ValueError(f"Unknown localization: {how}")

    if method in ("boys", "pipek", "edmiston", "ibo"):
        C_occ_loc = _localize(C_occ, method)
        C_vir_loc = vvo.livvo(mol, C_occ_loc, C_vir) if loc_vir else C_vir
        return np.hstack([C_occ_loc, C_vir_loc])

    if method in ("canonical", None, ""):
        return C

    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# FCIDUMP Generation
# ============================================================================

def gen_fcidump(geom: list, basis: str = "sto-3g", out: str = "FCIDUMP",
                charge: int = 0, spin: int = 0, loc: str = None,
                verbose: bool = False, loc_vir: bool = False,
                benchmark: bool = False) -> None:
    """
    Generate FCIDUMP file from molecular geometry.
  
    Args:
        geom: Molecular geometry [(atom, (x,y,z)), ...]
        basis: Basis set name
        out: Output filename
        charge: Molecular charge
        spin: Number of unpaired electrons (2S)
        loc: Orbital transform method
        verbose: Verbose output
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

        # Build molecule
        log.write("=" * 60 + "\n")
        log.write("MOLECULAR GEOMETRY\n")
        log.write("=" * 60 + "\n")
        mol = gto.Mole()
        mol.atom = geom
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.verbose = 4 if verbose else 2
        mol.build()

        log.write(f"Charge: {charge}\n")
        log.write(f"Spin (2S): {spin}\n")
        log.write(f"Basis: {basis}\n")
        log.write(f"N_electrons: {mol.nelectron}\n")
        log.write(f"N_orbitals: {mol.nao_nr()}\n\n")
        log.write("Atomic coordinates (Angstrom):\n")
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
        log.write(f"Tolerance: 1e-12\n\n")
        log.flush()

        fcidump.from_scf(mf, out, tol=1e-12)

        log.write("FCIDUMP file generated successfully\n\n")
        log.write("=" * 60 + "\n")
        log.write("END OF LOG\n")
        log.write("=" * 60 + "\n")

    # Console output
    print(f"\nGenerated: {out}")
    print(f"Log file: {log_filename}")
    print(f"  Basis: {basis}")
    print(f"  E(SCF) = {mf.e_tot:.12f}")

    if benchmark:
        print("\n  Benchmark energies:")
        if "mp2" in bench_results and bench_results["mp2"]:
            print(f"    E(MP2)  = {bench_results['mp2']:.12f}")
        if "cisd" in bench_results and bench_results["cisd"]:
            print(f"    E(CISD) = {bench_results['cisd']:.12f}")
        if "ccsd" in bench_results and bench_results["ccsd"]:
            print(f"    E(CCSD) = {bench_results['ccsd']:.12f}")
        if "ccsd_t" in bench_results and bench_results["ccsd_t"]:
            print(f"    E(CCSD(T)) = {bench_results['ccsd_t']:.12f}")
        if "fci" in bench_results and bench_results["fci"]:
            print(f"    E(FCI)  = {bench_results['fci']:.12f}")


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Generate FCIDUMP files with orbital transforms")

    p.add_argument("molecule", help="Molecule type (h2, h2o, c2h4, c2h6, ...)")
    p.add_argument("--basis", default="sto-3g", help="Basis set")
    p.add_argument("--output", default="FCIDUMP", help="Output filename")
    p.add_argument("--charge", type=int, default=0, help="Charge")
    p.add_argument("--spin", type=int, default=0, help="Unpaired electrons (2S)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument("--benchmark", action="store_true",
                   help="Run benchmark calculations")

    # Geometry parameters
    p.add_argument("--distance", type=float, help="Bond distance (Angstrom)")
    p.add_argument("--n_atoms", type=int, help="Number of atoms")
    p.add_argument("--radius", type=float, help="Ring radius")
    p.add_argument("--cc_dist", type=float, help="C-C bond distance")
    p.add_argument("--cc_angle", type=float, default=0,
                   help="C-C rotation angle (degrees)")

    # Orbital transforms
    p.add_argument("--localize",
                   choices=["canonical", "boys", "pipek", "edmiston", "ibo",
                           "no-mp2", "no-cisd", "no-ccsd",
                           "mp2_no", "cisd_no", "ccsd_no"],
                   help="Orbital transform method")
    p.add_argument("--localize_virtual", action="store_true",
                   help="Localize virtuals with LIVVO")

    return p.parse_args()


def main():
    args = parse_args()
    mol = args.molecule.lower()

    # Geometry dispatch
    geom_map = {
        "h2": lambda: _diatomic("H", "H", args.distance or 0.74),
        "b2": lambda: _diatomic("B", "B", args.distance or 2.0),
        "c2": lambda: _diatomic("C", "C", args.distance or 1.2425),
        "n2": lambda: _diatomic("N", "N", args.distance or 1.098),
        "o2": lambda: _diatomic("O", "O", args.distance or 1.208),
        "o3": lambda: geom_o3(),
        "licl": lambda: _diatomic("Li", "Cl", args.distance or 2.02),
        "li2o": lambda: geom_li2o(),
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
    }

    if mol not in geom_map:
        print(f"Error: Unsupported molecule '{mol}'")
        print(f"Supported: {', '.join(sorted(geom_map.keys()))}")
        sys.exit(1)

    geom = geom_map[mol]()

    gen_fcidump(
        geom=geom,
        basis=args.basis,
        out=args.output,
        charge=args.charge,
        spin=args.spin,
        loc=args.localize,
        verbose=args.verbose,
        loc_vir=args.localize_virtual,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
