#!/usr/bin/env python
"""
FCIDUMP.py - Generate FCIDUMP files for calculations and tests

This script generates FCIDUMP files for various molecular geometries and
supports different orbital localization schemes. It uses PySCF for the
quantum chemistry calculations.

Usage:
    python FCIDUMP.py molecule [options]

Example:
    python FCIDUMP.py h2 --basis sto-3g --output H2.FCIDUMP
    python FCIDUMP.py h2o --basis cc-pvdz --output H2O.FCIDUMP --symmetry
    python FCIDUMP.py hchain --n_atoms 10 --distance 2.0 --basis sto-3g --localize boys --output H10.FCIDUMP
"""

import os
import sys
import math
import argparse
import numpy as np
from functools import reduce
from pyscf import gto, scf, lo, symm, ao2mo
from pyscf.tools import fcidump


def round_geometry(geometry, decimals=4):
    """
    Round coordinates in a molecular geometry to a specified number of decimal places.
    
    Args:
        geometry: List of (atom, (x, y, z)) tuples
        decimals: Number of decimal places to round to
        
    Returns:
        Geometry with rounded coordinates
    """
    new_geometry = []
    for atom, (x, y, z) in geometry:
        x_rounded = round(x, decimals)
        y_rounded = round(y, decimals)
        z_rounded = round(z, decimals)
        new_geometry.append((atom, (x_rounded, y_rounded, z_rounded)))
    return new_geometry


def generate_hydrogen_chain(n, distance=1.4, decimals=4):
    """
    Generate a linear chain of hydrogen atoms.
    
    Args:
        n: Number of hydrogen atoms
        distance: Distance between neighboring atoms
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of a hydrogen chain
    """
    geometry = []
    for i in range(n):
        z = i * distance
        geometry.append(('H', (0.0, 0.0, z)))
    return round_geometry(geometry, decimals)


def generate_water_geometry(oh_distance=0.9572, decimals=4):
    """
    Generate the geometry of a water molecule.
    
    Args:
        oh_distance: O-H bond distance
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of a water molecule
    """
    hoh_angle_deg = 104.5
    hoh_angle_rad = math.radians(hoh_angle_deg)
    O = ("O", (0.0, 0.0, 0.0))
    x1 = oh_distance * math.sin(hoh_angle_rad/2)
    y1 = oh_distance * math.cos(hoh_angle_rad/2)
    H1 = ("H", (x1, y1, 0.0))
    x2 = -x1
    y2 = y1
    H2 = ("H", (x2, y2, 0.0))
    geometry = [O, H1, H2]
    return round_geometry(geometry, decimals)


def generate_diatomic_molecule(atom1, atom2, distance, decimals=4):
    """
    Generate a diatomic molecule.
    
    Args:
        atom1: Symbol of the first atom
        atom2: Symbol of the second atom
        distance: Bond distance
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of a diatomic molecule
    """
    atom1_position = (0.0, 0.0, 0.0)
    atom2_position = (0.0, 0.0, distance)
    
    geometry = [(atom1, atom1_position), (atom2, atom2_position)]
    return round_geometry(geometry, decimals)


def generate_hydrogen_ring(n, radius, decimals=4):
    """
    Generate a ring of hydrogen atoms uniformly distributed on a circle.

    Args:
        n: Number of hydrogen atoms in the ring.
        radius: Radius of the ring.
        decimals: Number of decimal places for coordinates.

    Returns:
        Geometry of the hydrogen ring as a list of ("H", (x, y, z)) tuples.
    """
    geometry = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        geometry.append(("H", (x, y, 0.0)))
    return round_geometry(geometry, decimals)


def generate_NH3_geometry(nh_distance=1.012, decimals=4):
    """
    Generate the geometry of an ammonia (NH3) molecule.
    
    Args:
        nh_distance: N–H bond distance (default ≈ 1.012 Å)
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of an NH3 molecule
    """
    r_xy = nh_distance / 1.08265
    h = 0.4147 * r_xy
    
    N = ("N", (0.0, 0.0, 0.0))
    H1 = ("H", (r_xy, 0.0, -h))
    H2 = ("H", (r_xy * math.cos(math.radians(120)), 
                r_xy * math.sin(math.radians(120)), -h))
    H3 = ("H", (r_xy * math.cos(math.radians(240)), 
                r_xy * math.sin(math.radians(240)), -h))
    
    geometry = [N, H1, H2, H3]
    return round_geometry(geometry, decimals)


def generate_CH4_geometry(ch_distance=1.09, decimals=4):
    """
    Generate the geometry of a methane (CH4) molecule.
    
    Args:
        ch_distance: C–H bond distance (default ≈ 1.09 Å)
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of a CH4 molecule
    """
    C = ("C", (0.0, 0.0, 0.0))
    directions = [
        (1,  1,  1),
        (1, -1, -1),
        (-1,  1, -1),
        (-1, -1,  1)
    ]
    
    H_atoms = []
    for vec in directions:
        norm = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        scaled = (ch_distance * vec[0] / norm,
                  ch_distance * vec[1] / norm,
                  ch_distance * vec[2] / norm)
        H_atoms.append(("H", scaled))
        
    geometry = [C] + H_atoms
    return round_geometry(geometry, decimals)


def generate_O2_geometry(distance=1.208, decimals=4):
    """
    Generate the geometry of an O2 molecule.
    
    Args:
        distance: O-O bond distance (default ≈ 1.208 Å)
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of an O2 molecule
    """
    return generate_diatomic_molecule("O", "O", distance, decimals)


def generate_N2_geometry(distance=1.098, decimals=4):
    """
    Generate the geometry of an N2 molecule.
    
    Args:
        distance: N-N bond distance (default ≈ 1.098 Å)
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of an N2 molecule
    """
    return generate_diatomic_molecule("N", "N", distance, decimals)


def generate_CO2_geometry(co_distance=1.16, decimals=4):
    """
    Generate the geometry of a CO2 molecule.
    
    Args:
        co_distance: C-O bond distance (default ≈ 1.16 Å)
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of a CO2 molecule
    """
    C = ("C", (0.0, 0.0, 0.0))
    O1 = ("O", (0.0, 0.0, co_distance))
    O2 = ("O", (0.0, 0.0, -co_distance))
    
    geometry = [C, O1, O2]
    return round_geometry(geometry, decimals)

def generate_Li2O_geometry(decimals=4):
    """
    Generate the geometry of a CO2 molecule.
    
    Args:
        co_distance: C-O bond distance (default ≈ 1.16 Å)
        decimals: Number of decimal places for coordinates
        
    Returns:
        Geometry of a CO2 molecule
    """
    Li1 = ("Li", (3.732, 0.25, 0.0))
    Li2 = ("Li", (2.0, 0.25, 0.0))
    O = ("O", (2.866, -0.25, 0.0))
    
    geometry = [Li1, Li2, O]
    return round_geometry(geometry, decimals)

def localize_orbitals(mol, mf, method):
    """
    Localize molecular orbitals using different methods.
    
    Args:
        mol: PySCF Mole object
        mf: PySCF SCF object
        method: Localization method ('boys', 'pipek', 'edmiston', etc.)
        
    Returns:
        Localized molecular orbital coefficients
    """
    mo_coeff = mf.mo_coeff
    
    if method.lower() == 'boys':
        loc_obj = lo.Boys(mol, mo_coeff)
        mo_loc = loc_obj.kernel()
    elif method.lower() == 'edmiston':
        loc_obj = lo.EdmistonRuedenberg(mol, mo_coeff)
        mo_loc = loc_obj.kernel()
    elif method.lower() == 'pipek':
        loc_obj = lo.PipekMezey(mol, mo_coeff)
        mo_loc = loc_obj.kernel()
    elif method.lower() == 'nao':
        mo_loc = lo.orth_ao(mf, 'nao')
    elif method.lower() == 'lowdin':
        mo_loc = lo.orth_ao(mf, 'lowdin')
    elif method.lower() == 'meta-lowdin':
        mo_loc = lo.orth_ao(mf, 'meta-lowdin')
    elif method.lower() == 'canonical':
        # Just return the original coefficients
        mo_loc = mo_coeff.copy()
    else:
        raise ValueError(f"Unsupported localization method: {method}")
    
    return mo_loc


def generate_fcidump(geometry, basis='sto-3g', output_file='FCIDUMP', 
                     charge=0, spin=0, symmetry=False, localize=None,
                     active_space=None, verbose=False, use_uhf=False):
    """
    Generate a FCIDUMP file for a molecule.
    
    Args:
        geometry: Molecular geometry
        basis: Basis set
        output_file: Output file name
        charge: Molecular charge
        spin: Number of unpaired electrons (2S)
        symmetry: Whether to use symmetry
        localize: Orbital localization method (None, 'boys', 'pipek', etc.)
        active_space: Tuple of (n_active_electrons, n_active_orbitals)
        verbose: Whether to print verbose output
        use_uhf: Whether to use UHF instead of RHF
    """
    # Build the molecule
    mol = gto.Mole()
    mol.atom = geometry
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = symmetry
    mol.verbose = 4 if verbose else 2
    mol.build()
    
    # Run SCF
    if use_uhf or spin > 0:
        mf = scf.UHF(mol)
    else:
        mf = scf.RHF(mol)
    
    mf.verbose = 4 if verbose else 0
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.run()
    
    # Localize orbitals if requested
    if localize:
        print(f"Localizing orbitals using {localize} method...")
        mo_coeff = localize_orbitals(mol, mf, localize)
        mf.mo_coeff = mo_coeff
    
    # Handle active space
    if active_space:
        n_active_electrons, n_active_orbitals = active_space
        if n_active_orbitals is None:
            n_active_orbitals = mol.nao_nr()
        
        # Assuming closed-shell for simplicity here
        n_inactive_orbs = (mol.nelectron - n_active_electrons) // 2
        n_active_orbs = n_active_orbitals
        
        active_space = (n_inactive_orbs, n_active_orbs)
        print(f"Using active space: {n_active_electrons} electrons in {n_active_orbitals} orbitals")
        print(f"Active space params: {active_space}")
    else:
        active_space = None
    
    # Write FCIDUMP file
    fcidump.from_scf(mf, output_file, tol=1e-10, molpro_orbsym=None)
    
    print(f"Generated FCIDUMP file: {output_file}")
    print(f"  Molecule: {mol.atom}")
    print(f"  Basis: {basis}")
    print(f"  E(SCF) = {mf.e_tot:.12f}")
    
    # Print orbital symmetries if using symmetry
    if symmetry:
        orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)
        print("  Orbital symmetries:", orbsym)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate FCIDUMP files for Dice/SHCI calculations')
    
    # Molecule selection
    parser.add_argument('molecule', type=str, help='Molecule type (h2, h2o, nh3, ch4, etc.)')
    
    # Basic options
    parser.add_argument('--basis', type=str, default='sto-3g', help='Basis set')
    parser.add_argument('--output', type=str, default='FCIDUMP', help='Output file name')
    parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    parser.add_argument('--spin', type=int, default=0, help='Number of unpaired electrons (2S)')
    parser.add_argument('--symmetry', action='store_true', help='Use symmetry')
    parser.add_argument('--uhf', action='store_true', help='Use unrestricted Hartree-Fock')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Molecular geometry parameters
    parser.add_argument('--distance', type=float, help='Bond distance (in Angstroms)')
    parser.add_argument('--n_atoms', type=int, help='Number of atoms (for chains, rings, etc.)')
    parser.add_argument('--radius', type=float, help='Radius (for rings)')
    
    # Orbital localization
    parser.add_argument('--localize', type=str, choices=[
        'boys', 'pipek', 'edmiston', 'nao', 'lowdin', 'meta-lowdin', 'canonical'],
        help='Orbital localization method')
    
    # Active space
    parser.add_argument('--active_e', type=int, help='Number of active electrons')
    parser.add_argument('--active_orb', type=int, help='Number of active orbitals')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set up geometry based on molecule type
    molecule = args.molecule.lower()
    
    if molecule == 'h2':
        distance = args.distance or 0.74
        geometry = generate_diatomic_molecule('H', 'H', distance)
    elif molecule == 'b2':
        distance = args.distance or 2.00
        geometry = generate_diatomic_molecule('B', 'B', distance)
    elif molecule == 'licl':
        distance = args.distance or 2.02
        geometry = generate_diatomic_molecule('Li', 'Cl', distance)
    elif molecule == 'li2o':
        geometry = generate_Li2O_geometry()
    elif molecule == 'c2':
        distance = args.distance or 2.00
        geometry = generate_diatomic_molecule('C', 'C', distance)
    elif molecule == 'hchain':
        n_atoms = args.n_atoms or 4
        distance = args.distance or 1.0
        geometry = generate_hydrogen_chain(n_atoms, distance)
    elif molecule == 'hring':
        n_atoms = args.n_atoms or 4
        radius = args.radius or 1.0
        geometry = generate_hydrogen_ring(n_atoms, radius)
    elif molecule == 'h6_chain':
        n_atoms = args.n_atoms or 6
        distance = args.distance or 1.0
        geometry = generate_hydrogen_chain(n_atoms, distance)
    elif molecule == 'h2o' or molecule == 'water':
        distance = args.distance or 0.9572
        geometry = generate_water_geometry(distance)
    elif molecule == 'nh3' or molecule == 'ammonia':
        distance = args.distance or 1.012
        geometry = generate_NH3_geometry(distance)
    elif molecule == 'ch4' or molecule == 'methane':
        distance = args.distance or 1.09
        geometry = generate_CH4_geometry(distance)
    elif molecule == 'o2':
        distance = args.distance or 1.208
        geometry = generate_O2_geometry(distance)
    elif molecule == 'n2':
        distance = args.distance or 1.098
        geometry = generate_N2_geometry(distance)
    elif molecule == 'co2':
        distance = args.distance or 1.16
        geometry = generate_CO2_geometry(distance)
    else:
        print(f"Error: Molecule '{molecule}' not supported.")
        print("Supported molecules: h2, h4_chain, h4_ring, h6_chain, h2o, nh3, ch4, o2, n2, co2")
        sys.exit(1)
    
    # Set up active space
    active_space = None
    if args.active_e and args.active_orb:
        active_space = (args.active_e, args.active_orb)
    
    # Generate FCIDUMP file
    generate_fcidump(
        geometry=geometry,
        basis=args.basis,
        output_file=args.output,
        charge=args.charge,
        spin=args.spin,
        symmetry=args.symmetry,
        localize=args.localize,
        active_space=active_space,
        verbose=args.verbose,
        use_uhf=args.uhf
    )


if __name__ == '__main__':
    main() 