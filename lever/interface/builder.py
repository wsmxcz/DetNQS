# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PySCF to FCIDUMP + HDF5 conversion pipeline.

Converts molecular specifications (atoms, basis, charge, spin) into:
  - Standard FCIDUMP file (Hamiltonian)
  - Compact HDF5 metadata (geometry, electron counts, HF guess)

File: lever/interface/builder.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import h5py

from . import pyscf_backend as backend


@dataclass(frozen=True)
class ExportedSystem:
    """Handle for exported quantum chemistry system."""
    fcidump_path: Path
    hdf5_path: Path
    n_orbitals: int
    n_alpha: int
    n_beta: int
    is_cas: bool
    n_core: int


def read_from_hdf5(h5_path: str) -> Dict[str, Any]:
    """
    Extract SystemConfig essentials from HDF5 file.
    
    Reads: fcidump_path, orbital counts, electron counts, CAS flags.
    """
    path = Path(h5_path)
    with h5py.File(path, "r") as f:
        g_sys = f["system"]
        n_orbitals = int(g_sys["n_orbitals"][()])
        n_alpha = int(g_sys["n_alpha"][()])
        n_beta = int(g_sys["n_beta"][()])

        is_cas = bool(g_sys.attrs.get("is_cas", False))
        n_core = int(g_sys["n_core_orbitals"][()]) if "n_core_orbitals" in g_sys else 0

        fcidump_path: str
        g_files = f.get("files", None)
        if g_files is not None and "fcidump_path" in g_files:
            raw = g_files["fcidump_path"][()]
            fcidump_path = raw.decode("utf8") if isinstance(raw, bytes) else str(raw)
        else:
            fcidump_path = str(path.with_suffix(".FCIDUMP"))

    return {
        "fcidump_path": fcidump_path,
        "n_orbitals": n_orbitals,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
        "meta_path": str(path),
        "is_cas": is_cas,
        "n_core": n_core,
    }


def _write_hdf5(
    *,
    hdf5_path: Path,
    mol,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    is_cas: bool,
    n_core: int,
    fcidump_path: Path,
    benchmarks: Optional[Dict[str, Optional[float]]],
) -> None:
    """
    Write compact HDF5 metadata with minimal schema.
    
    Schema:
      /system: orbital/electron counts, CAS flags
      /files: fcidump_path
      /geometry: atom symbols/coordinates
      /initialization: HF occupation patterns
      /benchmarks: optional energy values
    """
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "w") as f:
        # System metadata
        g_sys = f.create_group("system")
        g_sys.create_dataset("n_orbitals", data=int(n_orbitals))
        g_sys.create_dataset("n_alpha", data=int(n_alpha))
        g_sys.create_dataset("n_beta", data=int(n_beta))
        g_sys.create_dataset("charge", data=int(mol.charge))
        g_sys.create_dataset("spin", data=int(mol.spin))
        g_sys.create_dataset("n_core_orbitals", data=int(n_core))
        g_sys.attrs["is_cas"] = bool(is_cas)

        # File paths
        g_files = f.create_group("files")
        g_files.create_dataset("fcidump_path", data=str(fcidump_path.resolve()).encode("utf8"))

        # Molecular geometry
        g_geom = f.create_group("geometry")
        coords = mol.atom_coords(unit="Angstrom")
        symbols = np.array([mol.atom_symbol(i) for i in range(mol.natm)], dtype="S2")
        g_geom.create_dataset("atom_coords", data=coords)
        g_geom.create_dataset("atom_symbols", data=symbols)
        g_geom.attrs["unit"] = "Angstrom"

        # HF initialization data
        g_init = f.create_group("initialization")
        hf_occ_alpha = np.zeros((n_orbitals,), dtype=np.int8)
        hf_occ_beta = np.zeros((n_orbitals,), dtype=np.int8)
        hf_occ_alpha[:n_alpha] = 1
        hf_occ_beta[:n_beta] = 1
        g_init.create_dataset("hf_occ_alpha", data=hf_occ_alpha)
        g_init.create_dataset("hf_occ_beta", data=hf_occ_beta)

        # Bitstring encoding for determinants (n_orbitals ≤ 64)
        if n_orbitals <= 64:
            alpha_bits = sum(1 << i for i in range(n_alpha))
            beta_bits = sum(1 << i for i in range(n_beta))
            hf_det = np.array([alpha_bits, beta_bits], dtype="uint64")
            g_init.create_dataset("hf_det_bitstring", data=hf_det)

        # Optional benchmark energies
        if benchmarks:
            g_bench = f.create_group("benchmarks")
            for key, val in benchmarks.items():
                if val is not None:
                    g_bench.attrs[key] = float(val)


def load_initial_det(h5_path: str) -> Optional[np.ndarray]:
    """Load HF determinant bitstring from HDF5 file."""
    path = Path(h5_path)
    if not path.exists():
        return None

    with h5py.File(path, "r") as f:
        g_init = f.get("initialization")
        if g_init is None or "hf_det_bitstring" not in g_init:
            return None
        data = g_init["hf_det_bitstring"][()]

    return np.asarray(data, dtype="uint64").reshape(1, 2)


def load_benchmarks(h5_path: str) -> Dict[str, float]:
    """Load benchmark energies from HDF5 attributes."""
    path = Path(h5_path)
    if not path.exists():
        return {}

    out: Dict[str, float] = {}
    with h5py.File(path, "r") as f:
        g = f.get("benchmarks")
        if g is None:
            return {}
        for key, val in g.attrs.items():
            try:
                out[str(key)] = float(val)
            except Exception:
                continue
    return out


class MoleculeBuilder:
    """
    PySCF to FCIDUMP+HDF5 conversion pipeline.
    
    Usage:
        builder = (
            MoleculeBuilder(atom="N 0 0 0; N 0 0 1.1", basis="cc-pvdz")
            .run_scf()
            .apply_natural_orbitals("mp2")
            .set_active_space(10, 8)
            .localize_active_space("ibo")
            .run_benchmarks({"cisd", "ccsd"})
        )
        exported = builder.export("n2_cas")
    """

    def __init__(
        self,
        atom: Any,
        basis: str,
        *,
        charge: int = 0,
        spin: int = 0,
        symmetry: bool = False,
        work_dir: str | Path = ".",
    ) -> None:
        self._atom = atom
        self._basis = basis
        self._charge = int(charge)
        self._spin = int(spin)
        self._symmetry = bool(symmetry)
        self._work_dir = Path(work_dir)

        # PySCF objects
        self._mol = None
        self._mf = None

        # Current orbital representation
        self._C: Optional[np.ndarray] = None
        self._occ: Optional[np.ndarray] = None

        # Active space configuration
        self._natural_orbital_kind: str = "none"
        self._active_space: Optional[backend.ActiveSpaceSpec] = None
        self._localization_method: str = "canonical"
        self._loc_virtual: bool = False

        # Cached energies
        self._benchmarks: Dict[str, Optional[float]] = {}

    @property
    def work_dir(self) -> Path:
        """Output directory for exported files."""
        return self._work_dir

    def run_scf(self, method: str = "auto") -> "MoleculeBuilder":
        """Run SCF calculation and cache molecular objects."""
        self._mol, self._mf = backend.run_scf(
            atom=self._atom,
            basis=self._basis,
            charge=self._charge,
            spin=self._spin,
            symmetry=self._symmetry,
            scf_type=method,
        )
        self._C = np.array(self._mf.mo_coeff, copy=True)
        self._occ = np.array(self._mf.mo_occ, copy=True)
        self._natural_orbital_kind = "none"
        self._active_space = None
        return self

    def apply_natural_orbitals(self, kind: str = "none") -> "MoleculeBuilder":
        """Replace canonical orbitals with natural orbitals of specified kind."""
        if self._mf is None:
            raise RuntimeError("run_scf() required before apply_natural_orbitals()")

        kind = (kind or "none").lower()
        if kind == "none":
            self._C = np.array(self._mf.mo_coeff, copy=True)
            self._occ = np.array(self._mf.mo_occ, copy=True)
        else:
            C_no, occ_no = backend.compute_natural_orbitals(self._mf, kind=kind)
            self._C = C_no
            self._occ = occ_no

        self._natural_orbital_kind = kind
        self._active_space = None
        return self

    def set_active_space(self, n_elec: int, n_orb: int) -> "MoleculeBuilder":
        """
        Define CAS(n_elec, n_orb) active space.
        
        Algorithm:
          1. Sort orbitals by occupation number
          2. Partition: [core | active | virtual]
          3. Reorder MO coefficients accordingly
        
        Partition: n_core = (N_total - n_elec)/2, n_active = n_orb
        """
        if self._mol is None or self._C is None or self._occ is None:
            raise RuntimeError("run_scf() and apply_natural_orbitals() required")

        n_elec = int(n_elec)
        n_orb = int(n_orb)
        nelec_total = int(self._mol.nelectron)
        nmo = int(self._C.shape[1])

        if n_elec <= 0 or n_orb <= 0:
            raise ValueError("n_elec and n_orb must be positive")
        if n_elec > nelec_total:
            raise ValueError("Active electrons exceed total electrons")

        n_core_elec = nelec_total - n_elec
        if n_core_elec < 0 or (n_core_elec % 2) != 0:
            raise ValueError("Total - active electrons must be non-negative even")

        n_core = n_core_elec // 2
        if n_core + n_orb > nmo:
            raise ValueError("n_core + n_orb exceeds orbital count")

        # Sort by occupation: high → low
        idx_sorted = np.argsort(-self._occ)
        perm = np.concatenate([
            idx_sorted[:n_core],           # core
            idx_sorted[n_core:n_core+n_orb],  # active
            idx_sorted[n_core+n_orb:]      # virtual
        ])
        self._C = self._C[:, perm]
        self._occ = self._occ[perm]

        # Compute active space electron counts
        spin = int(self._mol.spin)
        n_alpha_total = (nelec_total + spin) // 2
        n_beta_total = nelec_total - n_alpha_total

        self._active_space = backend.ActiveSpaceSpec(
            n_core=n_core,
            n_active=n_orb,
            n_virt=nmo - n_core - n_orb,
            nelecas_alpha=n_alpha_total - n_core,
            nelecas_beta=n_beta_total - n_core,
        )
        return self

    def localize_active_space(
        self,
        method: str = "canonical",
        *,
        localize_virtual: bool = False,
    ) -> "MoleculeBuilder":
        """Apply orbital localization to active space or full space."""
        if self._mol is None or self._mf is None or self._C is None:
            raise RuntimeError("run_scf() required before localize_active_space()")

        method = (method or "canonical").lower()
        if method in ("canonical", ""):
            self._localization_method = "canonical"
            self._loc_virtual = False
            return self

        # Determine localization window
        if self._active_space is not None:
            n_core = self._active_space.n_core
            active_window = (n_core, n_core + self._active_space.n_active)
        else:
            active_window = None

        self._C = backend.localize_orbitals(
            self._mol,
            self._mf,
            self._C,
            method=method,
            loc_virtual=localize_virtual,
            active_window=active_window,
        )
        self._localization_method = method
        self._loc_virtual = bool(localize_virtual)
        return self

    def run_benchmarks(
        self,
        kinds: Optional[Iterable[str]] = None,
    ) -> "MoleculeBuilder":
        """Run quantum chemistry benchmarks and cache energies."""
        if self._mol is None or self._mf is None:
            raise RuntimeError("run_scf() required before run_benchmarks()")

        if kinds is None:
            kinds = {"hf", "mp2", "cisd", "ccsd", "ccsd_t", "fci"}

        kinds_set = {k.lower() for k in kinds}
        cas_mc = None

        # Build CASCI if active space defined
        if self._active_space is not None and "casci" in kinds_set:
            spec = self._active_space
            cas_mc = backend.build_casci(
                self._mf,
                self._C,
                ncas=spec.n_active,
                nelecas=(spec.nelecas_alpha, spec.nelecas_beta),
                ncore=spec.n_core,
            )

        self._benchmarks = backend.run_benchmarks(
            self._mol,
            self._mf,
            kinds=kinds_set,
            cas_mc=cas_mc,
        )
        return self

    def export(
        self,
        name: str = "system",
        *,
        write_fcidump: bool = True,
        write_hdf5: bool = True,
    ) -> ExportedSystem:
        """Finalize pipeline and write FCIDUMP + HDF5 files."""
        if self._mol is None or self._C is None:
            raise RuntimeError("run_scf() required before export()")

        self._work_dir.mkdir(parents=True, exist_ok=True)
        base = self._work_dir / name
        fcidump_path = base.with_suffix(".FCIDUMP")
        hdf5_path = base.with_suffix(".h5")

        # Determine Hamiltonian type and dimensions
        if self._active_space is None:
            # Full-space Hamiltonian
            n_orbitals = int(self._C.shape[1])
            nelec = int(self._mol.nelectron)
            spin = int(self._mol.spin)
            n_alpha = (nelec + spin) // 2
            n_beta = nelec - n_alpha
            is_cas = False
            n_core = 0

            if write_fcidump:
                backend.write_fcidump_full(self._mol, self._C, fcidump_path)
        else:
            # CAS Hamiltonian
            spec = self._active_space
            n_orbitals = spec.n_active
            n_alpha = spec.nelecas_alpha
            n_beta = spec.nelecas_beta
            is_cas = True
            n_core = spec.n_core

            if write_fcidump:
                mc = backend.build_casci(
                    self._mf,
                    self._C,
                    ncas=spec.n_active,
                    nelecas=(spec.nelecas_alpha, spec.nelecas_beta),
                    ncore=spec.n_core,
                )
                backend.write_fcidump_cas(mc, fcidump_path)

        if write_hdf5:
            _write_hdf5(
                hdf5_path=hdf5_path,
                mol=self._mol,
                n_orbitals=n_orbitals,
                n_alpha=n_alpha,
                n_beta=n_beta,
                is_cas=is_cas,
                n_core=n_core,
                fcidump_path=fcidump_path,
                benchmarks=self._benchmarks or None,
            )

        return ExportedSystem(
            fcidump_path=fcidump_path,
            hdf5_path=hdf5_path,
            n_orbitals=n_orbitals,
            n_alpha=n_alpha,
            n_beta=n_beta,
            is_cas=is_cas,
            n_core=n_core,
        )
