# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
High-level Python interface for generating LEVER Hamiltonian inputs.

Pipeline: SCF → natural orbitals → CAS selection → localization → export
FCIDUMP contains integral data; JSON provides system context.

File: lever/interface/builder.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import logging
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple, Union

import numpy as np
import pyscf

from . import metadata
from . import pyscf_driver as driver


# ============================================================================
# Public API
# ============================================================================

def export_system(
    atom: Union[str, list],
    basis: str,
    *,
    work_dir: Union[str, Path] = ".",
    name: str = "system",
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    symmetry: bool = False,
    scf: Optional[metadata.SCFConfig] = None,
    orbitals: Optional[metadata.OrbitalConfig] = None,
    benchmarks: Optional[Iterable[str]] = None,
) -> metadata.SystemMeta:
    """
    One-shot system builder: SCF → benchmarks → orbitals → export.

    Args:
        atom: Molecular geometry
        basis: Basis set identifier
        work_dir: Output directory
        name: System identifier
        charge: Total charge
        spin: Spin multiplicity 2S
        unit: Coordinate unit
        symmetry: Enable point group symmetry
        scf: SCF configuration
        orbitals: Orbital pipeline configuration
        benchmarks: Benchmark methods (e.g., ["hf", "fci", "ccsd"])

    Returns:
        SystemMeta describing active-space Hamiltonian
    """
    builder = MoleculeBuilder(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        unit=unit,
        symmetry=symmetry,
        work_dir=work_dir,
        name=name,
    )

    if scf is not None:
        builder.scf_config = scf
    if orbitals is not None:
        builder.orb_config = orbitals
    if benchmarks is not None:
        builder.benchmark_methods = set(benchmarks)

    return builder.run_all()


# ============================================================================
# Builder Implementation
# ============================================================================

class MoleculeBuilder:
    """
    Chainable PySCF orchestrator for LEVER Hamiltonian generation.

    Pipeline stages:
      1. SCF: Hartree-Fock with stability analysis
      2. Benchmarks: Post-HF methods (FCI, CCSD, etc.)
      3. Orbitals: Natural orbital transformation → CAS → localization
      4. Export: FCIDUMP + JSON metadata
    """

    def __init__(
        self,
        *,
        atom: Union[str, list, None] = None,
        basis: Optional[str] = None,
        charge: int = 0,
        spin: int = 0,
        unit: str = "Angstrom",
        symmetry: bool = False,
        work_dir: Union[str, Path] = ".",
        name: str = "system",
        info: Optional[metadata.MoleculeInfo] = None,
    ):
        if info is None:
            if atom is None or basis is None:
                raise ValueError("Either info or (atom, basis) must be provided")
            info = metadata.MoleculeInfo(
                atom=atom,
                basis=basis,
                charge=charge,
                spin=spin,
                symmetry=symmetry,
                unit=unit,
            )

        self.info: metadata.MoleculeInfo = info
        self.work_dir = Path(work_dir)
        self.name = name

        # Pipeline configurations
        self.scf_config = metadata.SCFConfig()
        self.orb_config = metadata.OrbitalConfig()
        self.benchmark_methods: Set[str] = {"hf"}

        # Computation cache
        self._scf_result: Optional[driver.ScfResult] = None
        self._benchmarks: Optional[Dict[str, metadata.BenchmarkItem]] = None
        self._mo_coeff: Optional[np.ndarray] = None
        self._mo_occ: Optional[np.ndarray] = None
        self._hf_occ_sorted: Optional[np.ndarray] = None
        self._dims: Optional[Tuple[int, int, int]] = None

        # Logging
        self.log_path = self.work_dir / "interface.log"
        self._header_logged = False
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Initialize file logger."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        logger_name = f"LEVER_Interface_{self.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_path, mode="a", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(fh)

    def _log_header(self) -> None:
        """Log system header once."""
        if self._header_logged:
            return
        self._header_logged = True

        log = self.logger.info
        log(f"--- LEVER Interface: {datetime.now().isoformat()} ---")
        log(f"System: {self.name}")
        log(f"Atom: {self.info.atom}")
        log(f"Basis: {self.info.basis} | Q={self.info.charge} S={self.info.spin}")

    def run_scf(
        self,
        cfg: Optional[metadata.SCFConfig] = None,
        **overrides,
    ) -> MoleculeBuilder:
        """Execute SCF with stability analysis."""
        log = self.logger.info

        if cfg is None:
            cfg = self.scf_config
        if overrides:
            cfg = cfg.model_copy(update=overrides)
        self.scf_config = cfg

        res = driver.run_scf(self.info, cfg, log_path=self.log_path)
        self._scf_result = res

        self._log_header()
        log(f"\n[Step 1] SCF ({cfg.type})")
        log(f"  Energy:    {res.mf.e_tot:.8f} Ha")
        log(f"  Converged: {res.converged} (Cycles: {res.n_cycle})")
        log(f"  <S^2>:     {res.s2:.4f}")

        if not res.converged:
            log("  WARNING: SCF did NOT converge")

        self._mo_coeff = res.mf.mo_coeff
        self._mo_occ = res.mf.mo_occ
        self._hf_occ_sorted = res.mf.mo_occ.copy()
        self._dims = None

        return self

    def run_benchmarks(
        self,
        methods: Optional[Iterable[str]] = None,
    ) -> MoleculeBuilder:
        """Compute post-HF benchmarks (FCI, CCSD, etc.)."""
        log = self.logger.info

        if self._scf_result is None:
            self.run_scf()

        res = self._scf_result
        assert res is not None

        if methods is not None:
            self.benchmark_methods = set(methods)

        log(f"\n[Step 2] Benchmarks: {sorted(self.benchmark_methods)}")

        if not res.converged:
            log("  (Skipping correlated methods - SCF failed)")
            bench_res = {
                "hf": metadata.BenchmarkItem(
                    energy=float(res.mf.e_tot),
                    status="failed",
                )
            }
        else:
            bench_res = driver.run_benchmarks(res, self.benchmark_methods)
            for method, result in bench_res.items():
                if result.status == "ok" and result.energy is not None:
                    val = f"{result.energy:.6f}"
                else:
                    val = result.status
                log(f"  {method.upper()}: {val}")

        self._benchmarks = bench_res
        return self

    def run_orbitals(
        self,
        cfg: Optional[metadata.OrbitalConfig] = None,
        **overrides,
    ) -> MoleculeBuilder:
        """
        Execute orbital pipeline: natural → CAS → localization.
        
        Natural orbitals: Diagonalize density matrix for improved convergence
        CAS selection: Partition orbitals into core/active/virtual spaces
        Localization: Apply Boys, Pipek-Mezey, or other schemes
        """
        log = self.logger.info

        if self._scf_result is None:
            self.run_scf()

        res = self._scf_result
        assert res is not None

        if cfg is None:
            cfg = self.orb_config
        if overrides:
            cfg = cfg.model_copy(update=overrides)
        self.orb_config = cfg

        if (cfg.active_orb > 0) ^ (cfg.active_elec > 0):
            raise ValueError("Active space requires both active_orb and active_elec")

        log("\n[Step 3] Orbital Pipeline")

        c_curr = res.mf.mo_coeff
        occ_curr = res.mf.mo_occ
        hf_occ_curr = res.mf.mo_occ.copy()

        # Natural orbitals: diagonalize 1-RDM for improved basis
        nat_type = cfg.natural_type
        if nat_type != "none":
            log(f"  > Natural Orbitals ({nat_type})")
            c_curr, occ_curr, hf_occ_curr = driver.make_natural(
                res, nat_type, hf_occ_curr
            )

        # CAS partitioning: select active space by occupation
        n_elec = cfg.active_elec
        n_orb = cfg.active_orb

        if n_orb > 0:
            log(f"  > Active Space: ({n_elec}e, {n_orb}o)")

            if (n_elec + res.mol.spin) % 2 != 0:
                raise ValueError(
                    f"Active electrons {n_elec} and spin {res.mol.spin} incompatible"
                )

            c_curr, occ_curr, hf_occ_curr, dims = driver.make_active(
                c_curr,
                occ_curr,
                hf_occ_curr,
                n_elec,
                n_orb,
                res.mol.nelectron,
            )
            n_core, n_act, n_virt = dims
            log(f"    Core={n_core} Active={n_act} Virt={n_virt}")
        else:
            # Full-space calculation
            n_core, n_act = 0, c_curr.shape[1]
            n_virt = 0
            idx = np.argsort(occ_curr)[::-1]
            c_curr = c_curr[:, idx]
            occ_curr = occ_curr[idx]
            hf_occ_curr = hf_occ_curr[idx]

        # Localization: transform orbitals for spatial locality
        loc_method = cfg.local_method
        if loc_method != "none":
            loc_window = cfg.local_window
            log(f"  > Localizing: {loc_method} (window={loc_window})")

            s_core = slice(0, n_core)
            s_act = slice(n_core, n_core + n_act)
            s_virt = slice(n_core + n_act, n_core + n_act + n_virt)

            c_curr = driver.localize_orbs(
                res.mol,
                c_curr,
                loc_method,
                loc_window,
                (s_core, s_act, s_virt),
                cfg.local_virtual,
            )

        self._mo_coeff = c_curr
        self._mo_occ = occ_curr
        self._hf_occ_sorted = hf_occ_curr
        self._dims = (n_core, n_act, n_virt)

        return self

    def export(self) -> metadata.SystemMeta:
        """
        Export FCIDUMP + system metadata.
        
        FCIDUMP format: Standard integral dump for quantum chemistry codes
        Metadata: System info, orbital space, SCF results, benchmarks
        """
        log = self.logger.info

        if self._scf_result is None:
            self.run_scf()
        if self._dims is None or self._mo_coeff is None or self._hf_occ_sorted is None:
            self.run_orbitals()

        res = self._scf_result
        assert res is not None
        c_curr = self._mo_coeff
        occ_curr = self._mo_occ
        hf_occ_curr = self._hf_occ_sorted
        n_core, n_act, n_virt = self._dims  # type: ignore[misc]

        if self._benchmarks is None and self.benchmark_methods:
            self.run_benchmarks(self.benchmark_methods)

        bench_res = self._benchmarks or {}

        # Write FCIDUMP: two-electron integrals in chemist's notation
        fcidump_name = "hamiltonian.fcidump"
        log(f"\n[Step 4] Exporting {fcidump_name}")
        driver.write_fcidump(
            res.mol,
            res.mf,
            c_curr,
            str(self.work_dir / fcidump_name),
            n_core,
            n_act,
        )

        # Reference state: bitstring representation of HF occupation
        final_hf_occ = hf_occ_curr[n_core : n_core + n_act]
        alpha_bits, beta_bits = driver.occ_to_bitstring(final_hf_occ)

        if n_core > 0:
            n_elec_active = self.orb_config.active_elec
            n_alpha = (n_elec_active + res.mol.spin) // 2
            n_beta = n_elec_active - n_alpha
        else:
            ne = res.mol.nelectron
            n_alpha = (ne + res.mol.spin) // 2
            n_beta = ne - n_alpha

        # Geometry in Angstrom
        atom_meta = [
            metadata.AtomMeta(
                symbol=res.mol.atom_symbol(i),
                coords=list(coords),
            )
            for i, coords in enumerate(res.mol.atom_coords(unit="Angstrom"))
        ]

        # SCF type detection
        import pyscf.scf
        scf_type = "ROHF" if isinstance(res.mf, pyscf.scf.rohf.ROHF) else "RHF"

        # LEVER version
        try:
            lever_version = importlib_metadata.version("lever")
        except importlib_metadata.PackageNotFoundError:
            lever_version = "unknown"

        # Assemble metadata
        system_info = metadata.SystemInfo(
            name=self.name,
            charge=self.info.charge,
            spin=self.info.spin,
            n_electrons=res.mol.nelectron,
            basis=self.info.basis,
            unit=self.info.unit,
            geometry=atom_meta,
        )

        orbital_space = metadata.OrbitalSpace(
            n_orb=n_act,
            n_alpha=int(n_alpha),
            n_beta=int(n_beta),
            n_core=int(n_core),
            is_cas=bool(n_core > 0),
            active_elec=self.orb_config.active_elec,
            active_orb=self.orb_config.active_orb,
        )

        scf_meta = metadata.ScfMeta(
            type=scf_type,
            energy=float(res.mf.e_tot),
            converged=res.converged,
            n_cycle=res.n_cycle,
            spin_s2=res.s2,
        )

        loc_meta = metadata.LocalizationMeta(
            method=self.orb_config.local_method,
            window=self.orb_config.local_window,
            local_virtual=self.orb_config.local_virtual,
        )

        pipeline_meta = metadata.PipelineMeta(
            natural_type=self.orb_config.natural_type,
            localization=loc_meta,
        )

        ref_state = metadata.ReferenceState(
            alpha_bits=int(alpha_bits),
            beta_bits=int(beta_bits),
            occ_vector=[int(x) for x in final_hf_occ],
        )

        files_meta = metadata.SystemFiles(
            fcidump=fcidump_name,
            log=self.log_path.name,
        )

        sys_meta = metadata.SystemMeta(
            lever_version=lever_version,
            pyscf_version=pyscf.__version__,
            system=system_info,
            orbital_space=orbital_space,
            scf=scf_meta,
            pipeline=pipeline_meta,
            reference_state=ref_state,
            benchmarks=bench_res,
            files=files_meta,
        )

        # Save metadata to JSON
        json_path = self.work_dir / "system.json"
        sys_meta.save(json_path)
        log(f"Metadata saved to {json_path.name}")

        # Cleanup logger
        for h in list(self.logger.handlers):
            h.close()
            self.logger.removeHandler(h)

        return sys_meta

    def run_all(self) -> metadata.SystemMeta:
        """Execute full pipeline: SCF → benchmarks → orbitals → export."""
        self.run_scf()
        if self.benchmark_methods:
            self.run_benchmarks(self.benchmark_methods)
        self.run_orbitals()
        return self.export()

    def run(self) -> metadata.SystemMeta:
        """Backward-compatible alias for run_all()."""
        return self.run_all()


__all__ = [
    "export_system",
    "MoleculeBuilder",
]
