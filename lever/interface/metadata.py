# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Molecular system and calculation metadata schemas.

Defines system-level data structures for quantum chemistry calculations,
separating physical definitions from pipeline implementation details.

File: lever/interface/metadata.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# User Configuration Models
# ============================================================================

class MoleculeInfo(BaseModel):
    """Physical system definition for PySCF initialization."""
    atom: Any = Field(..., description="PySCF geometry string or array")
    basis: str = Field(..., description="Basis set (e.g., 'sto-3g', '6-31g')")
    charge: int = 0
    spin: int = 0  # 2S = n_α - n_β
    symmetry: bool = False
    unit: Literal["Angstrom", "Bohr"] = "Angstrom"


class SCFConfig(BaseModel):
    """Self-consistent field solver parameters."""
    type: Literal["auto", "rhf", "rohf"] = "auto"
    tol: float = 1e-10          # Energy convergence: |ΔE| < tol
    grad_tol: float = 1e-8      # Gradient norm: ‖g‖ < grad_tol
    max_cycle: int = 50
    stability: Literal["none", "internal", "external", "both"] = "both"


class OrbitalConfig(BaseModel):
    """
    Orbital transformation pipeline.

    Pipeline order: Natural orbitals → CAS selection → Localization
    """
    natural_type: Literal["none", "mp2", "cisd", "ccsd"] = "none"

    # Active space: CAS(n_e, n_orb)
    active_elec: int = 0
    active_orb: int = 0

    # Localization scheme
    local_method: Literal["none", "boys", "pipek", "ibo", "edmiston"] = "none"
    local_window: Literal["active", "full"] = "active"
    local_virtual: bool = False


# ============================================================================
# System-level Metadata
# ============================================================================

class AtomMeta(BaseModel):
    """Atomic geometry entry (coordinates in Angstroms)."""
    symbol: str
    coords: List[float]  # [x, y, z]


class BenchmarkItem(BaseModel):
    """Reference calculation result."""
    energy: Optional[float] = None
    status: Literal["ok", "failed", "skipped"] = "skipped"


class SystemInfo(BaseModel):
    """Physical system identity and configuration."""
    name: str
    charge: int
    spin: int          # 2S
    n_electrons: int
    basis: str
    unit: Literal["Angstrom", "Bohr"]
    geometry: List[AtomMeta] = Field(default_factory=list)


class OrbitalSpace(BaseModel):
    """Hamiltonian dimensions and active space definition."""
    n_orb: int         # Spatial orbitals in FCIDUMP
    n_alpha: int       # α electrons in active space
    n_beta: int        # β electrons in active space
    n_core: int = 0    # Frozen core orbitals
    is_cas: bool = False

    # Active space: CAS(n_e, n_orb)
    active_elec: int = 0
    active_orb: int = 0


class ScfMeta(BaseModel):
    """SCF calculation diagnostics."""
    type: str
    energy: float
    converged: bool
    n_cycle: int
    spin_s2: float  # ⟨Ŝ²⟩


class LocalizationMeta(BaseModel):
    """Orbital localization configuration."""
    method: str
    window: str
    local_virtual: bool


class PipelineMeta(BaseModel):
    """Orbital transformation pipeline description."""
    natural_type: str
    localization: LocalizationMeta


class ReferenceState(BaseModel):
    """
    Reference determinant in active space.

    Determinants encoded as uint64 bitstrings: bit i = 1 ⇔ orbital i occupied.
    """
    alpha_bits: int
    beta_bits: int
    occ_vector: List[int]


class SystemFiles(BaseModel):
    """System artifact file paths."""
    fcidump: str = "hamiltonian.fcidump"
    log: str = "interface.log"


class SystemMeta(BaseModel):
    """
    Complete system provenance metadata.

    FCIDUMP contains full Hamiltonian; JSON provides context and diagnostics.
    """
    model_config = ConfigDict(extra="ignore")

    # Schema and software versions
    lever_version: str
    pyscf_version: str

    # System structure
    system: SystemInfo
    orbital_space: OrbitalSpace
    scf: ScfMeta
    pipeline: PipelineMeta
    reference_state: ReferenceState

    # Reference data and file paths
    benchmarks: Dict[str, BenchmarkItem] = Field(default_factory=dict)
    files: SystemFiles = Field(default_factory=SystemFiles)

    # I/O operations
    def save(self, path: Path | str) -> None:
        """Write metadata to JSON file."""
        with open(Path(path), "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path | str) -> SystemMeta:
        """Load metadata from JSON file."""
        with open(Path(path), "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


# ============================================================================
# Convenience Loaders
# ============================================================================

def load_initial_det(path: Path | str) -> np.ndarray:
    """
    Extract reference determinant as bitstrings.

    Returns:
        array([[alpha_bits, beta_bits]], dtype=uint64)
    """
    meta = SystemMeta.load(path)
    return np.array(
        [[meta.reference_state.alpha_bits, meta.reference_state.beta_bits]],
        dtype=np.uint64
    )


def load_benchmarks(path: Path | str) -> Dict[str, float]:
    """Extract converged benchmark energies."""
    meta = SystemMeta.load(path)
    return {
        k: v.energy
        for k, v in meta.benchmarks.items()
        if v.status == "ok" and v.energy is not None
    }


__all__ = [
    # Input models
    "MoleculeInfo",
    "SCFConfig",
    "OrbitalConfig",
    # System-level metadata
    "AtomMeta",
    "BenchmarkItem",
    "SystemInfo",
    "OrbitalSpace",
    "ScfMeta",
    "LocalizationMeta",
    "PipelineMeta",
    "ReferenceState",
    "SystemFiles",
    "SystemMeta",
    # Helpers
    "load_initial_det",
    "load_benchmarks",
]
