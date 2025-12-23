# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Molecular system from FCIDUMP integrals.

Parses orbital parameters, manages integral context with Heat-Bath screening,
and encodes Hartree-Fock determinant in 64-bit bitstrings.

File: lever/system.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import _lever_cpp  # nanobind bridge to C++ integral backend

_FCIDUMP_REQUIRED_KEYS = ("NORB", "NELEC")
_FCIDUMP_OPTIONAL_KEYS = ("MS2",)


def parse_fcidump_header(path: Path) -> dict[str, int]:
    """
    Extract orbital parameters from FCIDUMP &FCI namelist.

    Follows Psi4/PySCF convention: read KEY=VALUE integers between
    '&FCI' and '&END'/'/' delimiters.

    Args:
        path: FCIDUMP file path

    Returns:
        Dict with NORB, NELEC (required) and MS2 (default=0)

    Raises:
        FileNotFoundError: Missing file
        ValueError: Missing &FCI namelist or required keys
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FCIDUMP not found: {path}")

    header_lines = []
    inside = False

    with path.open("r") as f:
        for line in f:
            upper = line.upper()
            if not inside and "&FCI" in upper:
                inside = True

            if inside:
                header_lines.append(upper.strip())
                if "&END" in upper or "/" in upper:
                    break

    if not inside or not header_lines:
        raise ValueError(f"No &FCI namelist in FCIDUMP: {path}")

    # Extract KEY=INT pairs via regex
    header_text = " ".join(header_lines)
    params = {
        key.upper(): int(val)
        for key, val in re.findall(r"(\w+)\s*=\s*([+-]?\d+)", header_text)
    }

    missing = [k for k in _FCIDUMP_REQUIRED_KEYS if k not in params]
    if missing:
        raise ValueError(f"Missing required keys {missing} in {path}")

    params.setdefault("MS2", 0)  # Default to closed-shell singlet
    return params


@dataclass(eq=False)
class MolecularSystem:
    """
    Many-electron system defined by FCIDUMP integrals.

    Attributes:
        fcidump_path: Path to FCIDUMP file
        hb_threshold: Heat-Bath noise floor |<ij||ab>|_min
        build_heat_bath: Pre-build Heat-Bath table at construction
        n_orb: Number of spatial orbitals
        n_elec: Total electron count
        ms2: Twice the spin projection M_s (2·M_s = n_alpha - n_beta)
        e_nuc: Nuclear repulsion energy

    Spin configuration:
        n_alpha = (N_elec + M_s2) / 2
        n_beta  = (N_elec - M_s2) / 2 = N_elec - n_alpha
    """

    fcidump_path: Path
    hb_threshold: float = 1e-14
    build_heat_bath: bool = True

    # Auto-populated from FCIDUMP header
    n_orb: int = field(init=False)
    n_elec: int = field(init=False)
    ms2: int = field(init=False)
    e_nuc: float = field(init=False)

    _int_ctx: Optional[_lever_cpp.IntCtx] = field(
        init=False, default=None, repr=False
    )

    @classmethod
    def from_fcidump(
        cls,
        path: str | Path,
        hb_threshold: float = 1e-14,
        build_heat_bath: bool = True,
    ) -> MolecularSystem:
        """
        Construct system from FCIDUMP file.

        Args:
            path: FCIDUMP file path
            hb_threshold: Heat-Bath cutoff for |<ij||ab>|
            build_heat_bath: Build screening table immediately

        Returns:
            Configured molecular system

        Raises:
            ValueError: Invalid N_orb (must be in [1,64]) or spin parity
        """
        path = Path(path)
        header = parse_fcidump_header(path)

        n_orb = header["NORB"]
        n_elec = header["NELEC"]
        ms2 = header["MS2"]

        # Validate 64-bit determinant limits
        if not (1 <= n_orb <= 64):
            raise ValueError(f"NORB={n_orb} outside [1,64] range")
        if n_elec < 0:
            raise ValueError(f"NELEC={n_elec} must be non-negative")
        if (n_elec + ms2) % 2 != 0:
            raise ValueError(f"NELEC={n_elec} ± MS2={ms2} parity mismatch")

        # Initialize C++ integral backend
        int_ctx = _lever_cpp.IntCtx(str(path), int(n_orb))
        if build_heat_bath:
            int_ctx.hb_prepare(hb_threshold)

        instance = cls(
            fcidump_path=path,
            hb_threshold=hb_threshold,
            build_heat_bath=build_heat_bath,
        )
        instance._int_ctx = int_ctx
        instance.n_orb = int(n_orb)
        instance.n_elec = int(n_elec)
        instance.ms2 = int(ms2)
        instance.e_nuc = float(int_ctx.get_e_nuc())

        return instance

    @property
    def int_ctx(self) -> _lever_cpp.IntCtx:
        """Lazy-initialized C++ integral context with optional Heat-Bath table."""
        if self._int_ctx is None:
            self._int_ctx = _lever_cpp.IntCtx(
                str(self.fcidump_path), int(self.n_orb)
            )
            if self.build_heat_bath:
                self._int_ctx.hb_prepare(self.hb_threshold)
        return self._int_ctx

    @property
    def n_alpha(self) -> int:
        """Alpha electron count: n_alpha = (N_elec + M_s2) / 2"""
        return (self.n_elec + self.ms2) // 2

    @property
    def n_beta(self) -> int:
        """Beta electron count: n_beta = N_elec - n_alpha"""
        return self.n_elec - self.n_alpha

    @property
    def n_so(self) -> int:
        """Spin-orbital count: 2 × N_orb"""
        return 2 * self.n_orb

    def prepare_heat_bath(self, threshold: float | None = None) -> None:
        """
        Build or rebuild Heat-Bath screening table.

        Args:
            threshold: New cutoff (updates hb_threshold if provided)
        """
        if threshold is not None:
            self.hb_threshold = float(threshold)

        self.int_ctx.hb_prepare(self.hb_threshold)
        self.build_heat_bath = True

    def clear_heat_bath(self) -> None:
        """Release Heat-Bath table memory."""
        if self._int_ctx is not None:
            self._int_ctx.hb_clear()
        self.build_heat_bath = False

    def hf_determinant(self) -> np.ndarray:
        """
        Hartree-Fock determinant as 64-bit bitstring pair.

        Assumes lowest n_alpha (n_beta) orbitals occupied for alpha (beta) spin.

        Returns:
            Array [alpha_word, beta_word] of dtype uint64
        """
        alpha_word = np.uint64((1 << self.n_alpha) - 1)
        beta_word = np.uint64((1 << self.n_beta) - 1)
        return np.asarray([alpha_word, beta_word], dtype=np.uint64)

__all__ = ["MolecularSystem", "parse_fcidump_header"]
