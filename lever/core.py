# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python bindings for LEVER C++ computational backend.

Exposes nanobind-wrapped C++ routines for determinant generation,
Hamiltonian construction, and integral management.

File: lever/core.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

# Import the compiled C++ extension module.
from . import _lever_cpp

# --- Integral Management ---

IntCtx = _lever_cpp.IntCtx
"""Integral context manager for electron repulsion integrals (ERIs)."""

# --- Determinant Generation ---

gen_fci_dets = _lever_cpp.gen_fci_dets
"""Generate full CI determinants for a given electron configuration."""

gen_excited_dets = _lever_cpp.gen_excited_dets
"""Generate excited determinants with optional heat-bath screening."""

# --- Hamiltonian Construction ---

get_ham_diag = _lever_cpp.get_ham_diag
"""Compute diagonal Hamiltonian matrix elements."""

get_ham_block = _lever_cpp.get_ham_block
"""Build a Hamiltonian block <bra|H|ket> between two arbitrary determinant sets."""

get_ham_conns_SS = _lever_cpp.get_ham_conns_SS
"""Build same-space Hamiltonian connectivity (S↔S). Deprecated in new workflows."""

get_ham_conns_SC = _lever_cpp.get_ham_conns_SC
"""Build cross-space Hamiltonian connectivity (S↔C). Deprecated in new workflows."""

get_ham_conns_ST = _lever_cpp.get_ham_conns_ST
"""Build target-space Hamiltonian connectivity (S↔T). Deprecated in new workflows."""

get_ham_conns_SSSC = _lever_cpp.get_ham_conns_SSSC
"""Build combined SS+SC Hamiltonian connectivity in a single pass."""


# Define what symbols are exported when a user does `from lever.core import *`.
__all__ = [
    "IntCtx",
    "gen_fci_dets",
    "gen_excited_dets",
    "get_ham_diag",
    "get_ham_block",      # Added the new function
    "get_ham_conns_SS",
    "get_ham_conns_SC",
    "get_ham_conns_ST",
    "get_ham_conns_SSSC",
]