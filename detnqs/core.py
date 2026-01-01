# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Python bindings for detnqs C++ computational backend.

Provides determinant generation, Hamiltonian construction, and integral management.
"""

from __future__ import annotations

from . import _detnqs_cpp

# --- Integral Management ---

IntCtx = _detnqs_cpp.IntCtx
"""Integral context managing electron repulsion integrals (ERIs)."""

# --- Determinant Generation ---

gen_fci_dets = _detnqs_cpp.gen_fci_dets
"""Generate full CI determinant space."""

gen_excited_dets = _detnqs_cpp.gen_excited_dets
"""Generate single and double excitations from reference determinants."""

gen_complement_dets = _detnqs_cpp.gen_complement_dets
"""Generate complement space from reference determinants."""

prepare_det_batch = _detnqs_cpp.prepare_det_batch
"""Prepare determinant batch features on CPU."""

# --- Hamiltonian Construction ---

get_ham_diag = _detnqs_cpp.get_ham_diag
"""Compute diagonal Hamiltonian elements <D|H|D>."""

get_ham_ss = _detnqs_cpp.get_ham_ss
"""Compute H_SS block only (no C-space discovery)."""

get_ham_block = _detnqs_cpp.get_ham_block
"""Compute Hamiltonian blocks for predefined spaces."""

get_ham_conn = _detnqs_cpp.get_ham_conn
"""Build Hamiltonian with static screening (heat-bath on integrals)."""

get_ham_conn_amp = _detnqs_cpp.get_ham_conn_amp
"""Build Hamiltonian with dynamic amplitude screening."""

get_ham_eff = _detnqs_cpp.get_ham_eff
"""Assemble effective Hamiltonian via perturbative correction."""

compute_variational_energy = _detnqs_cpp.compute_variational_energy
"""Compute <Psi|H|Psi> on a fixed basis. Coeffs must be normalized in Python."""

compute_pt2 = _detnqs_cpp.compute_pt2
"""Compute EN-PT2 correction only. Requires e_ref from optimizer (electronic energy)."""

__all__ = [
    "IntCtx",
    "gen_fci_dets",
    "gen_excited_dets",
    "get_ham_diag",
    "get_ham_ss",
    "get_ham_block",
    "get_ham_conn",
    "get_ham_conn_amp",
    "get_ham_eff",
    "get_local_conn",
    "get_local_connections",
    "compute_variational_energy",
    "compute_pt2",
]