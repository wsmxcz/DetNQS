# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Python bindings for DetNQS C++ computational backend.

Provides:
  - Determinant generation (FCI, excitations, connected sets)
  - Hamiltonian construction (variational, perturbative, effective)
  - Integral management and energy evaluation

File: lever/cpp_ext/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from . import _detnqs_cpp

# --- Integral Management ---

IntCtx = _detnqs_cpp.IntCtx
"""Context managing electron repulsion integrals (ERIs) and one-body terms."""

# --- Determinant Generation ---

gen_fci_dets = _detnqs_cpp.gen_fci_dets
"""Generate complete Fock space for given (N_e, S_z)."""

gen_connected_dets = _detnqs_cpp.gen_connected_dets
"""Generate single/double excitations from reference configurations."""

gen_perturbative_dets = _detnqs_cpp.gen_perturbative_dets
"""
Construct perturbative set P from variational set V.
Returns P = C \\ V, all configurations coupled to V via H but not in V.
"""

prepare_det_batch = _detnqs_cpp.prepare_det_batch
"""Prepare determinant batch features for network input (CPU operation)."""

# --- Hamiltonian Construction ---

get_ham_diag = _detnqs_cpp.get_ham_diag
"""Compute diagonal matrix elements <x|H|x> for given configurations."""

get_ham_vv = _detnqs_cpp.get_ham_vv
"""
Construct H_VV block (variational-variational subspace).
Does not discover connected space C.
"""

get_ham_block = _detnqs_cpp.get_ham_block
"""
Compute generic Hamiltonian blocks H_XY for predefined row/column spaces.
Used for custom subspace projections.
"""

get_ham_conn = _detnqs_cpp.get_ham_conn
"""
Build target-space Hamiltonian with static heat-bath screening.
Discovers connected set C from variational set V using integral thresholds.
"""

get_ham_conn_amp = _detnqs_cpp.get_ham_conn_amp
"""
Build target-space Hamiltonian with dynamic amplitude-based screening.
Prunes connections using current wavefunction amplitudes.
"""

get_ham_eff = _detnqs_cpp.get_ham_eff
"""
Construct effective Hamiltonian H_eff on V via perturbative downfolding.
Incorporates P-space effects into variational subspace.
"""

compute_variational_energy = _detnqs_cpp.compute_variational_energy
"""
Evaluate <Psi|H|Psi> for normalized coefficients on fixed basis.
Normalization must be handled in Python layer.
"""

compute_pt2 = _detnqs_cpp.compute_pt2
"""
Compute Epstein-Nesbet PT2 correction Delta E_PT2.
Requires electronic reference energy e_ref from variational optimization.
"""

__all__ = [
    "IntCtx",
    "gen_fci_dets",
    "gen_connected_dets",
    "gen_perturbative_dets",
    "prepare_det_batch",
    "get_ham_diag",
    "get_ham_vv",
    "get_ham_block",
    "get_ham_conn",
    "get_ham_conn_amp",
    "get_ham_eff",
    "compute_variational_energy",
    "compute_pt2",
]
