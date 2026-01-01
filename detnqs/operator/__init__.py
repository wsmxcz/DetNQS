# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Operator layer: Hamiltonian construction, host kernels, and energy functionals.

Public surface:
  - Minimal COO / space containers and Hamiltonian dataclasses
  - S-space and proxy Hamiltonian builders
  - JAX-compatible SpMV operators with automatic dtype dispatch
  - Unified energy / gradient functional with mode-agnostic interface
"""

from .hamiltonian import (
    DiagonalInfo,
    SSHamiltonian,
    ProxyHamiltonian,
    build_ss_hamiltonian,
    build_proxy_hamiltonian,
    build_effective_hamiltonian,
)

from .kernel import (
    SSContraction,
    ProxyContraction,
    build_ss_operator,
    build_proxy_operator,
)

from .functional import (
    make_energy_step,
)

__all__ = [
    # hamiltonian
    "DiagonalInfo",
    "SSHamiltonian",
    "ProxyHamiltonian",
    "build_ss_hamiltonian",
    "build_proxy_hamiltonian",
    "build_effective_hamiltonian",
    # kernel
    "SSContraction",
    "ProxyContraction",
    "build_ss_operator",
    "build_proxy_operator",
    # functional
    "make_energy_step",
]
