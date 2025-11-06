# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
FCI energy computation for benchmarking LEVER results.

Provides exact ground-state energy via full CI diagonalization
for comparison with variational calculations.

File: lever/analysis/fci.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import core, engine
from .evaluator import EnergyEvaluator

if TYPE_CHECKING:
    from ..config import LeverConfig


def compute_fci_energy(
    config: LeverConfig,
    int_ctx: core.IntCtx
) -> float:
    """
    Compute exact FCI energy via full diagonalization.
    
    Generates complete FCI determinant basis and builds full Hamiltonian
    matrix for exact ground-state energy calculation.
    
    Args:
        config: LEVER configuration with system parameters
        int_ctx: Integral provider for Hamiltonian construction
        
    Returns:
        FCI ground-state energy (including nuclear repulsion)
    """
    sys = config.system
    
    # Generate full FCI determinant basis
    fci_dets = core.gen_fci_dets(sys.n_orbitals, sys.n_alpha, sys.n_beta)
    
    # Build full Hamiltonian matrix
    ham_fci, _ = engine.hamiltonian.get_ham_ss(
        S_dets=fci_dets,
        int_ctx=int_ctx,
        n_orbitals=sys.n_orbitals
    )
    
    # Exact diagonalization
    evaluator = EnergyEvaluator(
        int_ctx, 
        sys.n_orbitals, 
        int_ctx.get_e_nuc()
    )
    return evaluator.diagonalize(ham_fci)


__all__ = ["compute_fci_energy"]
