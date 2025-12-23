# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing analysis tools for LEVER results.

Provides optional analysis functions:
  - Convergence statistics from energy traces
  - PT2 correction (if C++ kernel available)
  - Variational energy evaluation

Note: Norm decomposition is now computed automatically in driver.

File: lever/analysis/metrics.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..driver import DriverResult
from ..system import MolecularSystem
from .trace import Trace


def convergence_stats(trace: Trace) -> dict[str, float]:
    """
    Compute convergence statistics from energy trace.
  
    Returns:
        - mean_delta: Average energy change between consecutive outers
        - max_delta: Maximum energy change
        - final_energy: Last recorded energy
        - n_outers: Number of outer loops executed
    """
    if trace.n_points == 0:
        return {}
  
    energies = trace.last_per_outer()
    if len(energies) < 2:
        return {
            "final_energy": float(trace.energies[-1]),
            "n_outers": trace.n_points,
        }
  
    sorted_outers = sorted(energies.keys())
    deltas = [
        abs(energies[sorted_outers[i]] - energies[sorted_outers[i-1]])
        for i in range(1, len(sorted_outers))
    ]
  
    return {
        "mean_delta": float(np.mean(deltas)),
        "max_delta": float(np.max(deltas)),
        "final_energy": float(energies[sorted_outers[-1]]),
        "n_outers": len(energies),
    }


def compute_pt2(
    result: DriverResult,
    system: MolecularSystem,
    *,
    screening: str = "heatbath",
    eps1: float = 1e-8,
) -> dict[str, float] | None:
    """
    Compute Epstein-Nesbet PT2 correction.
  
    Second-order perturbation theory:
        E_PT2 = sum_{a in C} |<a|H|ψ_S>|² / (E_S - <a|H|a>)
  
    Returns:
        Dict with e_var, e_pt2, e_total or None if kernel unavailable
    """
    try:
        from .. import core
      
        # Extract normalized S-space wavefunction
        n_s = result.detspace.size_S
        S_dets = np.asarray(result.detspace.S_dets, dtype=np.uint64)
      
        if S_dets.shape[0] == 0:
            return None
      
        indices = jnp.arange(n_s, dtype=jnp.int32)
        sign_s, logabs_s = result.state.forward(indices)
      
        logabs_s = jnp.real(logabs_s)
        shift = jnp.max(logabs_s)
        psi_s = sign_s * jnp.exp(logabs_s - shift)
        psi_s = np.asarray(psi_s, dtype=np.complex128)
      
        norm_s = np.sqrt(np.vdot(psi_s, psi_s).real)
        if norm_s < 1e-14:
            return None
        psi_s = psi_s / norm_s
      
        # Call C++ kernel
        e_var_elec, e_pt2 = core.compute_pt2(
            S_dets,
            psi_s,
            system.int_ctx,
            system.n_orb,
            use_heatbath=True,
            eps1=eps1,
        )
      
        e_var = float(e_var_elec) + system.e_nuc
        e_pt2 = float(e_pt2)
      
        return {
            "e_var": e_var,
            "e_pt2": e_pt2,
            "e_total": e_var + e_pt2,
        }
  
    except (ImportError, AttributeError):
        return None


def compute_variational(
    result: DriverResult,
    system: MolecularSystem,
) -> dict[str, float] | None:
    """
    Compute full variational energy on T-space.
  
    Evaluates: E_var = <ψ_T|H|ψ_T> / <ψ_T|ψ_T> + E_nuc
  
    Returns:
        Dict with e_var or None if kernel unavailable
    """
    try:
        from .. import core
      
        T_dets = np.asarray(result.detspace.T_dets, dtype=np.uint64)
        if T_dets.shape[0] == 0:
            return None
      
        # Evaluate full T-space wavefunction
        sign_t, logabs_t = result.state.forward()
        logabs_t = jnp.real(logabs_t)
      
        shift = jnp.max(logabs_t)
        psi_t = sign_t * jnp.exp(logabs_t - shift)
        psi_t = np.asarray(psi_t, dtype=np.complex128)
      
        # Call C++ kernel
        e_elec, norm = core.compute_variational_energy(
            T_dets,
            psi_t,
            system.int_ctx,
            system.n_orb,
            use_heatbath=True,
            eps1=1e-6,
        )
      
        if norm < 1e-14:
            return None
      
        e_var = float(e_elec / norm) + system.e_nuc
      
        return {"e_var": e_var}
  
    except (ImportError, AttributeError):
        return None


__all__ = [
    "convergence_stats",
    "compute_pt2",
    "compute_variational",
]
