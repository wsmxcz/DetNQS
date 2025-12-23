# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing analysis tools for LEVER results.

Provides optional analysis functions:
  - Convergence statistics from JSON trace files
  - PT2 correction via C++ kernel (Epstein-Nesbet)
  - Variational energy evaluation on full T-space

Note: These are designed for delayed execution after optimization,
      independent of the runtime driver loop.

File: lever/analysis/metrics.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..space import DetSpace
from ..state import State
from ..system import MolecularSystem


def convergence_stats(trace_path: str | Path) -> dict[str, float]:
    """
    Compute convergence statistics from JSONL trace file.
    
    Args:
        trace_path: Path to .jsonl trace file
    
    Returns:
        Dict with mean_delta, max_delta, final_energy, n_outers, total_time
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return {}
    
    energies = []
    timestamps = []
    
    with trace_path.open("r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                energies.append(record["energy"])
                timestamps.append(record["timestamp"])
    
    if not energies:
        return {}
    
    if len(energies) == 1:
        return {
            "final_energy": float(energies[0]),
            "n_outers": 1,
            "total_time": float(timestamps[0]),
        }
    
    deltas = [abs(energies[i] - energies[i-1]) for i in range(1, len(energies))]
    
    return {
        "mean_delta": float(np.mean(deltas)),
        "max_delta": float(np.max(deltas)),
        "final_energy": float(energies[-1]),
        "n_outers": len(energies),
        "total_time": float(timestamps[-1]),
    }


def compute_pt2(
    state: State,
    detspace: DetSpace,
    system: MolecularSystem,
    *,
    screening: str = "heatbath",
    eps1: float = 1e-8,
) -> dict[str, float] | None:
    """
    Compute Epstein-Nesbet PT2 correction.
    
    Second-order perturbation theory on C-space:
        E_PT2 = sum_{a in C} |<a|H|psi_S>|^2 / (E_S - <a|H|a>)
    
    Args:
        state: Current variational state
        detspace: Determinant space with S_dets
        system: Molecular system
        screening: Heat-Bath mode
        eps1: Screening threshold
    
    Returns:
        Dict with e_var, e_pt2, e_total (Hartree) or None if unavailable
    """
    try:
        from .. import core
    except (ImportError, AttributeError):
        return None
    
    S_dets = np.asarray(detspace.S_dets, dtype=np.uint64)
    n_s = S_dets.shape[0]
    
    if n_s == 0:
        return None
    
    # Evaluate S-space wavefunction: psi_S = sign * exp(logabs)
    indices = jnp.arange(n_s, dtype=jnp.int32)
    sign_s, logabs_s = state.forward(indices)
    
    logabs_s = jnp.real(logabs_s)
    shift = jnp.max(logabs_s)
    psi_s = sign_s * jnp.exp(logabs_s - shift)
    psi_s = np.asarray(psi_s, dtype=np.complex128)
    
    norm_s = np.sqrt(np.vdot(psi_s, psi_s).real)
    if norm_s < 1e-14:
        return None
    psi_s = psi_s / norm_s
    
    # Call C++ PT2 kernel
    use_heatbath = screening == "heatbath"
    e_var_elec, e_pt2 = core.compute_pt2(
        S_dets,
        psi_s,
        system.int_ctx,
        system.n_orb,
        use_heatbath=use_heatbath,
        eps1=eps1,
    )
    
    e_var = float(e_var_elec) + system.e_nuc
    e_pt2 = float(e_pt2)
    
    return {
        "e_var": e_var,
        "e_pt2": e_pt2,
        "e_total": e_var + e_pt2,
    }


def compute_variational(
    state: State,
    detspace: DetSpace,
    system: MolecularSystem,
    *,
    use_heatbath: bool = True,
    eps1: float = 1e-6,
) -> dict[str, float] | None:
    """
    Compute full variational energy on T-space.
    
    Evaluates Rayleigh quotient:
        E_var = <psi_T|H|psi_T> / <psi_T|psi_T> + E_nuc
    
    where T = S union C is the full target space.
    
    Args:
        state: Current variational state
        detspace: Determinant space with T_dets
        system: Molecular system
        use_heatbath: Enable Heat-Bath screening
        eps1: Screening threshold
    
    Returns:
        Dict with e_var (Hartree) or None if unavailable
    """
    try:
        from .. import core
    except (ImportError, AttributeError):
        return None
    
    T_dets = np.asarray(detspace.T_dets, dtype=np.uint64)
    n_t = T_dets.shape[0]
    
    if n_t == 0:
        return None
    
    # Evaluate full T-space wavefunction
    sign_t, logabs_t = state.forward()
    logabs_t = jnp.real(logabs_t)
    
    shift = jnp.max(logabs_t)
    psi_t = sign_t * jnp.exp(logabs_t - shift)
    psi_t = np.asarray(psi_t, dtype=np.complex128)
    
    # Call C++ variational energy kernel
    e_elec, norm = core.compute_variational_energy(
        T_dets,
        psi_t,
        system.int_ctx,
        system.n_orb,
        use_heatbath=use_heatbath,
        eps1=eps1,
    )
    
    if norm < 1e-14:
        return None
    
    e_var = float(e_elec / norm) + system.e_nuc
    
    return {"e_var": e_var}


def extract_norms(trace_path: str | Path) -> dict[str, np.ndarray]:
    """
    Extract norm evolution from JSONL trace.
    
    Args:
        trace_path: Path to .jsonl trace file
    
    Returns:
        Dict with steps, norm_s, norm_c, frac_s, frac_c arrays
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return {}
    
    steps = []
    norm_s = []
    norm_c = []
    
    with trace_path.open("r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                steps.append(record["outer_step"])
                norm_s.append(record["norm_s"])
                norm_c.append(record["norm_c"])
    
    steps = np.array(steps, dtype=int)
    norm_s = np.array(norm_s)
    norm_c = np.array(norm_c)
    
    # Compute normalized fractions: f_S = ||psi_S||^2 / (||psi_S||^2 + ||psi_C||^2)
    norm_tot = norm_s + norm_c
    norm_tot = np.where(norm_tot > 1e-14, norm_tot, 1.0)
    frac_s = norm_s / norm_tot
    frac_c = norm_c / norm_tot
    
    return {
        "steps": steps,
        "norm_s": norm_s,
        "norm_c": norm_c,
        "frac_s": frac_s,
        "frac_c": frac_c,
    }


__all__ = [
    "convergence_stats",
    "compute_pt2",
    "compute_variational",
    "extract_norms",
]
