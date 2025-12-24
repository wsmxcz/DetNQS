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
    e_ref_elec: float,
    screening: str = "heatbath",
    eps1: float = 1e-6,
) -> float | None:
    """
    Compute Epstein-Nesbet PT2 correction only (electronic part).

    Notes:
      - e_ref_elec must come from the LEVER optimization result (electronic energy).
      - Python guarantees psi_S is normalized before calling the C++ kernel.
      - C++ does NOT recompute e_var and does NOT do any norm handling.
    """
    try:
        from .. import core
    except (ImportError, AttributeError):
        return None

    S_dets = np.asarray(detspace.S_dets, dtype=np.uint64)
    n_s = S_dets.shape[0]
    if n_s == 0:
        return None

    # Evaluate unnormalized psi_S = sign * exp(logabs)
    indices = jnp.arange(n_s, dtype=jnp.int32)
    sign_s, logabs_s = state.forward(indices)
    sign_s, logabs_s = jnp.real(sign_s), jnp.real(logabs_s)

    # Normalize on Python side (domain-normalized)
    shift = jnp.max(logabs_s)
    psi_s = sign_s * jnp.exp(logabs_s - shift)
    psi_s = np.asarray(psi_s, dtype=np.float64)
    psi_s /= np.linalg.norm(psi_s)

    use_heatbath = (screening == "heatbath")

    # C++ returns ONLY e_pt2 (electronic correction)
    e_pt2 = core.compute_pt2(
        S_dets,
        psi_s,
        system.int_ctx,
        system.n_orb,
        e_ref_elec,              # <-- NEW: use optimized e_ref, not computed inside C++
        use_heatbath=use_heatbath,
        eps1=eps1,
    )

    return float(e_pt2)


def compute_variational(
    state: State,
    detspace: DetSpace,
    system: MolecularSystem,
    *,
    use_heatbath: bool = True,
    eps1: float = 1e-6,
) -> dict[str, float] | None:
    """
    Compute variational energies on S-space and optionally T-space.

    Workflow:
      1. Always compute E_var_S = <ψ_S|H_SS|ψ_S> on S-space
      2. If C-space exists, compute E_var_T = <ψ_T|H|ψ_T> on full T-space

    Args:
        state: Optimized network state
        detspace: Determinant space (may or may not have C)
        system: Molecular system
        use_heatbath: Enable Heat-Bath screening in C++ kernel
        eps1: Screening threshold

    Returns:
        Dict with 'e_var_s' (always) and 'e_var_t' (if C-space exists)

    Notes:
        - Python normalizes wavefunctions before C++ kernel
        - C++ returns only <ψ|H|ψ>, no norm handling
        - Energies include nuclear repulsion
    """
    try:
        from .. import core
    except (ImportError, AttributeError):
        return None

    result = {}

    # -------------------------------------------------------------------------
    # Always compute E_var_S on S-space
    # -------------------------------------------------------------------------
    S_dets = np.asarray(detspace.S_dets, dtype=np.uint64)
    if S_dets.shape[0] == 0:
        return None

    indices_s = jnp.arange(detspace.size_S, dtype=jnp.int32)
    sign_s, logabs_s = state.forward(indices_s)
    sign_s, logabs_s = jnp.real(sign_s), jnp.real(logabs_s)

    # Normalize on Python side
    shift_s = jnp.max(logabs_s) if logabs_s.size > 0 else 0.0
    psi_s = sign_s * jnp.exp(logabs_s - shift_s)
    psi_s = np.asarray(psi_s, dtype=np.float64)
    psi_s /= np.linalg.norm(psi_s)

    # C++ computes <ψ_S|H_SS|ψ_S>
    e_elec_s = core.compute_variational_energy(
        S_dets,
        psi_s,
        system.int_ctx,
        system.n_orb,
        use_heatbath=use_heatbath,
        eps1=eps1,
    )
    result["e_var_s"] = float(e_elec_s) + system.e_nuc

    # -------------------------------------------------------------------------
    # Optionally compute E_var_T on full T-space if C exists
    # -------------------------------------------------------------------------
    if detspace.has_C:
        T_dets = np.asarray(detspace.T_dets, dtype=np.uint64)
        sign_t, logabs_t = state.forward()
        sign_t, logabs_t = jnp.real(sign_t), jnp.real(logabs_t)

        # Normalize on Python side
        shift_t = jnp.max(logabs_t) if logabs_t.size > 0 else 0.0
        psi_t = sign_t * jnp.exp(logabs_t - shift_t)
        psi_t = np.asarray(psi_t, dtype=np.float64)
        psi_t /= np.linalg.norm(psi_t)

        # C++ computes <ψ_T|H|ψ_T>
        e_elec_t = core.compute_variational_energy(
            T_dets,
            psi_t,
            system.int_ctx,
            system.n_orb,
            use_heatbath=use_heatbath,
            eps1=eps1,
        )
        result["e_var_t"] = float(e_elec_t) + system.e_nuc

    return result


def extract_norms(trace_path: str | Path) -> dict[str, np.ndarray]:
    """
    Extract norm evolution from JSONL trace.
  
    Computes normalized fractions: f_S = ||psi_S||^2 / (||psi_S||^2 + ||psi_C||^2)
  
    Args:
        trace_path: Path to .jsonl trace file
  
    Returns:
        Dict with steps, norm_s, norm_c, frac_s, frac_c arrays
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return {}
  
    steps, norm_s, norm_c = [], [], []
  
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
  
    norm_tot = np.maximum(norm_s + norm_c, 1e-14)
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