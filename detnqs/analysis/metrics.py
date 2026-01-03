# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing analysis tools for DetNQS results.

Provides optional analysis functions:
  - Convergence statistics from JSON trace files
  - PT2 correction via C++ kernel (Epstein-Nesbet)
  - Variational energy evaluation on V-space and T-space

Note: These functions are independent of the runtime driver loop
      and designed for delayed execution after optimization.

File: detnqs/analysis/metrics.py
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
        Dict containing mean_delta, max_delta, final_energy, n_outers, total_time
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

    deltas = [abs(energies[i] - energies[i - 1]) for i in range(1, len(energies))]

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
    Compute Epstein-Nesbet PT2 correction on P-space (electronic part only).

    The PT2 energy correction is computed as:
        Delta E_PT2 = sum_{x in P} |<x|H|psi_V>|^2 / (E_0 - H_{xx})

    Args:
        state: Optimized network state
        detspace: Determinant space containing V and P
        system: Molecular system
        e_ref_elec: Reference electronic energy from variational optimization
        screening: Screening method ("heatbath" or "none")
        eps1: Screening threshold for heat-bath

    Returns:
        PT2 correction (electronic part), or None if C++ kernel unavailable
    """
    try:
        from .. import core
    except (ImportError, AttributeError):
        return None

    V_dets = np.asarray(detspace.V_dets, dtype=np.uint64)
    n_v = V_dets.shape[0]
    if n_v == 0:
        return None

    # Evaluate and normalize psi_V
    indices = jnp.arange(n_v, dtype=jnp.int32)
    sign_v, logabs_v = state.forward(indices)
    sign_v, logabs_v = jnp.real(sign_v), jnp.real(logabs_v)

    shift = jnp.max(logabs_v)
    psi_v = sign_v * jnp.exp(logabs_v - shift)
    psi_v = np.asarray(psi_v, dtype=np.float64)
    psi_v = psi_v / np.linalg.norm(psi_v)

    use_heatbath = screening == "heatbath"

    e_pt2 = core.compute_pt2(
        V_dets,
        psi_v,
        system.int_ctx,
        system.n_orb,
        e_ref_elec,
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
    Compute variational energies on V-space and optionally T-space.

    Workflow:
      1. Always compute E_var_V = <psi_V|H_VV|psi_V> on V-space
      2. If P-space exists, compute E_var_T = <psi_T|H|psi_T> on full T-space

    Args:
        state: Optimized network state
        detspace: Determinant space (may or may not have P)
        system: Molecular system
        use_heatbath: Enable heat-bath screening in C++ kernel
        eps1: Screening threshold

    Returns:
        Dict with 'e_var_v' (always) and 'e_var_t' (if P-space exists),
        both including nuclear repulsion

    Note:
        C++ kernel computes only <psi|H|psi>; normalization handled in Python.
    """
    try:
        from .. import core
    except (ImportError, AttributeError):
        return None

    result = {}

    # Compute E_var_V on V-space
    V_dets = np.asarray(detspace.V_dets, dtype=np.uint64)
    if V_dets.shape[0] == 0:
        return None

    indices_v = jnp.arange(detspace.size_V, dtype=jnp.int32)
    sign_v, logabs_v = state.forward(indices_v)
    sign_v, logabs_v = jnp.real(sign_v), jnp.real(logabs_v)

    shift_v = jnp.max(logabs_v) if logabs_v.size > 0 else 0.0
    psi_v = sign_v * jnp.exp(logabs_v - shift_v)
    psi_v = np.asarray(psi_v, dtype=np.float64)
    psi_v = psi_v / np.linalg.norm(psi_v)

    e_elec_v = core.compute_variational_energy(
        V_dets,
        psi_v,
        system.int_ctx,
        system.n_orb,
        use_heatbath=use_heatbath,
        eps1=eps1,
    )
    result["e_var_v"] = float(e_elec_v) + system.e_nuc

    # Optionally compute E_var_T on full T-space if P exists
    if detspace.has_P:
        T_dets = np.asarray(detspace.T_dets, dtype=np.uint64)
        sign_t, logabs_t = state.forward()
        sign_t, logabs_t = jnp.real(sign_t), jnp.real(logabs_t)

        shift_t = jnp.max(logabs_t) if logabs_t.size > 0 else 0.0
        psi_t = sign_t * jnp.exp(logabs_t - shift_t)
        psi_t = np.asarray(psi_t, dtype=np.float64)
        psi_t = psi_t / np.linalg.norm(psi_t)

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

    Computes normalized fractions:
        f_V = ||psi_V||^2 / (||psi_V||^2 + ||psi_P||^2)

    Args:
        trace_path: Path to .jsonl trace file

    Returns:
        Dict with steps, norm_v, norm_p, frac_v, frac_p arrays
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return {}

    steps, norm_v, norm_p = [], [], []

    with trace_path.open("r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                steps.append(record["outer_step"])
                norm_v.append(record["norm_v"])
                norm_p.append(record["norm_p"])

    steps = np.array(steps, dtype=int)
    norm_v = np.array(norm_v)
    norm_p = np.array(norm_p)

    norm_tot = np.maximum(norm_v + norm_p, 1e-14)
    frac_v = norm_v / norm_tot
    frac_p = norm_p / norm_tot

    return {
        "steps": steps,
        "norm_v": norm_v,
        "norm_p": norm_p,
        "frac_v": frac_v,
        "frac_p": frac_p,
    }


__all__ = [
    "convergence_stats",
    "compute_pt2",
    "compute_variational",
    "extract_norms",
]
