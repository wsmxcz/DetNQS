# lever/analysis/posthoc.py
# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc analysis: PT2 and variational evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .. import core
from ..driver import DriverResult
from ..space import DetSpace
from ..system import MolecularSystem
from .runtime import RuntimeRecorder


def evaluate_pt2(
    result: DriverResult,
    system: MolecularSystem,
    *,
    use_heatbath: bool = True,
    eps1: float = 1e-6,
) -> dict[str, float]:
    """
    Evaluate PT2 correction for variational/effective modes.

    Notes:
        This function is a thin wrapper around the low-level PT2 kernel.
        It expects a routine that can compute the Epstein-Nesbet PT2
        correction on the final S-space wavefunction.

        The PT2 core is not wired here because its API depends on the
        C++ implementation. Plug your existing PT2 routine into the
        marked section below.

    Args:
        result: Driver result with final state and detspace.
        system: Molecular system.
        use_heatbath: Enable Heat-Bath screening for C-space.
        eps1: Screening threshold.

    Returns:
        Dictionary with:
          - e_var:  variational S-space energy (if available)
          - e_pt2:  PT2 correction
          - e_tot:  e_var + e_pt2
    """
    # Example sketch:
    #
    # 1. Build final S determinant list.
    # 2. Evaluate psi_S from the neural state.
    # 3. Call your C++ PT2 kernel on (S_dets, psi_S, int_ctx, n_orb).
    #
    # Replace the NotImplementedError with a call such as:
    #
    #   e_var, e_pt2 = core.compute_variational_pt2(
    #       S_dets,
    #       psi_S,
    #       system.int_ctx,
    #       system.n_orb,
    #       use_heatbath,
    #       eps1,
    #   )
    #
    # and return {"e_var": e_var + system.e_nuc, ...}.

    raise NotImplementedError(
        "PT2 evaluation depends on the C++ PT2 kernel. "
        "Please connect your existing PT2 routine here."
    )


def _build_t_space(space: DetSpace) -> np.ndarray:
    """Return T = S ∪ C determinants as uint64 array."""
    s = np.asarray(space.S_dets, dtype=np.uint64)
    c = getattr(space, "C_dets", None)

    if c is None:
        return s

    c = np.asarray(c, dtype=np.uint64)
    if c.size == 0:
        return s

    return np.concatenate([s, c], axis=0)


def evaluate_proxy_variational(
    result: DriverResult,
    system: MolecularSystem,
) -> dict[str, float]:
    """
    Evaluate full variational energy on final T-space.

    This computes
        E_var = <psi_T|H|psi_T> / <psi_T|psi_T> + E_nuc

    using the C++ streaming kernel, with psi_T provided by the final
    neural state on the current determinant space.

    Args:
        result: Driver result after optimization.
        system: Molecular system.

    Returns:
        Dictionary with:
          - e_var:  total variational energy (electronic + nuclear)
          - norm_s: S-space norm contribution
          - norm_c: C-space norm contribution
    """
    space = result.detspace
    t_dets = _build_t_space(space)

    # Evaluate log-psi on full T-space from the final state.
    # This uses the state's internal determinant ordering.
    log_psi_t = result.state.forward(chunk_size=None)
    psi_t = np.asarray(jnp.exp(log_psi_t), dtype=np.complex128)

    if psi_t.shape[0] != t_dets.shape[0]:
        raise ValueError(
            f"Shape mismatch: psi_T has shape {psi_t.shape}, "
            f"but dets_T has shape {t_dets.shape}"
        )

    # Streaming variational energy on T-space
    e_el, norm = core.compute_variational_energy(
        t_dets,
        psi_t,
        system.int_ctx,
        system.n_orb,
        use_heatbath=True,
        eps1=1e-6,
    )

    if norm <= 1e-14:
        raise RuntimeError("Wavefunction norm is effectively zero.")

    e_var = float(e_el / norm + system.e_nuc)

    # Decompose norm into S / C parts
    n_s = int(space.size_S)
    psi_s = psi_t[:n_s]
    psi_c = psi_t[n_s:]

    norm_tot = float(np.vdot(psi_t, psi_t).real)
    if norm_tot <= 0.0:
        raise RuntimeError("Total norm is non-positive.")

    norm_s = float(np.vdot(psi_s, psi_s).real / norm_tot)
    norm_c = float(np.vdot(psi_c, psi_c).real / norm_tot)

    return {
        "e_var": e_var,
        "norm_s": norm_s,
        "norm_c": norm_c,
    }


def save_run_artifacts(
    run_dir: Path,
    result: DriverResult,
    recorder: RuntimeRecorder | None = None,
) -> None:
    """
    Save minimal artifacts for post-run analysis.

    Files:
      - history.json : energy trace and outer diagnostics
      - space.npz    : final S/C determinant space
      - params.msgpack (if flax is available): final NN parameters
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) history.json
    # ------------------------------------------------------------------
    energy_trace = [float(e) for e in result.energies]

    history: dict[str, Any] = {"energy_trace": energy_trace}

    if recorder is not None:
        outers_block = []
        for o in recorder.run.outers:
            outers_block.append(
                {
                    "outer": int(o.outer),
                    "n_steps": int(o.n_steps),
                    "final_energy": float(o.final_energy),
                    "variance": None if o.variance is None else float(o.variance),
                    "norm_s": None if o.norm_s is None else float(o.norm_s),
                    "norm_c": None if o.norm_c is None else float(o.norm_c),
                }
            )
        history["outer_cycles"] = outers_block

    with (run_dir / "history.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(history, f, indent=2)

    # ------------------------------------------------------------------
    # 2) space.npz  (final determinant space)
    # ------------------------------------------------------------------
    space = result.detspace
    s_dets = np.asarray(space.S_dets, dtype=np.uint64)
    c_dets = getattr(space, "C_dets", None)
    if c_dets is not None:
        c_dets = np.asarray(c_dets, dtype=np.uint64)
    else:
        c_dets = np.zeros((0, 2), dtype=np.uint64)

    np.savez_compressed(
        run_dir / "space.npz",
        S_dets=s_dets,
        C_dets=c_dets,
        n_s=int(getattr(space, "size_S", s_dets.shape[0])),
        n_c=int(getattr(space, "size_C", c_dets.shape[0])),
    )

    # ------------------------------------------------------------------
    # 3) params.msgpack (final NN parameters)
    # ------------------------------------------------------------------
    try:
        from flax import serialization  # type: ignore[import-not-found]
    except Exception:
        serialization = None

    if serialization is not None:
        params_path = run_dir / "params.msgpack"
        with params_path.open("wb") as f:
            f.write(serialization.to_bytes(result.state.params))


__all__ = [
    "evaluate_pt2",
    "evaluate_proxy_variational",
    "save_run_artifacts",
]