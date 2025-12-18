# lever/analysis/runtime.py
# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime recorder and diagnostics measurement."""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..operator import functional as func_mod
from ..space import DetSpace
from ..state import DeterministicState


@dataclass
class StepRecord:
    """Single optimization step record (minimal)."""
    outer: int
    inner: int
    energy: float


@dataclass
class OuterRecord:
    """Outer loop summary with detailed diagnostics."""
    outer: int
    n_steps: int
    final_energy: float
    variance: float | None = None
    norm_s: float | None = None
    norm_c: float | None = None


@dataclass
class RunRecord:
    """Complete run history."""
    steps: list[StepRecord] = field(default_factory=list)
    outers: list[OuterRecord] = field(default_factory=list)


def measure_energy(
    state: DeterministicState,
    ham: Any,
    op: Callable,
    *,
    detspace: DetSpace | None = None,
    mode: str = "auto",
    chunk_size: int | None = None,
) -> dict[str, Any]:
    """
    Measure energy with full diagnostics (not part of training loop).

    Internal helper for RuntimeRecorder.summarize_outer().
    """
    result = func_mod.compute_energy(
        state,
        ham,
        op,
        detspace=detspace,
        need_grad=False,
        compute_diagnostics=True,
        chunk_size=chunk_size,
        mode=mode,
    )

    # Convert result.value to Python float
    out: dict[str, Any] = {"energy": float(result.value)}

    # Keep scalar diagnostics as float, keep arrays as numpy arrays
    for k, v in result.diagnostics.items():
        v_np = np.asarray(v)
        if v_np.shape == ():  # scalar
            out[k] = float(v_np)
        else:  # vector or higher-rank diagnostics (e.g. weights_s, weights_c)
            out[k] = v_np

    return out


class RuntimeRecorder:
    """Lightweight recorder for optimization history."""

    def __init__(self) -> None:
        self.run = RunRecord()

    def step_callback(self, info: dict[str, Any]) -> None:
        """Callback for each inner step (minimal recording)."""
        self.run.steps.append(
            StepRecord(
                outer=int(info["outer"]),
                inner=int(info["inner"]),
                energy=float(info["energy"]),
            )
        )

    def summarize_outer(
        self,
        outer: int,
        *,
        state: DeterministicState,
        detspace: DetSpace,
        ham: Any,
        op: Callable,
        mode: str,
        chunk_size: int | None,
    ) -> None:
        """Compute detailed diagnostics at outer loop completion."""
        # Measure energy with full diagnostics (one call per outer cycle)
        stats = measure_energy(
            state,
            ham,
            op,
            detspace=detspace,
            mode=mode,
            chunk_size=chunk_size,
        )

        # Last energy and number of steps for this outer index
        final_energy = float("nan")
        for s in reversed(self.run.steps):
            if s.outer == outer:
                final_energy = s.energy
                break

        n_steps = sum(1 for s in self.run.steps if s.outer == outer)

        self.run.outers.append(
            OuterRecord(
                outer=outer,
                n_steps=n_steps,
                final_energy=final_energy,
                variance=stats.get("variance"),
                norm_s=stats.get("norm_s"),
                norm_c=stats.get("norm_c"),
            )
        )
