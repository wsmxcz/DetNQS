# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified experiment monitor with observer pattern for solver lifecycle.

Implements minimal-redundancy artifact storage: config.yaml and system.json
serve as single source of truth for reproducibility.

File: lever/utils/monitor.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.theme import Theme

from ..config import ExperimentConfig, ObjectSpec
from ..dtypes import LeverResult
from .config_utils import freeze_object
from .plot_utils import plot_convergence


# ============================================================================
# Console Theme (Two-Color Scheme)
# ============================================================================

LEVER_THEME = Theme({
    "lever.main": "cyan",           # Primary text and headers
    "lever.accent": "bright_yellow", # Emphasis values
})


class RunContext:
    """
    Experiment lifecycle manager with observer pattern.
    
    Artifact structure:
      - config.yaml: Complete reproducible configuration
      - system.json: Physical system and benchmarks
      - history.json: Energy trace {E_n} and cycle metrics
      - summary.json: Run-level index (references config/system)
      - core_space.npz: Final S/C determinant spaces and H_CC diagonal
      - params.msgpack: Optimized NN parameters
    """

    def __init__(self, root: Path, console: Console, logger: logging.Logger):
        self.root = root
        self.console = console
        self.logger = logger

    @classmethod
    def create(cls, system_name: str, root_dir: str = "runs") -> RunContext:
        """
        Initialize timestamped run directory.
        
        Pattern: {root_dir}/{YYYYMMDD_HHMMSS}_{system_name}/
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{system_name}"
        path = Path(root_dir) / run_id
        path.mkdir(parents=True, exist_ok=True)

        console = Console(theme=LEVER_THEME)

        # Configure logger without level prefix
        logger = logging.getLogger("LEVER")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        rich_h = RichHandler(
            console=console,
            show_path=False,
            show_time=False,
            show_level=False,  # Remove INFO/WARNING labels
            omit_repeated_times=False,
            markup=True,
        )
        rich_h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_h)

        console.print(
            f"[lever.main]Run:[/] [lever.accent]{path.absolute()}[/]"
        )

        return cls(path, console, logger)

    # ========================================================================
    # Logging API
    # ========================================================================

    def header(self, title: str) -> None:
        """Styled section separator."""
        self.console.rule(f"[lever.main]{title}[/]")

    def log_text(self, msg: str) -> None:
        """Output with rich markup support."""
        self.logger.info(msg, extra={"markup": True})

    def log_kv(self, label: str, value: str) -> None:
        """Key-value pair: label in main, value in accent."""
        self.log_text(f"[lever.main]{label}:[/] [lever.accent]{value}[/]")

    def warn(self, msg: str) -> None:
        """Warning message with accent prefix."""
        self.log_text(f"[lever.accent]⚠[/] [lever.main]{msg}[/]")

    # ========================================================================
    # Solver Event Handlers (Observer Pattern)
    # ========================================================================

    def on_solver_start(self, cfg: ExperimentConfig, model: Any) -> None:
        """Log solver config summary."""
        self.header("LEVER Solver")

        mode = cfg.runtime.compute_mode.value.upper()
        self.log_kv("Mode", mode)
        self.log_kv("System", cfg.system.name)

        try:
            summary = getattr(model, "summary", {})
            name = summary.get("name", type(model).__name__)
            n_params = summary.get("n_params")
        except Exception:
            name, n_params = type(model).__name__, None

        self.log_kv("Model", name)
        if n_params is not None:
            self.log_kv("Parameters", f"{n_params:,}")

    def on_cycle_start(self, cycle: int, max_outer: int, comp: dict) -> None:
        """
        Log space construction metrics.
        
        Sparsity: ρ = nnz(H_SS) / (n_S)²
        """
        n_s, n_c, nnz = comp["n_s"], comp["n_c"], comp["nnz_ss"]
        density = 100.0 * nnz / (n_s * n_s) if n_s > 0 else 0.0

        self.log_text(
            f"[lever.main]Cycle {cycle}/{max_outer}  "
            f"S={n_s}, C={n_c}, nnz={nnz} ({density:.1f}%)[/]"
        )

    def on_inner_progress(self, step: int, total: int, energy: float) -> None:
        """Inner optimization step."""
        self.log_text(
            f"[lever.main]  Inner {step}/{total}  E = {energy:.8f}[/]"
        )

    def on_cycle_end(self, entry: dict) -> None:
        """
        Cycle completion with norms.
        
        Normalization: ||ψ_S||² + ||ψ_C||² = 1 (proxy mode)
        """
        inner = entry["inner"]
        wf = entry["wavefunction"]
        outer = entry["outer"]
        t = entry["time_sec"]
        c = entry["cycle"]

        status = "✓" if inner["converged"] else "…"
        
        self.log_text(
            f"[lever.main]Cycle {c:>3} {status}  "
            f"E = [lever.accent]{inner['final_energy']:.8f}[/]  "
            f"(steps={inner['steps']}, Δ={outer['delta']:.2e}, "
            f"||S||²={wf['norm_s']:.6f}, ||C||²={wf['norm_c']:.6f}, "
            f"t={t:.1f}s)[/]"
        )
        self.console.print()

    def on_solver_end(
        self,
        converged: bool,
        total_time: float,
        max_cycles: int,
        final_delta: float,
        diagnostics: dict,
    ) -> None:
        """Final solver statistics."""
        self.header("Complete")

        status = "Converged" if converged else f"Max cycles ({max_cycles})"
        self.log_kv("Status", status)

        # Extract final metrics
        cycles = diagnostics.get("cycles", [])
        n_cycles = len(cycles)
        total_inner = sum(c["inner"]["steps"] for c in cycles)

        final_energy = 0.0
        n_s = n_c = nnz = 0

        if cycles:
            last = cycles[-1]
            final_energy = last["inner"]["final_energy"]
            comp = last["compile"]
            n_s = comp.get("n_s", 0)
            n_c = comp.get("n_c", 0)
            nnz = comp.get("nnz_ss", 0)

        self.log_kv("Energy", f"{final_energy:.8f} Ha")
        self.log_kv("Time", f"{total_time:.1f} s")
        self.log_kv("Iterations", f"{n_cycles} cycles ({total_inner} inner steps)")
        self.log_kv("Space", f"S={n_s:,}, C={n_c:,}")
        self.console.print()

    # ========================================================================
    # Artifact Recording
    # ========================================================================

    def record_experiment(
        self,
        config: ExperimentConfig,
        result: LeverResult,
        *,
        diagnostics: dict | None = None,
        meta: Any = None,
        model: Any = None,
        optimizer: Any = None,
        strategy: Any = None,
        energies: dict[str, float] | None = None,
        enable_plots: bool = True,
    ) -> None:
        """
        Snapshot artifacts with minimal redundancy.
        
        Philosophy: config.yaml + system.json = complete reproducibility
        """
        self.header("Artifacts")

        # Freeze config with object specs
        cfg_snapshot = config.model_copy()
        if model is not None:
            cfg_snapshot.model_spec = self._extract_spec(model)
        if optimizer is not None:
            cfg_snapshot.optimizer_spec = self._extract_spec(optimizer)
        if strategy is not None:
            cfg_snapshot.strategy_spec = self._extract_spec(strategy)

        config_path = self.root / "config.yaml"
        cfg_snapshot.save(config_path)
        self.log_kv("Config", config_path.name)

        self._save_artifacts(
            cfg_snapshot,
            result,
            energies,
            diagnostics,
            meta=meta,
            model=model,
        )

        if enable_plots and diagnostics:
            self._plot_convergence(result, diagnostics.get("cycles", []))
            self.log_kv("Plots", "convergence.pdf")

    def _extract_spec(self, obj: Any) -> ObjectSpec:
        """Extract serializable object spec."""
        try:
            return freeze_object(obj)
        except TypeError as e:
            self.warn(f"Config spec error for {obj}: {e}")
            return ObjectSpec(target=str(type(obj)), params={"error": str(e)})

    def _save_artifacts(
        self,
        config: ExperimentConfig,
        result: LeverResult,
        energies: dict[str, float] | None,
        diagnostics: dict | None,
        *,
        meta: Any = None,
        model: Any = None,
    ) -> None:
        """
        Save minimal-redundancy artifacts.
        
        Files:
          - history.json: Energy trace E(t) and cycle metrics
          - summary.json: Run index (references config/system)
          - core_space.npz: Final S/C spaces
          - params.msgpack: NN parameters
        """
        cycles: list[dict] = diagnostics.get("cycles", []) if diagnostics else []

        energy_trace = [float(x) for x in result.full_energy_history]
        total_inner = sum(int(c["inner"]["steps"]) for c in cycles) if cycles else 0

        # Extract final dimensions
        final_n_s = final_n_c = None
        space = result.final_space
        if space is not None:
            final_n_s = int(space.n_s)
            final_n_c = int(space.n_c)
        elif cycles:
            final_n_s = int(cycles[-1]["compile"]["n_s"])
            final_n_c = int(cycles[-1]["compile"]["n_c"])

        # History: time-series data
        outer_cycles: list[dict[str, Any]] = []
        for c in cycles:
            outer_cycles.append({
                "cycle": int(c["cycle"]),
                "time_sec": float(c["time_sec"]),
                "compile": {
                    "n_s": int(c["compile"]["n_s"]),
                    "n_c": int(c["compile"]["n_c"]),
                    "nnz_ss": int(c["compile"]["nnz_ss"]),
                },
                "inner": {
                    "steps": int(c["inner"]["steps"]),
                    "converged": bool(c["inner"]["converged"]),
                    "final_energy": float(c["inner"]["final_energy"]),
                },
                "wavefunction": {
                    "norm_s": float(c["wavefunction"]["norm_s"]),
                    "norm_c": float(c["wavefunction"]["norm_c"]),
                },
            })

        with open(self.root / "history.json", "w", encoding="utf-8") as f:
            json.dump({"energy_trace": energy_trace, "outer_cycles": outer_cycles}, f, indent=2)

        # Core space: determinants and diagonal
        if space is not None:
            try:
                np.savez_compressed(
                    self.root / "core_space.npz",
                    s_dets=np.asarray(result.final_s_dets, dtype=np.uint64),
                    c_dets=np.asarray(space.c_dets, dtype=np.uint64),
                    h_diag_c=np.asarray(space.h_diag_c, dtype=np.float64),
                    n_s=int(space.n_s),
                    n_c=int(space.n_c),
                )
                self.log_kv("Space", "core_space.npz")
            except Exception as e:
                self.warn(f"Space save failed: {e}")

        # Summary: run index
        model_summary = getattr(model, "summary", None) if model else None
        model_block = dict(model_summary) if isinstance(model_summary, dict) else {"name": "Unknown"}

        methods_block = {k: float(v) for k, v in (energies or {}).items()}

        summary_data: dict[str, Any] = {
            "system": {
                "name": config.system.name,
                "meta_file": "system.json",
            },
            "model": model_block,
            "energies": {"methods": methods_block},
            "metrics": {
                "converged": diagnostics.get("converged", False) if diagnostics else False,
                "total_time_sec": float(result.total_time),
                "outer_cycles": len(cycles),
                "inner_steps_total": total_inner,
                "final_space": {"n_s": final_n_s, "n_c": final_n_c} if final_n_s else None,
            },
            "files": {
                "config": "config.yaml",
                "history": "history.json",
            },
        }

        with open(self.root / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # NN parameters
        try:
            from flax import serialization
            with open(self.root / "params.msgpack", "wb") as f:
                f.write(serialization.to_bytes(result.final_params))
            self.log_kv("Params", "params.msgpack")
        except ImportError:
            pass

    def _plot_convergence(self, result: LeverResult, diag_cycles: list[dict]) -> None:
        """Generate convergence plot."""
        try:
            plot_convergence(self.root, result, diag_cycles)
        except Exception as e:
            self.warn(f"Plot failed: {e}")


__all__ = ["RunContext"]
