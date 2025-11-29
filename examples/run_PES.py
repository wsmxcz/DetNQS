# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bond dissociation potential energy surface (PES) scanner.

Performs sequential single-point variational calculations along a bond
coordinate. Outputs flat directory structure for each geometry point:
    scans/{system}/r_{dist:.2f}/{config,results,analysis}

File: examples/run_PES.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import csv
import gc
import logging
import shutil
from pathlib import Path

import jax
from rich.console import Console
from rich.logging import RichHandler

from lever import analysis, evolution, models, solver
from lever.config import (
    ComputeMode,
    ExperimentConfig,
    HamiltonianConfig,
    LoopConfig,
    RuntimeConfig,
    ScreenMode,
)
from lever.interface import MoleculeBuilder
from lever.optimizers import adam, cosine_decay_schedule
from lever.utils.monitor import RunContext


# ============================================================================
# Scan Configuration
# ============================================================================

SYSTEM_NAME = "N2_sto3g"
BASIS = "sto-3g"
CHARGE = 0
SPIN = 0

# Bond distances in Angstroms
DISTANCES = [0.8, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0]

# Output paths
SCAN_ROOT = Path("scans") / SYSTEM_NAME
CSV_PATH = SCAN_ROOT / "scan_results.csv"

# Reference methods for comparison
BENCH_METHODS = {"hf", "cisd", "ccsd", "fci"}


# ============================================================================
# Utilities
# ============================================================================

def setup_flat_run_context(work_dir: Path) -> RunContext:

    console = Console()
    logger = logging.getLogger(f"LEVER_{work_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
  
    rich_handler = RichHandler(
        console=console,
        show_path=False,
        show_time=False,
        markup=True
    )
    logger.addHandler(rich_handler)
  
    return RunContext(root=work_dir, console=console, logger=logger)


# ============================================================================
# Single-Point Calculation
# ============================================================================

def run_point(dist: float, point_dir: Path) -> dict:

    print(f"\n{'='*60}")
    print(f" Distance: {dist:.4f} Å")
    print(f" Directory: {point_dir}")
    print(f"{'='*60}")
  
    # Geometry: N₂ along z-axis
    atom_str = f"N 0 0 0; N 0 0 {dist:.4f}"
    point_name = f"n2_r{dist:.2f}"
  
    # PySCF: Generate integrals and reference energies
    meta = (
        MoleculeBuilder(
            atom=atom_str,
            basis=BASIS,
            charge=CHARGE,
            spin=SPIN,
            work_dir=point_dir,
            name=point_name,
        )
        .run_scf()
        .run_benchmarks(BENCH_METHODS)
        .export()
    )
  
    # LEVER configuration: dynamic screening, proxy mode
    cfg = ExperimentConfig.from_meta(
        meta,
        root_dir=point_dir,
        hamiltonian=HamiltonianConfig(
            screening_mode=ScreenMode.DYNAMIC,
            screen_eps=1e-6,
        ),
        loop=LoopConfig(
            max_outer=30,
            max_inner=1000,
            chunk_size=8192,
        ),
        runtime=RuntimeConfig(
            enable_x64=True,
            compute_mode=ComputeMode.PROXY,
        ),
    )
  
    # Neural network ansatz: generalized backflow with 2×64 hidden layers
    model = models.Backflow(
        n_orb=cfg.system.n_orb,
        n_alpha=cfg.system.n_alpha,
        n_beta=cfg.system.n_beta,
        seed=42,
        n_dets=1,
        generalized=True,
        hidden_dims=(64, 64),
    )
  
    # Optimizer: cosine-annealed Adam with L2 regularization
    lr_schedule = cosine_decay_schedule(
        init_value=5e-4,
        decay_steps=500,
        alpha=0.02
    )
    optimizer = adam(learning_rate=lr_schedule, weight_decay=1e-4)
  
    # Selection strategy: cumulative mass 99.999%, max 5000 determinants
    strategy = evolution.BasicStrategy(
        scorer=evolution.scores.AmplitudeScorer(),
        selector=evolution.CumulativeMassSelector(0.99999, max_size=5000),
    )
  
    # Initialize flat-structure run context
    run = setup_flat_run_context(point_dir)
  
    # Execute adaptive solver
    result, diag = solver.solve(cfg, model, strategy, optimizer, run_ctx=run)
  
    # Variational energy analysis
    evaluator = analysis.VariationalEvaluator(
        int_ctx=diag["int_ctx"],
        n_orb=cfg.system.n_orb,
        e_nuc=diag["e_nuc"],
    )
  
    energies = evaluator.analyze_result(
        result,
        model=model,
        compute_s_ci=True,
        compute_sc_var=True,
    )
  
    # Extract benchmark energies from metadata
    benchmarks = {}
    if meta.benchmarks:
        for method, item in meta.benchmarks.items():
            if item.energy is not None:
                benchmarks[method] = item.energy
  
    # Prepare CSV row
    n_s = result.final_space.s_dets.shape[0] if result.final_space else 0
    data = {
        "dist": dist,
        "n_s": n_s,
        "time": result.total_time,
        "E_LEVER": result.full_energy_history[-1],
        "E_VAR": energies.get("e_var"),
        "E_S_CI": energies.get("e_s_ci"),
        "E_HF": benchmarks.get("hf"),
        "E_CISD": benchmarks.get("cisd"),
        "E_CCSD": benchmarks.get("ccsd"),
        "E_FCI": benchmarks.get("fci"),
    }
  
    # Save artifacts: config, history, summary
    run.record_experiment(
        config=cfg,
        result=result,
        diagnostics=diag,
        meta=meta,
        model=model,
        optimizer=optimizer,
        strategy=strategy,
        energies=energies,
    )
  
    # Cleanup to prevent memory accumulation
    del result, model, evaluator, run
    return data


# ============================================================================
# Main Scan Loop
# ============================================================================

def main() -> None:
    # Reset scan directory
    if SCAN_ROOT.exists():
        shutil.rmtree(SCAN_ROOT)
    SCAN_ROOT.mkdir(parents=True)
  
    # Initialize CSV with header
    fieldnames = [
        "dist", "n_s", "time",
        "E_LEVER", "E_VAR", "E_S_CI",
        "E_HF", "E_CISD", "E_CCSD", "E_FCI",
    ]
  
    with open(CSV_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
  
    print(f"Starting PES Scan: {SYSTEM_NAME}")
    print(f"Output Root: {SCAN_ROOT}")
  
    # Sequential geometry loop
    for d in DISTANCES:
        point_dir = SCAN_ROOT / f"r_{d:.2f}"
        point_dir.mkdir(parents=True, exist_ok=True)
      
        try:
            row = run_point(d, point_dir)
          
            # Append to CSV
            with open(CSV_PATH, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
          
            print(f"✓ Point {d:.2f} Å completed")
      
        except Exception as e:
            print(f"✗ ERROR at {d:.2f} Å: {e}")
            import traceback
            traceback.print_exc()
      
        finally:
            # Force memory cleanup
            gc.collect()
            jax.clear_caches()
  
    print(f"\n✓ Scan complete. Results: {CSV_PATH}")


if __name__ == "__main__":
    main()