# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bond dissociation PES scanner.

Runs sequential single-point calculations along a bond coordinate.
Collects LEVER energies (Opt, Var, S-CI), space dimensions, and 
PySCF benchmarks (HF, CISD, CCSD, FCI) into a unified CSV report.

File: examples/run_PES.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import csv
import gc
import shutil
from pathlib import Path

import jax
import numpy as np
import optax

from lever import analysis, config, driver, evolution, models
from lever.interface import MoleculeBuilder, load_benchmarks
from lever.monitor import init_run
from lever.optimizers import adam

# ============================================================================
# Configuration
# ============================================================================

SYSTEM_NAME = "N2_sto3g"
BASIS = "sto-3g"
CHARGE = 0
SPIN = 0

# Scan parameters (Angstroms)
DISTANCES = [0.8, 1.0, 1.1, 1.5, 2.0, 2.5, 3.0]

# Output setup
SCAN_ROOT = Path("scan_runs") / SYSTEM_NAME
CSV_PATH = "scan_results.csv"

# Benchmarks to compute via PySCF
BENCH_METHODS = {"hf", "cisd", "ccsd", "fci"}

# ============================================================================
# Main Logic
# ============================================================================


def run_point(dist: float, work_dir: Path) -> dict:
    """Run single point: PySCF -> LEVER -> Analysis.
    
    Args:
        dist: Bond distance in Angstroms
        work_dir: Working directory for outputs
        
    Returns:
        Dictionary with energy results and metadata
    """
    print(f"\n{'='*60}")
    print(f" Processing Distance: {dist:.4f} A")
    print(f" Directory: {work_dir}")
    print(f"{'='*60}")

    # 1. Geometry string (N2 aligned along Z)
    atom_str = f"N 0 0 0; N 0 0 {dist:.4f}"

    # 2. PySCF: Generate integrals and benchmarks
    builder = (
        MoleculeBuilder(
            atom=atom_str,
            basis=BASIS,
            charge=CHARGE,
            spin=SPIN,
            work_dir=work_dir,
        )
        .run_scf()
        .run_benchmarks(BENCH_METHODS)
    )

    # Export FCIDUMP and HDF5
    exported = builder.export(f"n2_r{dist:.2f}")

    # 3. LEVER Configuration
    sys_cfg = config.SystemConfig.from_hdf5(str(exported.hdf5_path))

    ham_cfg = config.HamiltonianConfig(
        screening_mode=config.ScreenMode.DYNAMIC,
        screen_eps=1e-6,
    )

    loop_cfg = config.LoopConfig(
        max_outer=20,
        max_inner=500,
        chunk_size=8192
    )

    lever_cfg = config.LeverConfig(
        system=sys_cfg,
        hamiltonian=ham_cfg,
        loop=loop_cfg,
        compute_mode=config.ComputeMode.PROXY,
        precision=config.PrecisionConfig(enable_x64=True)
    )

    # 4. Model and Optimizer
    model = models.Backflow(
        n_orbitals=sys_cfg.n_orbitals,
        n_alpha=sys_cfg.n_alpha,
        n_beta=sys_cfg.n_beta,
        n_dets=1,
        generalized=True,
        hidden_dims=(64, 64),
        seed=42
    )

    lr_schedule = optax.cosine_decay_schedule(
        init_value=5e-4,
        decay_steps=500,
        alpha=1e-5 / 5e-4,
    )
    optimizer = adam(learning_rate=lr_schedule, weight_decay=1e-4)

    strategy = evolution.BasicStrategy(
        scorer=evolution.scores.AmplitudeScorer(),
        selector=evolution.CumulativeMassSelector(0.9999, max_size=5000)
    )

    # 5. Execute LEVER Optimization
    drv = driver.Driver(lever_cfg, model, strategy, optimizer)
    result = drv.run()

    # 6. Post-hoc Analysis (E_var and E_S_CI)
    evaluator = analysis.VariationalEvaluator(
        int_ctx=drv.int_ctx,
        n_orb=sys_cfg.n_orbitals,
        e_nuc=drv.int_ctx.get_e_nuc()
    )

    analysis_res = evaluator.analyze_result(
        result,
        model=model,
        compute_s_ci=True,
        compute_sc_var=True
    )

    # 7. Gather all data
    benchmarks = load_benchmarks(exported.hdf5_path)

    n_s = 0
    if result.final_space is not None:
        n_s = result.final_space.s_dets.shape[0]

    data = {
        "dist": dist,
        "n_s": n_s,
        "time": result.total_time,
        # LEVER Energies
        "E_LEVER": result.full_energy_history[-1],
        "E_VAR": analysis_res.get("e_var"),
        "E_S_CI": analysis_res.get("e_s_ci"),
        # Benchmarks
        "E_HF": benchmarks.get("hf"),
        "E_CISD": benchmarks.get("cisd"),
        "E_CCSD": benchmarks.get("ccsd"),
        "E_FCI": benchmarks.get("fci"),
    }

    # Cleanup heavy objects immediately
    del drv, result, model, evaluator, builder

    return data


def main():
    """Execute bond dissociation potential energy surface scan."""
    # Prepare clean root directory
    if SCAN_ROOT.exists():
        shutil.rmtree(SCAN_ROOT)
    SCAN_ROOT.mkdir(parents=True)

    # CSV Header setup
    fieldnames = [
        "dist", "n_s", "time",
        "E_LEVER", "E_VAR", "E_S_CI",
        "E_HF", "E_CISD", "E_CCSD", "E_FCI"
    ]

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    print(f"Starting Scan: {SYSTEM_NAME}")
    print(f"Root Directory: {SCAN_ROOT}")
    print(f"Output CSV: {CSV_PATH}")

    # Scan Loop
    for d in DISTANCES:
        # Create explicit subdirectory for this point
        point_dir = SCAN_ROOT / f"r_{d:.2f}"
        point_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging context for this specific run
        run_ctx = init_run(
            system_name=f"{SYSTEM_NAME}_r{d:.2f}",
            root_dir=str(SCAN_ROOT)
        )

        try:
            row_data = run_point(d, point_dir)

            # Write to CSV
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row_data)

            print(f"Saved point {d:.2f} A.")

        except Exception as e:
            print(f"ERROR at distance {d:.2f}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Close logging handles
            run_ctx.close()

            # Force memory cleanup (Critical for scans)
            gc.collect()
            jax.clear_caches()

    print("\nScan complete.")


if __name__ == "__main__":
    main()
