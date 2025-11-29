# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce variational quantum chemistry experiments from config snapshots.

Usage:
    python examples/reproduce.py runs/20251101_120000_N2_sto3g/config.yaml

File: examples/reproduce.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import sys
from pathlib import Path

from lever import analysis, solver
from lever.config import ExperimentConfig
from lever.utils.monitor import RunContext


def main(config_path: str) -> None:
    """
    Execute complete rerun from config snapshot.
    
    Pipeline:
        1. Load ExperimentConfig from YAML
        2. Rebuild model/optimizer/strategy from ObjectSpec
        3. Run variational solver
        4. Compute energies: E[ψ_S], E[ψ_SC], Var[H]
        5. Save artifacts to reruns/ directory
    
    Args:
        config_path: Path to serialized config.yaml
    """
    cfg_path = Path(config_path)
    
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading config: {cfg_path}")
    cfg = ExperimentConfig.load(cfg_path)
    
    # Rebuild components from ObjectSpec (exact reproducibility)
    model, optimizer, strategy = cfg.build_components()
    
    # Create new run context for rerun artifacts
    run = RunContext.create(
        system_name=cfg.system.name,
        root_dir="reruns"
    )
    
    print(f"Starting rerun: {run.run_dir}")
    
    # Execute variational solver
    result, diag = solver.solve(
        cfg, model, strategy, optimizer, run_ctx=run
    )
    
    # Post-hoc energy analysis
    evaluator = analysis.VariationalEvaluator(
        int_ctx=diag["int_ctx"],
        n_orb=cfg.system.n_orb,
        e_nuc=diag["e_nuc"],
    )
    
    energies = evaluator.analyze_result(
        result,
        model=model,
        compute_s_ci=True,   # CI energy in S-space
        compute_sc_var=True, # Variance Var[H]
    )
    
    # Persist rerun artifacts
    run.record_experiment(
        config=cfg,
        result=result,
        diagnostics=diag,
        meta=None,  # Original meta unavailable in rerun
        model=model,
        optimizer=optimizer,
        strategy=strategy,
        energies=energies,
    )
    
    print(f"✅ Rerun complete: {run.run_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reproduce.py <config.yaml>", file=sys.stderr)
        sys.exit(1)
    
    main(sys.argv[1])
