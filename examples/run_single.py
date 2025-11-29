# File: examples/n2_sto3g.py
# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER Example: N2 molecule in STO-3G basis.

Demonstrates Python-first workflow with unified RunContext and
system.json / FCIDUMP layout for quantum chemistry simulations.
"""

from __future__ import annotations

import jax

from lever import analysis, evolution, models, solver
from lever.config import (
    ExperimentConfig,
    HamiltonianConfig,
    LoopConfig,
    RuntimeConfig,
    ScreenMode,
    ComputeMode,
)
from lever.interface import MoleculeBuilder
from lever.utils.monitor import RunContext
from lever.optimizers import adam, cosine_decay_schedule


SYSTEM_NAME = "H2_ccpvtz"

def main() -> None:
    """Main workflow: setup → physics → configuration → execution → analysis."""
    
    # 0. Initialize run context with timestamped directory
    run = RunContext.create(system_name=SYSTEM_NAME, root_dir="runs")

    # 1. Define molecular system and generate Hamiltonian data
    meta = (
        MoleculeBuilder(
            atom="H 0 0 -1.0; H 0 0 1.0",
            basis="cc-pvtz",
            work_dir=run.root,
            name=SYSTEM_NAME,
        )
        .run_scf()
        .run_benchmarks({"fci"})
        .export()
    )

    # 2. Configure experiment with physics and runtime parameters
    cfg = ExperimentConfig.from_meta(
        meta,
        root_dir=run.root,
        hamiltonian=HamiltonianConfig(
            screening_mode=ScreenMode.DYNAMIC,
            screen_eps=1e-6,
        ),
        loop=LoopConfig(
            max_outer=20,
            max_inner=1000,
            chunk_size=8192,
        ),
        runtime=RuntimeConfig(
            enable_x64=True,
            compute_mode=ComputeMode.PROXY,
        ),
    )

    # 3. Model and Optimizer
    model = models.Backflow(
        n_orb=cfg.system.n_orb,
        n_alpha=cfg.system.n_alpha, 
        n_beta=cfg.system.n_beta,
        seed=42,
        n_dets=1,
        generalized=True,
        hidden_dims=(64, 64), 
    )

    lr_schedule = cosine_decay_schedule(
        init_value=5e-4,
        decay_steps=800,
        alpha=0.02,
    )
    optimizer = adam(learning_rate=lr_schedule, weight_decay=1e-4)

    # Evolution strategy
    strategy = evolution.BasicStrategy(
        scorer=evolution.scores.AmplitudeScorer(),
        selector=evolution.CumulativeMassSelector(0.9999, max_size=8192), 
    )

    # 4. Execute solver
    result, diag = solver.solve(cfg, model, strategy, optimizer, run_ctx=run)

    # 5. Analyze results and record experiment artifacts
    int_ctx = diag["int_ctx"]  # Integration context for expectation values
    e_nuc = diag["e_nuc"]      # Nuclear repulsion energy

    evaluator = analysis.VariationalEvaluator(
        int_ctx=int_ctx,
        n_orb=cfg.system.n_orb,
        e_nuc=e_nuc,
    )

    # Compute variational energies and wavefunction analysis
    energies = evaluator.analyze_result(
        result,
        model=model,
        compute_s_ci=True,
        compute_sc_var=True,
    )

    # Record complete experiment data for reproducibility
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


if __name__ == "__main__":
    main()
