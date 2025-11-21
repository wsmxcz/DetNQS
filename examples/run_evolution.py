# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER variational optimization with post-analysis diagnostics.

File: examples/run_evolution.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import jax
import jax.numpy as jnp

from lever import analysis, config, driver, evolution, models
from lever.optimizers import adam
from lever.monitor import init_run, summary, storage
from lever.monitor.plotting import plot_outer_convergence, plot_inner_convergence


def main() -> None:
    """Execute complete LEVER variational optimization workflow."""
    print("JAX devices:", jax.devices())
    jax.config.update("jax_platforms", "cuda")
    jax.config.update("jax_log_compiles", False)

    # System configuration
    sys_cfg = config.SystemConfig(
        fcidump_path="../benchmark/FCIDUMP/N2_sto3g.FCIDUMP",
        n_orbitals=10,
        n_alpha=7,
        n_beta=7
    )

    ham_cfg = config.HamiltonianConfig(
        screening_mode=config.ScreenMode.DYNAMIC,
        screen_eps=1e-6,
        diag_shift=0.0,
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
    )

    model = models.Backflow(
        n_orbitals=sys_cfg.n_orbitals,
        n_alpha=sys_cfg.n_alpha,
        n_beta=sys_cfg.n_beta,
        seed=42,
        n_dets=1,
        generalized=True,
        restricted=False,
        hidden_dims=(256,),
        param_dtype=jnp.complex64
    )

    # Initialize run: System name "N2_sto3g" automatically inferred from FCIDUMP
    run = init_run(cfg=lever_cfg, root_dir="runs")
    run.save_config(lever_cfg, model=model)

    optimizer = adam(learning_rate=5e-4, weight_decay=1e-4)

    amp_scorer = evolution.scores.AmplitudeScorer()
    mass_selector = evolution.CumulativeMassSelector(mass_threshold=0.9999)
    evo_strategy = evolution.BasicStrategy(
        scorer=amp_scorer,
        selector=mass_selector,
    )

    lever_driver = driver.Driver(lever_cfg, model, evo_strategy, optimizer)
    result = lever_driver.run()

    evaluator = analysis.VariationalEvaluator(
        int_ctx=lever_driver.int_ctx,
        n_orb=sys_cfg.n_orbitals,
        e_nuc=lever_driver.int_ctx.get_e_nuc()
    )

    energies = evaluator.analyze_result(
        result,
        model=model,
        compute_fci=True,
        compute_s_ci=True,
        compute_sc_var=True
    )

    storage.append_energies_to_summary(run.root, energies)
    
    # Summary now auto-detects system name from run metadata
    summary.print_summary_from_run(
        run_dir=run.root,
        energies=energies,
    )

    # Clean plotting calls: No redundant system_name needed
    plot_outer_convergence(
        run_dir=run.root,
        e_fci=energies["e_fci"],
        log_scale=True,
        save_path=run.root / "convergence_outer.pdf"
    )

    plot_inner_convergence(
        run_dir=run.root,
        e_fci=energies["e_fci"],
        log_scale=False,
        save_path=run.root / "convergence_inner.pdf"
    )

    run.close()


if __name__ == "__main__":
    main()