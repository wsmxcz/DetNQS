# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER optimization example.

File: examples/run_evolution.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import jax
import jax.numpy as jnp

from lever import core, analysis, config, driver, evolution, models
from lever.optimizers import adam

# JAX device configuration
print("JAX devices:", jax.devices())
jax.config.update("jax_platforms", "cuda")
jax.config.update("jax_log_compiles", False)


def main():
    # ========== System Definition ==========
    sys_cfg = config.SystemConfig(
        fcidump_path="../benchmark/FCIDUMP/H2O_631g.FCIDUMP",
        n_orbitals=13,
        n_alpha=5,
        n_beta=5
    )
    
    # ========== Hamiltonian Configuration ==========
    ham_cfg = config.HamiltonianConfig(
        screening_mode=config.ScreenMode.DYNAMIC,
        screen_eps=1e-6,
        diag_shift=0.5,
        reg_eps=1e-4
    )
    
    # ========== Loop Control ==========
    loop_cfg = config.LoopConfig(
        max_outer=10,
        outer_tol=1e-5,
        outer_patience=3,
        inner_steps=500,
        chunk_size=8192
    )

    
    # ========== Evaluation Configuration ==========
    eval_cfg = config.EvaluationConfig(
        var_energy_mode=config.EvalMode.NEVER,
        s_ci_energy_mode=config.EvalMode.FINAL,
        t_ci_energy_mode=config.EvalMode.NEVER
    )
    
    # ========== LEVER Configuration ==========
    lever_cfg = config.LeverConfig(
        system=sys_cfg,
        hamiltonian=ham_cfg,
        loop=loop_cfg,
        evaluation=eval_cfg,
        compute_mode=config.ComputeMode.PROXY,
    )

    # ========== Wavefunction Model ==========
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
    
    # ========== Optimizer ==========
    optimizer = adam(
        learning_rate=5e-4,
        weight_decay=1e-4
    )
    
    # ========== Evolution Strategy ==========
    evo_strategy = evolution.BasicStrategy(
        scorer=evolution.scores.AmpScorer(),
        selector=evolution.selectors.TopKSelector(k=400)
    )

    # ========== Run LEVER Workflow ==========
    lever_driver = driver.Driver(lever_cfg, model, evo_strategy, optimizer)
    results = lever_driver.run()

    # ========== Post-Analysis ==========
    e_fci = -76.12089
    
    # Print summary with FCI comparison
    analysis.print_summary(results, e_fci=e_fci, sys_name="H2O")
    
    # Plot convergence trajectory
    analysis.plot_convergence(
        results, 
        e_fci=e_fci,
        sys_name="H2O",
        save_path="convergence.pdf"
    )


if __name__ == "__main__":
    main()
