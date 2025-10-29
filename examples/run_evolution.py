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

from lever import analysis, config, driver, evolution, models
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
        max_cycles=30,
        max_steps=1000,
    )
    
    # ========== Evaluation Configuration ==========
    eval_cfg = config.EvaluationConfig(
        var_energy_mode=config.EvalMode.FINAL,
        s_ci_energy_mode=config.EvalMode.FINAL,
        t_ci_energy_mode=config.EvalMode.NEVER
    )
    
    # ========== LEVER Configuration ==========
    lever_cfg = config.LeverConfig(
        system=sys_cfg,
        hamiltonian=ham_cfg,
        loop=loop_cfg,
        evaluation=eval_cfg,
        compute_mode=config.ComputeMode.EFFECTIVE,
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
        hidden_dims=(32,),
        param_dtype=jnp.complex64 
    )
    
    # RBM:
    # model = models.RBM(
    #     n_orbitals=sys_cfg.n_orbitals,
    #     seed=42,
    #     alpha=2  # Hidden/visible ratio
    # )
    
    # ========== Optimizer ==========
    optimizer = adam(
        learning_rate=5e-4,
        weight_decay=1e-4    # L2 regularization
    )
    
    # Stochastic Reconfiguration
    # optimizer = sr(
    #     damping=1e-3,
    #     backend="dense",     # or "cg" for conjugate gradient
    #     learning_rate=0.05
    # )
    
    # ========== Evolution Strategy ==========
    evo_strategy = evolution.BasicStrategy(
        scorer=evolution.scores.AmpScorer(),
        selector=evolution.selectors.TopKSelector(k=3200)
    )

    # ========== Run LEVER Workflow ==========
    lever_driver = driver.Driver(lever_cfg, model, evo_strategy, optimizer)
    results = lever_driver.run()

    # ========== Post-Analysis ==========
    # suite = analysis.AnalysisSuite(results, lever_driver.controller.int_ctx)
    # suite.print_summary()
    # suite.plot_conv(sys_name="N2_STO-3G")


if __name__ == "__main__":
    main()
