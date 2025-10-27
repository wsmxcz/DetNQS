# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER optimization with EFFECTIVE mode.

File: examples/run_evolution.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import lever
from lever import config, models, evolution, driver, analysis

import jax
import jax.numpy as jnp

# JAX configuration
print("JAX devices:", jax.devices())
jax.config.update("jax_platforms", "cuda")
jax.config.update("jax_log_compiles", False)


def main():
    # System configuration
    sys_cfg = config.SystemConfig(
        fcidump_path="../benchmark/FCIDUMP/CH4_sto3g.FCIDUMP",
        n_orbitals=9, 
        n_alpha=5, 
        n_beta=5
    )
    
    # Optimization configuration
    opt_cfg = config.OptimizationConfig(
        seed=42,
        learning_rate=5e-4,
        s_space_size=400, 
        steps_per_cycle=200, 
        num_cycles=10,
        report_interval=50
    )
    
    # Evaluation configuration
    eval_cfg = config.EvaluationConfig(
        var_energy_mode=config.EvalMode.NEVER,
        s_ci_energy_mode=config.EvalMode.NEVER,
        t_ci_energy_mode=config.EvalMode.NEVER
    )
    
    # Screening configuration
    screen_cfg = config.ScreeningConfig(
        mode=config.ScreenMode.DYNAMIC,
        screen_eps=1e-6
    )
    
    # LEVER configuration
    lever_cfg = config.LeverConfig(
        system=sys_cfg, 
        optimization=opt_cfg, 
        evaluation=eval_cfg,
        screening=screen_cfg,
        compute_mode=config.ComputeMode.EFFECTIVE,
    )
    
    print(f"Mode: {lever_cfg.compute_mode.value}")

    # Model initialization
    model = models.Backflow(
        n_orbitals=sys_cfg.n_orbitals, 
        n_alpha=sys_cfg.n_alpha, 
        n_beta=sys_cfg.n_beta,
        seed=opt_cfg.seed, 
        n_dets=1, 
        generalized=True, 
        restricted=False,
        hidden_dims=(256,), 
        param_dtype=jnp.complex64
    )
    
    # Evolution strategy
    evo_strategy = evolution.BasicStrategy(
        scorer=evolution.scores.AmpScorer(),
        selector=evolution.selectors.TopKSelector(k=opt_cfg.s_space_size)
    )

    # Run LEVER
    lever_driver = driver.Driver(lever_cfg, model, evo_strategy)
    results = lever_driver.run()

    # Analysis
    suite = analysis.AnalysisSuite(results, lever_driver.int_ctx)
    suite.print_summary()
    suite.plot_conv(sys_name="H2O_STO-3G")


if __name__ == "__main__":
    main()
