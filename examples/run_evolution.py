# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER variational optimization with post-analysis diagnostics.

Demonstrates complete workflow: Hamiltonian screening → inner loop
variational optimization → outer loop S-space refinement → energy
diagnostics (E_LEVER, E_var, E_S-CI).

File: examples/run_evolution.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import jax
import jax.numpy as jnp

from lever import analysis, config, driver, evolution, models, plotting
from lever.optimizers import adam, sr

# Device setup
print("JAX devices:", jax.devices())
jax.config.update("jax_platforms", "cuda")
jax.config.update("jax_log_compiles", False)

# Debug constant capture (optional)
# jax.config.update('jax_captured_constants_warn_bytes', 64 * 1024**2)
# jax.config.update('jax_captured_constants_report_frames', 5)


def main():
    # System
    sys_cfg = config.SystemConfig(
        fcidump_path="../benchmark/FCIDUMP/H2O_631g.FCIDUMP",
        n_orbitals=13,
        n_alpha=5,
        n_beta=5
    )
    
    # Hamiltonian
    ham_cfg = config.HamiltonianConfig(
        screening_mode=config.ScreenMode.DYNAMIC,
        screen_eps=1e-6,
        diag_shift=0.0,
    )
    
    # Convergence control
    loop_cfg = config.LoopConfig(
        max_outer=10,
        outer_tol=1e-5,
        outer_patience=3,
        inner_steps=500,
        chunk_size=8192
    )
    
    lever_cfg = config.LeverConfig(
        system=sys_cfg,
        hamiltonian=ham_cfg,
        loop=loop_cfg,
        compute_mode=config.ComputeMode.PROXY,
    )

    # Wavefunction Ansatz
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
    
    optimizer = adam(learning_rate=5e-4, weight_decay=1e-4)
    
    # Evolution: amplitude-based top-Percentage selection
    evo_strategy = evolution.CumulativeMassStrategy(mass_threshold=0.999)
    
    lever_driver = driver.Driver(lever_cfg, model, evo_strategy, optimizer)
    result = lever_driver.run()

    # Post Analysis
    evaluator = analysis.VariationalEvaluator(
        int_ctx=lever_driver.int_ctx,
        n_orb=sys_cfg.n_orbitals,
        e_nuc=lever_driver.int_ctx.get_e_nuc()
    )
    
    energies = evaluator.analyze_result(
        result,
        model=model,
        compute_fci=False,  # Enable for FCI benchmark (expensive)
        compute_s_ci=True,
        compute_sc_var=True
    )
    
    # Optional: inject FCI reference
    # energies['e_fci'] = -109.099941428023  # N2/6-31G benchmark
    energies['e_fci'] = -76.12087435  # H2O/6-31G benchmark
    # energies['e_fci'] = -151.146949995833  # C2H4O/STO-3G benchmark


    plotting.print_summary(
        energies=energies,
        total_time=result.total_time,
        sys_name="System"
    )
    
if __name__ == "__main__":
    main()
