# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LEVER variational optimization example for N2 in STO-3G basis.

Demonstrates complete workflow: molecule setup → configuration → 
wavefunction optimization → analysis and visualization.

File: examples/run_evolution.py  
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import jax
import jax.numpy as jnp
import optax

from lever import analysis, config, driver, evolution, models
from lever.optimizers import adam
from lever.interface import MoleculeBuilder
from lever.monitor import init_run, summary, storage
from lever.monitor.plotting import plot_outer_convergence, plot_inner_convergence

# System identifier for file naming and metadata
SYSTEM_NAME = "N2_sto3g"


def main() -> None:
    """Execute complete LEVER variational optimization workflow."""
    
    # JAX runtime configuration
    jax.config.update("jax_disable_jit", False)
    jax.config.update("jax_log_compiles", False)
    
    # Precision setup - applied during LeverConfig construction
    precision_cfg = config.PrecisionConfig(enable_x64=True)
    print("x64 enabled:", jax.config.read("jax_enable_x64"))
    print("JAX devices:", jax.devices())
    
    # Initialize run directory and logging system
    run = init_run(cfg=None, system_name=SYSTEM_NAME, root_dir="runs")

    # 1) Molecular system setup using PySCF interface
    builder = (
        MoleculeBuilder(
            atom="N 0 0 0; N 0 0 10.0",  # N₂ molecule with 10.0 Bohr separation
            basis="sto-3g",
            charge=0,
            spin=0,
            work_dir=run.root,
        )
        .run_scf()
        .run_benchmarks({"hf", "mp2", "cisd", "ccsd", "ccsd_t", "fci"})
    )

    # Export FCIDUMP and HDF5 files with consistent naming
    exported = builder.export(SYSTEM_NAME)

    # 2) LEVER configuration from molecular data
    sys_cfg = config.SystemConfig.from_hdf5(str(exported.hdf5_path))

    ham_cfg = config.HamiltonianConfig(
        screening_mode=config.ScreenMode.DYNAMIC,
        screen_eps=1e-6,      # Dynamic screening threshold
        diag_shift=0.,       # No diagonal shift
    )

    loop_cfg = config.LoopConfig(
        max_outer=30,         # Maximum outer iterations
        max_inner=500,        # Maximum inner iterations  
        chunk_size=8192       # Batch size for sampling
    )
    
    lever_cfg = config.LeverConfig(
        system=sys_cfg,
        hamiltonian=ham_cfg,
        loop=loop_cfg,
        compute_mode=config.ComputeMode.PROXY,  # Full T-space treatment
        spin_flip_symmetry=False,
        precision=precision_cfg,
    )

    # 3) Neural wavefunction model construction
    model = models.Backflow(
        n_orbitals=sys_cfg.n_orbitals,
        n_alpha=sys_cfg.n_alpha,
        n_beta=sys_cfg.n_beta,
        seed=42,              # Random seed for reproducibility
        n_dets=1,             # Single determinant
        generalized=True,     # Generalized backflow transformations
        restricted=False,     # Unrestricted orbitals
        hidden_dims=(64, 64), # Neural network architecture
        precision=precision_cfg,
    )

    # Persist configuration and model metadata
    run.save_config(lever_cfg, model=model)
    
    # 4) Optimization and evolution setup
    lr_schedule = optax.cosine_decay_schedule(
        init_value=5e-4,      # Initial learning rate
        decay_steps=500,      # Decay over 500 steps
        alpha=0.01,
    )
    optimizer = adam(learning_rate=lr_schedule, weight_decay=1e-4)

    # Evolution strategy components
    amp_scorer = evolution.scores.AmplitudeScorer()  # |ψ|²-based scoring
    mass_selector = evolution.CumulativeMassSelector(
        mass_threshold=0.999,  # Select configurations covering 99.99% probability mass
        max_size=8192,          # Maximum selected configurations
    )
    evo_strategy = evolution.BasicStrategy(
        scorer=amp_scorer,
        selector=mass_selector,
    )

    # 5) Execute variational optimization
    lever_driver = driver.Driver(lever_cfg, model, evo_strategy, optimizer)
    result = lever_driver.run()

    # 6) Post-optimization analysis
    evaluator = analysis.VariationalEvaluator(
        int_ctx=lever_driver.int_ctx,
        n_orb=sys_cfg.n_orbitals,
        e_nuc=lever_driver.int_ctx.get_e_nuc()  # Nuclear repulsion energy
    )

    energies = evaluator.analyze_result(
        result,
        model=model,
        compute_s_ci=True,    # S-space CI analysis
        compute_sc_var=True,  # S-C variational analysis
    )

    storage.append_energies_to_summary(run.root, energies)
    
    # Print comprehensive summary
    summary.print_summary_from_run(
        run_dir=run.root,
        energies=energies,
    )

    # 7) Generate convergence plots
    plot_outer_convergence(
        run_dir=run.root,
        log_scale=True,                    # Log-scale for energy differences
        save_path=run.root / "convergence_outer.pdf",
    )

    plot_inner_convergence(
        run_dir=run.root,
        log_scale=False,                   # Linear scale for inner iterations
        save_path=run.root / "convergence_inner.pdf",
    )

    run.close()


if __name__ == "__main__":
    main()
