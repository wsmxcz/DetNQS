# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for LEVER quantum chemistry calculations.

Integrates selected CI with neural quantum states. Three modes:
  - Variational: NQS-enhanced sCI, solve H_SS with PT2 analysis
  - Effective: C-space down-folded into H_eff on S-space
  - Proxy: Diagonal approximation on T = S ∪ C

File: lever/examples/run_single.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from lever import models
from lever.analysis import CheckpointCallback, ConsoleCallback, JsonCallback
from lever.analysis.metrics import compute_pt2, compute_variational, convergence_stats
from lever.driver import AsymmetricDriver, EffectiveDriver, ProxyDriver, VariationalDriver
from lever.space import DetSpace
from lever.space.selector import TopKSelector
from lever.system import MolecularSystem


# Driver registry for mode selection
_DRIVER_MAP = {
    "variational": VariationalDriver,
    "effective": EffectiveDriver,
    "proxy": ProxyDriver,
    "asymmetric": AsymmetricDriver,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LEVER workflow."""
    parser = argparse.ArgumentParser(
        description="LEVER: Neural quantum states for selected CI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fcidump", type=Path, help="Path to FCIDUMP file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(_DRIVER_MAP.keys()),
        default="variational",
        help="Computational mode",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    return parser.parse_args()


def configure_jax() -> None:
    """Configure JAX runtime settings."""
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)
  
    # Suppress dtype conversion warnings
    warnings.filterwarnings("ignore", message="Explicitly requested dtype.*is not available")
  
    print(f"JAX devices: {jax.devices()}")


def print_summary(
    system: MolecularSystem,
    mode: str,
    stats: dict,
    output_dir: Path | None = None,
) -> None:
    """Print calculation summary."""
    print("\n" + "=" * 60)
    print(f"System: N_orb={system.n_orb}, N_elec={system.n_elec}, MS2={system.ms2}")
    print(f"Mode: {mode.upper()}")
    print(f"E_nuc:       {system.e_nuc:>18.8f} Ha")
    print(f"E_elec:      {stats['final_energy']:>18.8f} Ha")
    print(f"E_total:     {stats['final_energy'] + system.e_nuc:>18.8f} Ha")
    print(f"Runtime:     {stats['total_time']:>18.2f} s")
    if output_dir:
        print(f"Output:      {output_dir.resolve()}")
    print("=" * 60)


def run_post_analysis(driver, mode: str) -> None:
    """Execute post-hoc analysis based on mode."""
    print("\nPost-processing analysis...")
  
    if mode == "variational":
        # PT2 correction for variational mode
        pt2_result = compute_pt2(driver.state, driver.detspace, driver.system)
        if pt2_result is not None:
            print("\nPT2 correction:")
            print(f"  E_var:   {pt2_result['e_var']:>18.8f} Ha")
            print(f"  E_PT2:   {pt2_result['e_pt2']:>18.8f} Ha")
            print(f"  E_total: {pt2_result['e_total']:>18.8f} Ha")
  
    elif mode == "proxy":
        # Variational energy on full T-space
        var_result = compute_variational(driver.state, driver.detspace, driver.system)
        if var_result is not None:
            print("\nVariational energy on T-space:")
            print(f"  E_var:   {var_result['e_var']:>18.8f} Ha")


def main() -> None:
    """Execute LEVER workflow with specified mode and configuration."""
    args = parse_args()
    configure_jax()
  
    # Initialize molecular system
    system = MolecularSystem.from_fcidump(args.fcidump)
    hf_det = system.hf_determinant()
    detspace = DetSpace.initialize(hf_det)
  
    # Build neural network model: MLP + Slater2nd
    parametrizer = models.parametrizers.MLP(
        n_so=system.n_so,
        dim=256,
        depth=1,
        param_dtype=jnp.float64,
    )
  
    model = models.make_slater2nd(
        system=system,
        parametrizer=parametrizer,
        mapper="thouless",
        kmax=8,
        use_fast_kernel=True,
        param_dtype=jnp.float64,
    )
  
    # Configure optimizer and determinant selector
    optimizer = optax.adamw(learning_rate=5e-4)
    selector = TopKSelector(1024)
  
    # Build driver based on selected mode
    driver = _DRIVER_MAP[args.mode].build(
        system=system,
        detspace=detspace,
        model=model,
        optimizer=optimizer,
        selector=selector,
        chunk_size=8192,
    )
  
    # Setup callbacks
    callbacks = [ConsoleCallback(every=1)]
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        callbacks.extend([
            JsonCallback(args.output / "trace.jsonl"),
            CheckpointCallback(args.output / "checkpoints", interval=5, keep_last=5),
        ])
  
    # Run optimization
    driver.run(callbacks=callbacks)
  
    # Print summary if trace file exists
    if args.output is not None:
        trace_file = args.output / "trace.jsonl"
        stats = convergence_stats(trace_file)
        if stats:
            print_summary(system, args.mode, stats, args.output)
  
    # Post-hoc analysis
    run_post_analysis(driver, args.mode)


if __name__ == "__main__":
    main()