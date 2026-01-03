# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for DetNQS quantum chemistry calculations.

Computational modes:
  - Variational: NQS-enhanced sCI on H_VV with optional PT2 analysis
  - Effective:   Löwdin down-folded H_eff on V-space
  - Proxy:       Diagonal approximation on T = V ∪ P
  - Asymmetric:  VMC-style truncated estimator on V-space

File: detnqs/examples/run_single.py
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

from detnqs import models
from detnqs.analysis import CheckpointCallback, ConsoleCallback, JsonCallback
from detnqs.analysis.metrics import compute_pt2, compute_variational, convergence_stats
from detnqs.driver import AsymmetricDriver, EffectiveDriver, ProxyDriver, VariationalDriver
from detnqs.space import DetSpace
from detnqs.space.selector import TopKSelector, ThresholdSelector
from detnqs.system import MolecularSystem


_DRIVER_MAP = {
    "variational": VariationalDriver,
    "effective": EffectiveDriver,
    "proxy": ProxyDriver,
    "asymmetric": AsymmetricDriver,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mode, input, and output paths."""
    parser = argparse.ArgumentParser(
        description="DetNQS: Neural quantum states for selected CI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("fcidump", type=Path, help="Path to FCIDUMP integral file")
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(_DRIVER_MAP.keys()),
        default="variational",
        help="Computational mode",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output directory for logs and checkpoints"
    )
    return parser.parse_args()


def configure_jax() -> None:
    """Set JAX runtime: GPU priority, float64, suppress compile warnings."""
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)

    warnings.filterwarnings(
        "ignore", message="Explicitly requested dtype.*is not available"
    )

    print(f"JAX devices: {jax.devices()}")


def print_summary(
    system: MolecularSystem,
    mode: str,
    stats: dict,
    output_dir: Path | None = None,
) -> None:
    """Print final energy and system metadata."""
    print("\n" + "=" * 60)
    print(f"System: N_o={system.n_orb}, N_e={system.n_elec}, MS2={system.ms2}")
    print(f"Mode: {mode.upper()}")
    print(f"E_nuc:       {system.e_nuc:>18.8f} Ha")
    print(f"E_elec:      {stats['final_energy']:>18.8f} Ha")
    print(f"E_total:     {stats['final_energy'] + system.e_nuc:>18.8f} Ha")
    print(f"Runtime:     {stats['total_time']:>18.2f} s")
    if output_dir:
        print(f"Output:      {output_dir.resolve()}")
    print("=" * 60)


def run_post_analysis(
    driver,
    e_ref_elec: float,
    mode: str,
) -> None:
    """
    Execute mode-specific post-processing.

    - Variational: Compute PT2 correction over P-space
    - Proxy: Evaluate variational energies on V and T

    Args:
        driver: Optimized driver with final detspace
        e_ref_elec: Electronic energy from optimization
        mode: Computational mode identifier
    """
    print("\nPost-processing analysis...")

    if mode == "variational":
        # PT2 from P-space: E_total = E_var + Delta E_PT2
        e_pt2 = compute_pt2(
            state=driver.state,
            detspace=driver.detspace,
            system=driver.system,
            e_ref_elec=e_ref_elec,
        )
        if e_pt2 is not None:
            e_var_total = e_ref_elec + driver.system.e_nuc
            e_total = e_var_total + e_pt2
            print("\nPT2 correction:")
            print(f"  E_var:   {e_var_total:>18.8f} Ha")
            print(f"  E_PT2:   {e_pt2:>18.8f} Ha")
            print(f"  E_total: {e_total:>18.8f} Ha")

    elif mode == "proxy":
        # Variational energies on V and T spaces
        var_result = compute_variational(
            state=driver.state,
            detspace=driver.detspace,
            system=driver.system,
        )
        if var_result is not None:
            print("\nVariational energy analysis:")
            print(f"  E_var(V): {var_result['e_var_v']:>18.8f} Ha")
            if "e_var_t" in var_result:
                print(f"  E_var(T): {var_result['e_var_t']:>18.8f} Ha")


def main() -> None:
    """Execute DetNQS workflow: load system, optimize ansatz, analyze results."""
    args = parse_args()
    configure_jax()

    # L0: Load molecular system and initialize Hartree-Fock reference
    system = MolecularSystem.from_fcidump(args.fcidump)
    hf_det = system.hf_determinant()
    detspace = DetSpace.initialize(hf_det)

    # Build neural ansatz: MLP parametrizer + Slater2nd architecture
    parametrizer = models.parametrizers.MLP(
        n_so=system.n_so,
        dim=256,
        depth=1,
        param_dtype=jnp.float64,
    )

    model = models.make_slater2nd(
        system=system,
        parametrizer=parametrizer,
        mapper="full",
        use_fast_kernel=True,
        param_dtype=jnp.float64,
    )

    # Configure optimizer and determinant selector
    optimizer = optax.adamw(learning_rate=5e-4)
    selector = TopKSelector(40, stream=True)

    # L1: Instantiate driver for selected computational mode
    driver = _DRIVER_MAP[args.mode].build(
        system=system,
        detspace=detspace,
        model=model,
        optimizer=optimizer,
        selector=selector,
        chunk_size=8192,
    )

    # Register callbacks for monitoring and checkpointing
    callbacks = [ConsoleCallback(every=1)]
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        callbacks.extend(
            [
                JsonCallback(args.output / "trace.jsonl"),
                CheckpointCallback(
                    args.output / "checkpoints", interval=5, keep_last=5
                ),
            ]
        )

    # L2: Run outer loop (detspace evolution) + inner loop (parameter updates)
    driver.run(callbacks=callbacks)

    # Extract convergence statistics and post-process
    if args.output is not None:
        trace_file = args.output / "trace.jsonl"
        stats = convergence_stats(trace_file)
        if stats:
            print_summary(system, args.mode, stats, args.output)
            run_post_analysis(
                driver=driver,
                e_ref_elec=stats["final_energy"],
                mode=args.mode,
            )
    else:
        print("\nOptimization completed (no trace file for summary).")


if __name__ == "__main__":
    main()