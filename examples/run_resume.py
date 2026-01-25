# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Resume optimization from DetNQS checkpoint.

Restores complete driver state (V-space, parameters, optimizer) and continues
outer-loop evolution from the saved iteration.

Usage:
    python run_resume.py checkpoint.npz [--fcidump path] [--output dir]

Adjustable parameters (edit in script):
    - selector: Determinant selection strategy
    - max_outer: Additional outer iterations
    - chunk_size: Memory-efficient chunking

File: detnqs/examples/run_resume.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2026
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from detnqs import models
from detnqs.analysis import CheckpointCallback, CheckpointManager, ConsoleCallback, JsonCallback
from detnqs.analysis.metrics import compute_pt2, compute_variational, convergence_stats
from detnqs.space.selector import TopKSelector
from detnqs.system import MolecularSystem


_DRIVER_MAP = {
    "variational": "detnqs.driver.VariationalDriver",
    "effective": "detnqs.driver.EffectiveDriver",
    "proxy": "detnqs.driver.ProxyDriver",
    "asymmetric": "detnqs.driver.AsymmetricDriver",
}


def configure_jax() -> None:
    """Configure JAX: GPU priority, float64, suppress warnings."""
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)
    warnings.filterwarnings("ignore", message="Explicitly requested dtype.*is not available")
    print(f"JAX devices: {jax.devices()}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Resume DetNQS optimization from checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Checkpoint .npz file")
    parser.add_argument(
        "--fcidump", type=Path, default=None, help="Override FCIDUMP path if relocated"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output directory (default: checkpoint parent)"
    )
    return parser.parse_args()


def print_summary(
    system: MolecularSystem,
    mode: str,
    e_total: float,
    runtime: float,
    output_dir: Path,
) -> None:
    """Print final energy, system metadata, and timing."""
    print("\n" + "=" * 60)
    print(f"System: N_o={system.n_orb}, N_e={system.n_elec}, MS2={system.ms2}")
    print(f"Mode: {mode.upper()}")
    print(f"E_nuc:       {system.e_nuc:>18.8f} Ha")
    print(f"E_total:     {e_total:>18.8f} Ha")
    print(f"Runtime:     {runtime:>18.2f} s")
    print(f"Output:      {output_dir.resolve()}")
    print("=" * 60)


def run_post_analysis(driver, e_ref: float, mode: str) -> None:
    """
    Execute mode-specific post-analysis.

    - Variational: Decomposed PT2 correction over P-space
    - Proxy: Variational energies on V and T
    """
    if mode == "variational":
        print("\nComputing decomposed PT2 correction...")
        t_start = time.time()
        pt2_result = compute_pt2(
            state=driver.state,
            detspace=driver.detspace,
            system=driver.system,
            e_ref=e_ref,
        )
        t_pt2 = time.time() - t_start

        if pt2_result is not None:
            e_pt2_int = pt2_result["e_pt2_internal"]
            e_pt2_ext = pt2_result["e_pt2_external"]
            e_pt2_tot = pt2_result["e_pt2_total"]
            n_ext = pt2_result["n_ext"]
            e_total_pt2 = e_ref + e_pt2_tot

            print(f"\nPT2 results (computed in {t_pt2:.2f}s):")
            print(f"  E_var:                  {e_ref:>18.8f} Ha")
            print(f"  Delta E_PT2 (internal): {e_pt2_int:>18.8f} Ha")
            print(f"  Delta E_PT2 (external): {e_pt2_ext:>18.8f} Ha")
            print(f"  Delta E_PT2 (total):    {e_pt2_tot:>18.8f} Ha")
            print(f"  E_total(PT2):           {e_total_pt2:>18.8f} Ha")
            print(f"  |P|:                    {n_ext}")

    elif mode == "proxy":
        print("\nComputing variational energies...")
        var_result = compute_variational(
            state=driver.state,
            detspace=driver.detspace,
            system=driver.system,
        )
        if var_result:
            print("\nVariational energy analysis:")
            print(f"  E_var(V): {var_result['e_var_v']:>18.8f} Ha")
            if "e_var_t" in var_result:
                print(f"  E_var(T): {var_result['e_var_t']:>18.8f} Ha")


def main() -> None:
    """Resume optimization from checkpoint and continue evolution."""
    args = parse_args()
    configure_jax()

    # =====================================================================
    # User-configurable parameters (edit here)
    # =====================================================================
  
    # Determinant selector
    selector = TopKSelector(8192, stream=True)
  
    # Additional outer iterations beyond checkpoint
    max_outer = 30
  
    # Memory-efficient chunking for forward/backward passes
    chunk_size = 8192
  
    # =====================================================================

    # Load checkpoint metadata
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path.name}")
    data = CheckpointManager.load(ckpt_path)
    meta = data["metadata"]

    # Locate FCIDUMP and rebuild molecular system
    fcidump_path = Path(args.fcidump or meta["fcidump_path"])
    if not fcidump_path.exists():
        raise FileNotFoundError(
            f"FCIDUMP not found: {fcidump_path}\n"
            f"Use --fcidump to specify correct path"
        )

    print(f"Loading molecular system: {fcidump_path.name}")
    system = MolecularSystem.from_fcidump(fcidump_path)
    print(f"System: N_o={system.n_orb}, N_e={system.n_elec}, MS2={system.ms2}")

    # Rebuild model architecture matching checkpoint
    parametrizer = models.parametrizers.MLP(
        n_so=system.n_so,
        dim=256,
        depth=2,
        param_dtype=jnp.float64,
    )

    model = models.make_backflow(
        system=system,
        parametrizer=parametrizer,
        mapper="full",
        use_fast_kernel=True,
        param_dtype=jnp.float64,
    )

    # Rebuild optimizer
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=800, alpha=0.05)
    optimizer = optax.adamw(learning_rate=schedule)

    # Resolve driver class from metadata
    mode = meta["mode"]
    if mode not in _DRIVER_MAP:
        raise ValueError(f"Unknown mode in checkpoint: {mode}")

    module_path, class_name = _DRIVER_MAP[mode].rsplit(".", 1)
    import importlib
    driver_cls = getattr(importlib.import_module(module_path), class_name)

    # Resume driver state
    print(f"\nResuming {mode} mode from step {meta['outer_step']}")
    driver = CheckpointManager.resume(
        path=ckpt_path,
        driver_cls=driver_cls,
        model=model,
        optimizer=optimizer,
        fcidump_path=fcidump_path,
        selector=selector,
        max_outer=max_outer,
        chunk_size=chunk_size,
    )

    print(f"Restored: |V| = {driver.detspace.size_V}, params shape verified")

    # Setup output directory
    output_dir = args.output or ckpt_path.parent.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register callbacks
    callbacks = [
        ConsoleCallback(every=1),
        JsonCallback(output_dir / "trace_resume.jsonl"),
        CheckpointCallback(output_dir / "checkpoints_resume", interval=5, keep_last=5),
    ]

    # Continue optimization
    print(f"\nResuming optimization for {max_outer} additional iterations...")
    t_start = time.time()
    driver.run(callbacks=callbacks)
    runtime = time.time() - t_start

    # Extract final statistics
    stats = convergence_stats(output_dir / "trace_resume.jsonl")
    if stats:
        e_total = stats["final_energy"]
        print_summary(system, mode, e_total, runtime, output_dir)
        run_post_analysis(driver, e_total, mode)
    else:
        print(f"\nOptimization completed in {runtime:.2f}s (no trace file for summary).")


if __name__ == "__main__":
    main()