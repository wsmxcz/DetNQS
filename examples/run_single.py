# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for LEVER quantum chemistry calculations.

LEVER integrates selected CI with neural quantum states for full-CI
in orbital space. Three operational modes:
  - Variational: NQS-enhanced sCI, solve H_SS with PT2 correction
  - Effective: C-space effects down-folded into H_eff on S-space
  - Proxy: Diagonal approximation on full T = S ∪ C space

Workflow design:
  L0: Static C++ layer (integrals, heat-bath tables)
  L1: Outer loop (S/C space evolution, H reconstruction)
  L2: Inner loop (JIT-compiled network optimization)

File: lever/examples/run_single.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from lever import models
from lever.driver import AsymmetricDriver, EffectiveDriver, ProxyDriver, VariationalDriver
from lever.space import DetSpace
from lever.space.selector import ThresholdSelector, TopFractionSelector, TopKSelector
from lever.system import MolecularSystem


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LEVER execution."""
    parser = argparse.ArgumentParser(
        description="LEVER: Neural quantum states for selected CI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fcidump",
        type=Path,
        help="Path to FCIDUMP file containing molecular integrals",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("variational", "effective", "proxy", "asymmetric"),
        default="variational",
        help="Computational mode: variational|effective|proxy|asymmetric",
    )
    return parser.parse_args()


def main() -> None:
    """Execute LEVER workflow with timing diagnostics."""
    args = parse_args()

    # Configure JAX runtime
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)
    print(f"JAX devices: {jax.devices()}")
    warnings.filterwarnings("ignore", message="Explicitly requested dtype.*is not available")

    start_time = time.perf_counter()

    # L0: Initialize molecular system from FCIDUMP
    system = MolecularSystem.from_fcidump(args.fcidump)

    # L1: Initialize determinant space with HF reference
    hf_det = system.hf_determinant()
    detspace = DetSpace.initialize(hf_det)

    # Construct NQS parametrizer
    parametrizer = models.slater.MLP(
        n_so=system.n_so,
        dim=256,
        depth=1,
        param_dtype=jnp.float64,
    )

    # Build Slater-type wavefunction model
    model = models.make_slater(
        system=system,
        parametrizer=parametrizer,
        update="additive",
        n_det=1,
        rank=4,
        param_dtype=jnp.float64,
    )

    # Configure optimizer and selector
    optimizer = optax.adamw(learning_rate=5e-4)
    selector = TopKSelector(256)

    # Select driver based on computational mode
    driver_cls = {
        "variational": VariationalDriver,
        "effective": EffectiveDriver,
        "proxy": ProxyDriver,
        "asymmetric": AsymmetricDriver,
    }[args.mode]

    driver = driver_cls.build(
        system=system,
        detspace=detspace,
        model=model,
        optimizer=optimizer,
        selector=selector,
        chunk_size=8192,
        use_gpu_kernel=False,
    )

    # L2: Execute optimization with outer/inner loop structure
    result = driver.run()

    # Compute final total energy: E_total = E_elec + E_nuc
    e_elec_final = result.energies[-1]
    e_total_final = e_elec_final + result.e_nuc
    elapsed = time.perf_counter() - start_time

    # Print summary
    print("\n" + "=" * 60)
    print(f"System: N_orb={system.n_orb}, N_elec={system.n_elec}, MS2={system.ms2}")
    print(f"Mode: {args.mode.upper()}")
    print(f"E_nuc:   {result.e_nuc:>18.8f} Ha")
    print(f"E_elec:  {e_elec_final:>18.8f} Ha")
    print(f"E_total: {e_total_final:>18.8f} Ha")
    print(f"Runtime: {elapsed:>18.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()