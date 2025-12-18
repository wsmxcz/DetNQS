# lever/examples/run_single.py
# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Command-line interface for LEVER quantum chemistry calculations.

LEVER (Low-rank Enhanced Variational Eigensolver with RBM) integrates
selected CI with neural quantum states for full-configuration interaction
in orbital space. Three operational modes are supported:
  - Variational: NQS-enhanced selected CI solving H_SS with PT2 correction
  - Effective: Down-folded C-space effects into effective Hamiltonian H_eff
  - Proxy: Diagonal approximation on full T = S ∪ C space

Workflow follows three-layer design:
  L0: Static C++ layer (integrals, heat-bath tables)
  L1: Outer loop (DetSpace evolution, Hamiltonian reconstruction)
  L2: Inner loop (JIT-compiled network optimization)

Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
File: lever/examples/run_single.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from lever.driver import EffectiveDriver, ProxyDriver, VariationalDriver, AsymmetricDriver
from lever import models
from lever.space import DetSpace
from lever.space.selector import TopKSelector, ThresholdSelector, TopFractionSelector
from lever.system import MolecularSystem


def parse_args() -> argparse.Namespace:
    """Parse LEVER command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LEVER: Neural quantum states for selected CI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fcidump",
        type=Path,
        help="FCIDUMP file with molecular integrals"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("variational", "effective", "proxy", "asymmetric"),
        default="variational",
        help="Computational mode selection"
    )
    return parser.parse_args()


def main() -> None:
    """Execute LEVER workflow with timing diagnostics."""
    args = parse_args()
  
    # JAX configuration
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)
    print(f"JAX devices: {jax.devices()}")
  
    start_time = time.perf_counter()
  
    # L0: Molecular system initialization
    system = MolecularSystem.from_fcidump(args.fcidump)

    # L1: Determinant space initialization
    hf_det = system.hf_determinant()
    detspace = DetSpace.initialize(hf_det)

    # User constructs parametrizer
    parametrizer = models.slater.MLP(
        n_so=system.n_so,
        dim=256,
        depth=1,
        param_dtype=jnp.float64,
    )

    # Construct Slater wavefunction
    model = models.make_slater(
        system=system,
        parametrizer=parametrizer,
        update="additive",
        n_det=1,
        rank=4,
        param_dtype=jnp.float64,
    )

    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=500, alpha=0.02)
    optimizer = optax.adamw(learning_rate=schedule)
    selector = TopKSelector(256)
    # selector = ThresholdSelector(threshold=1e-8)
    # selector = TopFractionSelector(0.99999)

    # Driver selection for compute mode
    driver_cls = {
        "variational": VariationalDriver,
        "effective": EffectiveDriver,
        "proxy": ProxyDriver,
        "asymmetric": AsymmetricDriver
    }[args.mode]
  
    driver = driver_cls.build(
        system=system,
        detspace=detspace,
        model=model,
        optimizer=optimizer,
        selector=selector,
        chunk_size=None,
        use_gpu_kernel=False,
    )
  
    # L2: Execute optimization cycles
    result = driver.run()
  
    # Final energy calculation: E_total = E_elec + E_nuc
    e_elec_final = result.energies[-1]
    e_total_final = e_elec_final + result.e_nuc
    elapsed = time.perf_counter() - start_time
  
    # Summary output
    print("\n" + "=" * 60)
    print(f"System: NORB={system.n_orb}, NELEC={system.n_elec}, MS2={system.ms2}")
    print(f"Mode: {args.mode.upper()}")
    print(f"E_nuc:   {result.e_nuc:>18.8f} Ha")
    print(f"E_elec:  {e_elec_final:>18.8f} Ha")
    print(f"E_total: {e_total_final:>18.8f} Ha")
    print(f"Runtime: {elapsed:>18.2f} s")
    print("=" * 60)


if __name__ == "__main__":
    main()