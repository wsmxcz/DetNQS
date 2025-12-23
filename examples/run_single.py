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
from lever.analysis import save
from lever.analysis.metrics import compute_pt2, compute_variational
from lever.driver import AsymmetricDriver, EffectiveDriver, ProxyDriver, VariationalDriver
from lever.space import DetSpace
from lever.space.selector import TopKSelector
from lever.system import MolecularSystem


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LEVER: Neural quantum states for selected CI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fcidump",
        type=Path,
        help="Path to FCIDUMP file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("variational", "effective", "proxy", "asymmetric"),
        default="variational",
        help="Computational mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    return parser.parse_args()


def main() -> None:
    """Execute LEVER workflow."""
    args = parse_args()

    # Configure JAX
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)
    print(f"JAX devices: {jax.devices()}")
    warnings.filterwarnings("ignore", message="Explicitly requested dtype.*is not available")

    # Initialize system
    system = MolecularSystem.from_fcidump(args.fcidump)
    hf_det = system.hf_determinant()
    detspace = DetSpace.initialize(hf_det)

    # Build model
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
        kmax=8,
        use_fast_kernel=True,
        param_dtype=jnp.float64,
    )

    # Configure optimizer and selector
    optimizer = optax.adamw(learning_rate=5e-4)
    selector = TopKSelector(1024)

    # Select driver
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
        verbose=True,
    )

    # Execute optimization
    result = driver.run()
    trace = result.trace

    # Extract final metrics
    e_elec = trace.energies[-1]
    e_total = e_elec + result.e_nuc
    frac_s, frac_c = trace.norm_fractions()

    # Print summary
    print("\n" + "=" * 60)
    print(f"System: N_orb={system.n_orb}, N_elec={system.n_elec}, MS2={system.ms2}")
    print(f"Mode: {args.mode.upper()}")
    print(f"E_nuc:       {result.e_nuc:>18.8f} Ha")
    print(f"E_elec:      {e_elec:>18.8f} Ha")
    print(f"E_total:     {e_total:>18.8f} Ha")
    print(f"||ψ_S||²:    {frac_s[-1]:>18.6f}")
    print(f"||ψ_C||²:    {frac_c[-1]:>18.6f}")
    print(f"Runtime:     {trace.total_time:>18.2f} s")
    print("=" * 60)

    # Post-hoc analysis and save
    if args.output is not None:
        print(f"\nSaving results to {args.output}")
        analysis = {}

        # Mode-specific analysis
        if args.mode == "variational":
            pt2_result = compute_pt2(result, system)
            if pt2_result is not None:
                analysis["pt2"] = pt2_result
                print(f"\nPT2 correction:")
                print(f"  E_var:   {pt2_result['e_var']:>18.8f} Ha")
                print(f"  E_PT2:   {pt2_result['e_pt2']:>18.8f} Ha")
                print(f"  E_total: {pt2_result['e_total']:>18.8f} Ha")

        elif args.mode == "proxy":
            var_result = compute_variational(result, system)
            if var_result is not None:
                analysis["variational"] = var_result
                print(f"\nVariational energy on T-space:")
                print(f"  E_var:   {var_result['e_var']:>18.8f} Ha")

        save(
            args.output,
            result=result,
            trace=trace,
            analysis=analysis if analysis else None,
        )
        print("Results saved successfully")


if __name__ == "__main__":
    main()
