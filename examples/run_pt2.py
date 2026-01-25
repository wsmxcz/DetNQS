# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PT2 correction calculator from checkpoint files.

Computes Epstein-Nesbet second-order perturbation correction:
    Delta E_PT2 = Sum_k |<det_k|(H - E_0)|psi_0>|^2 / (E_0 - H_kk)

Usage:
    python run_pt2.py checkpoint.npz [--fcidump path] [--screening heatbath]

File: detnqs/examples/run_pt2.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2026
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp

from detnqs import models
from detnqs.analysis import CheckpointManager
from detnqs.analysis.metrics import compute_pt2
from detnqs.space import DetSpace
from detnqs.state import State
from detnqs.system import MolecularSystem


def configure_jax() -> None:
    """Configure JAX: prioritize GPU, enable float64, suppress warnings."""
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_log_compiles", False)
    warnings.filterwarnings("ignore", message="Explicitly requested dtype.*is not available")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PT2 calculation."""
    parser = argparse.ArgumentParser(
        description="Compute PT2 correction from DetNQS checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint .npz file",
    )
    parser.add_argument(
        "--fcidump",
        type=Path,
        default=None,
        help="Override FCIDUMP path if file moved",
    )
    parser.add_argument(
        "--screening",
        type=str,
        choices=["heatbath", "none"],
        default="heatbath",
        help="Screening method for perturbative space",
    )
    parser.add_argument(
        "--eps1",
        type=float,
        default=1e-6,
        help="Screening threshold for heat-bath",
    )
    return parser.parse_args()


def main() -> None:
    """Load checkpoint, rebuild state, compute PT2 correction."""
    args = parse_args()
    configure_jax()

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path.name}")
    data = CheckpointManager.load(ckpt_path)

    # Extract metadata and locate FCIDUMP
    meta = data["metadata"]
    fcidump_path = Path(args.fcidump or meta["fcidump_path"])
  
    if not fcidump_path.exists():
        raise FileNotFoundError(
            f"FCIDUMP not found: {fcidump_path}\n"
            f"Use --fcidump to specify correct path"
        )

    # Rebuild molecular system
    print(f"Loading molecular system: {fcidump_path.name}")
    system = MolecularSystem.from_fcidump(fcidump_path)
    print(f"System: N_o={system.n_orb}, N_e={system.n_elec}, MS2={system.ms2}")

    # Initialize heat-bath table if screening enabled
    if args.screening == "heatbath":
        print("\nBuilding heat-bath table...")
        try:
            system.int_ctx.hb_prepare(1e-15)
            print("Heat-bath table ready")
        except Exception as e:
            print(f"Warning: Heat-bath initialization failed: {e}")
            print("Falling back to combinatorial generation (no screening)")
            args.screening = "none"

    # Rebuild determinant space from variational set V_k
    detspace = DetSpace.initialize(data["V_dets"])
    print(f"Variational space: |V_k| = {detspace.size_V}")

    # Create minimal state (PT2 only requires forward pass)
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

    state = State.init(system=system, detspace=detspace, model=model)
    state = state.replace(params=data["params"])

    # Extract reference energy
    e_ref_total = float(meta.get("energy", 0.0))
  
    print(f"\nReference energy (total): {e_ref_total:.8f} Ha")
    print(f"Nuclear repulsion:        {system.e_nuc:.8f} Ha")
    print(f"Reference energy (elec):  {e_ref_total - system.e_nuc:.8f} Ha")

    # Compute PT2 correction
    print(f"\nComputing PT2 correction (screening={args.screening}, eps1={args.eps1:.2e})...")
  
    pt2_result = compute_pt2(
        state=state,
        detspace=detspace,
        system=system,
        e_ref=e_ref_total,
        screening=args.screening,
        eps1=args.eps1,
    )
  
    if pt2_result is None:
        print("\nError: PT2 computation failed (C++ kernel unavailable)")
        return

    # Extract PT2 components
    e_pt2_int = pt2_result["e_pt2_internal"]
    e_pt2_ext = pt2_result["e_pt2_external"]
    e_pt2_tot = pt2_result["e_pt2_total"]
    n_ext = pt2_result["n_ext"]
  
    e_total_pt2 = e_ref_total + e_pt2_tot

    # Report decomposed results
    print("\n" + "=" * 60)
    print("PT2 Results (Decomposed):")
    print(f"  E_var (total):         {e_ref_total:>18.8f} Ha")
    print(f"  Delta E_PT2 (internal):{e_pt2_int:>18.8f} Ha")
    print(f"  Delta E_PT2 (external):{e_pt2_ext:>18.8f} Ha")
    print(f"  Delta E_PT2 (total):   {e_pt2_tot:>18.8f} Ha")
    print(f"  E_var+PT2 (total):     {e_total_pt2:>18.8f} Ha")
    print(f"  |P_k|:                 {n_ext}")
    print("=" * 60)


if __name__ == "__main__":
    main()