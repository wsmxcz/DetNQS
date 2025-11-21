# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Energy summary formatter for LEVER quantum chemistry calculations.

Formats final results with reference comparisons and error metrics.
Displays energies in Hartree with optional milli-Hartree deviations.

File: lever/monitor/summary.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

from . import storage as store

PathLike = Union[str, Path]


def print_summary_from_run(
    run_dir: PathLike,
    energies: Dict[str, float] | None = None,
    sys_name: str | None = None,
) -> None:
    """
    Print formatted energy summary table with reference comparisons.
    
    Energy hierarchy: E_S-CI ≤ E_VAR ≤ E_FCI (variational principle)
    Error metric: ΔE = E - E_FCI [mHa]
    
    Args:
        run_dir: Directory containing summary.json and meta.json
        energies: Optional override/additional energy diagnostics
        sys_name: System identifier for header
    """
    summary = store.load_summary(run_dir)
    meta = store.load_meta(run_dir)

    # Resolve system name from metadata or use default
    sys_name = (
        str(meta["system_name"]) 
        if meta and "system_name" in meta 
        else sys_name or "System"
    )

    # Merge stored energies with user overrides
    energies_all: Dict[str, float] = dict(summary.get("energies", {}))
    if energies:
        energies_all.update((k, float(v)) for k, v in energies.items())

    e_fci = energies_all.get("e_fci")
    has_ref = e_fci is not None

    # Energy metrics display configuration: (label, data_key, description)
    metrics = [
        ("E_LEVER", "e_lever", "Final Opt. Energy"),
        ("E_VAR", "e_var", "Final Var. (S U C)"),
        ("E_S-CI", "e_s_ci", "Exact Diag. (S)"),
        ("E_VMC", "e_vmc", "Stochastic VMC"),
        ("E_FCI", "e_fci", "Reference FCI"),
    ]

    total_time = summary["total_time"]
    space = summary.get("space", {})

    # Print summary header
    print(f"\n{'='*75}")
    print(f" LEVER RESULTS SUMMARY: {sys_name}")
    print(f"{'='*75}")
    print(f" Total Time        : {total_time:.2f} s")
    print(f" Outer Cycles      : {summary['n_cycles']}")
    print(f" Optimization Steps: {summary['n_steps']}")
    
    if space:
        print(
            " Final Space       : "
            f"S={space.get('n_s', 0)}, "
            f"C={space.get('n_c', 0)}, "
            f"T={space.get('n_t', 0)}"
        )
    
    print(f"{'-'*75}")

    # Table header
    header = f" {'Metric':<10} | {'Description':<20} | {'Energy (Ha)':<16}"
    if has_ref:
        header += f" | {'Error (mHa)':<12}"
    print(header)
    print(f"{'-'*75}")

    # Print energy rows
    for label, key, desc in metrics:
        if key not in energies_all:
            continue

        val = energies_all[key]
        
        # VMC special formatting with error bars: E(δE) [μHa precision]
        if key == "e_vmc" and "e_vmc_err" in energies_all:
            err = energies_all["e_vmc_err"]
            val_str = f"{val:.6f}({int(err * 1e6):d})"
        else:
            val_str = f"{val:.8f}"

        row = f" {label:<10} | {desc:<20} | {val_str:<16}"

        # Reference error column (milli-Hartree)
        if has_ref:
            if key == "e_fci":
                diff_str = "0.0000"
            else:
                diff = (val - e_fci) * 1000.0  # Ha → mHa
                diff_str = f"{diff:+.4f}"
            row += f" | {diff_str:<12}"

        print(row)

    print(f"{'='*75}\n")
