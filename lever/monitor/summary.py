# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Energy summary formatter for quantum chemistry results.

Reads enhanced summary.json to display system configuration,
energy results, and benchmark comparisons.

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
    Print formatted quantum chemistry energy summary table.
    
    Displays system configuration, run statistics, and energy table
    with reference FCI errors when available.
    
    Args:
        run_dir: Directory containing summary.json and config files
        energies: Optional runtime energy overrides
        sys_name: Optional system name override
    """
    summary = store.load_summary(run_dir)
    config_data = store.load_config(run_dir) or {}
    
    # Resolve system name from config or summary
    if sys_name is None:
        run_info = config_data.get("run_info", {})
        sys_name = run_info.get("system_name", summary.get("system", {}).get("name", "System"))

    # Collect all energies with runtime overrides
    energies_all: Dict[str, float] = dict(summary.get("energies", {}))
    if energies:
        energies_all.update((k, float(v)) for k, v in energies.items())

    # Standardize energy key names
    _standardize_energy_keys(energies_all)
    
    e_fci = energies_all.get("e_fci")
    has_ref = e_fci is not None

    # Print header and system information
    _print_header(sys_name)
    _print_system_info(summary)
    _print_run_stats(summary)
    
    # Print energy comparison table
    _print_energy_table(energies_all, has_ref, e_fci)


def _standardize_energy_keys(energies: Dict[str, float]) -> None:
    """Map legacy energy keys to standardized names."""
    key_map = {
        "hf": "e_hf", "mp2": "e_mp2", "cisd": "e_cisd", 
        "ccsd": "e_ccsd", "ccsd_t": "e_ccsd_t", "casci": "e_casci", "fci": "e_fci"
    }
    for legacy_key, std_key in key_map.items():
        if legacy_key in energies and std_key not in energies:
            energies[std_key] = energies[legacy_key]


def _print_header(sys_name: str) -> None:
    """Print summary header with system name."""
    print(f"\n{'='*75}")
    print(f" LEVER SUMMARY: {sys_name}")
    print(f"{'='*75}")


def _print_system_info(summary: Dict) -> None:
    """Print system configuration details."""
    sys_info = summary.get("system", {})
    if not sys_info:
        return
        
    n_orb = sys_info.get("n_orbitals", "?")
    n_a = sys_info.get("n_alpha", "?")
    n_b = sys_info.get("n_beta", "?")
    print(f" System Config     : {n_orb} orbitals, ({n_a}a, {n_b}b) electrons")


def _print_run_stats(summary: Dict) -> None:
    """Print runtime statistics and space dimensions."""
    print(f" Total Time        : {summary.get('total_time', 0):.2f} s")
    print(f" Cycles / Steps    : {summary.get('n_cycles', 0)} / {summary.get('n_steps', 0)}")
    
    space_info = summary.get("space", {})
    if space_info:
        print(f" Final Space       : S={space_info.get('n_s', 0)}, C={space_info.get('n_c', 0)}")
    
    print(f"{'-'*75}")


def _print_energy_table(energies: Dict[str, float], has_ref: bool, e_fci: float | None) -> None:
    """Print formatted energy comparison table."""
    # Define energy metrics with labels and descriptions
    metrics = [
        ("E_LEVER",   "e_lever",   "Final Opt. Energy"),
        ("E_VAR",     "e_var",     "Final Var. (S ∪ C)"),
        ("E_S-CI",    "e_s_ci",    "Exact Diag. (S)"),
        ("E_VMC",     "e_vmc",     "Stochastic VMC"),
        ("E_HF",      "e_hf",      "Hartree–Fock"),
        ("E_MP2",     "e_mp2",     "MP2"),
        ("E_CISD",    "e_cisd",    "CISD"),
        ("E_CCSD",    "e_ccsd",    "CCSD"),
        ("E_CCSD(T)", "e_ccsd_t",  "CCSD(T)"),
        ("E_FCI",     "e_fci",     "Reference FCI"),
    ]

    # Print table header
    header = f" {'Metric':<10} | {'Description':<20} | {'Energy (Ha)':<16}"
    if has_ref:
        header += f" | {'Error (mHa)':<12}"
    print(header)
    print(f"{'-'*75}")

    # Print each energy row
    for label, key, desc in metrics:
        if key not in energies:
            continue
            
        energy_val = energies[key]
        energy_str = _format_energy_value(key, energy_val, energies)
        row = f" {label:<10} | {desc:<20} | {energy_str:<16}"
        
        if has_ref and key != "e_fci":
            error_mha = (energy_val - e_fci) * 1000.0
            row += f" | {error_mha:+.4f}"
        elif has_ref:
            row += f" | {'0.0000':<12}"
            
        print(row)

    print(f"{'='*75}\n")


def _format_energy_value(key: str, value: float, energies: Dict[str, float]) -> str:
    """
    Format energy value with special handling for VMC uncertainties.
    
    VMC energies display as: value(uncertainty×10⁶)
    Other energies use standard 8-digit precision.
    """
    if key == "e_vmc" and "e_vmc_err" in energies:
        uncertainty = energies["e_vmc_err"]
        return f"{value:.6f}({int(uncertainty * 1e6):d})"
    return f"{value:.8f}"
