# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualization and reporting tools for LEVER convergence.
"""

import matplotlib.pyplot as plt
import numpy as np

def print_summary(energies: dict[str, float], total_time: float, sys_name: str = "System"):
    """
    Print a formatted summary of energy results in a tabular style.
    
    Displays:
      - E_LEVER: Final optimization energy (Proxy/Effective/Asymmetric).
      - E_VAR:   Exact variational energy on S U C (Streaming).
      - E_S-CI:  Exact diagonalization energy on S space.
      - E_VMC:   Stochastic post-training energy (if available).
      - E_FCI:   Reference Full CI energy (if available).
    
    If E_FCI is provided, calculates absolute errors in mHa.
    """
    e_fci = energies.get('e_fci')
    has_ref = e_fci is not None

    # Define the metrics to display: (Label, Key, Description)
    metrics = [
        ("E_LEVER", 'e_lever', "Final Opt. Energy"),
        ("E_VAR",   'e_var',   "Final Var. (S U C)"),
        ("E_S-CI",  'e_s_ci',  "Exact Diag. (S)"),
        ("E_VMC",   'e_vmc',   "Stochastic VMC"),
        ("E_FCI",   'e_fci',   "Reference FCI"),
    ]

    print(f"\n{'='*75}")
    print(f" LEVER RESULTS SUMMARY: {sys_name}")
    print(f"{'='*75}")
    print(f" Total Time       : {total_time:.2f} s")
    print(f"{'-'*75}")
    
    # Table Header
    header = f" {'Metric':<10} | {'Description':<20} | {'Energy (Ha)':<16}"
    if has_ref:
        header += f" | {'Error (mHa)':<12}"
    print(header)
    print(f"{'-'*75}")

    for label, key, desc in metrics:
        if key not in energies:
            continue
            
        val = energies[key]
        
        # Format Energy Column
        # Special handling for VMC to show error bars if available
        if key == 'e_vmc' and 'e_vmc_err' in energies:
            err = energies['e_vmc_err']
            val_str = f"{val:.6f}({int(err*1e6):d})" # e.g., -76.123456(78)
        else:
            val_str = f"{val:.8f}"
            
        row = f" {label:<10} | {desc:<20} | {val_str:<16}"
        
        # Format Error Column (if FCI exists)
        if has_ref:
            if key == 'e_fci':
                diff_str = "0.0000"
            else:
                diff = (val - e_fci) * 1000.0
                diff_str = f"{diff:+.4f}"
            row += f" | {diff_str:<12}"
            
        print(row)

    print(f"{'='*75}\n")


def plot_convergence(
    result,
    e_fci: float | None = None,
    sys_name: str = "System",
    save_path: str | None = None
):
    """Plot energy convergence history."""
    history = result.full_energy_history
    cycles = result.cycle_boundaries
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot trace
    ax.plot(history, label="LEVER Optimization", alpha=0.8)
    
    # Plot cycle boundaries
    for i, boundary in enumerate(cycles[1:-1]):
        ax.axvline(boundary, color='gray', linestyle='--', alpha=0.5)
        if i == 0:
            ax.text(boundary, min(history), " Cycle Boundaries", 
                   rotation=90, verticalalignment='bottom', color='gray')
            
    # Plot FCI reference
    if e_fci is not None:
        ax.axhline(e_fci, color='red', linestyle=':', label="FCI Reference")
        
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(f"Convergence Trajectory: {sys_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()