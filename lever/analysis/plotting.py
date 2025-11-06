# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convergence visualization and summary reporting for LEVER calculations.

Provides energy trajectory plotting and formatted result summaries
with FCI comparison.

File: lever/analysis/plotting.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..dtypes import LeverResult


def print_summary(
    result: LeverResult,
    e_fci: float | None = None,
    sys_name: str | None = None
) -> None:
    """
    Print formatted energy summary with optional FCI comparison.
    
    Args:
        result: LEVER calculation result
        e_fci: FCI reference energy (optional)
        sys_name: System identifier (inferred from fcidump if None)
    """
    final_e = result.full_energy_history[-1]
    
    # Infer system name from config
    if sys_name is None:
        sys_name = Path(result.config.system.fcidump_path).stem

    print("\n" + "="*50)
    print(f"LEVER Analysis Summary: {sys_name}")
    print("="*50)
    
    if e_fci is not None:
        print(f"\nFCI Reference: {e_fci:.8f} Ha")
        print("\nLEVER Energies:")
        
        # Optimization energy
        err_mha = (final_e - e_fci) * 1e3
        print(f"  Optimization: {final_e:.8f} Ha  (Δ = {err_mha:+.4f} mHa)")
    else:
        print(f"\nFinal Energy: {final_e:.8f} Ha")

    # Variational energy (if available)
    if result.var_energy_history:
        e_var = result.var_energy_history[-1]
        print(f"  Variational:  {e_var:.8f} Ha", end="")
        if e_fci is not None:
            err_var = (e_var - e_fci) * 1e3
            print(f"  (Δ = {err_var:+.4f} mHa)")
        else:
            print()
    
    # S-space CI energy (if available)
    if result.s_ci_energy_history:
        e_s = result.s_ci_energy_history[-1]
        print(f"  S-space CI:   {e_s:.8f} Ha", end="")
        if e_fci is not None:
            err_s = (e_s - e_fci) * 1e3
            print(f"  (Δ = {err_s:+.4f} mHa)")
        else:
            print()
    
    print(f"\nWall Time: {result.total_time:.2f} s")
    print(f"Total Cycles: {len(result.cycle_boundaries)}")
    print("="*50 + "\n")


def plot_convergence(
    result: LeverResult,
    e_fci: float | None = None,
    sys_name: str | None = None,
    save_path: str | None = None
) -> None:
    """
    Generate dual-panel convergence plot.
    
    Top panel: Energy trajectory with chemical accuracy band (±1.6 mHa)
    Bottom panel: Logarithmic absolute error |E_LEVER - E_FCI|
    
    Args:
        result: LEVER calculation result
        e_fci: FCI reference energy for error plotting
        sys_name: System identifier (inferred from fcidump if None)
        save_path: Output file path (displays if None)
    """
    # Infer system name from config
    if sys_name is None:
        sys_name = Path(result.config.system.fcidump_path).stem

    # Configure plot style
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'lines.linewidth': 2.0,
        'grid.alpha': 0.3
    })
    
    # Extract cycle-end energies
    energy = np.array(result.full_energy_history)
    cycle_ends = [bound - 1 for bound in result.cycle_boundaries[1:]]
    cycle_energies = energy[cycle_ends]
    cycles = np.arange(1, len(cycle_ends) + 1)
    
    chem_accuracy = 1.6e-3  # Chemical accuracy: 1.6 mHa
    
    # Determine if we can plot error panel
    has_fci = e_fci is not None
    
    if has_fci:
        # Create dual-panel layout
        fig, (ax_energy, ax_error) = plt.subplots(
            2, 1,
            figsize=(8, 6),
            sharex=True,
            height_ratios=[2, 1],
            gridspec_kw={'hspace': 0.15}
        )
        
        errors = np.abs(cycle_energies - e_fci)
        _plot_energy_panel(ax_energy, cycles, cycle_energies, result, 
                          sys_name, e_fci, chem_accuracy)
        _plot_error_panel(ax_error, cycles, errors, chem_accuracy)
    else:
        # Single energy panel without FCI reference
        fig, ax_energy = plt.subplots(figsize=(8, 4))
        _plot_energy_panel(ax_energy, cycles, cycle_energies, result, 
                          sys_name, None, chem_accuracy)
        ax_energy.set_xlabel('Evolution Cycle')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def _plot_energy_panel(
    ax: plt.Axes,
    cycles: np.ndarray,
    energies: np.ndarray,
    result: LeverResult,
    sys_name: str,
    e_fci: float | None,
    chem_acc: float
) -> None:
    """Plot energy trajectory with optional FCI reference and accuracy band."""
    
    if e_fci is not None:
        # Chemical accuracy band
        ax.axhspan(
            e_fci - chem_acc,
            e_fci + chem_acc,
            alpha=0.15,
            color='green',
            label='Chem. Acc. (±1.6 mHa)'
        )
        
        # FCI reference line
        ax.axhline(
            e_fci,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'FCI: {e_fci:.6f} Ha'
        )
    
    # Optimization trajectory
    ax.plot(
        cycles,
        energies,
        'o-',
        color='steelblue',
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=1.5,
        label='LEVER (opt)'
    )

    # Variational energies (if available)
    if result.var_energy_history:
        x_var = _align_to_cycles(result.var_energy_history, cycles)
        if x_var:
            ax.plot(
                x_var,
                result.var_energy_history,
                's-',
                color='orange',
                markersize=6,
                markeredgecolor='white',
                markeredgewidth=1.0,
                label='LEVER (var)'
            )
  
    # S-space CI energies (if available)
    if result.s_ci_energy_history:
        x_s = _align_to_cycles(result.s_ci_energy_history, cycles)
        if x_s:
            ax.plot(
                x_s,
                result.s_ci_energy_history,
                'D-',
                color='purple',
                markersize=5,
                markeredgecolor='white',
                markeredgewidth=1.0,
                label='S-space CI'
            )

    # Format panel
    if e_fci is not None:
        ax.set_ylim(e_fci - 5e-3, e_fci + 15e-3)
    ax.set_ylabel('Total Energy (Ha)')
    ax.set_title(f'LEVER Evolution: {sys_name}')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)


def _plot_error_panel(
    ax: plt.Axes,
    cycles: np.ndarray,
    errors: np.ndarray,
    chem_acc: float
) -> None:
    """Plot logarithmic error evolution with accuracy threshold."""
    ax.semilogy(
        cycles,
        errors,
        's-',
        color='crimson',
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=1.5,
        label='Absolute Error'
    )
    
    # Chemical accuracy threshold
    ax.axhline(
        chem_acc,
        color='green',
        linestyle='--',
        alpha=0.6,
        label=f'Chem. Acc. ({chem_acc*1e3:.1f} mHa)'
    )
    
    ax.set_xlabel('Evolution Cycle')
    ax.set_ylabel(r'$|E_{\mathrm{LEVER}} - E_{\mathrm{FCI}}|$ (Ha)')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_xticks(cycles)


def _align_to_cycles(hist: list, cycles: np.ndarray) -> list:
    """
    Map data history to cycle x-coordinates.
    
    Handles: full history (1:1 mapping), single final value, or empty.
    """
    n = len(hist)
    if n == len(cycles):
        return cycles.tolist()
    elif n == 1:
        return [cycles[-1]]
    else:
        return []


__all__ = ["plot_convergence", "print_summary"]
