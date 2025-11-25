# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized convergence plotting utilities for LEVER.

Plots outer-cycle convergence and full inner-loop trajectories with 
reference energy benchmarks and cycle-aware visualization.

File: lever/monitor/plotting.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from . import storage as store

PathLike = Union[str, Path]

# Color palette for consistent visualization
PALETTE = {
    'baseline':   '#E29578',
    'primary':    '#264653',
    'secondary':  '#2A9D8F',
    'accent':     '#457B9D',
    'bar_fill':   '#E9ECEF',
    'bar_edge':   '#6C757D',
    'grid':       '#E5E5E5',
    'text':       '#333333'
}

def _apply_style() -> None:
    """Apply global Matplotlib style configuration."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.edgecolor': PALETTE['text'],
        'axes.labelcolor': PALETTE['text'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 110,
        'grid.color': PALETTE['grid'],
        'legend.frameon': False,
    })

def _get_system_name(run_dir: PathLike, override: str | None) -> str:
    """Extract system name from run configuration or use override."""
    if override:
        return override
    
    cfg_data = store.load_config(run_dir)
    if cfg_data and "run_info" in cfg_data:
        return cfg_data["run_info"].get("system_name", "System")
        
    return "System"

def _resolve_benchmark(run_dir: PathLike, manual_val: float | None) -> float | None:
    """
    Resolve benchmark energy from summary data.
    
    Searches for exact reference energies in order: 'e_fci', 'fci', 'e_exact'.
    """
    if manual_val is not None:
        return manual_val
        
    summary = store.load_summary(run_dir)
    energies = summary.get("energies", {})
    
    for key in ["e_fci", "fci", "e_exact"]:
        if key in energies:
            return float(energies[key])
    return None

def _format_axis(ax: plt.Axes, axis: str = "y", log: bool = False) -> None:
    """Configure axis formatting with scientific notation control."""
    if log:
        return
    formatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(False)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)

def _enable_minor_ticks(ax: plt.Axes) -> None:
    """Enable minor ticks and configure grid styling."""
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(True, which="major", linestyle=":", linewidth=0.9, 
            color=PALETTE['grid'], alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.6, 
            color=PALETTE['grid'], alpha=0.4)

def plot_outer_convergence(
    run_dir: PathLike,
    benchmark_energy: float | None = None,
    system_name: str | None = None,
    log_scale: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Plot outer-loop convergence vs cycle index.
    
    Visualizes energy convergence at the end of each outer cycle.
    Log scale shows absolute deviation |E - E_ref| from reference.
    
    Args:
        run_dir: Directory containing run history data
        benchmark_energy: Reference energy (e.g., FCI)
        system_name: Override system label
        log_scale: Plot absolute deviation in log scale
        save_path: Optional output file path
    """
    _apply_style()
    sys_label = _get_system_name(run_dir, system_name)
    e_ref = _resolve_benchmark(run_dir, benchmark_energy)

    history, cycles = store.load_history(run_dir)
    if history.size == 0:
        return

    # Extract cycle-end energies
    end_indices = np.append(cycles[1:] - 1, len(history) - 1)
    valid_indices = end_indices[end_indices < len(history)]
    y_vals = history[valid_indices]
    x_vals = np.arange(1, len(y_vals) + 1)

    # Prepare plot data: either raw energies or |E - E_ref|
    if log_scale:
        ref_val = e_ref if e_ref is not None else np.min(y_vals)
        plot_data = np.maximum(np.abs(y_vals - ref_val), 1e-16)
        ylabel = r"$|E - E_{\mathrm{ref}}|$ (Ha)"
    else:
        plot_data = y_vals
        ylabel = r"Total Energy (Ha)"

    fig, ax = plt.subplots(figsize=(7, 5))

    # Reference line for linear scale
    if not log_scale and e_ref is not None:
        ax.axhline(e_ref, color=PALETTE['baseline'], linestyle="--", 
                  label="Benchmark", zorder=1)

    # Main convergence trajectory
    ax.plot(x_vals, plot_data, marker="o", color=PALETTE['primary'], 
            label="LEVER", zorder=3)

    ax.set_xlabel("Outer Cycle")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Scale configuration
    if log_scale:
        ax.set_yscale("log")
        _enable_minor_ticks(ax)
    else:
        _format_axis(ax, "y")
        _enable_minor_ticks(ax)

    # System label annotation
    ax.text(0.03, 0.97, sys_label, transform=ax.transAxes, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                     edgecolor=PALETTE['primary'], alpha=0.9))

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def plot_inner_convergence(
    run_dir: PathLike,
    benchmark_energy: float | None = None,
    system_name: str | None = None,
    log_scale: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Plot full inner-loop optimization trajectory.
    
    Shows complete optimization history with cycle boundaries and 
    alternating background shading for cycle visualization.
    
    Args:
        run_dir: Directory containing run history
        benchmark_energy: Reference energy for deviation calculation
        system_name: Override system label
        log_scale: Plot absolute deviation in log scale
        save_path: Optional output file path
    """
    _apply_style()
    sys_label = _get_system_name(run_dir, system_name)
    e_ref = _resolve_benchmark(run_dir, benchmark_energy)

    history, cycles = store.load_history(run_dir)
    if history.size == 0:
        return

    steps = np.arange(1, len(history) + 1)

    # Prepare plot data
    if log_scale:
        ref_val = e_ref if e_ref is not None else np.min(history)
        plot_data = np.maximum(np.abs(history - ref_val), 1e-16)
        ylabel = r"$|E - E_{\mathrm{ref}}|$ (Ha)"
    else:
        plot_data = history
        ylabel = r"Total Energy (Ha)"

    fig, ax = plt.subplots(figsize=(10, 5))

    # Cycle boundary visualization with alternating shading
    cycle_bounds = list(cycles) + [len(history)]
    for i in range(len(cycle_bounds) - 1):
        start, end = cycle_bounds[i], cycle_bounds[i + 1]
        color = PALETTE['bar_fill'] if i % 2 == 0 else "white"
        ax.axvspan(start, end, facecolor=color, alpha=0.6, zorder=0)
        
        # Cycle label for sufficiently large cycles
        if (end - start) > len(history) * 0.02:
            mid = (start + end) / 2
            ax.text(mid, 1.01, f"C{i+1}", transform=ax.get_xaxis_transform(), 
                    ha="center", va="bottom", fontsize=9, color="#666")

    # Reference line for linear scale
    if not log_scale and e_ref is not None:
        ax.axhline(e_ref, color=PALETTE['baseline'], linestyle="--", 
                  label="Benchmark", zorder=2)

    # Main optimization trajectory
    ax.plot(steps, plot_data, color=PALETTE['primary'], 
            label="Optimization", zorder=3)

    ax.set_xlabel("Optimization Step")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, len(history))

    # Scale configuration
    if log_scale:
        ax.set_yscale("log")
        _enable_minor_ticks(ax)
    else:
        _format_axis(ax, "y")
        _enable_minor_ticks(ax)

    # System label annotation
    ax.text(0.02, 0.97, sys_label, transform=ax.transAxes, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                     edgecolor=PALETTE['primary'], alpha=0.9))

    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

__all__ = ["plot_outer_convergence", "plot_inner_convergence"]
