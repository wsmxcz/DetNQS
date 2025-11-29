# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Publication-ready convergence visualization utilities.

Generates two-panel diagnostic plots:
- Top: Outer cycle energy trajectory with core space evolution
- Bottom: Full inner optimization history with cycle boundaries

File: lever/utils/plot_utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..dtypes import LeverResult


# ============================================================================
# Color Palette
# ============================================================================

PALETTE = {
    "baseline": "#E29578",      # Cycle boundary markers
    "primary": "#264653",       # Energy line (outer)
    "secondary": "#2A9D8F",     # Energy line (inner)
    "accent": "#457B9D",        # Unused (reserved)
    "bar_fill": "#E9ECEF",      # Core space bars
    "bar_edge": "#6C757D",      # Bar borders
    "grid": "#E5E5E5",          # Grid lines
    "text": "#333333",          # Axis labels
}


# ============================================================================
# Matplotlib Style Configuration
# ============================================================================

def _configure_plot_style() -> None:
    """Set publication-ready matplotlib defaults."""
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.edgecolor": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "text.color": PALETTE["text"],
        "figure.dpi": 300,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


_configure_plot_style()


# ============================================================================
# Convergence Plot
# ============================================================================

def plot_convergence(
    output_dir: Path,
    result: LeverResult,
    diag_cycles: List[Dict[str, Any]],
) -> None:
    """
    Generate two-panel convergence diagnostic plot.
    
    Layout:
    ┌─────────────────────────────────────┐
    │ Top: Outer cycle energy + core size │  Dual-axis plot
    │   - Line: Energy trajectory         │  (energy vs. n_s)
    │   - Bars: Core space dimension      │
    ├─────────────────────────────────────┤
    │ Bottom: Inner optimization history  │  Full energy trace
    │   - Dashed lines: Cycle boundaries  │  with cycle markers
    └─────────────────────────────────────┘
    
    Args:
        output_dir: Directory for convergence.pdf
        result: Solver output with energy history
        diag_cycles: Per-cycle diagnostics (compile, inner, wavefunction)
    
    Raises:
        ValueError: If energy history or cycle data is invalid
    """
    hist = np.asarray(result.full_energy_history, dtype=float)
    bounds = result.cycle_boundaries

    # Validate input data
    if len(bounds) < 2 or hist.size == 0 or not diag_cycles:
        raise ValueError("Insufficient data for convergence plot")

    # Extract outer cycle energies (last point of each cycle)
    outer_indices = [b - 1 for b in bounds[1:] if 0 < b <= len(hist)]
    if not outer_indices:
        raise ValueError("No valid outer cycle boundaries")

    outer_energies = hist[outer_indices]
    cycles = np.arange(1, len(outer_energies) + 1)

    # Extract core space sizes
    core_sizes = np.array(
        [c["compile"]["n_s"] for c in diag_cycles[: len(cycles)]],
        dtype=float,
    )

    # Create figure with two vertically stacked panels
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(6.5, 6.5),
        sharex=False,
        constrained_layout=True,
    )

    # ====================================================================
    # Top panel: Outer convergence (dual axis)
    # ====================================================================
    _plot_outer_convergence(ax_top, cycles, outer_energies, core_sizes)

    # ====================================================================
    # Bottom panel: Inner trajectory
    # ====================================================================
    _plot_inner_trajectory(ax_bottom, hist, bounds)

    # Save figure
    fig.savefig(output_dir / "convergence.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_outer_convergence(
    ax: plt.Axes,
    cycles: np.ndarray,
    energies: np.ndarray,
    core_sizes: np.ndarray,
) -> None:
    """
    Plot outer cycle convergence with dual y-axes.
    
    Left axis: Energy trajectory (line plot)
    Right axis: Core space size n_s (bar chart)
    
    Args:
        ax: Primary matplotlib axes (energy)
        cycles: Outer cycle indices [1, 2, ..., N]
        energies: Energy at end of each cycle
        core_sizes: Core space dimension n_s per cycle
    """
    # Create secondary axis for core space
    ax_energy = ax
    ax_core = ax.twinx()

    # Plot core space bars (secondary axis)
    bar_handle = ax_core.bar(
        cycles,
        core_sizes,
        width=0.45,
        color=PALETTE["bar_fill"],
        edgecolor=PALETTE["bar_edge"],
        linewidth=1.0,
        alpha=0.7,
        label="Core size $n_s$",
        zorder=1,
    )

    ax_core.set_ylabel("Core space size $n_s$", labelpad=8, fontweight="medium")
    ax_core.tick_params(axis="y", colors=PALETTE["bar_edge"])
    ax_core.spines["top"].set_visible(False)

    # Plot energy line (primary axis, overlay on top)
    ax_energy.set_zorder(ax_core.get_zorder() + 1)
    ax_energy.patch.set_visible(False)  # Transparent background

    (line_handle,) = ax_energy.plot(
        cycles,
        energies,
        marker="o",
        linestyle="-",
        color=PALETTE["primary"],
        linewidth=1.8,
        markersize=5,
        label="Energy",
        zorder=10,
    )

    # Configure primary axis
    ax_energy.set_xlabel("Outer cycle", labelpad=8, fontweight="medium")
    ax_energy.set_ylabel("Energy (Ha)", labelpad=8, fontweight="medium")
    ax_energy.grid(True, linestyle="--", alpha=0.5, zorder=5)
    ax_energy.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Set energy axis limits with 10% margin
    e_min, e_max = energies.min(), energies.max()
    margin = (e_max - e_min) * 0.1 if e_max > e_min else 1e-3
    ax_energy.set_ylim(e_min - margin, e_max + margin)

    # Title and unified legend
    ax.set_title("Outer convergence", pad=10, fontweight="bold")

    handles = [line_handle, bar_handle]
    labels = [h.get_label() for h in handles]
    ax_energy.legend(
        handles,
        labels,
        loc="best",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.8,
    )


def _plot_inner_trajectory(
    ax: plt.Axes,
    energy_history: np.ndarray,
    cycle_boundaries: list[int],
) -> None:
    """
    Plot full inner optimization trajectory with cycle markers.
    
    Args:
        ax: Matplotlib axes
        energy_history: Full energy trace [step_0, ..., step_N]
        cycle_boundaries: Indices marking cycle transitions
    """
    # Plot continuous energy trajectory
    ax.plot(
        np.arange(len(energy_history)),
        energy_history,
        linestyle="-",
        color=PALETTE["secondary"],
        linewidth=1.5,
        alpha=0.9,
    )

    # Configure axes
    ax.set_xlabel("Inner optimization step", labelpad=8, fontweight="medium")
    ax.set_ylabel("Energy (Ha)", labelpad=8, fontweight="medium")
    ax.set_title("Inner energy trajectory", pad=10, fontweight="bold")
    ax.grid(True, alpha=0.5)

    # Mark cycle boundaries (exclude first and last)
    for boundary in cycle_boundaries[1:-1]:
        if 0 < boundary < len(energy_history):
            ax.axvline(
                boundary,
                color=PALETTE["baseline"],
                linestyle="--",
                alpha=0.5,
                linewidth=1.0,
            )


__all__ = ["plot_convergence"]
