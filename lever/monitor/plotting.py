"""
Optimized convergence plotting utilities for LEVER.

File: lever/monitor/plotting.py
Author: Zheng (Alex) Che
Updated: November, 2025
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from . import storage as store

PathLike = Union[str, Path]

# Colorblind-friendly scientific palette
COLORS = {
    "primary": "#2E86AB",      # Deep blue
    "secondary": "#A23B72",    # Magenta
    "reference": "#F18F01",    # Orange
    "grid": "#E1E1E1",         # Light grey
    "cycle_odd": "#F5F5F5",    # Very light grey
    "cycle_even": "#FFFFFF",   # White
}

# Global style tuned for publication-quality figures
PLOT_STYLE = {
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Lines and markers
    "lines.linewidth": 2.2,
    "lines.markersize": 7.5,
    "lines.markeredgewidth": 1.5,

    # Axes and ticks
    "axes.linewidth": 1.4,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.color": "#333333",
    "ytick.color": "#333333",

    # Figure / saving
    "figure.constrained_layout.use": True,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def _apply_style() -> None:
    """Apply global Matplotlib style."""
    plt.rcParams.update(PLOT_STYLE)


def _get_system_name(run_dir: PathLike, override: str | None) -> str:
    """Return system name from metadata, or an override if given."""
    if override:
        return override
    meta = store.load_meta(run_dir)
    return meta.get("system_name", "System") if meta else "System"


def _format_axis(ax: plt.Axes, axis: str = "y") -> None:
    """
    Use plain scalar formatting (no offset) for a cleaner numeric axis.
    """
    formatter = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    formatter.set_scientific(False)
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _enable_minor_ticks(ax: plt.Axes) -> None:
    """Enable minor ticks and light grid for better readability."""
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.9,
        color=COLORS["grid"],
        alpha=0.8,
    )
    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.6,
        color=COLORS["grid"],
        alpha=0.4,
    )


def plot_outer_convergence(
    run_dir: PathLike,
    e_fci: float | None = None,
    system_name: str | None = None,
    log_scale: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Plot outer-loop convergence vs cycle index.

    Args:
        run_dir: Directory containing run history.
        e_fci: Reference FCI energy (optional).
        system_name: Override label for the system (optional).
        log_scale: If True, plot |E − E_ref| on a log scale.
        save_path: If given, save to this path instead of showing the plot.
    """
    _apply_style()
    sys_label = _get_system_name(run_dir, system_name)

    # Load cycle-end energies
    history, cycles = store.load_history(run_dir)
    if history.size == 0:
        return

    end_indices = np.append(cycles[1:] - 1, len(history) - 1)
    valid_indices = end_indices[end_indices < len(history)]

    y_vals = history[valid_indices]
    x_vals = np.arange(1, len(y_vals) + 1)

    # Choose plotted quantity
    if log_scale:
        ref = e_fci if e_fci is not None else np.min(y_vals)
        plot_data = np.maximum(np.abs(y_vals - ref), 1e-16)
        ylabel = r"$|E - E_{\mathrm{ref}}|$ (Ha)"
    else:
        plot_data = y_vals
        ylabel = r"Total Energy (Ha)"

    # Slightly larger figure for readability
    fig, ax = plt.subplots(figsize=(7.5, 5.3))

    # Reference line
    if not log_scale and e_fci is not None:
        ax.axhline(
            e_fci,
            color=COLORS["reference"],
            linestyle="--",
            linewidth=2.2,
            label="FCI Reference",
            zorder=1,
            alpha=0.9,
        )

    # Convergence trajectory
    ax.plot(
        x_vals,
        plot_data,
        marker="o",
        color=COLORS["primary"],
        linewidth=2.6,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        label="LEVER Energy",
        zorder=3,
    )

    # Axes formatting
    ax.set_xlabel("Outer Cycle", fontweight="semibold")
    ax.set_ylabel(ylabel, fontweight="semibold")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    if log_scale:
        ax.set_yscale("log")
        _enable_minor_ticks(ax)
    else:
        _format_axis(ax, "y")
        _enable_minor_ticks(ax)

    # System name box
    ax.text(
        0.03,
        0.97,
        sys_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=COLORS["primary"],
            linewidth=1.6,
            alpha=0.96,
        ),
    )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="best",
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            borderaxespad=0.8,
        )

    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_inner_convergence(
    run_dir: PathLike,
    e_fci: float | None = None,
    system_name: str | None = None,
    log_scale: bool = False,
    save_path: str | None = None,
) -> None:
    """
    Plot the full inner-loop optimization trajectory with cycle shading.

    Args:
        run_dir: Directory containing run history.
        e_fci: Reference FCI energy (optional).
        system_name: Override label for the system (optional).
        log_scale: If True, plot |E − E_ref| on a log scale.
        save_path: If given, save to this path instead of showing the plot.
    """
    _apply_style()
    sys_label = _get_system_name(run_dir, system_name)

    # Load full trajectory and cycle boundaries
    history, cycles = store.load_history(run_dir)
    if history.size == 0:
        return

    steps = np.arange(1, len(history) + 1)

    # Choose plotted quantity
    if log_scale:
        ref = e_fci if e_fci is not None else np.min(history)
        plot_data = np.maximum(np.abs(history - ref), 1e-16)
        ylabel = r"$|E - E_{\mathrm{ref}}|$ (Ha)"
    else:
        plot_data = history
        ylabel = r"Total Energy (Ha)"

    # Wide figure for long trajectories
    fig, ax = plt.subplots(figsize=(13, 5.2))

    # Background shading for cycles
    cycle_boundaries = list(cycles) + [len(history)]
    for i in range(len(cycle_boundaries) - 1):
        start, end = cycle_boundaries[i], cycle_boundaries[i + 1]
        color = COLORS["cycle_odd"] if i % 2 == 0 else COLORS["cycle_even"]
        ax.axvspan(
            start,
            end,
            facecolor=color,
            edgecolor=None,
            alpha=0.6,
            zorder=0,
        )

        # Label cycles along the top axis when there is enough width
        mid = 0.5 * (start + end)
        if (end - start) > len(history) * 0.015:
            ax.text(
                mid,
                1.015,
                f"Cycle {i + 1}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
                color="#555555",
                fontweight="semibold",
            )

    # Reference energy line
    if not log_scale and e_fci is not None:
        ax.axhline(
            e_fci,
            color=COLORS["reference"],
            linestyle="--",
            linewidth=2.2,
            label="FCI Reference",
            zorder=2,
            alpha=0.9,
        )

    # Variational trajectory
    ax.plot(
        steps,
        plot_data,
        color=COLORS["primary"],
        linewidth=2.4,
        label="Optimization Energy",
        zorder=3,
    )

    # Axes formatting
    ax.set_xlabel("Optimization Step", fontweight="semibold")
    ax.set_ylabel(ylabel, fontweight="semibold")
    ax.set_xlim(1, len(history))

    if log_scale:
        ax.set_yscale("log")
        _enable_minor_ticks(ax)
    else:
        _format_axis(ax, "y")
        _enable_minor_ticks(ax)

        # Focus y-range on the stable part of the trajectory
        if len(plot_data) > 50:
            stable = plot_data[int(len(plot_data) * 0.3):]
            y_min, y_max = float(np.min(stable)), float(np.max(stable))
            if y_max > y_min:
                margin = 0.18 * (y_max - y_min)
                ax.set_ylim(y_min - margin, y_max + margin)

    # System name box
    ax.text(
        0.02,
        0.97,
        sys_label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor=COLORS["primary"],
            linewidth=1.6,
            alpha=0.96,
        ),
    )

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            borderaxespad=0.8,
        )

    # Save or show
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)


__all__ = ["plot_outer_convergence", "plot_inner_convergence"]