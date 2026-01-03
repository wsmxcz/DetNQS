# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Observer pattern for runtime monitoring and data persistence.

Provides callback hooks at outer loop boundaries:
  - ConsoleCallback: Formatted table output
  - JsonCallback: Line-delimited JSON streaming
  - JupyterCallback: Dynamic inline visualization

File: detnqs/analysis/callbacks.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..driver import BaseDriver


class BaseCallback(ABC):
    """Abstract base for outer loop observers."""

    def on_run_start(self, driver: BaseDriver) -> None:
        """Invoked before first outer loop."""
        pass

    @abstractmethod
    def on_outer_end(self, step: int, stats: dict[str, Any], driver: BaseDriver) -> None:
        """Invoked after each outer loop iteration."""
        pass

    def on_run_end(self, driver: BaseDriver) -> None:
        """Invoked after optimization completes."""
        pass


class ConsoleCallback(BaseCallback):
    """Stdout progress table with energy and wave function norms."""

    def __init__(self, every: int = 1):
        """
        Args:
            every: Print interval in outer steps.
        """
        self.every = every
        self._header_printed = False

    def _print_header(self) -> None:
        """Print formatted table header."""
        sep = "=" * 95
        print(f"\n{sep}")
        print(
            f"{'Outer':>6} | {'|V|':>8} | {'|P|':>8} | "
            f"{'Energy':>18} | {'||ψ_V||²':>10} | {'||ψ_P||²':>10} | {'Time':>10}"
        )
        print(sep)

    def on_outer_end(self, step: int, stats: dict, driver: BaseDriver) -> None:
        """Print current iteration stats."""
        if step % self.every != 0:
            return

        if not self._header_printed:
            self._print_header()
            self._header_printed = True

        # Extract wave function norms
        norm_v = stats["norm_v"]
        norm_p = stats["norm_p"]
        norm_tot = norm_v + norm_p

        # Compute normalized fractions
        frac_v = norm_v / norm_tot if norm_tot > 1e-14 else 1.0
        frac_p = norm_p / norm_tot if norm_tot > 1e-14 else 0.0

        print(
            f"{step:6d} | {stats['size_v']:8d} | {stats['size_p']:8d} | "
            f"{stats['energy']:18.10f} | {frac_v:10.6f} | {frac_p:10.6f} | "
            f"{stats['timestamp']:10.2f}s"
        )

    def on_run_end(self, driver: BaseDriver) -> None:
        """Print footer separator."""
        print("=" * 95)


class JsonCallback(BaseCallback):
    """Stream stats to line-delimited JSON file for post-analysis."""

    def __init__(self, path: str | Path):
        """
        Args:
            path: Output file path.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def on_outer_end(self, step: int, stats: dict, driver: BaseDriver) -> None:
        """Append serialized stats to file."""
        record = {k: self._to_json(v) for k, v in stats.items()}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _to_json(obj: Any) -> Any:
        """Convert numpy/jax arrays to native Python types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [JsonCallback._to_json(x) for x in obj]
        return obj


class JupyterCallback(BaseCallback):
    """Live energy convergence plot for interactive notebooks."""

    def __init__(self, plot_interval: int = 1, figsize: tuple[int, int] = (8, 5)):
        """
        Args:
            plot_interval: Update plot every N outer steps.
            figsize: Matplotlib figure dimensions.
        """
        self.plot_interval = plot_interval
        self.energies: list[float] = []
        self.steps: list[int] = []

        try:
            from IPython.display import clear_output, display
            import matplotlib.pyplot as plt

            self.display = display
            self.clear_output = clear_output
            self.plt = plt
            self.fig, self.ax = plt.subplots(figsize=figsize)
        except ImportError as e:
            raise RuntimeError("JupyterCallback requires IPython and matplotlib") from e

    def on_outer_end(self, step: int, stats: dict, driver: BaseDriver) -> None:
        """Update convergence plot."""
        if step % self.plot_interval != 0:
            return

        self.steps.append(step)
        self.energies.append(stats["energy"])

        self.clear_output(wait=True)
        self.ax.clear()
        self.ax.plot(self.steps, self.energies, marker="o", linestyle="-")
        self.ax.set_xlabel("Outer Step")
        self.ax.set_ylabel("Energy (Ha)")
        self.ax.set_title("Energy Convergence")
        self.ax.grid(alpha=0.3)
        self.display(self.fig)

    def on_run_end(self, driver: BaseDriver) -> None:
        """Close matplotlib figure."""
        self.plt.close(self.fig)


__all__ = ["BaseCallback", "ConsoleCallback", "JsonCallback", "JupyterCallback"]