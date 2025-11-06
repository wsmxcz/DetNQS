# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing and analysis tools for LEVER calculations.

Provides energy evaluation, FCI benchmarking, and convergence visualization.

File: lever/analysis/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from .evaluator import EnergyEvaluator
from .fci import compute_fci_energy
from .plotting import plot_convergence, print_summary

__all__ = [
    "EnergyEvaluator",
    "compute_fci_energy",
    "plot_convergence",
    "print_summary",
]
