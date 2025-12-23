# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analysis module for LEVER optimization results.

Provides:
  - Trace: Lightweight runtime trajectory recorder
  - Metrics: Post-hoc analysis tools (PT2, variational)
  - I/O: Persistent storage utilities

File: lever/analysis/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from .io import load, save
from .metrics import (
    compute_pt2,
    compute_variational,
    convergence_stats,
)
from .trace import Trace

__all__ = [
    # Core data structures
    "Trace",
    # I/O utilities
    "save",
    "load",
    # Analysis tools
    "convergence_stats",
    "compute_pt2",
    "compute_variational",
]