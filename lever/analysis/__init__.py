# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analysis module for LEVER optimization workflows.

Provides runtime monitoring and post-processing tools:
  - Callbacks: Observer pattern for runtime tracking (Console, JSON, Jupyter, Checkpoint)
  - Checkpoint: State persistence and resumption for long jobs
  - Metrics: Post-hoc analysis (PT2, variational energy, convergence stats)

File: lever/analysis/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from .callbacks import (
    BaseCallback,
    ConsoleCallback,
    JupyterCallback,
    JsonCallback,
)
from .checkpoint import CheckpointCallback, CheckpointManager
from .metrics import (
    compute_pt2,
    compute_variational,
    convergence_stats,
    extract_norms,
)

__all__ = [
    # Callback system
    "BaseCallback",
    "ConsoleCallback",
    "JsonCallback",
    "JupyterCallback",
    # Checkpoint management
    "CheckpointCallback",
    "CheckpointManager",
    # Post-processing analysis
    "convergence_stats",
    "compute_pt2",
    "compute_variational",
    "extract_norms",
]
