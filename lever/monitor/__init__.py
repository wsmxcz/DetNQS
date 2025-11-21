# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monitor module initialization and global state management.

File: lever/monitor/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .logger import MonitorLogger
from .storage import RunContext

# Re-export submodules
from . import storage  # noqa: F401
from . import plotting  # noqa: F401
from . import summary  # noqa: F401

if TYPE_CHECKING:
    from ..config import LeverConfig  # type: ignore

_RUN_CONTEXT: Optional[RunContext] = None
_LOGGER_INSTANCE: Optional[MonitorLogger] = None


def get_logger() -> MonitorLogger:
    """Get the global MonitorLogger instance."""
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = MonitorLogger()
    return _LOGGER_INSTANCE


def _infer_system_name(cfg: "LeverConfig | None", override: str | None) -> str:
    """
    Infer system name with priority: override > FCIDUMP stem > 'System'.
    """
    if override is not None and override.strip():
        return override.strip()

    if cfg is not None:
        try:
            # Attempt to extract filename from FCIDUMP path (e.g., "N2_sto3g")
            fcidump = getattr(cfg.system, "fcidump_path", None)
            if fcidump:
                return Path(str(fcidump)).stem
        except Exception:
            pass

    return "System"


def init_run(
    cfg: "LeverConfig | None" = None,
    *,
    system_name: str | None = None,
    root_dir: str = "runs",
) -> RunContext:
    """
    Initialize run context, infer system name, and bind logger.
    
    The inferred system name is stored in metadata, allowing downstream
    utilities (plotting, summary) to access it without redundancy.
    """
    global _RUN_CONTEXT

    # Auto-determine name to avoid repetitive arguments later
    name = _infer_system_name(cfg, system_name)
    
    ctx = RunContext(root_dir=root_dir, system_name=name)

    logger = get_logger()
    logger.bind_file(ctx.log_file)
    
    ctx.save_meta()
    _RUN_CONTEXT = ctx
    return ctx


def get_run() -> Optional[RunContext]:
    """Get current RunContext if initialized."""
    return _RUN_CONTEXT


__all__ = [
    "MonitorLogger",
    "RunContext",
    "get_logger",
    "init_run",
    "get_run",
]