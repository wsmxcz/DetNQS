# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monitor module initialization and global context management.

Provides centralized logging and run tracking for quantum chemistry simulations.

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
    """Get singleton MonitorLogger instance."""
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = MonitorLogger()
    return _LOGGER_INSTANCE


def _infer_system_name(cfg: LeverConfig | None, override: str | None) -> str:
    """
    Infer system name from override or config.
    
    Priority: override → fcidump filename → default
    """
    if override is not None and override.strip():
        return override.strip()

    if cfg is not None:
        try:
            fcidump = getattr(cfg.system, "fcidump_path", None)
            if fcidump:
                return Path(str(fcidump)).stem
        except Exception:
            pass
    return "System"


def init_run(
    cfg: LeverConfig | None = None,
    *,
    system_name: str | None = None,
    root_dir: str = "runs",
) -> RunContext:
    """
    Initialize run context and bind logger.
    
    Creates run directory structure and sets up logging pipeline.
    """
    global _RUN_CONTEXT

    name = _infer_system_name(cfg, system_name)
    ctx = RunContext(root_dir=root_dir, system_name=name)

    logger = get_logger()
    logger.bind_file(ctx.log_file)
    
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
