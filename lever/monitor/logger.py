# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified logging system for monitoring and progress tracking.

Provides colorized console output with ANSI code stripping for file logging.
Supports section headers, convergence messages, and final summaries.

File: lever/monitor/logger.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import re
from typing import TextIO


# ANSI escape sequence regex for clean file logging
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class MonitorLogger:
    """Unified logger for stdout and optional file output with color support."""
    
    BLUE = "\033[94m"
    RESET = "\033[0m"

    def __init__(self, file: TextIO | None = None) -> None:
        """Initialize logger with optional file handle."""
        self._file = file

    def bind_file(self, file: TextIO | None) -> None:
        """Attach or replace file handle for persistent logging."""
        self._file = file

    def _write(self, msg: str) -> None:
        """
        Write message to stdout and file (with ANSI stripping).
        
        Color codes preserved for console, stripped for file logging.
        Uses best-effort file writing to avoid disrupting main execution.
        """
        print(msg)
        
        if self._file is not None:
            try:
                clean_msg = _ANSI_ESCAPE_RE.sub("", msg)
                self._file.write(clean_msg + "\n")
                self._file.flush()
            except Exception:
                # Best-effort file logging; never break main execution
                pass

    def _blue(self, text: str) -> str:
        """Wrap text in bright blue color for emphasis."""
        return f"{self.BLUE}{text}{self.RESET}"

    def info(self, msg: str) -> None:
        """Generic informational message."""
        self._write(msg)

    def header(self, title: str) -> None:
        """Print centered section header with border lines."""
        line = "=" * 60
        self._write(f"\n{line}")
        self._write(f"{title:^60}")
        self._write(line)

    def separator(self) -> None:
        """Print blank line separator."""
        self._write("")

    def converged(self, cycle: int, delta: float) -> None:
        """Print convergence message with cycle count and energy delta."""
        self._write(f"\n{self._blue('Converged')} at cycle {cycle} (ΔE = {delta:.2e})")

    def max_cycles_reached(self, max_cycles: int) -> None:
        """Print message when maximum outer cycles are reached."""
        self._write(f"\nReached max cycles ({max_cycles})")

    def final_summary(self, energy: float, n_cycles: int, total_time: float) -> None:
        """Print final summary with energy, cycle count, and total time."""
        line = "=" * 60
        self._write(f"\n{line}")
        self._write(f"Final: {self._blue(f'E = {energy:.10f}')}")
        self._write(f"Cycles: {n_cycles} | Time: {total_time:.2f}s")
        self._write(line + "\n")

    def warning(self, message: str) -> None:
        """Print warning message."""
        self._write(f"Warning: {message}")

    def error(self, message: str) -> None:
        """Print error message."""
        self._write(f"ERROR: {message}")
