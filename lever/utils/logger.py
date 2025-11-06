# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simplified logging utilities for LEVER.

File: lever/utils/logger.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations


class Logger:
    """Simplified logger with compact formatting and color highlights."""
    
    BLUE = "\033[94m"
    RESET = "\033[0m"
    
    def _blue(self, text: str) -> str:
        """Wrap text in bright blue color."""
        return f"{self.BLUE}{text}{self.RESET}"
    
    # ========== Headers ==========
    def header(self, title: str):
        """Print section header."""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
    
    def separator(self):
        """Print blank line separator."""
        print()
    
    # ========== Convergence Messages ==========
    def converged(self, cycle: int, delta: float):
        """Print convergence message."""
        print(f"\n{self._blue('Converged')} at cycle {cycle} (ΔE = {delta:.2e})")
    
    def max_cycles_reached(self, max_cycles: int):
        """Print max cycles reached."""
        print(f"\nReached max cycles ({max_cycles})")
    
    def final_summary(self, energy: float, n_cycles: int, total_time: float):
        """Print final summary."""
        print(f"\n{'='*60}")
        print(f"Final: {self._blue(f'E = {energy:.10f}')}")
        print(f"Cycles: {n_cycles} | Time: {total_time:.2f}s")
        print(f"{'='*60}\n")
    
    # ========== Warnings ==========
    def warning(self, message: str):
        """Print warning."""
        print(f"Warning: {message}")
    
    def error(self, message: str):
        """Print error."""
        print(f"ERROR: {message}")


_LOGGER_INSTANCE = None

def get_logger() -> Logger:
    """Get global logger singleton."""
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = Logger()
    return _LOGGER_INSTANCE


__all__ = ["Logger", "get_logger"]
