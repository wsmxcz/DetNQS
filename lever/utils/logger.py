# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Centralized logging for LEVER workflows.

Provides structured, formatted output with color-coded severity levels
for optimization progress tracking.

File: lever/utils/logger.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

import logging
import sys
from typing import Optional


class ColorFormatter(logging.Formatter):
    """ANSI color formatter for terminal output."""
  
    # ANSI color codes
    COLORS = {
        'DEBUG':    '\033[36m',  # Cyan
        'INFO':     '\033[37m',  # White
        'WARNING':  '\033[33m',  # Yellow
        'ERROR':    '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET':    '\033[0m',   # Reset
        'BOLD':     '\033[1m',   # Bold
        'DIM':      '\033[2m',   # Dim
        'CYAN':     '\033[96m',  # Bright cyan
        'GREEN':    '\033[92m',  # Bright green
        'YELLOW':   '\033[93m',  # Bright yellow
    }
  
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color codes."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
      
        # Color only the level name
        levelname_colored = f"{color}{record.levelname:<7}{reset}"
        record.levelname = levelname_colored
      
        return super().format(record)


class LeverLogger:
    """
    Structured logger for LEVER driver and engine.
  
    Output hierarchy:
      L0: Section headers (bold cyan, no indent)
      L1: Subsections (cyan, 2-space indent)
      L2: Key-value pairs (4-space indent, aligned colon)
      L3: Nested details (6-space indent)
  
    Color scheme:
      Titles:  Cyan (structural elements)
      Values:  Yellow (numerical data)
      Status:  Green (completion markers)
      Hints:   Warning level (performance notes)
    """
  
    def __init__(
        self,
        name: str = "lever",
        level: int = logging.INFO,
        use_color: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.use_color = use_color and sys.stdout.isatty()
      
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(level)
          
            if self.use_color:
                formatter = ColorFormatter(
                    '%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%H:%M:%S'
                )
            else:
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)-7s | %(message)s',
                    datefmt='%H:%M:%S'
                )
          
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
  
    def _colorize(self, text: str, color: str) -> str:
        """Apply color formatting if enabled."""
        if not self.use_color:
            return text
        colors = {
            'cyan': '\033[96m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'reset': '\033[0m'
        }
        code = colors.get(color, colors['reset'])
        return f"{code}{text}{colors['reset']}"
  
    def _format_number(self, value: float, precision: int = 2) -> str:
        """Format number with color and comma separators."""
        if isinstance(value, int) or value > 1e6:
            return self._colorize(f"{value:,.0f}", 'yellow')
        elif abs(value) < 1e-3:
            return self._colorize(f"{value:.{precision}e}", 'yellow')
        else:
            return self._colorize(f"{value:.{precision}f}", 'yellow')
  
    # ==================== Section Headers ====================
  
    def header(self, title: str):
        """Print L0 section header with visual separation."""
        sep = "=" * 70
        self.logger.info("")
        self.logger.info(self._colorize(sep, 'cyan'))
        self.logger.info(self._colorize(f"  {title}", 'bold'))
        self.logger.info(self._colorize(sep, 'cyan'))
  
    def subheader(self, title: str):
        """Print L1 subsection header."""
        self.logger.info("")
        self.logger.info(self._colorize(f"  {title}", 'cyan'))
  
    # ==================== Cycle Progress ====================
  
    def cycle_start(self, current: int, total: int):
        """Log cycle start (simplified, no progress bar)."""
        progress = f"[{current}/{total}]"
        self.logger.info("")
        self.logger.info(
            self._colorize(f"  Optimization Cycle {progress}", 'cyan')
        )
  
    # ==================== Configuration Info ====================
  
    def config_info(self, items: dict[str, str]):
        """Log configuration items in aligned table."""
        max_key = max(len(k) for k in items.keys())
      
        for key, value in items.items():
            value_colored = self._colorize(value, 'yellow')
            self.logger.info(f"    {key:<{max_key}}  :  {value_colored}")
  
    # ==================== Space Dimensions ====================
  
    def space_dimensions(self, n_s: int, n_c: int):
        """Log Hilbert space dimensions in compact format."""
        self.logger.info(
            f"    Determinant space: "
            f"S = {self._format_number(n_s, 0)} | "
            f"C = {self._format_number(n_c, 0)}"
        )
  
    def hamiltonian_sparsity(
        self,
        n_s: int,
        nnz_ss: int,
        nnz_sc: Optional[int] = None
    ):
        """Log Hamiltonian sparsity in compact format."""
        if nnz_sc is not None:
            # Proxy mode: H_SS and H_SC
            density_ss = 100.0 * nnz_ss / (n_s * n_s) if n_s > 0 else 0.0
            self.logger.info(
                f"    Hamiltonian: "
                f"H_SS {self._format_number(nnz_ss, 0)} nnz "
                f"({self._format_number(density_ss, 2)}%) | "
                f"H_SC {self._format_number(nnz_sc, 0)} nnz"
            )
        else:
            # Effective mode: H_eff only
            density = 100.0 * nnz_ss / (n_s * n_s) if n_s > 0 else 0.0
            self.logger.info(
                f"    Hamiltonian: "
                f"H_eff {self._format_number(nnz_ss, 0)} nnz "
                f"({self._format_number(density, 2)}%)"
            )
  
    # ==================== Optimization Progress ====================
  
    def optimization_header(self):
        """Print optimization section header."""
        self.logger.info("    Parameter optimization")
  
    def optimization_step(self, step: int, energy: float, total_steps: int):
        """Log optimization step (no progress bar)."""
        if step % 10 == 0 or step == total_steps:  # Log every 10 steps
            self.logger.info(
                f"      Step {step:>4}/{total_steps:<4}  "
                f"E = {self._format_number(energy, 8)} Ha"
            )
  
    # ==================== Bootstrap & Assembly ====================
  
    def bootstrap_amplitudes(self, min_val: float, max_val: float):
        """Log bootstrap amplitude range in compact format."""
        self.logger.info(
            f"    Bootstrap |psi_S|: "
            f"[{self._format_number(min_val, 3)}, "
            f"{self._format_number(max_val, 3)}]"
        )
  
    def hamiltonian_assembled(self, h_type: str, nnz: int):
        """Log Hamiltonian assembly completion."""
        status = self._colorize("ASSEMBLED", 'green')
        self.logger.info(
            f"    {status}  {h_type} with {self._format_number(nnz, 0)} nonzeros"
        )
  
    # ==================== Wavefunction Cache ====================
  
    def wavefunction_cache(self, norm_s: float, norm_c: float):
        """Log wavefunction cache statistics in compact format."""
        total = norm_s + norm_c
        s_weight = 100.0 * norm_s / total if abs(total) > 1e-10 else 0.0
      
        self.logger.info(
            f"    Wavefunction cache: "
            f"||S||² = {self._format_number(norm_s, 4)}, "
            f"||C||² = {self._format_number(norm_c, 4)} "
            f"(S-weight: {self._format_number(s_weight, 1)}%)"
        )
  
    # ==================== Energy Diagnostics ====================
  
    def energy_table(self, energies: dict[str, float]):
        """Log diagnostic energy values in aligned table."""
        self.logger.info("    Energy diagnostics")
      
        max_label = max(len(label) for label in energies.keys())
      
        for label, value in energies.items():
            self.logger.info(
                f"      {label:<{max_label}}  {self._format_number(value, 8)} Ha"
            )
  
    # ==================== Warnings & Performance ====================
  
    def recompilation_warning(self, old_shape: tuple, new_shape: tuple):
        """Warn about JIT recompilation due to shape change."""
        self.logger.warning(
            f"  JIT recompilation triggered: shape {old_shape} -> {new_shape}"
        )
  
    def performance_note(self, message: str):
        """Log performance hint as warning."""
        self.logger.warning(f"  Performance hint: {message}")
  
    # ==================== Timing ====================
  
    def timing(self, stage: str, seconds: float):
        """Log stage timing with formatted duration."""
        if seconds < 60:
            time_str = f"{seconds:.2f} s"
        elif seconds < 3600:
            time_str = f"{seconds/60:.1f} min"
        else:
            time_str = f"{seconds/3600:.2f} h"
      
        status = self._colorize("COMPLETED", 'green')
        self.logger.info(f"  {status}  {stage} in {self._colorize(time_str, 'yellow')}")
  
    # ==================== Final Summary ====================
  
    def final_summary(
        self,
        total_time: float,
        final_energy: float,
        num_cycles: int
    ):
        """Print final workflow summary with key metrics."""
        self.header("LEVER Workflow Summary")
      
        self.logger.info(f"    Final energy              {self._format_number(final_energy, 8)} Ha")
        self.logger.info(f"    Total cycles              {self._format_number(num_cycles, 0)}")
      
        if total_time < 60:
            time_str = f"{total_time:.2f} s"
        elif total_time < 3600:
            time_str = f"{total_time/60:.1f} min ({total_time:.1f} s)"
        else:
            hours = total_time / 3600
            time_str = f"{hours:.2f} h ({total_time:.1f} s)"
      
        self.logger.info(f"    Total runtime             {self._colorize(time_str, 'yellow')}")
        self.logger.info("")


# Global singleton
_logger_instance: Optional[LeverLogger] = None


def get_logger(level: int = logging.INFO) -> LeverLogger:
    """Get or create global LEVER logger singleton."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = LeverLogger(level=level)
    return _logger_instance


__all__ = ["LeverLogger", "get_logger"]