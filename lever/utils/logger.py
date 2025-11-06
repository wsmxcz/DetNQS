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
    """Simplified logger without timestamps and log levels."""
    
    # ========== Headers ==========
    def header(self, title: str):
        """Print section header."""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
    
    def optimization_header(self):
        """Print optimization start header."""
        pass  # Silent by default, can enable if needed
        # print("Starting parameter optimization...")
    
    # ========== Configuration ==========
    def config_info(self, items: dict):
        """Print configuration key-value pairs."""
        for key, value in items.items():
            print(f"{key:20s}: {value}")
    
    def system_info(self, fcidump: str, n_orb: int, n_elec: tuple):
        """Print system information."""
        print(f"System:   {fcidump}")
        print(f"Orbitals: {n_orb}")
        print(f"Electrons: α={n_elec[0]}, β={n_elec[1]}")
    
    # ========== Space & Hamiltonian ==========
    def space_dimensions(self, n_s: int, n_c: int):
        """Print space dimensions."""
        print(f"S-space: {n_s:6d} determinants")
        print(f"C-space: {n_c:6d} determinants")
    
    def hamiltonian_sparsity(self, n_s: int, nnz_ss: int, nnz_sc: int = None):
        """Print Hamiltonian sparsity statistics."""
        density_ss = 100.0 * nnz_ss / (n_s * n_s)
        print(f"H_SS: {nnz_ss:8d} nonzeros ({density_ss:.3f}% dense)")
        
        if nnz_sc is not None:
            print(f"H_SC: {nnz_sc:8d} nonzeros")
    
    def bootstrap_amplitudes(self, min_val: float, max_val: float):
        """Print bootstrap amplitude range."""
        print(f"Bootstrap amplitudes: [{min_val:.6f}, {max_val:.6f}]")
    
    # ========== Compilation & Caching ==========
    def recompilation_warning(self, old_shape: tuple, new_shape: tuple):
        """Warn about JIT recompilation due to shape change."""
        print(f"Warning: Shape change {old_shape} → {new_shape} triggers recompilation")
    
    def wavefunction_cache(self, norm_s: float, norm_c: float):
        """Print wavefunction cache statistics."""
        pass  # Silent by default
        # print(f"Cache: ‖ψ_S‖² = {norm_s:.6f}, ‖ψ_C‖² = {norm_c:.6f}")
    
    def compilation_start(self):
        """Print compilation start."""
        pass  # Silent
    
    def compilation_complete(self):
        """Print compilation complete."""
        pass  # Silent
    
    # ========== Diagnostics ==========
    def diagnostic_energy(self, label: str, energy: float):
        """Print diagnostic energy."""
        print(f"{label:6s} energy: {energy:.10f}")
    
    def variational_energy(self, energy: float):
        """Print variational energy."""
        print(f"Var    energy: {energy:.10f}")
    
    def s_ci_energy(self, energy: float):
        """Print S-CI energy."""
        print(f"S-CI   energy: {energy:.10f}")
    
    def t_ci_energy(self, energy: float):
        """Print T-CI energy."""
        print(f"T-CI   energy: {energy:.10f}")
    
    # ========== Cycle Progress ==========
    def cycle_start(self, cycle: int, max_cycles: int):
        """Print cycle start marker."""
        pass  # Silent, replaced by cycle_complete
    
    def cycle_complete(self, cycle: int, energy: float, steps: int, time_elapsed: float):
        """Print cycle completion summary."""
        print(
            f"Cycle {cycle:3d} | "
            f"E = {energy:.10f} | "
            f"Steps = {steps:4d} | "
            f"Time = {time_elapsed:.2f}s"
        )
    
    # ========== Fitting ==========
    def fitting_start(self, max_steps: int):
        """Print fitting start."""
        pass  # Silent
    
    def fitting_complete(self, final_step: int, converged: bool):
        """Print fitting complete."""
        pass  # Silent, info included in cycle_complete
    
    def fitting_progress(self, step: int, max_steps: int, energy: float):
        """Print fitting progress (for debugging)."""
        pass  # Never used with silent mode
    
    # ========== Evolution ==========
    def evolution_start(self):
        """Print evolution start."""
        pass  # Silent
    
    def evolution_complete(self, n_added: int, n_removed: int):
        """Print evolution complete."""
        pass  # Silent, can enable if needed
        # print(f"Evolution: +{n_added}, -{n_removed}")
    
    def no_evolution(self):
        """Print no evolution message."""
        pass  # Silent
    
    # ========== Convergence & Finalization ==========
    def outer_loop_converged(self, cycle: int, delta: float):
        """Print outer loop convergence message."""
        print(f"Outer loop converged at cycle {cycle} (ΔE = {delta:.2e})")
    
    def max_cycles_reached(self, max_cycles: int):
        """Print max cycles reached."""
        print(f"Reached maximum cycles ({max_cycles})")
    
    def final_summary(self, total_time: float, final_energy: float, n_cycles: int):
        """Print final workflow summary."""
        print(f"\n{'='*60}")
        print(f"LEVER workflow completed")
        print(f"Final energy:  {final_energy:.10f}")
        print(f"Total cycles:  {n_cycles}")
        print(f"Total time:    {total_time:.2f}s")
        print(f"{'='*60}\n")
    
    # ========== Errors & Warnings ==========
    def warning(self, message: str):
        """Print warning message."""
        print(f"Warning: {message}")
    
    def error(self, message: str):
        """Print error message."""
        print(f"ERROR: {message}")


_LOGGER_INSTANCE = None

def get_logger() -> Logger:
    """Get global logger singleton."""
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = Logger()
    return _LOGGER_INSTANCE


__all__ = ["Logger", "get_logger"]
