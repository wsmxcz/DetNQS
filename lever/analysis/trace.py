# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight trajectory recorder for optimization runs.

Stores essential runtime data:
  - Energy values per outer loop
  - Timestamps for performance tracking
  - Space sizes and norm decomposition for evolution monitoring

File: lever/analysis/trace.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trace:
    """Immutable energy trajectory container with norm tracking."""
  
    outers: np.ndarray      # Outer loop indices [n_points]
    energies: np.ndarray    # Electronic energies [n_points]
    timestamps: np.ndarray  # Cumulative runtime in seconds [n_points]
    size_s: np.ndarray      # S-space sizes [n_points]
    size_c: np.ndarray      # C-space sizes [n_points]
    norm_s: np.ndarray      # ||ψ_S||² [n_points]
    norm_c: np.ndarray      # ||ψ_C||² [n_points]
  
    @staticmethod
    def empty() -> "Trace":
        """Create empty trace."""
        return Trace(
            outers=np.array([], dtype=int),
            energies=np.array([]),
            timestamps=np.array([]),
            size_s=np.array([], dtype=int),
            size_c=np.array([], dtype=int),
            norm_s=np.array([]),
            norm_c=np.array([]),
        )
  
    def append(
        self,
        outer: int,
        energy: float,
        timestamp: float,
        n_s: int,
        n_c: int,
        norm_s: float,
        norm_c: float,
    ) -> "Trace":
        """Append new data point (functional update)."""
        return Trace(
            outers=np.append(self.outers, outer),
            energies=np.append(self.energies, energy),
            timestamps=np.append(self.timestamps, timestamp),
            size_s=np.append(self.size_s, n_s),
            size_c=np.append(self.size_c, n_c),
            norm_s=np.append(self.norm_s, norm_s),
            norm_c=np.append(self.norm_c, norm_c),
        )
  
    def last_per_outer(self) -> dict[int, float]:
        """Extract final energy for each outer loop."""
        if len(self.energies) == 0:
            return {}
      
        result = {}
        for outer, energy in zip(self.outers, self.energies):
            result[int(outer)] = float(energy)
      
        return result
    
    def norm_fractions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized fractions: ||ψ_S||²/||ψ_T||², ||ψ_C||²/||ψ_T||².
        
        Returns:
            (frac_s, frac_c): Arrays of normalized fractions
        """
        norm_tot = self.norm_s + self.norm_c
        norm_tot = np.where(norm_tot > 1e-14, norm_tot, 1.0)
        
        frac_s = self.norm_s / norm_tot
        frac_c = self.norm_c / norm_tot
        
        return frac_s, frac_c
  
    @property
    def n_points(self) -> int:
        """Number of recorded points."""
        return len(self.energies)
  
    @property
    def total_time(self) -> float:
        """Total runtime."""
        return float(self.timestamps[-1]) if len(self.timestamps) > 0 else 0.0


__all__ = ["Trace"]
