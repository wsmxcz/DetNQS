# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant importance scoring strategies.

Provides amplitude-based and energy-based scoring functions
for space evolution.

File: lever/evolution/scores.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ..dtypes import ScoreResult

if TYPE_CHECKING:
    from ..dtypes import OuterCtx, PsiCache


class AmpScorer:
    """Scores determinants by wavefunction amplitude |ψ_i|."""
    
    def score(self, ctx: OuterCtx, psi_cache: PsiCache) -> ScoreResult:
        """
        Compute amplitude-based scores from cache.
        
        Args:
            ctx: Outer context for determinant merging
            psi_cache: Cached wavefunction amplitudes
        
        Returns:
            ScoreResult with |ψ| scores for S ∪ C
        """
        psi_all = np.array(psi_cache.psi_all)
        dets_all = np.concatenate([ctx.space.s_dets, ctx.space.c_dets], axis=0)
        scores = np.abs(psi_all)
        return ScoreResult(scores=scores, dets=dets_all, meta={})


__all__ = ["AmpScorer"]
