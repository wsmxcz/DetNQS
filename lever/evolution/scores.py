# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant importance scoring with OuterCtx.

File: lever/evolution/scores.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from ..utils.dtypes import ScoreResult

if TYPE_CHECKING:
    from ..utils.dtypes import OuterCtx, PyTree, PsiCache


class AmpScorer:
    """Scores by wavefunction amplitude |ψ_i|."""
    
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
