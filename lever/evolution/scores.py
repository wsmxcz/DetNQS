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

from .base import ScoreResult

if TYPE_CHECKING:
    from ..engine import OuterCtx
    from ..engine.utils import PyTree


class AmpScorer:
    """Scores by wavefunction amplitude |ψ_i|."""
    
    def score(self, ctx: OuterCtx, params: PyTree) -> ScoreResult:
        """
        Compute amplitude-based scores.
        
        Args:
            ctx: Outer context with cached features
            params: Converged parameters
            
        Returns:
            ScoreResult with |ψ| scores for S ∪ C
        """
        # Use cached batch_logpsi (no redundant network call)
        log_all = ctx.logpsi_fn(params)
        psi_all = np.array(jnp.exp(log_all))
        
        # Merge determinant sets
        dets_all = np.concatenate([ctx.space.s_dets, ctx.space.c_dets], axis=0)
        
        # Score = absolute amplitude
        scores = np.abs(psi_all)
        
        return ScoreResult(scores=scores, dets=dets_all, meta={})


__all__ = ["AmpScorer"]
