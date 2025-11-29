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
from ..utils.config_utils import capture_config

if TYPE_CHECKING:
    from ..dtypes import OuterCtx, PsiCache

@capture_config
class AmplitudeScorer:
    """
    Scores determinants by probability mass |ψ_i|².

    The returned scores are non-negative and sum to the total norm
    of the wavefunction over S ∪ C.
    """

    def score(self, ctx: OuterCtx, psi_cache: PsiCache) -> ScoreResult:
        """
        Compute probability-mass-based scores from cache.

        Args:
            ctx: Outer context providing S and C determinants
            psi_cache: Cached wavefunction amplitudes

        Returns:
            ScoreResult with |ψ_i|² scores for S ∪ C
        """
        psi_all = np.asarray(psi_cache.psi_all)
        dets_all = np.concatenate([ctx.space.s_dets, ctx.space.c_dets], axis=0)

        # Use probability mass as the score
        scores = np.abs(psi_all) ** 2
        return ScoreResult(scores=scores, dets=dets_all, meta={})


# Backward-compatible alias
AmpScorer = AmplitudeScorer

__all__ = ["AmplitudeScorer", "AmpScorer"]