# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant importance scoring implementations.

Provides concrete scorer classes for different physical metrics:
- AmplitudeScorer: Scores by wavefunction amplitude |ψ_i|
- PT2Scorer: Scores by second-order perturbation energy contribution

File: lever/evolution/scores.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ScoreResult

if TYPE_CHECKING:
    from ..engine import Evaluator


class AmplitudeScorer:
    """
    Scores determinants by wavefunction amplitude magnitude.
    
    Score formula: s_i = |ψ_i|
    
    Used for identifying important configurations in variational wavefunctions.
    """
    
    def compute_scores(self, evaluator: Evaluator) -> ScoreResult:
        """
        Compute amplitude-based scores for all determinants.
        
        Args:
            evaluator: Evaluation context with cached wavefunction amplitudes
            
        Returns:
            ScoreResult containing:
              - scores: |ψ_i| for all determinants
              - dets: Combined S and C space determinants
              - metadata: Empty dict
        """
        # Extract wavefunction components from cached results
        psi_s, psi_c = evaluator.wavefunction
        psi_all = np.concatenate([np.asarray(psi_s), np.asarray(psi_c)])
        
        # Merge determinant sets from both spaces
        dets_all = np.concatenate([
            evaluator.space.s_dets,
            evaluator.space.c_dets
        ], axis=0)

        # Score = absolute amplitude
        scores = np.abs(psi_all)

        return ScoreResult(scores=scores, dets=dets_all, metadata={})


__all__ = ["AmplitudeScorer"]
