# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Update rules for step size control.

Phase 1: Constant learning rate
Phase 4+: Line search, trust region

File: lever/optimizers/rule.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from ..utils.dtypes import PyTree


class ConstantRule:
    """
    Fixed learning rate: α = constant.
    
    Simplest update rule, compatible with all direction providers.
    """
    
    def __init__(self, learning_rate: float):
        """
        Args:
            learning_rate: Fixed step size α
        """
        self.learning_rate = learning_rate
    
    def __call__(
        self,
        direction: PyTree,
        params: PyTree,
        energy: float | None = None
    ) -> float:
        """
        Return fixed learning rate.
        
        Args:
            direction: Search direction (unused)
            params: Current parameters (unused)
            energy: Current energy (unused)
            
        Returns:
            Fixed learning rate
        """
        return self.learning_rate


__all__ = ["ConstantRule"]
