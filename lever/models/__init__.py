# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wavefunction model components and factory functions.

Provides neural quantum state ansätze: Slater determinants, Pfaffians,
RBMs, Jastrow factors, and their compositions. All models map occupation
basis vectors to complex log-amplitudes log ψ(s).

File: lever/models/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from .slater import make_slater

__all__ = [
    "make_slater",
]