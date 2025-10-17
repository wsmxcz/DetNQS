# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Restricted Boltzmann Machine (RBM) wavefunction ansätze.

Implements complex-parameter and modulus-phase RBM variants for
quantum many-body wavefunction representation.

File: lever/models/rbm.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from . import utils


class RBM(nn.Module):
    """
    Complex-parameter Restricted Boltzmann Machine.
    
    Computes log-amplitude via energy-based model:
        log ψ(s) = Σⱼ log cosh(bₕⱼ + Σᵢ Wᵢⱼ sᵢ) + s · bᵥ
    
    where s are visible units (occupation numbers), and hidden units
    capture non-local correlations through weight matrix W.
    
    Attributes:
        alpha: Hidden-to-visible ratio (n_hidden = α × n_visible)
        param_dtype: Parameter dtype (complex128 for holomorphic model)
        use_visible_bias: Include learnable visible bias bᵥ
    """
    
    alpha: float = 1.0
    param_dtype: Any = jnp.complex128
    use_visible_bias: bool = True
    
    __lever_is_holomorphic__ = True  # Guide auto-detection in base.py

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate log ψ(s) for occupation vector s.
        
        Args:
            s: Visible configuration, shape (n_visible,)
            
        Returns:
            Complex log-amplitude scalar
        """
        n_vis = s.shape[-1]
        n_hid = int(self.alpha * n_vis)
        sigma = 1.0 / jnp.sqrt(float(n_vis + n_hid))

        # Initialize parameters with variance scaling
        W = self.param("W", utils.c_init(sigma), (n_vis, n_hid), self.param_dtype)
        b_h = self.param("b_h", utils.c_init(sigma), (n_hid,), self.param_dtype)
        
        # Hidden activations: θⱼ = bₕⱼ + Σᵢ Wᵢⱼ sᵢ
        theta = b_h + jnp.dot(s.astype(self.param_dtype), W)
        log_psi = jnp.sum(utils.log_cosh(theta), axis=-1)

        if self.use_visible_bias:
            b_v = self.param("b_v", utils.c_init(sigma), (n_vis,), self.param_dtype)
            log_psi += jnp.dot(s.astype(self.param_dtype), b_v)
            
        return log_psi


class RBMModPhase(nn.Module):
    """
    Twin-RBM for separate modulus and phase modeling.
    
    Decomposes wavefunction into real-valued components:
        log ψ(s) = RBM_mod(s) + i × RBM_phase(s)
    
    Uses independent real-parameter RBMs to avoid holomorphic constraints
    and improve optimization landscape. Numerically stable for variational
    Monte Carlo with separate amplitude/phase optimization.
    
    Attributes:
        alpha: Hidden-to-visible ratio per RBM
        param_dtype: Parameter dtype (float64 for real networks)
        use_hidden_bias: Include hidden bias in Dense layers
        kernel_init: Weight matrix initializer
        bias_init: Bias vector initializer
    """
    
    alpha: float = 1.0
    param_dtype: Any = jnp.float64
    use_hidden_bias: bool = True
    kernel_init: Callable = initializers.lecun_normal()
    bias_init: Callable = initializers.zeros
    
    __lever_is_holomorphic__ = False  # Non-holomorphic by design

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate log ψ(s) with separate modulus/phase networks.
        
        Args:
            s: Visible configuration, shape (n_visible,)
            
        Returns:
            Complex log-amplitude with real/imag from independent RBMs
        """
        n_vis = s.shape[-1]
        n_hid = int(self.alpha * n_vis)

        # Modulus network: log |ψ(s)|
        mod_dense = nn.Dense(
            features=n_hid,
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="ModulusRBM",
        )
        modulus = jnp.sum(utils.log_cosh(mod_dense(s)), axis=-1)

        # Phase network: arg ψ(s)
        phase_dense = nn.Dense(
            features=n_hid,
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="PhaseRBM",
        )
        phase = jnp.sum(utils.log_cosh(phase_dense(s)), axis=-1)

        return modulus + 1j * phase


__all__ = ["RBM", "RBMModPhase"]
