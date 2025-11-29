# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Slater determinant wavefunction models for quantum many-body systems.

Implements single and multi-determinant expansions using neural orbital matrices.
Supports generalized (spin-orbital) and spatial (restricted/unrestricted) formulations.

File: lever/models/slater.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from . import utils


class Slater(nn.Module):
    """
    Neural Slater determinant wavefunction model.
    
    Computes log-amplitude ψ(s) = log(Σ_i c_i det(M_i[s,:])) where:
      - s: occupation number vector (2×n_orbitals for α/β spins)
      - M_i: learnable orbital matrices (single or multi-determinant)
      - c_i: optional expansion coefficients
    
    Orbital matrix configurations:
      - Generalized: Single matrix for all electrons (spin-orbital basis)
      - Restricted: Shared spatial orbitals for α/β (RHF-like)
      - Unrestricted: Separate α/β orbital matrices (UHF-like)
    """
    n_orb: int
    n_alpha: int
    n_beta: int
    n_dets: int = 1
    generalized: bool = False
    restricted: bool = True
    use_log_coeffs: bool = True
    param_dtype: Any = jnp.complex128
    kernel_init: Callable = initializers.orthogonal()

    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate log-amplitude for occupation vector.
        
        Args:
            s: Occupation numbers, shape (2×n_orbitals,) as [α_occs, β_occs]
        
        Returns:
            Complex scalar log(ψ(s))
        """
        n_elec = self.n_alpha + self.n_beta
        is_multi = self.n_dets > 1

        orbital_params = self._init_orbital_params(is_multi)
        eval_det = self._build_det_evaluator(n_elec, orbital_params, is_multi)
        
        return eval_det(s) if not is_multi else self._combine_multi_dets(eval_det, s)

    def _init_orbital_params(self, is_multi: bool) -> tuple | jnp.ndarray:
        """Initialize learnable orbital matrices."""
        n_elec = self.n_alpha + self.n_beta
        det_prefix = (self.n_dets,) if is_multi else ()

        if self.generalized:
            # Spin-orbital formulation: (n_spin_orbitals, n_elec)
            shape = (*det_prefix, 2 * self.n_orb, n_elec)
            return self.param("M_gen", self.kernel_init, shape, self.param_dtype)
        
        if self.restricted:
            # Shared spatial orbitals: (n_orb, max(n_α, n_β))
            k = max(self.n_alpha, self.n_beta)
            shape = (*det_prefix, self.n_orb, k)
            return self.param("M_spatial", self.kernel_init, shape, self.param_dtype)
        
        # Unrestricted: separate α/β matrices
        shape_a = (*det_prefix, self.n_orb, self.n_alpha)
        shape_b = (*det_prefix, self.n_orb, self.n_beta)
        M_a = self.param("M_alpha", self.kernel_init, shape_a, self.param_dtype)
        M_b = self.param("M_beta", self.kernel_init, shape_b, self.param_dtype)
        return M_a, M_b

    def _build_det_evaluator(
        self, 
        n_elec: int, 
        params: tuple | jnp.ndarray,
        is_multi: bool
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Construct determinant evaluation function with precomputed indices."""
        
        def _extract_indices(s: jnp.ndarray):
            """Compute occupied row indices for configuration s."""
            if self.generalized:
                rows = jnp.nonzero(s, size=n_elec, fill_value=-1)[0]
                return rows, None

            α_occ = s[:self.n_orb]
            β_occ = s[self.n_orb:]
            rows_α = jnp.nonzero(α_occ, size=self.n_alpha, fill_value=-1)[0]
            rows_β = jnp.nonzero(β_occ, size=self.n_beta, fill_value=-1)[0]
            return rows_α, rows_β

        def _single_det_log(M_params, rows_α, rows_β, s: jnp.ndarray) -> jnp.ndarray:
            """Evaluate log det for single determinant using precomputed indices."""
            if self.generalized:
                rows, _ = rows_α, rows_β
                A = M_params[rows, :]
                return utils.logdet_c(A)

            if self.restricted:
                # Shared spatial orbitals: log det_α + log det_β
                A_α = M_params[rows_α, :self.n_alpha]
                A_β = M_params[rows_β, :self.n_beta]
                return utils.logdet_c(A_α) + utils.logdet_c(A_β)

            # Unrestricted: separate α/β matrices
            M_α, M_β = M_params
            A_α = M_α[rows_α, :]
            A_β = M_β[rows_β, :]
            return utils.logdet_c(A_α) + utils.logdet_c(A_β)

        if not is_multi:
            def eval_det(s: jnp.ndarray) -> jnp.ndarray:
                rows_α, rows_β = _extract_indices(s)
                return _single_det_log(params, rows_α, rows_β, s)
            return eval_det

        # Multi-determinant: batched logdet computation
        if self.generalized:
            def eval_det(s: jnp.ndarray) -> jnp.ndarray:
                rows_α, _ = _extract_indices(s)
                A = params[:, rows_α, :]  # (n_dets, n_elec, n_elec)
                return utils.logdet_c(A)
            return eval_det

        if self.restricted:
            def eval_det(s: jnp.ndarray) -> jnp.ndarray:
                rows_α, rows_β = _extract_indices(s)
                M = params  # (n_dets, n_orb, k)
                A_α = M[:, rows_α, :self.n_alpha]
                A_β = M[:, rows_β, :self.n_beta]
                return utils.logdet_c(A_α) + utils.logdet_c(A_β)
            return eval_det

        # Unrestricted multi-determinant
        M_α, M_β = params
        def eval_det(s: jnp.ndarray) -> jnp.ndarray:
            rows_α, rows_β = _extract_indices(s)
            A_α = M_α[:, rows_α, :]
            A_β = M_β[:, rows_β, :]
            return utils.logdet_c(A_α) + utils.logdet_c(A_β)
        return eval_det

    def _combine_multi_dets(self, eval_det: Callable, s: jnp.ndarray) -> jnp.ndarray:
        """
        Combine multiple determinants with learned coefficients.
        
        Computes: log(Σ_i exp(log_c_i + log_det_i))
        """
        log_dets = eval_det(s)  # Shape: (n_dets,)
        
        if self.use_log_coeffs:
            log_coeffs = self.param(
                "log_coeffs",
                nn.initializers.zeros,
                (self.n_dets,),
                self.param_dtype
            )
            log_dets = log_dets + log_coeffs
        
        return utils.logsumexp_c(log_dets, axis=0)


__all__ = ["Slater"]
