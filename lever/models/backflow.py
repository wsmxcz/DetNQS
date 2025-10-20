# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backflow network: configuration-dependent Slater determinant via MLP.

Implements orbital matrix correction ΔM(s) for electron correlation capture.
Real-valued residual architecture with complex projection head for stability.

Architecture: ψ = Σᵢ cᵢ det(Mᵢ + ΔMᵢ(s))
  - Real MLP backbone processes electron occupancy patterns
  - Complex output via Re/Im channel split maintains numerical precision
  - Multi-determinant expansion with learned coefficients (optional)

File: lever/models/backflow.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from .utils import c_orthogonal_init, logdet_c, logsumexp_c


class BackflowMLP(nn.Module):
    """
    Configuration-dependent Slater determinant with backflow correction.

    Core equation: log ψ(s) = log det(M + ΔM(s))
    Multi-det: log ψ(s) = log Σᵢ cᵢ exp(log det(Mᵢ + ΔMᵢ(s)))

    Attributes:
        n_orbitals: Number of spatial orbitals
        n_alpha, n_beta: Spin-up/down electron counts (α/β)
        n_dets: Determinants in expansion (1 for single-det)
        generalized: Unified spin-orbital matrix (no spin separation)
        restricted: Shared spatial orbitals for α/β
        use_log_coeffs: Learn expansion coefficients log(cᵢ)
        hidden_dims: Residual block widths
        param_dtype: Complex type for orbital matrices
        kernel_init, bias_init: Weight initializers
    """

    n_orbitals: int
    n_alpha: int
    n_beta: int
    n_dets: int = 1
    generalized: bool = False
    restricted: bool = True
    use_log_coeffs: bool = True
    hidden_dims: tuple[int, ...] = (256,)
    param_dtype: Any = jnp.complex64
    kernel_init: Callable = initializers.glorot_normal()
    bias_init: Callable = initializers.zeros_init()

    def _mlp(self, name: str, out_dim: int, s: jnp.ndarray) -> jnp.ndarray:
        """
        Real residual MLP → complex projection via Re/Im split.

        Flow: s → [residual blocks] → GELU → split(2×out_dim) → complex assembly
        Scaling: Var[Re] = Var[Im] = 1/2 → Var[|z|²] = 1

        Args:
            name: Parameter namespace prefix
            out_dim: Number of complex outputs
            s: Real occupancy vector

        Returns:
            Complex array (..., out_dim) with controlled variance
        """
        real_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
        x = s.astype(real_dtype)

        # Residual blocks with pre-activation
        for li, h in enumerate(self.hidden_dims):
            residual = x
            x = nn.gelu(x)
            x = nn.Dense(
                h,
                param_dtype=real_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"{name}_dense_{li}",
            )(x)

            # Dimension matching for skip connection
            if residual.shape[-1] != h:
                residual = nn.Dense(
                    h,
                    param_dtype=real_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    name=f"{name}_skip_{li}",
                )(residual)

            x = x + residual

        # Complex projection: small initial corrections
        x = nn.gelu(x)
        x = nn.Dense(
            2 * out_dim,
            param_dtype=real_dtype,
            kernel_init=initializers.variance_scaling(
                scale=0.01, mode="fan_avg", distribution="truncated_normal"
            ),
            bias_init=self.bias_init,
            name=f"{name}_out",
        )(x)

        # Maintain unit variance: split and scale
        x = x / jnp.sqrt(2.0)
        re, im = jnp.split(x, 2, axis=-1)
        target_dtype = jnp.real(jnp.zeros(1, dtype=self.param_dtype)).dtype

        return jax.lax.complex(re.astype(target_dtype), im.astype(target_dtype))

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        """
        Compute log-amplitude via backflow-corrected determinant(s).

        Single-det: log ψ = log|det(M + ΔM(s))[occupied rows]|
        Multi-det: log ψ = log Σᵢ cᵢ exp(log ψᵢ)

        Args:
            s: Binary occupancy (2×n_orbitals,)
               Layout: [α₁,...,αₙ, β₁,...,βₙ]

        Returns:
            Complex log-amplitude log ψ(s)
        """
        n_e = self.n_alpha + self.n_beta
        is_multi = self.n_dets > 1

        # Initialize base orbitals and construct evaluator
        orbitals = self._init_orbitals(is_multi)
        eval_fn = self._build_evaluator(n_e, orbitals, is_multi, s)

        # Combine determinants if needed
        if not is_multi:
            return eval_fn(s)

        return self._combine_dets(eval_fn, s)

    def _init_orbitals(self, is_multi: bool) -> tuple | jnp.ndarray:
        """Initialize learnable base orbital matrices."""
        n_e = self.n_alpha + self.n_beta
        det_shape = (self.n_dets,) if is_multi else ()

        if self.generalized:
            # Spin-orbital: (2n × n_e) unified matrix
            shape = (*det_shape, 2 * self.n_orbitals, n_e)
            return self.param("M_gen", c_orthogonal_init, shape, self.param_dtype)

        if self.restricted:
            # Restricted: shared spatial (n × k) where k = max(n_α, n_β)
            k = max(self.n_alpha, self.n_beta)
            shape = (*det_shape, self.n_orbitals, k)
            return self.param("M_spatial", c_orthogonal_init, shape, self.param_dtype)

        # Unrestricted: separate α/β matrices
        shape_α = (*det_shape, self.n_orbitals, self.n_alpha)
        shape_β = (*det_shape, self.n_orbitals, self.n_beta)
        M_α = self.param("M_alpha", c_orthogonal_init, shape_α, self.param_dtype)
        M_β = self.param("M_beta", c_orthogonal_init, shape_β, self.param_dtype)
        return M_α, M_β

    def _build_evaluator(
        self,
        n_e: int,
        orbitals: tuple | jnp.ndarray,
        is_multi: bool,
        s: jnp.ndarray,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Construct backflow determinant evaluator: s → log det(M + ΔM(s)).

        Returns:
            Single-det: s → complex scalar
            Multi-det: s → array of log-determinants (n_dets,)
        """

        def eval_single(M_base, idx: int, s: jnp.ndarray) -> jnp.ndarray:
            """Evaluate log|det(M + ΔM(s))| for one determinant."""
            suffix = f"_det{idx}" if is_multi else ""

            if self.generalized:
                # Generalized: single (2n × n_e) correction
                out_dim = (2 * self.n_orbitals) * n_e
                Δ = self._mlp(f"BF_gen{suffix}", out_dim, s).reshape(
                    (2 * self.n_orbitals, n_e)
                )
                M_eff = M_base + Δ

                # Extract occupied spin-orbital rows
                rows = jnp.nonzero(s, size=n_e, fill_value=-1)[0]
                return logdet_c(M_eff[rows, :])

            # Split α/β occupancies
            α_occ = s[: self.n_orbitals]
            β_occ = s[self.n_orbitals :]
            rows_α = jnp.nonzero(α_occ, size=self.n_alpha, fill_value=-1)[0]
            rows_β = jnp.nonzero(β_occ, size=self.n_beta, fill_value=-1)[0]

            if self.restricted:
                # Restricted: shared (n × k) spatial correction
                k = max(self.n_alpha, self.n_beta)
                out_dim = self.n_orbitals * k
                Δ = self._mlp(f"BF_res{suffix}", out_dim, s).reshape(
                    (self.n_orbitals, k)
                )
                M_eff = M_base + Δ

                # Product: det(α) × det(β)
                A_α = M_eff[rows_α, : self.n_alpha]
                A_β = M_eff[rows_β, : self.n_beta]
                return logdet_c(A_α) + logdet_c(A_β)

            # Unrestricted: separate α/β corrections
            M_α_base, M_β_base = M_base
            out_dim = self.n_orbitals * (self.n_alpha + self.n_beta)
            Δ = self._mlp(f"BF_unres{suffix}", out_dim, s).reshape(
                (self.n_orbitals, self.n_alpha + self.n_beta)
            )

            Δ_α = Δ[:, : self.n_alpha]
            Δ_β = Δ[:, self.n_alpha :]

            A_α = (M_α_base + Δ_α)[rows_α, :]
            A_β = (M_β_base + Δ_β)[rows_β, :]
            return logdet_c(A_α) + logdet_c(A_β)

        if not is_multi:
            return lambda s: eval_single(orbitals, 0, s)

        # Multi-det: vectorize over determinant index
        def eval_all(s: jnp.ndarray) -> jnp.ndarray:
            if self.generalized or self.restricted:
                return jax.vmap(lambda i, M: eval_single(M, i, s), in_axes=(0, 0))(
                    jnp.arange(self.n_dets), orbitals
                )
            # Unrestricted: vmap over paired α/β
            M_α, M_β = orbitals
            return jax.vmap(lambda i, Ma, Mb: eval_single((Ma, Mb), i, s), in_axes=(0, 0, 0))(
                jnp.arange(self.n_dets), M_α, M_β
            )

        return eval_all

    def _combine_dets(
        self, eval_fn: Callable, s: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Combine determinants: log(Σᵢ cᵢ exp(log ψᵢ)).

        Args:
            eval_fn: Returns log-determinants (n_dets,)
            s: Occupancy vector

        Returns:
            Combined log-amplitude
        """
        log_dets = eval_fn(s)  # Shape: (n_dets,)

        if self.use_log_coeffs:
            log_c = self.param(
                "log_coeffs", nn.initializers.zeros, (self.n_dets,), self.param_dtype
            )
            log_dets = log_dets + log_c

        return logsumexp_c(log_dets, axis=0)


__all__ = ["BackflowMLP"]
