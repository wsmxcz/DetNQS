# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backflow networks for configuration-dependent Slater determinants.

Implements orbital correction ΔM(s) via MLP to capture electron correlation:
  ψ(s) = Σᵢ cᵢ det(Mᵢ + ΔMᵢ(s))

Two variants:
  - BackflowMLP: Real MLP with complex projection (numerically stable)
  - cBackflowMLP: Native complex arithmetic with Wirtinger gradients

File: lever/models/backflow.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers

from .utils import (
    c_orthogonal_init,
    logdet_c,
    logsumexp_c,
    complex_glorot_init,
    complex_he_init,
)


# ============================================================================
# Real-valued Backflow Network
# ============================================================================


class BackflowMLP(nn.Module):
    """
    Real MLP with complex projection for orbital correction.
    
    Architecture: s → [residual blocks] → GELU → Re/Im split → complex ΔM
    Maintains numerical stability through controlled variance scaling.
    
    Attributes:
        n_orb: Number of spatial orbitals
        n_alpha, n_beta: Spin-up/down electron counts
        n_dets: Number of determinants in expansion
        generalized: Use unified spin-orbital matrix
        restricted: Share spatial orbitals between spins
        use_log_coeffs: Learn determinant expansion coefficients
        hidden_dims: MLP hidden layer dimensions
        param_dtype: Complex type for orbital matrices
        kernel_init, bias_init: Weight initializers
    """

    n_orb: int
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
        """Generate complex correction via real residual MLP + Re/Im projection."""
        real_dtype = jnp.float32 if self.param_dtype == jnp.complex64 else jnp.float64
        x = s.astype(real_dtype)

        # Residual blocks with pre-activation normalization
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

            # Dimension-matching skip connection
            if residual.shape[-1] != h:
                residual = nn.Dense(
                    h,
                    param_dtype=real_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    name=f"{name}_skip_{li}",
                )(residual)

            x = x + residual

        # Project to complex with controlled variance
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

        # Split and assemble: Var[Re] = Var[Im] = 1/2 → Var[|z|²] = 1
        x = x / jnp.sqrt(2.0)
        re, im = jnp.split(x, 2, axis=-1)
        target_dtype = jnp.real(jnp.zeros(1, dtype=self.param_dtype)).dtype

        return jax.lax.complex(re.astype(target_dtype), im.astype(target_dtype))

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        """
        Compute log-amplitude: log ψ(s) via backflow determinants.
        
        Args:
            s: Binary occupancy [α₁...αₙ β₁...βₙ] shape (2n,)
        
        Returns:
            Complex log-amplitude
        """
        n_e = self.n_alpha + self.n_beta
        is_multi = self.n_dets > 1

        orbitals = self._init_orbitals(is_multi)
        eval_fn = self._build_evaluator(n_e, orbitals, is_multi, s)

        if not is_multi:
            return eval_fn(s)

        return self._combine_dets(eval_fn, s)

    def _init_orbitals(self, is_multi: bool) -> tuple | jnp.ndarray:
        """Initialize base orbital matrices with orthonormal columns."""
        n_e = self.n_alpha + self.n_beta
        det_shape = (self.n_dets,) if is_multi else ()

        if self.generalized:
            shape = (*det_shape, 2 * self.n_orb, n_e)
            return self.param("M_gen", c_orthogonal_init, shape, self.param_dtype)

        if self.restricted:
            k = max(self.n_alpha, self.n_beta)
            shape = (*det_shape, self.n_orb, k)
            return self.param("M_spatial", c_orthogonal_init, shape, self.param_dtype)

        # Unrestricted: separate α/β orbital spaces
        shape_α = (*det_shape, self.n_orb, self.n_alpha)
        shape_β = (*det_shape, self.n_orb, self.n_beta)
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
        """Build function: s -> log det(M + ΔM(s))."""

        def _extract_indices(s: jnp.ndarray):
            """Compute occupied row indices for configuration s."""
            if self.generalized:
                rows = jnp.nonzero(s, size=n_e, fill_value=-1)[0]
                return rows, None

            α_occ = s[: self.n_orb]
            β_occ = s[self.n_orb:]
            rows_α = jnp.nonzero(α_occ, size=self.n_alpha, fill_value=-1)[0]
            rows_β = jnp.nonzero(β_occ, size=self.n_beta, fill_value=-1)[0]
            return rows_α, rows_β

        if not is_multi:
            def eval_single(s: jnp.ndarray) -> jnp.ndarray:
                rows_α, rows_β = _extract_indices(s)

                if self.generalized:
                    out_dim = (2 * self.n_orb) * n_e
                    Δ = self._mlp("BF_gen", out_dim, s).reshape(
                        (2 * self.n_orb, n_e)
                    )
                    M_eff = orbitals + Δ
                    rows = rows_α
                    return logdet_c(M_eff[rows, :])

                if self.restricted:
                    k = max(self.n_alpha, self.n_beta)
                    out_dim = self.n_orb * k
                    Δ = self._mlp("BF_res", out_dim, s).reshape(
                        (self.n_orb, k)
                    )
                    M_eff = orbitals + Δ
                    A_α = M_eff[rows_α, : self.n_alpha]
                    A_β = M_eff[rows_β, : self.n_beta]
                    return logdet_c(A_α) + logdet_c(A_β)

                # Unrestricted
                M_α_base, M_β_base = orbitals
                out_dim = self.n_orb * (self.n_alpha + self.n_beta)
                Δ = self._mlp("BF_unres", out_dim, s).reshape(
                    (self.n_orb, self.n_alpha + self.n_beta)
                )
                Δ_α = Δ[:, : self.n_alpha]
                Δ_β = Δ[:, self.n_alpha :]
                A_α = (M_α_base + Δ_α)[rows_α, :]
                A_β = (M_β_base + Δ_β)[rows_β, :]
                return logdet_c(A_α) + logdet_c(A_β)

            return eval_single

        # Multi-determinant case
        def eval_multi(s: jnp.ndarray) -> jnp.ndarray:
            rows_α, rows_β = _extract_indices(s)

            if self.generalized:
                out_dim_per_det = (2 * self.n_orb) * n_e
                total_dim = self.n_dets * out_dim_per_det
                Δ_flat = self._mlp("BF_gen", total_dim, s)
                Δ = Δ_flat.reshape(self.n_dets, 2 * self.n_orb, n_e)
                M_eff = orbitals + Δ
                rows = rows_α
                A = M_eff[:, rows, :]
                return logdet_c(A)

            if self.restricted:
                k = max(self.n_alpha, self.n_beta)
                out_dim_per_det = self.n_orb * k
                total_dim = self.n_dets * out_dim_per_det
                Δ_flat = self._mlp("BF_res", total_dim, s)
                Δ = Δ_flat.reshape(self.n_dets, self.n_orb, k)
                M_eff = orbitals + Δ
                A_α = M_eff[:, rows_α, : self.n_alpha]
                A_β = M_eff[:, rows_β, : self.n_beta]
                return logdet_c(A_α) + logdet_c(A_β)

            # Unrestricted multi-determinant
            M_α_base, M_β_base = orbitals
            out_dim_per_det = self.n_orb * (self.n_alpha + self.n_beta)
            total_dim = self.n_dets * out_dim_per_det
            Δ_flat = self._mlp("BF_unres", total_dim, s)
            Δ = Δ_flat.reshape(self.n_dets, self.n_orb, self.n_alpha + self.n_beta)
            Δ_α = Δ[:, :, : self.n_alpha]
            Δ_β = Δ[:, :, self.n_alpha :]
            A_α = (M_α_base + Δ_α)[:, rows_α, :]
            A_β = (M_β_base + Δ_β)[:, rows_β, :]
            return logdet_c(A_α) + logdet_c(A_β)

        return eval_multi

    def _combine_dets(self, eval_fn: Callable, s: jnp.ndarray) -> jnp.ndarray:
        """Multi-determinant combination via log-sum-exp."""
        log_dets = eval_fn(s)

        if self.use_log_coeffs:
            log_c = self.param(
                "log_coeffs", nn.initializers.zeros, (self.n_dets,), self.param_dtype
            )
            log_dets = log_dets + log_c

        return logsumexp_c(log_dets, axis=0)


# ============================================================================
# Complex-valued Backflow Network
# ============================================================================


class ComplexDense(nn.Module):
    """Complex linear layer with explicit dtype handling."""

    features: int
    param_dtype: Any = jnp.complex64
    kernel_init: Callable = complex_glorot_init()
    bias_init: Callable = initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.astype(self.param_dtype)
        return nn.Dense(
            self.features,
            dtype=self.param_dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)


class ModReLU(nn.Module):
    """
    Modulus ReLU: f(z) = ReLU(|z| + b) * z/|z|.
    
    Phase-preserving amplitude gating with learnable per-channel bias.
    """

    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        features = z.shape[-1]
        eps = 1e-6 if z.real.dtype == jnp.float32 else 1e-12
        
        b = self.param('b', initializers.zeros, (features,), z.real.dtype)
        
        amplitude = jnp.abs(z)
        phase = z / (amplitude + eps)
        gated_amplitude = jnp.maximum(amplitude + b, 0.0)
        
        return gated_amplitude * phase


class ComplexRMSNorm(nn.Module):
    """Complex RMS normalization with learnable per-channel scale."""

    eps: float = 1e-6
    param_dtype: Any = jnp.complex64

    @nn.compact
    def __call__(self, z: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(jnp.abs(z) ** 2, axis=-1, keepdims=True) + self.eps)
        features = z.shape[-1]
        gamma = self.param('gamma', initializers.ones, (features,), self.param_dtype)
        
        return (z / rms) * gamma


class ComplexResidualMLP(nn.Module):
    """Residual MLP with native complex arithmetic."""

    hidden_dims: tuple[int, ...] = (256,)
    param_dtype: Any = jnp.complex64
    kernel_init: Callable = complex_glorot_init()
    bias_init: Callable = initializers.zeros_init()

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Promote real input to complex
        if not jnp.issubdtype(x.dtype, jnp.complexfloating):
            x = x.astype(jnp.float32) + 0j
        
        h = x
        
        for layer_idx, width in enumerate(self.hidden_dims):
            residual = h
            
            # Pre-activation normalization
            h = ComplexRMSNorm(param_dtype=self.param_dtype)(h)
            h = ModReLU(param_dtype=self.param_dtype)(h)
            
            # Linear transform
            h = ComplexDense(
                width,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                name=f"cdense_{layer_idx}",
            )(h)
            
            # Dimension-matching skip connection
            if residual.shape[-1] != width:
                residual = ComplexDense(
                    width,
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    name=f"skip_{layer_idx}",
                )(residual)
            
            h = h + residual
        
        # Final refinement
        h = ComplexRMSNorm(param_dtype=self.param_dtype)(h)
        h = ModReLU(param_dtype=self.param_dtype)(h)
        
        return h


class cBackflowMLP(nn.Module):
    """
    Complex backflow network with native Wirtinger gradients.
    
    Directly learns amplitude-phase corrections through complex MLP.
    Uses Rayleigh-uniform initialization for proper complex statistics.
    
    Attributes:
        n_orb: Number of spatial orbitals
        n_alpha, n_beta: Spin-up/down electron counts
        n_dets: Number of determinants in expansion
        generalized: Use unified spin-orbital matrix
        restricted: Share spatial orbitals between spins
        use_log_coeffs: Learn determinant expansion coefficients
        hidden_dims: Complex MLP hidden dimensions
        param_dtype: Complex type for all parameters
        kernel_init, bias_init: Weight initializers
        logdet_jitter: Diagonal regularization for stability
    """

    n_orb: int
    n_alpha: int
    n_beta: int
    n_dets: int = 1
    generalized: bool = False
    restricted: bool = True
    use_log_coeffs: bool = True
    hidden_dims: tuple[int, ...] = (256,)
    param_dtype: Any = jnp.complex64
    kernel_init: Callable = complex_glorot_init()
    bias_init: Callable = initializers.zeros_init()
    logdet_jitter: float = 0.0

    def _mlp(self, name: str, out_dim: int, s: jnp.ndarray) -> jnp.ndarray:
        """Generate complex correction via residual network."""
        x = ComplexResidualMLP(
            hidden_dims=self.hidden_dims,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name=f"{name}_cmlp",
        )(s)
        
        # Output projection with He scaling
        z = ComplexDense(
            out_dim,
            param_dtype=self.param_dtype,
            kernel_init=complex_he_init(),
            bias_init=self.bias_init,
            name=f"{name}_out",
        )(x)
        
        return z

    def _safe_logdet(self, A: jax.Array) -> jax.Array:
        """Compute log-determinant with optional jitter for numerical stability."""
        if self.logdet_jitter > 0:
            n = A.shape[-1]
            A = A + jnp.eye(n, dtype=A.dtype) * self.logdet_jitter
        return logdet_c(A)

    @nn.compact
    def __call__(self, s: jax.Array) -> jax.Array:
        """
        Compute log-amplitude via complex backflow determinants.
        
        Args:
            s: Binary occupancy [α₁...αₙ β₁...βₙ] shape (2n,)
        
        Returns:
            Complex log-amplitude
        """
        n_e = self.n_alpha + self.n_beta
        is_multi = self.n_dets > 1

        orbitals = self._init_orbitals(is_multi)
        eval_fn = self._build_evaluator(n_e, orbitals, is_multi, s)

        if not is_multi:
            return eval_fn(s)

        return self._combine_dets(eval_fn, s)

    def _init_orbitals(self, is_multi: bool) -> tuple | jnp.ndarray:
        """Initialize base orbital matrices with orthonormal columns."""
        n_e = self.n_alpha + self.n_beta
        det_shape = (self.n_dets,) if is_multi else ()

        if self.generalized:
            shape = (*det_shape, 2 * self.n_orb, n_e)
            return self.param("M_gen", c_orthogonal_init, shape, self.param_dtype)

        if self.restricted:
            k = max(self.n_alpha, self.n_beta)
            shape = (*det_shape, self.n_orb, k)
            return self.param("M_spatial", c_orthogonal_init, shape, self.param_dtype)

        # Unrestricted: separate α/β orbital spaces
        shape_α = (*det_shape, self.n_orb, self.n_alpha)
        shape_β = (*det_shape, self.n_orb, self.n_beta)
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
        """Build function: s -> log det(M + ΔM(s)) with complex arithmetic."""

        def _extract_indices(s: jnp.ndarray):
            """Compute occupied row indices for configuration s."""
            if self.generalized:
                rows = jnp.nonzero(s, size=n_e, fill_value=0)[0]
                return rows, None

            α_occ = s[: self.n_orb]
            β_occ = s[self.n_orb :]
            rows_α = jnp.nonzero(α_occ, size=self.n_alpha, fill_value=0)[0]
            rows_β = jnp.nonzero(β_occ, size=self.n_beta, fill_value=0)[0]
            return rows_α, rows_β

        if not is_multi:
            def eval_single(s: jnp.ndarray) -> jnp.ndarray:
                rows_α, rows_β = _extract_indices(s)

                if self.generalized:
                    out_dim = (2 * self.n_orb) * n_e
                    Δ = self._mlp("BF_gen", out_dim, s).reshape(
                        (2 * self.n_orb, n_e)
                    )
                    M_eff = orbitals + Δ
                    rows = rows_α
                    return self._safe_logdet(M_eff[rows, :])

                if self.restricted:
                    k = max(self.n_alpha, self.n_beta)
                    out_dim = self.n_orb * k
                    Δ = self._mlp("BF_res", out_dim, s).reshape(
                        (self.n_orb, k)
                    )
                    M_eff = orbitals + Δ
                    A_α = M_eff[rows_α, : self.n_alpha]
                    A_β = M_eff[rows_β, : self.n_beta]
                    return self._safe_logdet(A_α) + self._safe_logdet(A_β)

                # Unrestricted
                M_α_base, M_β_base = orbitals
                out_dim = self.n_orb * (self.n_alpha + self.n_beta)
                Δ = self._mlp("BF_unres", out_dim, s).reshape(
                    (self.n_orb, self.n_alpha + self.n_beta)
                )
                Δ_α = Δ[:, : self.n_alpha]
                Δ_β = Δ[:, self.n_alpha :]
                A_α = (M_α_base + Δ_α)[rows_α, :]
                A_β = (M_β_base + Δ_β)[rows_β, :]
                return self._safe_logdet(A_α) + self._safe_logdet(A_β)

            return eval_single

        # Multi-determinant case
        def eval_multi(s: jnp.ndarray) -> jnp.ndarray:
            rows_α, rows_β = _extract_indices(s)

            if self.generalized:
                out_dim_per_det = (2 * self.n_orb) * n_e
                total_dim = self.n_dets * out_dim_per_det
                Δ_flat = self._mlp("BF_gen", total_dim, s)
                Δ = Δ_flat.reshape(self.n_dets, 2 * self.n_orb, n_e)
                M_eff = orbitals + Δ
                rows = rows_α
                A = M_eff[:, rows, :]
                return self._safe_logdet(A)

            if self.restricted:
                k = max(self.n_alpha, self.n_beta)
                out_dim_per_det = self.n_orb * k
                total_dim = self.n_dets * out_dim_per_det
                Δ_flat = self._mlp("BF_res", total_dim, s)
                Δ = Δ_flat.reshape(self.n_dets, self.n_orb, k)
                M_eff = orbitals + Δ
                A_α = M_eff[:, rows_α, : self.n_alpha]
                A_β = M_eff[:, rows_β, : self.n_beta]
                return self._safe_logdet(A_α) + self._safe_logdet(A_β)

            # Unrestricted multi-determinant
            M_α_base, M_β_base = orbitals
            out_dim_per_det = self.n_orb * (self.n_alpha + self.n_beta)
            total_dim = self.n_dets * out_dim_per_det
            Δ_flat = self._mlp("BF_unres", total_dim, s)
            Δ = Δ_flat.reshape(self.n_dets, self.n_orb, self.n_alpha + self.n_beta)
            Δ_α = Δ[:, :, : self.n_alpha]
            Δ_β = Δ[:, :, self.n_alpha :]
            A_α = (M_α_base + Δ_α)[:, rows_α, :]
            A_β = (M_β_base + Δ_β)[:, rows_β, :]
            return self._safe_logdet(A_α) + self._safe_logdet(A_β)

        return eval_multi

    def _combine_dets(self, eval_fn: Callable, s: jnp.ndarray) -> jnp.ndarray:
        """Multi-determinant combination via log-sum-exp."""
        log_dets = eval_fn(s)

        if self.use_log_coeffs:
            log_c = self.param(
                "log_coeffs", nn.initializers.zeros, (self.n_dets,), self.param_dtype
            )
            log_dets = log_dets + log_c

        return logsumexp_c(log_dets, axis=0)


__all__ = ["BackflowMLP", "cBackflowMLP"]
