# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lazy evaluator with domain-aware caching for LEVER wavefunction states.

Implements cached evaluation chain:
  Features → log(ψ) → ψ → H·ψ contractions

Supports three compute modes:
  - EFFECTIVE: S-space only (H_eff = H_SS + H_SC·D⁻¹·H_CS)
  - ASYMMETRIC/PROXY: Full T-space (S ∪ C)

File: lever/engine/evaluator.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import kernels
from .utils import Contractions, masks_to_vecs
from ..config import ComputeMode

if TYPE_CHECKING:
    from ..config import LeverConfig
    from .utils import HamOp, PyTree, SpaceRep


def _compute_contractions_numpy(
    ψ_S_np: np.ndarray,
    ψ_C_np: np.ndarray,
    ham_ss_rows: np.ndarray,
    ham_ss_cols: np.ndarray,
    ham_ss_vals: np.ndarray,
    ham_sc_rows: np.ndarray | None,
    ham_sc_cols: np.ndarray | None,
    ham_sc_vals: np.ndarray | None,
    size_S: np.ndarray | int,
    size_C: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sparse matrix-vector products via Numba kernels:
      N_SS = H_SS @ ψ_S
      N_SC = H_SC @ ψ_C  (if H_SC exists)
      N_CS = H_CS @ ψ_S  (if H_CS exists, H_CS = H_SC^T)

    Returns zeros for N_SC/N_CS when ham_sc=None (EFFECTIVE mode).
    """
    # Type conversion for Numba
    ψ_S = np.asarray(ψ_S_np, dtype=np.complex128)
    ψ_C = np.asarray(ψ_C_np, dtype=np.complex128)
    rows_ss = np.asarray(ham_ss_rows, dtype=np.int64)
    cols_ss = np.asarray(ham_ss_cols, dtype=np.int64)
    vals_ss = np.asarray(ham_ss_vals, dtype=np.float64)
    size_S_int = int(size_S)
    size_C_int = int(size_C)

    # H_SS @ ψ_S (always required)
    N_SS = kernels.coo_matvec(rows_ss, cols_ss, vals_ss, ψ_S, size_S_int)

    # H_SC @ ψ_C and H_CS @ ψ_S (skip for EFFECTIVE)
    if ham_sc_rows is not None and ham_sc_cols is not None and ham_sc_vals is not None:
        rows_sc = np.asarray(ham_sc_rows, dtype=np.int64)
        cols_sc = np.asarray(ham_sc_cols, dtype=np.int64)
        vals_sc = np.asarray(ham_sc_vals, dtype=np.float64)
        N_SC, N_CS = kernels.coo_dual_contract(
            rows_sc, cols_sc, vals_sc, ψ_S, ψ_C, size_S_int, size_C_int
        )
    else:
        N_SC = np.zeros(size_S_int, dtype=np.complex128)
        N_CS = np.zeros(size_C_int, dtype=np.complex128)

    return N_SS, N_SC, N_CS


def _compute_contractions_s_only(
    ψ_S_np: np.ndarray,
    ham_rows: np.ndarray,
    ham_cols: np.ndarray,
    ham_vals: np.ndarray,
    size_S: np.ndarray | int,
) -> np.ndarray:
    """
    S-only sparse matrix-vector product: N_SS = H_eff @ ψ_S.
    
    Dedicated callback for EFFECTIVE mode to avoid C-space overhead.
    """
    ψ_S = np.asarray(ψ_S_np, dtype=np.complex128)
    rows = np.asarray(ham_rows, dtype=np.int64)
    cols = np.asarray(ham_cols, dtype=np.int64)
    vals = np.asarray(ham_vals, dtype=np.float64)
    size_S_int = int(size_S)
    
    N_SS = kernels.coo_matvec(rows, cols, vals, ψ_S, size_S_int)
    return N_SS


class Evaluator:
    """
    Lazy evaluation context with domain-aware computation and caching.

    Evaluation chain (cached via @functools.cached_property):
      1. features_{S,C} ← occupancy vectors
      2. log(ψ) ← neural network(features)
      3. ψ ← exp(normalized log(ψ))
      4. contractions ← sparse H·ψ products

    Domain logic:
      - EFFECTIVE: S-only evaluation (||ψ_S||² = 1), unless force_full_space=True
      - ASYMMETRIC/PROXY: Full T-space (||ψ_S||² + ||ψ_C||² = 1)
    """

    def __init__(
        self,
        params: PyTree,
        logpsi_fn: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
        space: SpaceRep,
        n_orbitals: int,
        ham_ss: HamOp,
        ham_sc: HamOp | None,
        config: LeverConfig,
        features_S: jnp.ndarray | None = None,
        features_C: jnp.ndarray | None = None,
        force_full_space: bool = False,
    ) -> None:
        """
        Initialize evaluator for single optimization step.

        Args:
            params: Neural network parameters
            logpsi_fn: Network function: (params, features) → log(ψ)
            space: Determinant space representation
            n_orbitals: Number of spatial orbitals
            ham_ss: ⟨S|H|S⟩ block (COO format)
            ham_sc: ⟨S|H|C⟩ block (None for EFFECTIVE)
            config: Global configuration
            features_S/C: Pre-computed occupancy vectors (optional)
            force_full_space: Override for T-space evolution in EFFECTIVE mode
        """
        self.params = params
        self.logpsi_fn = logpsi_fn
        self.space = space
        self.n_orbitals = n_orbitals
        self.ham_ss = ham_ss
        self.ham_sc = ham_sc
        self.config = config
        self._features_S = features_S
        self._features_C = features_C

        # Domain-aware mode: EFFECTIVE → S-only, unless overridden
        self._is_effective = (
            config.compute_mode == ComputeMode.EFFECTIVE and not force_full_space
        )
        
        # ✅ Solution B: Create SpMV closures with captured COO data
        self._spmv_fn = self._make_spmv_closure()

    def _make_spmv_closure(self) -> Callable:
        """
        Create SpMV closure with statically captured COO matrices.
        
        Reduces pure_callback overhead by capturing sparse data in closure,
        only passing wavefunction arrays as runtime arguments.
        """
        if self.ham_sc is None:
            # S-only mode (EFFECTIVE)
            rows = self.ham_ss.rows
            cols = self.ham_ss.cols
            vals = self.ham_ss.vals
            size_S = self.space.size_S
            result_shape = jax.ShapeDtypeStruct((size_S,), jnp.complex128)
            
            def spmv_s_only(ψ_S: jnp.ndarray) -> Contractions:
                """SpMV for EFFECTIVE mode: N_SS = H_eff @ ψ_S"""
                N_SS = jax.pure_callback(
                    _compute_contractions_s_only,
                    result_shape,
                    ψ_S, rows, cols, vals, size_S,
                )
                return Contractions(N_SS=N_SS, N_SC=None, N_CS=None, N_CC=None)
            
            return spmv_s_only
        
        else:
            # Full-space mode (PROXY/ASYMMETRIC)
            rows_ss = self.ham_ss.rows
            cols_ss = self.ham_ss.cols
            vals_ss = self.ham_ss.vals
            rows_sc = self.ham_sc.rows
            cols_sc = self.ham_sc.cols
            vals_sc = self.ham_sc.vals
            size_S = self.space.size_S
            size_C = self.space.size_C
            H_diag_C = jnp.asarray(self.space.H_diag_C)
            
            result_shapes = (
                jax.ShapeDtypeStruct((size_S,), jnp.complex128),  # N_SS
                jax.ShapeDtypeStruct((size_S,), jnp.complex128),  # N_SC
                jax.ShapeDtypeStruct((size_C,), jnp.complex128),  # N_CS
            )
            
            def spmv_full(ψ_S: jnp.ndarray, ψ_C: jnp.ndarray) -> Contractions:
                """SpMV for full-space: N_SS, N_SC, N_CS, N_CC"""
                N_SS, N_SC, N_CS = jax.pure_callback(
                    _compute_contractions_numpy,
                    result_shapes,
                    ψ_S, ψ_C,
                    rows_ss, cols_ss, vals_ss,
                    rows_sc, cols_sc, vals_sc,
                    size_S, size_C,
                )
                N_CC = H_diag_C * ψ_C
                return Contractions(N_SS=N_SS, N_SC=N_SC, N_CS=N_CS, N_CC=N_CC)
            
            return spmv_full

    # ========================================================================
    # Level 0: Neural network input features
    # ========================================================================

    @functools.cached_property
    def features_S(self) -> jnp.ndarray:
        """S-space occupancy vectors: [N_S, 2*n_orb] (spin-up || spin-down)."""
        if self._features_S is not None:
            return self._features_S
        s_dets_dev = jnp.asarray(self.space.s_dets)
        return masks_to_vecs(s_dets_dev, self.n_orbitals)

    @functools.cached_property
    def features_C(self) -> jnp.ndarray:
        """C-space occupancy vectors (empty array for EFFECTIVE)."""
        if self._is_effective:
            return jnp.empty((0, 2 * self.n_orbitals), dtype=jnp.float32)
        if self._features_C is not None:
            return self._features_C
        c_dets_dev = jnp.asarray(self.space.c_dets)
        return masks_to_vecs(c_dets_dev, self.n_orbitals)

    @functools.cached_property
    def all_features(self) -> jnp.ndarray:
        """
        Concatenated features for batch evaluation.

        EFFECTIVE: S-only [N_S, 2*n_orb]
        Others: S ∥ C [N_S+N_C, 2*n_orb]
        """
        if self._is_effective:
            return self.features_S
        return jnp.concatenate([self.features_S, self.features_C], axis=0)

    # ========================================================================
    # Level 1: Wavefunction amplitudes
    # ========================================================================

    @functools.cached_property
    def logpsi_all(self) -> jnp.ndarray:
        """
        Normalized log-amplitudes with gradient-free normalization.

        Normalization:
          EFFECTIVE: log(ψ_S) - 0.5·log(Σ|ψ_S|²)
          Others:    log(ψ) - 0.5·log(Σ|ψ_S|² + Σ|ψ_C|²)

        Ensures ||ψ|| = 1 without backprop through normalization constant.
        """
        log_all = self.logpsi_fn(self.params, self.all_features)

        if not self.config.normalize_wf:
            return log_all

        if self._is_effective:
            # S-only normalization: ||ψ_S||² = 1
            log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all))
            log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
            return log_all - 0.5 * log_norm_sq
        else:
            # Joint normalization: ||ψ_S||² + ||ψ_C||² = 1
            n_S = self.space.size_S
            log_S, log_C = log_all[:n_S], log_all[n_S:]
            log_norm_sq = jnp.logaddexp(
                jax.scipy.special.logsumexp(2.0 * jnp.real(log_S)),
                jax.scipy.special.logsumexp(2.0 * jnp.real(log_C)),
            )
            log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
            return log_all - 0.5 * log_norm_sq

    @functools.cached_property
    def wavefunction(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Exponential amplitudes: ψ = exp(log(ψ)).

        Returns:
            (ψ_S, ψ_C) where ψ_C is empty for EFFECTIVE mode
        """
        ψ_all = jnp.exp(self.logpsi_all)
        if self._is_effective:
            return ψ_all, jnp.empty(0, dtype=ψ_all.dtype)
        n_S = self.space.size_S
        return ψ_all[:n_S], ψ_all[n_S:]

    def get_batch_logpsi_fn(self) -> Callable[[PyTree], jnp.ndarray]:
        """
        Pure function for batch evaluation: params → normalized log(ψ).

        Closure captures features and normalization logic, enabling
        efficient parameter sweeps without re-initializing Evaluator.
        """
        logpsi_fn = self.logpsi_fn
        all_features = self.all_features
        normalize = self.config.normalize_wf
        is_effective = self._is_effective
        n_S = self.space.size_S

        def _batch_logpsi(params: PyTree) -> jnp.ndarray:
            log_all = logpsi_fn(params, all_features)
            if not normalize:
                return log_all

            if is_effective:
                log_norm_sq = jax.scipy.special.logsumexp(2.0 * jnp.real(log_all))
                log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
                return log_all - 0.5 * log_norm_sq
            else:
                log_S, log_C = log_all[:n_S], log_all[n_S:]
                log_norm_sq = jnp.logaddexp(
                    jax.scipy.special.logsumexp(2.0 * jnp.real(log_S)),
                    jax.scipy.special.logsumexp(2.0 * jnp.real(log_C)),
                )
                log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
                return log_all - 0.5 * log_norm_sq

        return _batch_logpsi

    # ========================================================================
    # Level 2: Hamiltonian contractions
    # ========================================================================

    def compute_contractions_from_psi(
        self, 
        ψ_S: jnp.ndarray, 
        ψ_C: jnp.ndarray | None = None
    ) -> Contractions:
        """
        ✅ Solution A: Compute contractions from externally provided ψ.
        
        Avoids redundant network forward pass by accepting pre-computed
        wavefunction amplitudes (e.g., from VJP forward pass).
        
        Args:
            ψ_S: S-space wavefunction amplitudes
            ψ_C: C-space amplitudes (required for PROXY/ASYMMETRIC)
        
        Returns:
            Contractions: Hamiltonian-wavefunction products
        """
        if self._is_effective:
            # S-only mode: delegate to closure
            return self._spmv_fn(ψ_S)
        else:
            # Full-space mode: require ψ_C
            if ψ_C is None:
                raise ValueError("ψ_C required for full-space contractions")
            return self._spmv_fn(ψ_S, ψ_C)

    @functools.cached_property
    def contractions(self) -> Contractions:
        """
        Cached contractions (compatibility interface).
        
        Note: Optimization paths should use compute_contractions_from_psi()
        to avoid redundant network forward passes. This property retained
        for non-optimization uses (e.g., analysis, evolution).
        """
        ψ_S, ψ_C = self.wavefunction
        return self.compute_contractions_from_psi(ψ_S, ψ_C)


__all__ = ["Evaluator"]
