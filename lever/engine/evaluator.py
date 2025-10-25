# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lazy evaluator for LEVER wavefunction states with atomic result caching.

Caches fundamental computational results (features, amplitudes, Hamiltonian
contractions) while delegating higher-level quantities to pure functions.

File: lever/engine/evaluator.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import kernels
from .config import DEFAULT_CONFIG
from .utils import Contractions, masks_to_vecs

if TYPE_CHECKING:
    from .config import EngineConfig
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
    Compute H_SS@ψ_S, H_SC@ψ_C, H_CS@ψ_S via Numba kernels.
  
    Handles both ASYMMETRIC/PROXY modes (ham_sc provided) and EFFECTIVE mode
    (ham_sc=None, returns zero for N_SC and N_CS).
  
    Returns:
        (N_SS, N_SC, N_CS): Hamiltonian-wavefunction products
    """
    # Sanitize arrays for Numba compatibility
    ψ_S = np.asarray(ψ_S_np, dtype=np.complex128)
    ψ_C = np.asarray(ψ_C_np, dtype=np.complex128)

    rows_ss = np.asarray(ham_ss_rows, dtype=np.int64)
    cols_ss = np.asarray(ham_ss_cols, dtype=np.int64)
    vals_ss = np.asarray(ham_ss_vals, dtype=np.float64)

    size_S_int = int(size_S)
    size_C_int = int(size_C)

    # Always compute H_SS@ψ_S
    N_SS = kernels.coo_matvec(rows_ss, cols_ss, vals_ss, ψ_S, size_S_int)

    # Conditional H_SC and H_CS computation
    if ham_sc_rows is not None and ham_sc_cols is not None and ham_sc_vals is not None:
        rows_sc = np.asarray(ham_sc_rows, dtype=np.int64)
        cols_sc = np.asarray(ham_sc_cols, dtype=np.int64)
        vals_sc = np.asarray(ham_sc_vals, dtype=np.float64)
        N_SC, N_CS = kernels.coo_dual_contract(
            rows_sc, cols_sc, vals_sc, ψ_S, ψ_C, size_S_int, size_C_int
        )
    else:
        # EFFECTIVE mode: no S-C coupling
        N_SC = np.zeros(size_S_int, dtype=np.complex128)
        N_CS = np.zeros(size_C_int, dtype=np.complex128)

    return N_SS, N_SC, N_CS


class Evaluator:
    """
    Lazy evaluation context with atomic result caching.
  
    Provides consistent, cached view of wavefunction and contractions for
    single parameter set. Physical computations delegated to pure functions.
    """

    def __init__(
        self,
        params: PyTree,
        logpsi_fn: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
        space: SpaceRep,
        n_orbitals: int,
        ham_ss: HamOp,
        ham_sc: HamOp | None = None,
        config: EngineConfig = DEFAULT_CONFIG,
    ) -> None:
        """
        Initialize evaluator for single optimization step.

        Args:
            params: PyTree of model parameters
            logpsi_fn: Function computing log-amplitudes log(ψ)
            space: Determinant space representation (host memory)
            n_orbitals: Number of spatial orbitals
            ham_ss: S-space Hamiltonian H_SS or effective H_eff (host memory)
            ham_sc: S-C coupling Hamiltonian H_SC (None for EFFECTIVE mode)
            config: Engine computation configuration
        """
        self.params = params
        self.logpsi_fn = logpsi_fn
        self.space = space
        self.n_orbitals = n_orbitals
        self.ham_ss = ham_ss
        self.ham_sc = ham_sc
        self.config = config

    # ========================================================================
    # Level 0: Input Features
    # ========================================================================

    @functools.cached_property
    def all_features(self) -> jnp.ndarray:
        """Concatenated neural network features for all S and C determinants."""
        s_dets_dev = jnp.asarray(self.space.s_dets)
        c_dets_dev = jnp.asarray(self.space.c_dets)
        s_vecs = masks_to_vecs(s_dets_dev, self.n_orbitals)
        c_vecs = masks_to_vecs(c_dets_dev, self.n_orbitals)
        return jnp.concatenate([s_vecs, c_vecs], axis=0)

    # ========================================================================
    # Level 1: Wavefunction Amplitudes
    # ========================================================================

    @functools.cached_property
    def logpsi_all(self) -> jnp.ndarray:
        """
        Normalized log-amplitudes log(ψ) for all determinants.
      
        Applies joint L2 normalization: ||ψ_S||² + ||ψ_C||² = 1.
        Gradient flow through norm stopped to preserve variational derivatives.
        """
        log_all = self.logpsi_fn(self.params, self.all_features)

        if self.config.normalize_wf:
            n_S = self.space.size_S
            log_S, log_C = log_all[:n_S], log_all[n_S:]

            # Joint norm in log-space: log(||ψ_S||² + ||ψ_C||²)
            log_norm_sq = jnp.logaddexp(
                jax.scipy.special.logsumexp(2.0 * jnp.real(log_S)),
                jax.scipy.special.logsumexp(2.0 * jnp.real(log_C)),
            )
            # Stop gradient to avoid affecting variational derivatives
            log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
            log_all -= 0.5 * log_norm_sq

        return log_all

    @functools.cached_property
    def wavefunction(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Wavefunction amplitudes (ψ_S, ψ_C) as JAX arrays."""
        n_S = self.space.size_S
        ψ_all = jnp.exp(self.logpsi_all)
        return ψ_all[:n_S], ψ_all[n_S:]

    def get_batch_logpsi_fn(self) -> Callable[[PyTree], jnp.ndarray]:
        """
        Pure function computing normalized log(ψ) for all determinants.
      
        Captures static data via closure for cross-module consistency.
        """
        logpsi_fn = self.logpsi_fn
        all_features = self.all_features
        n_S = self.space.size_S
        config = self.config

        def _batch_logpsi(params: PyTree) -> jnp.ndarray:
            log_all = logpsi_fn(params, all_features)
            if config.normalize_wf:
                log_S, log_C = log_all[:n_S], log_all[n_S:]
                log_norm_sq = jnp.logaddexp(
                    jax.scipy.special.logsumexp(2.0 * jnp.real(log_S)),
                    jax.scipy.special.logsumexp(2.0 * jnp.real(log_C)),
                )
                log_norm_sq = jax.lax.stop_gradient(log_norm_sq)
                log_all -= 0.5 * log_norm_sq
            return log_all

        return _batch_logpsi

    # ========================================================================
    # Level 2: Hamiltonian Contractions (JAX-Numba Boundary)
    # ========================================================================

    @functools.cached_property
    def contractions(self) -> Contractions:
        """
        All Hψ products: (H_SS@ψ_S, H_SC@ψ_C, H_CS@ψ_S, H_CC_diag⊙ψ_C).
      
        Modes:
          • ASYMMETRIC/PROXY: Compute H_SC and H_CS via Numba
          • EFFECTIVE: Skip H_SC/H_CS (returns zeros), H_SS is actually H_eff
      
        C-space always uses diagonal approximation: N_CC = H_diag_C ⊙ ψ_C
        """
        ψ_S, ψ_C = self.wavefunction
        complex_dtype = ψ_S.dtype

        # Define output shapes for three contractions from Numba
        result_shapes = (
            jax.ShapeDtypeStruct((self.space.size_S,), complex_dtype),  # N_SS
            jax.ShapeDtypeStruct((self.space.size_S,), complex_dtype),  # N_SC
            jax.ShapeDtypeStruct((self.space.size_C,), complex_dtype),  # N_CS
        )

        # Prepare ham_sc arguments (None for EFFECTIVE mode)
        if self.ham_sc is not None:
            ham_sc_args = (self.ham_sc.rows, self.ham_sc.cols, self.ham_sc.vals)
        else:
            ham_sc_args = (None, None, None)

        # Offload to Numba kernels
        N_SS, N_SC, N_CS = jax.pure_callback(
            _compute_contractions_numpy,
            result_shapes,
            ψ_S,
            ψ_C,
            self.ham_ss.rows,
            self.ham_ss.cols,
            self.ham_ss.vals,
            *ham_sc_args,
            self.space.size_S,
            self.space.size_C,
        )

        # C-space diagonal approximation (PROXY mode)
        H_diag_C = jnp.asarray(self.space.H_diag_C)
        N_CC = H_diag_C * ψ_C

        return Contractions(N_SS=N_SS, N_SC=N_SC, N_CS=N_CS, N_CC=N_CC)


__all__ = ["Evaluator"]
