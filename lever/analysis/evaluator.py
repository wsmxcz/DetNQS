# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Energy evaluation toolkit for active-space calculations.

Provides exact diagonalization and variational energy computation
using Davidson iterative solver and Rayleigh quotient methods.

File: lever/analysis/evaluator.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.sparse as sp
from pyscf import lib

from .. import core, engine
from ..dtypes import COOMatrix, PsiCache

if TYPE_CHECKING:
    from ..dtypes import PyTree


class EnergyEvaluator:
    """
    Energy evaluation toolkit for quantum chemistry calculations.
    
    Supports exact diagonalization via Davidson solver and variational
    energy computation via Rayleigh quotient.
    """
    
    def __init__(self, int_ctx: core.IntCtx, n_orb: int, e_nuc: float):
        """
        Initialize evaluation context.
        
        Args:
            int_ctx: Integral provider for Hamiltonian construction
            n_orb: Number of active orbitals
            e_nuc: Nuclear repulsion energy
        """
        self.int_ctx = int_ctx
        self.n_orb = n_orb
        self.e_nuc = e_nuc

    def diagonalize(self, ham: COOMatrix) -> float:
        """
        Compute ground-state energy via Davidson diagonalization.
        
        Algorithm: Iterative subspace method with diagonal preconditioner
        P(r,e) = r / (H_diag - e + τ·sign(H_diag - e))
        
        Initial guess: [argmin(H_diag), random orthogonal vector]
        
        Args:
            ham: Hamiltonian in COO format
            
        Returns:
            Total ground-state energy E₀ + E_nuc
        """
        # Handle trivial dimensions
        if ham.shape[0] == 0:
            return self.e_nuc
        if ham.shape[0] == 1:
            return float(ham.vals[0]) + self.e_nuc

        # Build CSR matrix for efficient SpMV
        H = sp.coo_matrix(
            (ham.vals, (ham.rows, ham.cols)),
            shape=ham.shape
        ).tocsr()
        H.sum_duplicates()

        diag = H.diagonal()

        # Initial guess: lowest diagonal + orthogonal random
        i_min = int(np.argmin(diag.real))
        x0 = np.zeros(H.shape[0], dtype=diag.dtype)
        x0[i_min] = 1.0
        
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(H.shape[0]).astype(diag.real.dtype)
        x1 -= (x1 @ x0.real) * x0
        x1 /= np.linalg.norm(x1)
        X0 = [x0, x1.astype(x0.dtype, copy=False)]

        # Diagonal preconditioner with safety margin
        tau = 1e-3 * (np.linalg.norm(diag, ord=np.inf) + 1.0)
        
        def precond(r: np.ndarray, e: float, _x: np.ndarray) -> np.ndarray:
            denom = diag - e
            safe_denom = np.where(
                np.abs(denom) > 1e-8,
                denom,
                np.sign(denom) * tau
            )
            return r / safe_denom

        # Davidson iteration
        e_vals, _ = lib.linalg_helper.davidson(
            lambda v: H @ v,
            X0,
            precond,
            nroots=1,
            max_space=50,
            max_cycle=200,
            tol=1e-10
        )
        return float(np.atleast_1d(e_vals)[0]) + self.e_nuc

    def variational_energy(
        self,
        params: PyTree,
        logpsi_fn: Callable,
        ham: COOMatrix,
        dets: np.ndarray
    ) -> float:
        """
        Compute variational energy via Rayleigh quotient.
        
        E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ where ψ = exp(log_psi)
        
        Args:
            params: Neural network parameters
            logpsi_fn: log|ψ(det)| evaluator
            ham: Hamiltonian operator
            dets: Determinant basis (bool masks)
            
        Returns:
            Total variational energy E + E_nuc
        """
        from jax import numpy as jnp
      
        # Evaluate wavefunction amplitudes
        t_vecs = engine.utils.masks_to_vecs(jnp.asarray(dets), self.n_orb)
        log_psi = logpsi_fn(params, t_vecs)
        psi = np.array(jnp.exp(log_psi))
      
        # Compute H|ψ⟩
        h_psi = engine.kernels.coo_matvec(
            ham.rows, ham.cols, ham.vals, psi, len(dets)
        )

        # Rayleigh quotient
        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc

    def variational_energy_from_cache(
        self,
        psi_cache: PsiCache,
        ham: COOMatrix,
        dets: np.ndarray
    ) -> float:
        """
        Compute variational energy from cached amplitudes.
        
        Avoids redundant model forward pass by reusing cached ψ.
        
        Args:
            psi_cache: Cached wavefunction amplitudes
            ham: Hamiltonian operator
            dets: Determinant basis (must match cache order)
            
        Returns:
            Total variational energy E + E_nuc
        """
        psi = np.array(psi_cache.psi_all)
        
        if len(psi) != len(dets):
            raise ValueError(
                f"Cache size {len(psi)} != determinant count {len(dets)}"
            )
        
        # Compute H|ψ⟩
        h_psi = engine.kernels.coo_matvec(
            ham.rows, ham.cols, ham.vals, psi, len(dets)
        )

        # Rayleigh quotient
        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc


__all__ = ["EnergyEvaluator"]
