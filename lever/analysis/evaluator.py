# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Energy evaluation toolkit for active-space quantum chemistry.

Provides exact diagonalization, variational energy computation, and
post-calculation analysis with FCI benchmarking.

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
from ..dtypes import COOMatrix, PsiCache, ComputeMode

if TYPE_CHECKING:
    from ..config import LeverConfig
    from ..dtypes import PyTree, LeverResult


class EnergyEvaluator:
    """
    Energy evaluator with exact diagonalization and variational methods.
    
    Implements Davidson eigensolvers and Rayleigh quotient energy computation.
    """
    
    def __init__(self, int_ctx: core.IntCtx, n_orb: int, e_nuc: float):
        """
        Initialize with integral context and nuclear energy.
        
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
        Exact ground-state energy via diagonalization.
        
        Method selection:
          - Dense (scipy.linalg.eigh): n < 1000 or density > 10%
          - Davidson (pyscf): large sparse matrices
        
        Args:
            ham: Hamiltonian in COO format
            
        Returns:
            Ground-state energy E₀ + E_nuc
        """
        if ham.shape[0] == 0:
            return self.e_nuc
        if ham.shape[0] == 1:
            return float(ham.vals[0]) + self.e_nuc
        
        n = ham.shape[0]
        
        # Convert to CSR sparse matrix
        H_sparse = sp.coo_matrix(
            (ham.vals, (ham.rows, ham.cols)), shape=ham.shape
        ).tocsr()
        H_sparse.sum_duplicates()
        
        density = ham.nnz / (n * n)
        use_dense = (n < 1000) or (density > 0.1)
        
        if use_dense:
            from scipy.linalg import eigh
            H_dense = H_sparse.toarray()
            e_vals, _ = eigh(H_dense, lower=True)
            return float(e_vals[0]) + self.e_nuc
        
        # Davidson solver with diagonal preconditioner
        diag = H_sparse.diagonal()
        
        # Initial guess: ground state + orthogonal random vector
        i_min = int(np.argmin(diag.real))
        x0 = np.zeros(n, dtype=diag.dtype)
        x0[i_min] = 1.0
        
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(n).astype(diag.real.dtype)
        x1 -= (x1 @ x0.real) * x0
        x1 /= np.linalg.norm(x1)
        X0 = [x0, x1.astype(x0.dtype, copy=False)]
        
        # Preconditioner: P(r) = r / (diag - E)
        tau = 1e-3 * (np.linalg.norm(diag, ord=np.inf) + 1.0)
        
        def precond(r: np.ndarray, e: float, _x: np.ndarray) -> np.ndarray:
            denom = diag - e
            safe_denom = np.where(
                np.abs(denom) > 1e-8, denom, np.sign(denom) * tau
            )
            return r / safe_denom
        
        e_vals, _ = lib.linalg_helper.davidson(
            lambda v: H_sparse @ v, X0, precond,
            nroots=1, max_space=50, max_cycle=200, tol=1e-10
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
        Variational energy via Rayleigh quotient: E = <ψ|H|ψ> / <ψ|ψ>.
        
        Args:
            params: Neural network parameters
            logpsi_fn: Log-amplitude function (params, features) → log|ψ>
            ham: Hamiltonian operator
            dets: Determinant basis (boolean occupation masks)
            
        Returns:
            Variational energy E + E_nuc
        """
        from jax import numpy as jnp
        from ..utils.features import dets_to_features
      
        # Compute wavefunction amplitudes
        features = dets_to_features(dets, self.n_orb)
        log_psi = logpsi_fn(params, features)
        psi = np.array(jnp.exp(log_psi))
      
        # H|ψ> via sparse matrix-vector product
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
        Compute energy from cached wavefunction amplitudes.
        
        Args:
            psi_cache: Cached amplitudes |ψ>
            ham: Hamiltonian operator
            dets: Determinant basis (must match cache)
            
        Returns:
            Variational energy E + E_nuc
        """
        psi = np.array(psi_cache.psi_all)
        
        if len(psi) != len(dets):
            raise ValueError(
                f"Cache size {len(psi)} != basis size {len(dets)}"
            )
        
        h_psi = engine.kernels.coo_matvec(
            ham.rows, ham.cols, ham.vals, psi, len(dets)
        )
        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc

    def compute_fci_energy(self, config: LeverConfig) -> float:
        """
        Full CI energy via complete basis diagonalization.
        
        Args:
            config: System configuration
            
        Returns:
            FCI ground-state energy (includes E_nuc)
        """
        sys = config.system
        
        # Generate complete FCI basis
        fci_dets = core.gen_fci_dets(sys.n_orbitals, sys.n_alpha, sys.n_beta)
        
        # Build and diagonalize full Hamiltonian
        ham_fci, _ = engine.hamiltonian.get_ham_ss(
            S_dets=fci_dets,
            int_ctx=self.int_ctx,
            n_orbitals=sys.n_orbitals
        )
        
        return self.diagonalize(ham_fci)

    def analyze_result(
        self,
        result: LeverResult,
        model,
        *,
        compute_fci: bool = False,
        compute_var: bool = False,
        compute_s_ci: bool = False
    ) -> dict[str, float]:
        """
        Post-convergence energy diagnostics.
        
        Args:
            result: Converged optimization result
            model: Compiled wavefunction model
            compute_fci: Include FCI benchmark (expensive)
            compute_var: Include full T-space variational energy
            compute_s_ci: Include S-space CI energy
            
        Returns:
            Energy dictionary with requested components
        """
        energies = {'e_lever': result.full_energy_history[-1]}
        
        if compute_fci:
            energies['e_fci'] = self.compute_fci_energy(result.config)
        
        if compute_var or compute_s_ci:
            ctx, psi = self._rebuild_final_context(result, model)
            
            if compute_var:
                energies['e_var'] = self._compute_var_energy(ctx, psi, result)
            
            if compute_s_ci:
                energies['e_s_ci'] = self.diagonalize(ctx.ham_ss)
        
        return energies

    def _rebuild_final_context(self, result: LeverResult, model):
        """Reconstruct final optimization state for diagnostics."""
        from ..workflow.compiler import Compiler
        from ..dtypes import OuterState
        from jax import numpy as jnp
        
        compiler = Compiler(
            config=result.config,
            model=model,
            int_ctx=self.int_ctx
        )
        
        # Extract electronic energy for EFFECTIVE mode reference
        e_ref = None
        if result.config.compute_mode == ComputeMode.EFFECTIVE:
            final_energy = result.full_energy_history[-1]
            e_ref = final_energy - self.e_nuc
        
        final_state = OuterState(
            cycle=len(result.cycle_boundaries) - 1,
            s_dets=result.final_s_dets,
            params=result.final_params,
            e_ref=e_ref
        )
        
        ctx, _ = compiler.compile(final_state)
        
        # Evaluate wavefunction amplitudes
        logpsi_fn = ctx.log_psi_fn[1] if isinstance(ctx.log_psi_fn, tuple) else ctx.log_psi_fn
        log_psi = logpsi_fn(result.final_params, ctx.features_s, ctx.features_c)
        psi = np.array(jnp.exp(log_psi))
        
        return ctx, psi

    def _compute_var_energy(self, ctx, psi: np.ndarray, result: LeverResult) -> float:
        """Compute variational energy over full T-space."""
        # Build full T-space Hamiltonian
        t_dets = np.concatenate([ctx.space.s_dets, ctx.space.c_dets])
        ham_full, _ = engine.hamiltonian.get_ham_ss(
            S_dets=t_dets,
            int_ctx=self.int_ctx,
            n_orbitals=result.config.system.n_orbitals
        )
        
        # Rayleigh quotient: E = <ψ|H|ψ> / <ψ|ψ>
        h_psi = engine.kernels.coo_matvec(
            ham_full.rows, ham_full.cols, ham_full.vals, psi, len(t_dets)
        )
        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc


__all__ = ["EnergyEvaluator"]
