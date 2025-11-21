# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Exact and deterministic energy evaluation.

Implements:
1. Exact Diagonalization (FCI/CASCI via Davidson/Lanczos)
2. Deterministic Variational Energy (sum over fixed determinant list)

File: lever/analysis/exact.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
from pyscf import lib

from .. import core, engine
from ..dtypes import COOMatrix, SpaceRep, PsiCache

if TYPE_CHECKING:
    from ..config import LeverConfig
    from ..dtypes import LeverResult


class VariationalEvaluator:
    """
    Fixed-basis quantum chemistry evaluator.
    
    Provides exact diagonalization and deterministic variational energy
    computations without explicit Hamiltonian construction.
    """
    
    def __init__(self, int_ctx: core.IntCtx, n_orb: int, e_nuc: float):
        self.int_ctx = int_ctx
        self.n_orb = n_orb
        self.e_nuc = e_nuc

    def diagonalize(self, ham: COOMatrix) -> float:
        """
        Compute exact ground-state energy E₀ = min eig(H).
        
        Algorithm: Dense solver for small/dense matrices, Davidson for large sparse.
        """
        if ham.shape[0] == 0:
            return self.e_nuc
        if ham.shape[0] == 1:
            return float(ham.vals[0]) + self.e_nuc
        
        n = ham.shape[0]
        
        # Convert to CSR format for efficient linear algebra
        H_sparse = sp.coo_matrix(
            (ham.vals, (ham.rows, ham.cols)), shape=ham.shape
        ).tocsr()
        H_sparse.sum_duplicates()
        
        # Heuristic selection: dense solver for small/dense matrices
        density = ham.nnz / (n * n) if n > 0 else 0
        use_dense = (n < 2000) or (density > 0.1)
        
        if use_dense:
            from scipy.linalg import eigh
            H_dense = H_sparse.toarray()
            e_vals, _ = eigh(H_dense, subset_by_index=[0, 0])
            return float(e_vals[0]) + self.e_nuc
        
        return self._run_davidson(H_sparse)

    def _run_davidson(self, H_sparse: sp.csr_matrix) -> float:
        """Davidson diagonalization for sparse matrices with diagonal preconditioner."""
        n = H_sparse.shape[0]
        diag = H_sparse.diagonal()
        
        # Initial guess: lowest diagonal element
        i_min = int(np.argmin(diag.real))
        x0 = np.zeros(n, dtype=diag.dtype)
        x0[i_min] = 1.0
        
        def precond(r, e, x0):
            return r / (diag - e + 1e-12)

        e_vals, _ = lib.linalg_helper.davidson(
            lambda v: H_sparse @ v,
            x0,
            precond,
            nroots=1,
            max_cycle=100,
            tol=1e-8,
            verbose=0
        )
        return float(np.atleast_1d(e_vals)[0]) + self.e_nuc

    def compute_fci_energy(self, config: LeverConfig) -> float:
        """Compute Full CI energy (exponential scaling, small systems only)."""
        sys = config.system
        fci_dets = core.gen_fci_dets(sys.n_orbitals, sys.n_alpha, sys.n_beta)
        
        ham_fci, _ = engine.hamiltonian.get_ham_ss(
            S_dets=fci_dets,
            int_ctx=self.int_ctx,
            n_orbitals=sys.n_orbitals
        )
        return self.diagonalize(ham_fci)

    def compute_sc_variational_energy(
        self,
        result: LeverResult,
        *,
        use_heatbath: bool = False,
        eps1: float = 1e-6
    ) -> float:
        """
        Compute variational energy E = ⟨Ψ|H|Ψ⟩/⟨Ψ|Ψ⟩ on S∪C space.
        
        Uses C++ streaming kernel to evaluate expectation value without
        constructing full Hamiltonian matrix.
        
        Args:
            result: LeverResult with final_space and final_psi_cache
            use_heatbath: Enable Heat-Bath screening for double excitations
            eps1: Threshold for screening matrix elements
            
        Returns:
            Total variational energy (including nuclear repulsion)
        """
        if result.final_space is None or result.final_psi_cache is None:
            raise ValueError(
                "LeverResult missing final space/psi_cache. "
                "Ensure Driver.run() captures final state."
            )

        space: SpaceRep = result.final_space
        cache: PsiCache = result.final_psi_cache

        # Construct T = S ∪ C determinant list
        if space.c_dets.shape[0] == 0:
            dets_T = space.s_dets
        else:
            dets_T = np.concatenate([space.s_dets, space.c_dets], axis=0)

        # Convert coefficients to NumPy complex128
        coeffs = np.array(cache.psi_all, dtype=np.complex128)

        if coeffs.shape[0] != dets_T.shape[0]:
            raise ValueError(
                f"Shape mismatch: coeffs {coeffs.shape} vs dets {dets_T.shape}"
            )

        # C++ streaming kernel for efficient energy evaluation
        e_el, norm = core.compute_variational_energy(
            dets_T,
            coeffs,
            self.int_ctx,
            self.n_orb,
            use_heatbath,
            eps1
        )

        if norm <= 1e-14:
            raise RuntimeError("Wavefunction norm is effectively zero.")

        return e_el / norm + self.e_nuc

    def analyze_result(
        self,
        result: LeverResult,
        model: Any,
        *,
        compute_fci: bool = False,
        compute_s_ci: bool = False,
        compute_sc_var: bool = True
    ) -> dict[str, float]:
        """Perform post-calculation analysis on final result."""
        energies = {'e_lever': result.full_energy_history[-1]}
        
        # Exact variational energy on S ∪ C
        if compute_sc_var and result.final_space is not None:
            energies['e_var'] = self.compute_sc_variational_energy(result)
        
        # S-space CI (Davidson diagonalization)
        if compute_s_ci:
            ham_s, _ = engine.hamiltonian.get_ham_ss(
                S_dets=result.final_s_dets,
                int_ctx=self.int_ctx,
                n_orbitals=result.config.system.n_orbitals
            )
            energies['e_s_ci'] = self.diagonalize(ham_s)
            
        # Full CI benchmark (expensive)
        if compute_fci:
            energies['e_fci'] = self.compute_fci_energy(result.config)
            
        return energies
