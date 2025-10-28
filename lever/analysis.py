# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing and visualization tools for LEVER calculations.

Provides energy evaluation via Davidson diagonalization and convergence
analysis with comparison to FCI reference.

File: lever/analysis.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from pyscf import lib

from . import core, engine
from .utils.dtypes import HamOp, PsiCache

if TYPE_CHECKING:
    from .driver import DriverResults


class EvalSuite:
    """
    Energy evaluation toolkit for active-space calculations.
    
    Supports exact diagonalization and variational energy computation.
    """
    
    def __init__(self, int_ctx: core.IntCtx, n_orb: int, e_nuc: float):
        """
        Initialize evaluation context.
        
        Args:
            int_ctx: Integral provider
            n_orb: Number of active orbitals
            e_nuc: Nuclear repulsion energy
        """
        self.int_ctx = int_ctx
        self.n_orb = n_orb
        self.e_nuc = e_nuc

    def diag_ham(self, ham_op: HamOp) -> float:
        """
        Compute ground-state energy via Davidson diagonalization.
        
        Algorithm: Iterative subspace method with preconditioner
        P(r,e) = r / (H_diag - e + τ·sign(H_diag - e)), τ ~ 1e-3·||H_diag||_∞
        
        Initial guesses: [argmin(H_diag), random orthogonal vector]
        
        Args:
            ham_op: Hamiltonian in COO format
            
        Returns:
            Total ground-state energy E₀ + E_nuc
        """
        # Handle trivial dimensions
        if ham_op.shape[0] == 0:
            return self.e_nuc
        if ham_op.shape[0] == 1:
            return float(ham_op.vals[0]) + self.e_nuc

        # Build CSR matrix for efficient SpMV
        H = sp.coo_matrix(
            (ham_op.vals, (ham_op.rows, ham_op.cols)),
            shape=ham_op.shape
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

        # Preconditioner: P(r, e) = r / (diag - e + τ·sign(diag - e))
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

    def var_energy(
        self,
        params,
        logpsi_fn: Callable,
        ham_op: HamOp,
        dets: np.ndarray
    ) -> float:
        """
        Compute variational energy via Rayleigh quotient.
        
        E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩ where ψ = exp(log_psi)
        
        Args:
            params: Neural network parameters
            logpsi_fn: log|ψ(det)| evaluator
            ham_op: Hamiltonian operator
            dets: Determinant basis (bool masks)
            
        Returns:
            Total variational energy
        """
        from jax import numpy as jnp
      
        # Evaluate wavefunction amplitudes
        t_vecs = engine.utils.masks_to_vecs(jnp.asarray(dets), self.n_orb)
        log_psi = logpsi_fn(params, t_vecs)
        psi = np.array(jnp.exp(log_psi))
      
        # Compute H|ψ⟩
        h_psi = engine.kernels.coo_matvec(
            ham_op.rows, ham_op.cols, ham_op.vals, psi, len(dets)
        )

        # Rayleigh quotient
        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc

    def var_energy_from_cache(
        self,
        psi_cache: PsiCache,
        ham_op: HamOp,
        dets: np.ndarray
    ) -> float:
        """
        Compute variational energy from cached amplitudes.
        
        Avoids redundant model forward pass by reusing cached ψ.
        
        Args:
            psi_cache: Cached wavefunction amplitudes
            ham_op: Hamiltonian operator
            dets: Determinant basis (must match cache order)
            
        Returns:
            Total variational energy E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ + E_nuc
        """
        psi = np.array(psi_cache.psi_all)
        
        if len(psi) != len(dets):
            raise ValueError(
                f"Cache size {len(psi)} != determinant count {len(dets)}"
            )
        
        # Compute H|ψ⟩
        h_psi = engine.kernels.coo_matvec(
            ham_op.rows, ham_op.cols, ham_op.vals, psi, len(dets)
        )

        # Rayleigh quotient
        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc


class AnalysisSuite:
    """
    Post-run analysis and convergence visualization.
    
    Compares LEVER energies against exact FCI reference.
    """
    
    def __init__(self, results: DriverResults, int_ctx: core.IntCtx):
        """
        Initialize analysis suite.
        
        Args:
            results: LEVER driver output
            int_ctx: Integral provider for FCI computation
        """
        self.res = results
        self.cfg = results.config
        self.int_ctx = int_ctx
        self.e_nuc = int_ctx.get_e_nuc()
        self.e_fci = self._compute_fci()

    def print_summary(self) -> None:
        """Print formatted energy summary with FCI comparison."""
        res = self.res
        final_e = res.full_hist[-1]

        print("\n" + "="*50)
        print("LEVER Analysis Summary")
        print("="*50)
        print(f"\nFCI Reference: {self.e_fci:.8f} Ha")
        print("\nLEVER Energies:")
        
        # Optimization energy
        err_mha = (final_e - self.e_fci) * 1e3
        print(f"  Optimization: {final_e:.8f} Ha  (Δ = {err_mha:+.4f} mHa)")

        # Variational energy (if available)
        if res.var_hist:
            e_var = res.var_hist[-1]
            err_var = (e_var - self.e_fci) * 1e3
            print(f"  Variational:  {e_var:.8f} Ha  (Δ = {err_var:+.4f} mHa)")
        
        # S-space CI energy (if available)
        if res.s_ci_hist:
            e_s = res.s_ci_hist[-1]
            err_s = (e_s - self.e_fci) * 1e3
            print(f"  S-space CI:   {e_s:.8f} Ha  (Δ = {err_s:+.4f} mHa)")
        
        print(f"\nWall Time: {res.total_time:.2f} s")
        print("="*50 + "\n")

    def plot_conv(self, sys_name: str | None = None) -> None:
        """
        Generate dual-panel convergence plot.
        
        Top panel: Energy trajectory with chemical accuracy band (±1.6 mHa)
        Bottom panel: Logarithmic absolute error |E_LEVER - E_FCI|
        
        Args:
            sys_name: System identifier (inferred from fcidump if None)
        """
        res = self.res
        
        # Infer system name from fcidump path
        if sys_name is None:
            from pathlib import Path
            sys_name = Path(self.cfg.system.fcidump_path).stem

        # Configure plot style
        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'lines.linewidth': 2.0,
            'grid.alpha': 0.3
        })
    
        # Create dual-panel layout
        fig, (ax_energy, ax_error) = plt.subplots(
            2, 1,
            figsize=(8, 6),
            sharex=True,
            height_ratios=[2, 1],
            gridspec_kw={'hspace': 0.15}
        )
    
        # Extract cycle-end energies
        energy = np.array(res.full_hist)
        cycle_ends = [bound - 1 for bound in res.cycle_bounds[1:]]
        cycle_energies = energy[cycle_ends]
        cycles = np.arange(1, len(cycle_ends) + 1)
        errors = np.abs(cycle_energies - self.e_fci)
        
        chem_accuracy = 1.6e-3  # Chemical accuracy: 1.6 mHa

        # Plot panels
        self._plot_energy_panel(
            ax_energy, cycles, cycle_energies, sys_name, chem_accuracy
        )
        self._plot_error_panel(ax_error, cycles, errors, chem_accuracy)
        
        plt.show()

    def _plot_energy_panel(
        self,
        ax: plt.Axes,
        cycles: np.ndarray,
        energies: np.ndarray,
        sys_name: str,
        chem_acc: float
    ) -> None:
        """Plot energy trajectory with FCI reference and accuracy band."""
        res = self.res
        
        # Chemical accuracy band
        ax.axhspan(
            self.e_fci - chem_acc,
            self.e_fci + chem_acc,
            alpha=0.15,
            color='green',
            label='Chem. Acc. (±1.6 mHa)'
        )
        
        # FCI reference line
        ax.axhline(
            self.e_fci,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label=f'FCI: {self.e_fci:.6f} Ha'
        )
        
        # Optimization trajectory
        ax.plot(
            cycles,
            energies,
            'o-',
            color='steelblue',
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='LEVER (opt)'
        )
    
        # Variational energies (if available)
        if res.var_hist:
            x_var = self._align_to_cycles(res.var_hist, cycles)
            if x_var:
                ax.plot(
                    x_var,
                    res.var_hist,
                    's-',
                    color='orange',
                    markersize=6,
                    markeredgecolor='white',
                    markeredgewidth=1.0,
                    label='LEVER (var)'
                )
      
        # S-space CI energies (if available)
        if res.s_ci_hist:
            x_s = self._align_to_cycles(res.s_ci_hist, cycles)
            if x_s:
                ax.plot(
                    x_s,
                    res.s_ci_hist,
                    'D-',
                    color='purple',
                    markersize=5,
                    markeredgecolor='white',
                    markeredgewidth=1.0,
                    label='S-space CI'
                )

        # Format panel
        ax.set_ylim(self.e_fci - 5e-3, self.e_fci + 15e-3)
        ax.set_ylabel('Total Energy (Ha)')
        ax.set_title(f'LEVER Evolution: {sys_name}')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    def _plot_error_panel(
        self,
        ax: plt.Axes,
        cycles: np.ndarray,
        errors: np.ndarray,
        chem_acc: float
    ) -> None:
        """Plot logarithmic error evolution with accuracy threshold."""
        ax.semilogy(
            cycles,
            errors,
            's-',
            color='crimson',
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='Absolute Error'
        )
        
        # Chemical accuracy threshold
        ax.axhline(
            chem_acc,
            color='green',
            linestyle='--',
            alpha=0.6,
            label=f'Chem. Acc. ({chem_acc*1e3:.1f} mHa)'
        )
        
        ax.set_xlabel('Evolution Cycle')
        ax.set_ylabel(r'$|E_{\mathrm{LEVER}} - E_{\mathrm{FCI}}|$ (Ha)')
        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_xticks(cycles)

    def _align_to_cycles(self, hist: list, cycles: np.ndarray) -> list:
        """
        Map data history to cycle x-coordinates.
        
        Handles: full history (1:1 mapping), single final value, or empty.
        """
        n = len(hist)
        if n == len(cycles):
            return cycles.tolist()
        elif n == 1:
            return [cycles[-1]]
        else:
            return []

    def _compute_fci(self) -> float:
        """
        Compute exact FCI energy via full diagonalization.
        
        Returns:
            FCI ground-state energy (including nuclear repulsion)
        """
        sys = self.cfg.system
        
        # Generate full FCI determinant basis
        fci_dets = core.gen_fci_dets(sys.n_orbitals, sys.n_alpha, sys.n_beta)
        
        # Build full Hamiltonian matrix
        ham_fci, _ = engine.hamiltonian.get_ham_ss(
            S_dets=fci_dets,
            int_ctx=self.int_ctx,
            n_orbitals=sys.n_orbitals
        )
        
        # Exact diagonalization
        evaluator = EvalSuite(self.int_ctx, sys.n_orbitals, self.e_nuc)
        return evaluator.diag_ham(ham_fci)


__all__ = ["EvalSuite", "AnalysisSuite"]
