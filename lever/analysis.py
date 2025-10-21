# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analysis and visualization tools for LEVER results.

Computes reference energies, processes driver output, and generates convergence plots.

File: lever/analysis/__init__.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from pyscf import lib

from . import core, engine
from .driver import DriverResults


class EvaluationSuite:
    """Helper for computing energies during active calculations."""
    
    def __init__(self, int_ctx: core.IntCtx, n_orbitals: int, e_nuc: float):
        self.int_ctx = int_ctx
        self.n_orbitals = n_orbitals
        self.e_nuc = e_nuc

    def diagonalize_hamiltonian(self, ham_op):
        """
        Davidson with proper 3-arg preconditioner and block initial guesses.
        """
        if ham_op.shape[0] == 0:
            return self.e_nuc
        if ham_op.shape[0] == 1:
            return float(ham_op.vals[0]) + self.e_nuc

        # build CSR once
        H = sp.coo_matrix(
            (ham_op.vals, (ham_op.rows, ham_op.cols)),
            shape=ham_op.shape
        ).tocsr()
        H.sum_duplicates()

        # diagonal (may be complex for general Hermitian)
        diag = H.diagonal()

        # better initial guesses: index of min diagonal + one random orthogonal vec
        i0 = int(np.argmin(diag.real))  # use real part as heuristic
        x0 = np.zeros(H.shape[0], dtype=diag.dtype)
        x0[i0] = 1.0
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(H.shape[0]).astype(diag.real.dtype)
        x1 = x1 - (x1 @ x0.real) * x0  # orthogonalize
        x1 = x1 / np.linalg.norm(x1)
        X0 = [x0, x1.astype(x0.dtype, copy=False)]

        # 3-argument preconditioner: (r, e, x) -> approx (D - e I)^{-1} r
        def precond_fun(r, e, _x_unused):
            denom = diag - e
            # damping to avoid division by tiny numbers
            tau = 1e-3 * (np.linalg.norm(diag, ord=np.inf) + 1.0)
            safe = np.where(np.abs(denom) > 1e-8, denom, np.sign(denom) * tau)
            return r / safe

        e, _ = lib.linalg_helper.davidson(
            lambda v: H @ v,              # matvec
            X0,                           # block initial guesses
            precond_fun,                  # 3-arg preconditioner
            nroots=1,
            max_space=50,
            max_cycle=200,
            tol=1e-10
        )
        return float(np.atleast_1d(e)[0]) + self.e_nuc

    def compute_variational_energy(
        self, variables: Any, logpsi_fn: Callable, ham_op: engine.HamOp,
        dets: np.ndarray
    ) -> float:
        """
        Computes variational energy E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩.
        
        Algorithm: Convert parameters to amplitudes ψ, compute Hψ via sparse matvec,
        then evaluate Rayleigh quotient.
        """
        from jax import numpy as jnp
      
        t_vecs = engine.utils.masks_to_vecs(jnp.asarray(dets), self.n_orbitals)
        log_psi = logpsi_fn(variables, t_vecs)
        psi = np.array(jnp.exp(log_psi))
      
        h_psi = engine.kernels.coo_matvec(
            ham_op.rows, ham_op.cols, ham_op.vals, psi, len(dets)
        )

        e_elec = np.vdot(psi, h_psi).real / np.vdot(psi, psi).real
        return e_elec + self.e_nuc


class AnalysisSuite:
    """Processes and visualizes results from completed LEVER calculations."""
    
    def __init__(self, results: DriverResults, int_ctx: core.IntCtx):
        self.results = results
        self.config = results.config
        self.int_ctx = int_ctx
        self.e_fci = self._compute_fci_energy()

    def print_summary(self):
        """Prints final calculation summary with energy comparisons."""
        res = self.results
        final_opt_E = res.full_history[-1]

        print("\n--- Final Analysis ---")
        print(f"Reference FCI Energy: {self.e_fci:.8f} Ha")
        print("\nLEVER Final Energies:")
        
        opt_error_mHa = (final_opt_E - self.e_fci) * 1e3
        print(f"  Optimization:   {final_opt_E:.8f} Ha "
              f"(Δ = {opt_error_mHa:+.4f} mHa)")

        if res.var_energy_history:
            final_var_E = res.var_energy_history[-1]
            var_error_mHa = (final_var_E - self.e_fci) * 1e3
            print(f"  Variational(T): {final_var_E:.8f} Ha "
                  f"(Δ = {var_error_mHa:+.4f} mHa)")
        
        if res.s_ci_energy_history:
            final_s_ci_E = res.s_ci_energy_history[-1]
            s_ci_error_mHa = (final_s_ci_E - self.e_fci) * 1e3
            print(f"  S-space CI:     {final_s_ci_E:.8f} Ha "
                  f"(Δ = {s_ci_error_mHa:+.4f} mHa)")
        
        print(f"\nTotal runtime: {res.total_time:.2f} seconds")

    def plot_convergence(self, system_name: str | None = None):
        """
        Generates dual-panel convergence plot.
        
        Top: Energy trajectory vs cycle with FCI reference
        Bottom: Absolute error on log scale
        """
        res = self.results
        
        if system_name is None:
            from pathlib import Path
            system_name = Path(self.config.system.fcidump_path).stem

        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'lines.linewidth': 2.0,
            'grid.alpha': 0.3,
        })
    
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True, 
            height_ratios=[2, 1], gridspec_kw={'hspace': 0.15}
        )
    
        energy = np.array(res.full_history)
        cycle_end_steps = [b - 1 for b in res.cycle_boundaries[1:]]
        cycle_end_energies = energy[cycle_end_steps]
        cycle_indices = np.arange(1, len(cycle_end_steps) + 1)
        cycle_end_errors = np.abs(cycle_end_energies - self.e_fci)
        
        chem_acc = 1.6e-3

        ax1.axhspan(
            self.e_fci - chem_acc, self.e_fci + chem_acc,
            alpha=0.15, color='green', label='Chemical Accuracy (±1.6 mHa)'
        )
        ax1.axhline(
            self.e_fci, color='black', linestyle='--', linewidth=1.5,
            label=f'FCI: {self.e_fci:.6f} Ha'
        )
        ax1.plot(
            cycle_indices, cycle_end_energies, 'o-', color='steelblue',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5, 
            label='LEVER (opt)'
        )
    
        if res.var_energy_history:
            x_var = self._get_plot_indices(res.var_energy_history, cycle_indices)
            if len(x_var) > 0:
                ax1.plot(
                    x_var, res.var_energy_history, 's-', color='orange',
                    markersize=6, markeredgecolor='white', markeredgewidth=1.0,
                    label='LEVER (var)'
                )
      
        if res.s_ci_energy_history:
            x_sci = self._get_plot_indices(res.s_ci_energy_history, cycle_indices)
            if len(x_sci) > 0:
                ax1.plot(
                    x_sci, res.s_ci_energy_history, 'D-', color='purple',
                    markersize=5, markeredgecolor='white', markeredgewidth=1.0,
                    label='S-space CI'
                )

        ax1.set_ylim(self.e_fci - 5e-3, self.e_fci + 15e-3)
        ax1.set_ylabel('Total Energy (Ha)')
        ax1.set_title(f'LEVER Evolution: {system_name}')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax1.ticklabel_format(style='plain', axis='y', useOffset=False)

        ax2.semilogy(
            cycle_indices, cycle_end_errors, 's-', color='crimson',
            markersize=8, markeredgecolor='white', markeredgewidth=1.5,
            label='Absolute Error'
        )
        ax2.axhline(chem_acc, color='green', linestyle='--', alpha=0.6)
        ax2.set_xlabel('Evolution Cycle')
        ax2.set_ylabel(r'$|E_{\mathrm{LEVER}} - E_{\mathrm{FCI}}|$ (Ha)')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        ax2.set_xticks(cycle_indices)
    
        plt.tight_layout()
        plt.show()

    def _get_plot_indices(self, data_history: list, cycle_indices: np.ndarray) -> list:
        """Maps data history to appropriate x-axis indices for plotting."""
        num_points = len(data_history)
        
        if num_points == len(cycle_indices):
            return cycle_indices.tolist()
        elif num_points == 1:
            return [cycle_indices[-1]]
        else:
            return []

    def _compute_fci_energy(self) -> float:
        """Computes exact FCI ground state energy via full Hamiltonian diagonalization."""
        sys = self.config.system
        
        fci_dets = core.gen_fci_dets(sys.n_orbitals, sys.n_alpha, sys.n_beta)
        
        ham_fci, _, _ = engine.hamiltonian.get_ham_proxy(
            S_dets=fci_dets, int_ctx=self.int_ctx, n_orbitals=sys.n_orbitals,
            use_heatbath=False
        )
        
        eval_suite = EvaluationSuite(
            self.int_ctx, sys.n_orbitals, self.int_ctx.get_e_nuc()
        )
        return eval_suite.diagonalize_hamiltonian(ham_fci)


__all__ = ["EvaluationSuite", "AnalysisSuite"]
