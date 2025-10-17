# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pfaffian wavefunction models for quantum many-body systems with pairing correlations.

Implements BCS-like pairing states using Pfaffian wavefunctions in occupation basis.
Core ansätze:
  - AGP (Antisymmetrized Geminal Power): Singlet-only pairing via det(G)
  - Full Pfaffian: Singlet + triplet pairing via Pf(antisymmetric matrix)

Mathematical background:
  AGP:       log ψ(s) = log det(G[R_α, R_β])
  Pfaffian:  log ψ(s) = log Pf([[F_αα, Φ], [-Φᵀ, F_ββ]])
  
where R_α, R_β are occupied orbital indices for spin-up/down electrons.

File: lever/models/pfaffian.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: October, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from jax import lax

from . import utils


# --- Pfaffian-specific utilities ---

def _antisym(A: jnp.ndarray) -> jnp.ndarray:
    """
    Extract strictly antisymmetric part of a square matrix.
    
    For matrix A, computes: (A - Aᵀ) / 2
    
    Args:
        A: Square matrix, shape (..., n, n)
        
    Returns:
        Antisymmetric matrix with diagonal zeros
    """
    return 0.5 * (A - jnp.swapaxes(A, -1, -2))


def logpfaff_c(A: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically stable log-Pfaffian for complex antisymmetric matrices.
    
    Computes log(Pf(A)) via iterative rank-2 antisymmetric reduction.
    Algorithm based on Parlett-Reid (1970) with masked updates to avoid
    dynamic shape changes in JIT-compiled code.
    
    Mathematical properties:
      - Pf(A)² = det(A) for antisymmetric A
      - Pf is only defined for even-dimensional matrices
      - Row/column swaps multiply Pf by -1
      - Pf(empty matrix) = 1 by convention
    
    Args:
        A: Antisymmetric matrix, shape (n, n) with n even
        
    Returns:
        Complex scalar log(Pf(A))
        Returns 0 for empty matrices (Pf = 1)
        Returns -∞ if A is singular or n is odd
        
    Algorithm outline:
      1. For each 2×2 pivot block:
         - Find largest off-diagonal element (k, j) in remaining submatrix
         - Swap rows/columns to bring it to position (k, k+1)
         - Perform rank-2 antisymmetric update on trailing block
         - Accumulate log(pivot) and phase from swaps
      2. Product of all pivots gives Pf(A)
      
    References:
      - Parlett & Reid (1970): On the solution of a system of linear equations
        whose matrix is symmetric but not definite
      - Wimmer (2012): Efficient computation of the Pfaffian for dense
        and banded skew-symmetric matrices (arXiv:1102.3440)
    """
    n = A.shape[-1]
    
    # Handle empty matrix: Pf(∅) = 1 by convention → log(1) = 0
    # This occurs when evaluating empty C-space in FCI calculations
    if n == 0:
        return jnp.array(0.0 + 0.0j)
    
    # Pfaffian only defined for even dimensions
    if (n % 2) != 0:
        return jnp.array(-jnp.inf + 0.0j)
    
    # Ensure strict antisymmetry and complex dtype
    M = _antisym(A).astype(jnp.complex128)
    
    # Safe row/col swap using JAX lax slice updates (no advanced indexing)
    # This avoids gather/scatter issues in JIT-compiled fori_loop contexts
    def _swap_rows_cols(mat: jnp.ndarray, i: int, j: int) -> jnp.ndarray:
        """
        Swap rows i<->j and columns i<->j using dynamic slice operations.
        
        Uses lax.dynamic_slice_in_dim and lax.dynamic_update_slice_in_dim
        to avoid advanced indexing issues with .at[indices].set() in JIT context.
        
        Args:
            mat: Matrix to modify
            i, j: Row/column indices to swap
            
        Returns:
            Matrix with swapped rows and columns
        """
        def perform_swap(X):
            # Swap rows i <-> j
            # Extract row i (shape: (1, n))
            row_i = lax.dynamic_slice_in_dim(X, i, 1, axis=0)
            # Extract row j
            row_j = lax.dynamic_slice_in_dim(X, j, 1, axis=0)
            # Update: put row_j at position i
            X = lax.dynamic_update_slice_in_dim(X, row_j, i, axis=0)
            # Update: put row_i at position j
            X = lax.dynamic_update_slice_in_dim(X, row_i, j, axis=0)
            
            # Swap columns i <-> j
            # Extract column i (shape: (n, 1))
            col_i = lax.dynamic_slice_in_dim(X, i, 1, axis=1)
            # Extract column j
            col_j = lax.dynamic_slice_in_dim(X, j, 1, axis=1)
            # Update: put col_j at position i
            X = lax.dynamic_update_slice_in_dim(X, col_j, i, axis=1)
            # Update: put col_i at position j
            X = lax.dynamic_update_slice_in_dim(X, col_i, j, axis=1)
            
            return X
        
        # Only swap if indices differ (identity operation otherwise)
        return lax.cond(i == j, lambda X: X, perform_swap, mat)
    
    def reduction_step(i: int, carry: tuple) -> tuple:
        """
        Single iteration of Pfaffian reduction.
        
        Processes 2×2 block starting at row/col index k = 2*i.
        Updates remaining (n-k) × (n-k) trailing block via rank-2 correction.
        
        Args:
            i: Iteration index (processes block at k = 2*i)
            carry: (matrix, log_accumulator, singularity_flag)
            
        Returns:
            Updated carry tuple
        """
        k = 2 * i
        matrix, log_acc, is_singular = carry
        
        # Early exit if already detected singularity
        def skip_iteration(state):
            return state
        
        def perform_reduction(state):
            M_cur, log_cur, _ = state
            row_k = M_cur[k, :]
            
            # Find largest off-diagonal element in row k (beyond diagonal)
            magnitudes = jnp.abs(row_k)
            valid_mask = jnp.arange(n) > k  # Only consider indices > k
            masked_mags = jnp.where(valid_mask, magnitudes, -jnp.inf)
            j_max = jnp.argmax(masked_mags)
            
            # Check if we have any valid candidates for pivot
            # This prevents meaningless swaps on [-inf] argmax results
            has_valid_pivot = jnp.any(valid_mask)
            
            # Swap columns/rows if needed to move pivot to (k, k+1)
            def apply_swap(swap_state):
                M_swap, log_swap = swap_state
                # Single transposition (k+1) <-> j_max using safe slice operations
                M_swap = _swap_rows_cols(M_swap, k + 1, j_max)
                # Swap introduces phase factor -1 in Pfaffian
                log_swap = log_swap + jnp.log(-1.0 + 0.0j)
                return M_swap, log_swap
            
            def skip_swap(swap_state):
                return swap_state
            
            M_cur, log_cur = lax.cond(
                jnp.logical_and(has_valid_pivot, j_max != (k + 1)),
                apply_swap,
                skip_swap,
                (M_cur, log_cur)
            )
            
            # Extract pivot element
            pivot = M_cur[k, k + 1]
            tolerance = 1e-16
            pivot_is_zero = jnp.abs(pivot) < tolerance
            
            # Handle singular case
            def mark_singular(sing_state):
                M_s, log_s = sing_state
                return (M_s, log_s, True)
            
            # Perform rank-2 update on trailing block
            def perform_update(update_state):
                M_upd, log_upd = update_state
                
                # Indices for masking: only update rows/cols >= k+2
                idx_range = jnp.arange(n)
                trailing_mask = (idx_range >= (k + 2)).astype(M_upd.dtype)
                
                # Extract relevant vectors (full length for broadcasting)
                u_full = M_upd[k, :]      # Row k
                v_full = M_upd[k + 1, :]  # Row k+1
                
                # Rank-2 antisymmetric update: (u⊗v - v⊗u) / pivot
                delta_full = (jnp.outer(u_full, v_full) - 
                             jnp.outer(v_full, u_full)) / pivot
                
                # Apply mask to restrict update to trailing block
                mask_2d = (trailing_mask[:, None] * 
                          trailing_mask[None, :]).astype(M_upd.dtype)
                
                # Update matrix while preserving antisymmetry
                M_updated = _antisym(M_upd - delta_full * mask_2d)
                
                # Accumulate log(pivot) contribution
                log_upd = log_upd + jnp.log(pivot + 0j)
                
                return (M_updated, log_upd, False)
            
            # Extra robustness: only proceed with update if we have a valid
            # non-zero pivot. This prevents numerical issues with degenerate
            # matrices or edge cases in small Hilbert spaces.
            can_proceed = jnp.logical_and(has_valid_pivot, 
                                         jnp.logical_not(pivot_is_zero))
            
            return lax.cond(
                can_proceed,
                perform_update,
                mark_singular,
                (M_cur, log_cur)
            )
        
        return lax.cond(
            is_singular,
            skip_iteration,
            perform_reduction,
            (matrix, log_acc, is_singular)
        )
    
    # Initialize reduction: (matrix, log_accumulator, singularity_flag)
    initial_state = (M, jnp.array(0.0 + 0.0j), False)
    
    # Perform n/2 reduction steps
    _, final_log, is_singular = lax.fori_loop(
        0, n // 2, reduction_step, initial_state
    )
    
    # Return -∞ if singular, otherwise accumulated log-Pfaffian
    return jnp.where(is_singular, jnp.array(-jnp.inf + 0.0j), final_log)


# --- Pfaffian Wavefunction Module ---

class Pfaffian(nn.Module):
    """
    Unified Pfaffian wavefunction ansatz for pairing states.
    
    Supports two formulations:
    
    1. AGP (singlet_only=True):
       Antisymmetrized Geminal Power - BCS ground state with singlet pairs.
       log ψ(s) = log det(G[R_α, R_β])
       where G is the pairing matrix (n_orbitals × n_orbitals).
       
       Physical interpretation:
         - Represents perfect singlet pairing between α and β spins
         - Equivalent to number-projected BCS state
         - Each geminal (row of G) describes spatial correlation of a pair
    
    2. Full Pfaffian (singlet_only=False):
       Includes triplet pairing correlations via antisymmetric blocks.
       log ψ(s) = log Pf([[F_αα,  Φ   ],
                          [-Φᵀ,   F_ββ ]])
       where:
         - F_αα: α-α triplet pairing (antisymmetric)
         - F_ββ: β-β triplet pairing (antisymmetric)
         - Φ:    α-β singlet pairing
         
       Physical interpretation:
         - F_αα, F_ββ enable same-spin (triplet) correlations
         - Φ provides opposite-spin (singlet) correlations
         - Can represent systems with both singlet and triplet superconductivity
    
    Multi-term expansion:
      - n_terms ≥ 1: Linear combination of Pfaffians/AGPs
      - Combine via log-sum-exp for numerical stability
      - Optional learnable complex coefficients
      - Generalizes single Pfaffian to multi-reference states
    
    Attributes:
        n_orbitals: Number of spatial orbitals
        n_alpha, n_beta: Electron counts per spin
        n_terms: Number of terms in expansion (1 = single Pfaffian)
        singlet_only: Use AGP formulation (True) or full Pfaffian (False)
        use_log_coeffs: Learn expansion coefficients (n_terms > 1 only)
        param_dtype: Parameter dtype (complex recommended for flexibility)
        kernel_init: Weight initializer (default: lecun_normal)
    
    Examples:
        >>> # BCS-like singlet pairing for 4 orbitals, 2 up + 2 down electrons
        >>> agp = Pfaffian(n_orbitals=4, n_alpha=2, n_beta=2, singlet_only=True)
        
        >>> # Full Pfaffian with triplet correlations
        >>> pf = Pfaffian(n_orbitals=4, n_alpha=2, n_beta=2, singlet_only=False)
        
        >>> # Multi-reference: 3 AGP states with learnable mixing coefficients
        >>> multi_agp = Pfaffian(n_orbitals=4, n_alpha=2, n_beta=2, 
        ...                      singlet_only=True, n_terms=3)
    
    References:
        - Bajdich et al., PRL 96, 130201 (2006): AGP in QMC
        - Neuscamman, Phys. Rev. B 93, 245112 (2016): Pfaffian wavefunctions
    """
    
    n_orbitals: int
    n_alpha: int
    n_beta: int
    n_terms: int = 1
    singlet_only: bool = True
    use_log_coeffs: bool = True
    param_dtype: Any = jnp.complex128
    kernel_init: Callable = initializers.lecun_normal()
    
    @nn.compact
    def __call__(self, s: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate log-amplitude for occupation vector.
        
        Args:
            s: Occupation numbers, shape (2×n_orbitals,) as [α_occs, β_occs]
               Each element is 0 (unoccupied) or 1 (occupied)
        
        Returns:
            Complex scalar log(ψ(s))
        """
        n_orb = self.n_orbitals
        is_multi = self.n_terms > 1
        
        # Split spin sectors
        alpha_occ = s[:n_orb]
        beta_occ = s[n_orb:2*n_orb]
        
        # Extract occupied orbital indices
        # fill_value=-1 handles edge cases, but should not be accessed
        rows_alpha = jnp.nonzero(alpha_occ, size=self.n_alpha, fill_value=-1)[0]
        rows_beta = jnp.nonzero(beta_occ, size=self.n_beta, fill_value=-1)[0]
        
        if self.singlet_only:
            # AGP mode: single pairing matrix G
            log_vals = self._eval_agp(rows_alpha, rows_beta, is_multi)
        else:
            # Full Pfaffian: triplet + singlet blocks
            log_vals = self._eval_pfaffian(rows_alpha, rows_beta, is_multi)
        
        # Multi-term combination
        if not is_multi:
            return log_vals
        
        if self.use_log_coeffs:
            log_coeffs = self.param(
                "log_coeffs",
                lambda k, sh, dt: jnp.zeros(sh, dt),
                (self.n_terms,),
                self.param_dtype
            )
            log_vals = log_vals + log_coeffs
        
        return utils.logsumexp_c(log_vals, axis=0)
    
    def _eval_agp(
        self, 
        rows_alpha: jnp.ndarray, 
        rows_beta: jnp.ndarray,
        is_multi: bool
    ) -> jnp.ndarray:
        """
        Evaluate AGP (Antisymmetrized Geminal Power) amplitude.
        
        AGP wavefunction: ψ_AGP = det(G[R_α, R_β])
        where G is the geminal (pairing) matrix.
        
        The determinant structure ensures proper antisymmetry:
          - Each column corresponds to a β electron
          - Each row corresponds to an α electron
          - G[i,j] describes pairing amplitude between α-orbital i and β-orbital j
        
        Args:
            rows_alpha: Indices of occupied α orbitals (shape: (n_alpha,))
            rows_beta: Indices of occupied β orbitals (shape: (n_beta,))
            is_multi: Whether to use multi-term expansion
            
        Returns:
            log(det(G)) or array of log-determinants (n_terms,)
        """
        n_orb = self.n_orbitals
        shape = ((self.n_terms, n_orb, n_orb) if is_multi 
                 else (n_orb, n_orb))
        
        G = self.param("G", self.kernel_init, shape, self.param_dtype)
        
        def eval_single_agp(G_term: jnp.ndarray) -> jnp.ndarray:
            """Extract occupied block and compute log-determinant."""
            # G[R_α, R_β]: rows from α sector, columns from β sector
            # This indexing extracts the (n_alpha × n_beta) submatrix
            # corresponding to the occupied orbitals
            block = G_term[rows_alpha, :][:, rows_beta]
            return utils.logdet_c(block)
        
        if not is_multi:
            return eval_single_agp(G)
        
        # Vectorize over term axis for multi-term expansion
        return jax.vmap(eval_single_agp)(G)
    
    def _eval_pfaffian(
        self,
        rows_alpha: jnp.ndarray,
        rows_beta: jnp.ndarray,
        is_multi: bool
    ) -> jnp.ndarray:
        """
        Evaluate full Pfaffian amplitude with triplet correlations.
        
        Full Pfaffian structure:
          A_occ = [[F_αα[R_α, R_α],  Φ[R_α, R_β]    ],
                   [-Φᵀ[R_β, R_α],   F_ββ[R_β, R_β] ]]
        
        where:
          - F_αα, F_ββ are antisymmetric (same-spin pairing)
          - Φ is general (opposite-spin pairing)
        
        The antisymmetric block structure ensures:
          - Proper fermionic antisymmetry
          - Pf(A)² = det(A) relation holds
          - A_occ is (n_alpha + n_beta) × (n_alpha + n_beta) antisymmetric
        
        Physical interpretation:
          - F_αα describes α-α pairing (triplet with S_z = +1)
          - F_ββ describes β-β pairing (triplet with S_z = -1)
          - Φ describes α-β pairing (singlet and triplet with S_z = 0)
        
        Args:
            rows_alpha: Occupied α orbital indices (shape: (n_alpha,))
            rows_beta: Occupied β orbital indices (shape: (n_beta,))
            is_multi: Whether to use multi-term expansion
            
        Returns:
            log(Pf(A_occ)) or array of log-Pfaffians (n_terms,)
        """
        n_orb = self.n_orbitals
        shape = ((self.n_terms, n_orb, n_orb) if is_multi 
                 else (n_orb, n_orb))
        
        # Initialize three pairing blocks
        F_aa = self.param("F_aa", self.kernel_init, shape, self.param_dtype)
        F_bb = self.param("F_bb", self.kernel_init, shape, self.param_dtype)
        Phi = self.param("Phi", self.kernel_init, shape, self.param_dtype)
        
        def eval_single_pfaffian(params_term: tuple) -> jnp.ndarray:
            """Construct occupied-space Pfaffian matrix and evaluate."""
            F_aa_t, F_bb_t, Phi_t = params_term
            
            # Extract occupied blocks
            # Upper-left: α-α antisymmetric pairing
            A11 = _antisym(F_aa_t)[rows_alpha, :][:, rows_alpha]
            
            # Lower-right: β-β antisymmetric pairing
            A22 = _antisym(F_bb_t)[rows_beta, :][:, rows_beta]
            
            # Upper-right: α-β singlet pairing
            A12 = Phi_t[rows_alpha, :][:, rows_beta]
            
            # Lower-left: -Φᵀ (antisymmetry requirement)
            A21 = -jnp.transpose(A12)
            
            # Assemble full antisymmetric matrix
            # Structure: [[A11, A12],
            #            [A21, A22]]
            # This is (n_alpha + n_beta) × (n_alpha + n_beta) and antisymmetric
            upper_block = jnp.concatenate([A11, A12], axis=1)
            lower_block = jnp.concatenate([A21, A22], axis=1)
            A_occ = jnp.concatenate([upper_block, lower_block], axis=0)
            
            return logpfaff_c(A_occ)
        
        if not is_multi:
            return eval_single_pfaffian((F_aa, F_bb, Phi))
        
        # Vectorize over term axis for multi-term expansion
        return jax.vmap(
            lambda Faa, Fbb, Ph: eval_single_pfaffian((Faa, Fbb, Ph))
        )(F_aa, F_bb, Phi)


__all__ = ["Pfaffian", "logpfaff_c"]
