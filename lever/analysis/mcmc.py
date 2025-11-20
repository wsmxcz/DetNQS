# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-training MCMC inference engine (PostMCMC).

Implements Variational Monte Carlo (VMC) sampling restricted to the union of
Selected (S) and Connected (C) spaces to prevent C-space leakage.

Features:
  - Restricted Sampling: x ~ |Ψ(x)|² s.t. x ∈ S ∪ C
  - Vectorized Proposal: Efficient bitwise excitation logic on CPU
  - Unique-First Evaluation: Deduplicates samples before E_L computation
  - Error Analysis: Power-of-2 blocking method for autocorrelation correction

File: lever/analysis/mcmc.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from lever.core import IntCtx, get_local_connections
from lever.utils.features import dets_to_features

if TYPE_CHECKING:
    from lever.config import LeverConfig
    from lever.models import WavefunctionModel

Array = jax.Array


# ============================================================================
# Statistics & Helpers
# ============================================================================

@dataclass(frozen=True)
class MCMCResult:
    """Container for VMC energy statistics."""
    mean: float
    error: float
    variance: float
    acc_rate: float
    n_samples: int


def blocking_analysis(data: np.ndarray) -> Tuple[float, float]:
    """
    Estimate mean and standard error via Flyvbjerg-Petersen blocking method.
    Truncates data to nearest power of 2 for cleaner block averaging.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n < 2:
        return float(x[0]) if n == 1 else np.nan, np.nan

    # Truncate to power of 2
    n_pow2 = 1 << (n.bit_length() - 1)
    if n_pow2 < 2: return np.mean(x), np.nan
    curr = x[:n_pow2].copy()
    
    mean = float(curr.mean())
    best_stderr = np.sqrt(np.var(curr, ddof=1) / curr.size)

    while curr.size >= 4:
        # Block averaging: x'_i = 0.5 * (x_{2i} + x_{2i+1})
        curr = 0.5 * (curr[0::2] + curr[1::2])
        
        err = np.sqrt(np.var(curr, ddof=1) / curr.size)
        best_stderr = max(best_stderr, err)

    return mean, float(best_stderr)


# ============================================================================
# Proposal Logic (CPU)
# ============================================================================

def apply_excitation_numpy(
    dets: np.ndarray, 
    n_orb: int, 
    rng: np.random.Generator, 
    p_double: float
) -> np.ndarray:
    """
    Apply symmetric Single/Double excitations via vectorized bitwise ops.
    Optimized to update occupancy masks incrementally.
    """
    n_w = dets.shape[0]
    new_dets = dets.copy()
    
    # 1. Select Move Type (Single vs Double)
    is_double = rng.random(n_w) < p_double
    
    # 2. Precompute Initial Occupancy Masks
    idx_rng = np.arange(n_orb, dtype=np.uint64)
    alpha, beta = new_dets[:, 0], new_dets[:, 1]
    occ_a = ((alpha[:, None] >> idx_rng) & 1).astype(bool)
    occ_b = ((beta[:, None]  >> idx_rng) & 1).astype(bool)

    # Helper to apply single excitation to a subset of walkers
    def _do_single_excitation(mask):
        if not np.any(mask): return

        # Filter valid channels for active walkers
        valid_a = (occ_a[mask].any(axis=1) & (~occ_a[mask]).any(axis=1))
        valid_b = (occ_b[mask].any(axis=1) & (~occ_b[mask]).any(axis=1))
        
        # Select channel A/B
        n_active = np.sum(mask)
        choose_a = rng.random(n_active) < 0.5
        choose_a = np.where(valid_a & valid_b, choose_a, valid_a)
        
        # Target masks
        t_occ  = np.where(choose_a[:, None], occ_a[mask], occ_b[mask])
        t_virt = np.where(choose_a[:, None], ~occ_a[mask], ~occ_b[mask])
        
        # Sample indices
        def sample_idx(m):
            r = rng.random(m.shape, dtype=np.float32)
            return np.argmax(np.where(m, r, -1.0), axis=1).astype(np.uint64)

        idx_i, idx_a = sample_idx(t_occ), sample_idx(t_virt)
        
        # Apply updates
        flip = (np.uint64(1) << idx_i) | (np.uint64(1) << idx_a)
        
        # Map active indices back to full array
        active_indices = np.flatnonzero(mask)
        idx_update_a = active_indices[choose_a]
        idx_update_b = active_indices[~choose_a]
        
        # Update bitstrings
        new_dets[idx_update_a, 0] ^= flip[choose_a]
        new_dets[idx_update_b, 1] ^= flip[~choose_a]
        
        # Incremental update of occupancy masks (avoids recomputing shift)
        # Update A channel
        if len(idx_update_a) > 0:
            occ_a[idx_update_a, idx_i[choose_a]] = False
            occ_a[idx_update_a, idx_a[choose_a]] = True
        # Update B channel
        if len(idx_update_b) > 0:
            occ_b[idx_update_b, idx_i[~choose_a]] = False
            occ_b[idx_update_b, idx_a[~choose_a]] = True

    # Step 1: Single Excitation (All walkers)
    _do_single_excitation(np.ones(n_w, dtype=bool))
    
    # Step 2: Double Excitation (Subset only)
    _do_single_excitation(is_double)

    return new_dets


# ============================================================================
# MCMC Driver
# ============================================================================

class PostMCMC:
    """
    Hybrid CPU/GPU engine for VMC sampling and energy estimation.
    Uses JAX for neural network inference and NumPy for sampling logic.
    """
    
    def __init__(
        self, 
        config: LeverConfig, 
        int_ctx: IntCtx, 
        model: WavefunctionModel, 
        params: Any,
        infer_batch_size: int = 16384, 
        conn_chunk_size: int = 10000
    ):
        self.cfg = config
        self.int_ctx = int_ctx
        self.n_orb = config.system.n_orbitals
        self.bs = infer_batch_size
        self.conn_bs = conn_chunk_size

        # Compile JAX evaluators
        # log_prob = 2 * Re[log(Ψ)]
        @jax.jit
        def _psi(d): 
            return jnp.exp(model.log_psi(params, dets_to_features(d, self.n_orb)))
        
        @jax.jit
        def _logp(d): 
            return 2.0 * jnp.real(model.log_psi(params, dets_to_features(d, self.n_orb)))

        self._jit_psi = _psi
        self._jit_logp = _logp

    def _eval_log_prob(self, dets: np.ndarray) -> np.ndarray:
        """Evaluate log-probability on GPU, return writable NumPy array."""
        # copy=True prevents 'read-only buffer' errors during MCMC updates
        return np.array(self._jit_logp(dets), copy=True)

    def _eval_psi(self, dets: np.ndarray) -> np.ndarray:
        """Batched evaluation of wavefunction amplitudes."""
        n = len(dets)
        if n == 0: return np.array([], dtype=np.complex128)
        
        res = []
        for i in range(0, n, self.bs):
            batch = dets[i : i + self.bs]
            res.append(self._jit_psi(batch))
            
        return np.array(jnp.concatenate(res), copy=True)

    def _compute_batch_eloc(self, samples: np.ndarray, hb: bool, eps: float) -> np.ndarray:
        """
        Compute Local Energy E_L(x) = Σ_{x'} H_{xx'} Ψ(x')/Ψ(x).
        
        Args:
            samples: Unique determinants to evaluate.
        """
        n = len(samples)
        if n == 0: return np.array([])

        # 1. Get Hamiltonian Connections (CPU)
        conn = get_local_connections(samples, self.int_ctx, self.n_orb, hb, eps)
        kets = conn["dets"].astype(np.uint64).reshape(-1, 2)
        vals = conn["values"]
        offsets = conn["offsets"]

        # 2. Evaluate Wavefunction (GPU)
        # Combine bra (samples) and ket (connections) to minimize inference
        all_dets = np.vstack((samples, kets))
        uniq, inv = np.unique(all_dets, axis=0, return_inverse=True)
        psi_uniq = self._eval_psi(uniq)
        
        psi_samp = psi_uniq[inv[:n]]
        psi_kets = psi_uniq[inv[n:]]

        # 3. Contract E_L numerator
        num = np.zeros(n, dtype=np.complex128)
        row_idx = np.repeat(np.arange(n), np.diff(offsets))
        np.add.at(num, row_idx, vals * psi_kets)

        # 4. Divide by Ψ(x) (Safe division)
        mask_zero = np.abs(psi_samp) < 1e-14
        denom = np.where(mask_zero, 1.0, psi_samp)
        return np.where(mask_zero, 0.0, num / denom)

    def run(
        self, 
        s_dets: np.ndarray, 
        *, 
        n_walkers: int = 1024, 
        n_steps: int = 10000, 
        burn_in: int = 1000, 
        thinning: int = 10, 
        seed: int = 42, 
        use_heatbath: bool = False, 
        screen_eps: float = 1e-6, 
        p_double: float = 0.5,
        verbose: bool = True
    ) -> MCMCResult:
        """
        Execute MCMC workflow with S ∪ C restriction.
        
        Algorithm:
          1. Construct allowed space S ∪ C (hash set).
          2. Run Metropolis-Hastings, rejecting moves outside S ∪ C.
          3. Deduplicate resulting chain.
          4. Compute E_L for unique states and map back.
        """
        if verbose:
            print(f"Starting Post-MCMC (Restricted S∪C) | Walkers={n_walkers} Steps={n_steps}")

        # --- 1. Build Valid Space (S U C) ---
        if verbose: print("  Constructing valid space map...")
        conn = get_local_connections(s_dets, self.int_ctx, self.n_orb, use_heatbath, screen_eps)
        c_dets = conn["dets"].astype(np.uint64).reshape(-1, 2)
        valid_dets = np.unique(np.vstack((s_dets, c_dets)), axis=0)
        
        # Use raw bytes (void view) for efficient vectorized lookup
        void_dt = np.dtype((np.void, 16))
        valid_view = np.ascontiguousarray(valid_dets).view(void_dt).ravel()
        # Sort for np.isin binary search
        valid_view.sort()
        
        if verbose:
            print(f"  Valid Space Size: {len(valid_view)} (S={len(s_dets)})")

        # --- 2. Initialize Walkers ---
        rng = np.random.default_rng(seed)
        # Bootstrap from S-space
        curr = s_dets[rng.choice(len(s_dets), n_walkers, replace=True)].astype(np.uint64).copy()
        curr_logp = self._eval_log_prob(curr)
        
        samples = []
        total_acc = 0.0
        
        # --- 3. Sampling Loop ---
        iterator = tqdm(range(n_steps), desc="MCMC", unit="step", disable=not verbose)
        for step in iterator:
            # A. Propose
            prop = apply_excitation_numpy(curr, self.n_orb, rng, p_double)
            
            # B. Check Constraints (S ∪ C)
            prop_view = np.ascontiguousarray(prop).view(void_dt).ravel()
            # Vectorized membership check (assumes valid_view is sorted)
            is_valid = np.isin(prop_view, valid_view, assume_unique=True)
            
            # Revert invalid proposals immediately
            prop[~is_valid] = curr[~is_valid]
            
            # C. Evaluate & Metropolis
            prop_logp = self._eval_log_prob(prop)
            ratio = prop_logp - curr_logp
            
            # Accept logic: u < ratio
            accept = np.log(rng.random(n_walkers)) < ratio
            # Ensure invalid moves (which were reverted) are counted as rejections
            accept[~is_valid] = False 
            
            # Update state
            curr[accept] = prop[accept]
            curr_logp[accept] = prop_logp[accept]
            
            # Stats
            total_acc += accept.mean()
            if verbose and step % 100 == 0: 
                iterator.set_postfix(acc=f"{accept.mean():.1%}")
            
            # Collect
            if step >= burn_in and (step - burn_in) % thinning == 0:
                samples.append(curr.copy())

        # --- 4. Analysis (Unique-First Optimization) ---
        avg_acc = total_acc / max(n_steps, 1)
        all_samples = np.concatenate(samples) if samples else np.zeros((0, 2), dtype=np.uint64)
        n_total = len(all_samples)
        
        if n_total == 0: 
            return MCMCResult(np.nan, np.nan, np.nan, avg_acc, 0)

        # Deduplicate for expensive E_L computation
        if verbose: print(f"  Analysis: {n_total} raw samples. Deduplicating...")
        unique_dets, inverse_idx = np.unique(all_samples, axis=0, return_inverse=True)
        
        if verbose:
            ratio = n_total / len(unique_dets)
            print(f"  Unique Determinants: {len(unique_dets)} (Compression: {ratio:.1f}x)")

        # Compute E_loc for unique states
        unique_elocs = []
        iterator_eloc = tqdm(
            range(0, len(unique_dets), self.conn_bs), 
            desc="E_loc", 
            disable=not verbose
        )
        for i in iterator_eloc:
            chunk = unique_dets[i : i + self.conn_bs]
            unique_elocs.append(self._compute_batch_eloc(chunk, use_heatbath, screen_eps))
        
        # Map back to full chain
        full_elocs = np.concatenate(unique_elocs)[inverse_idx]
        e_real = np.real(full_elocs) + self.int_ctx.get_e_nuc()
        
        # Statistical Analysis
        mean, err = blocking_analysis(e_real)
        var = np.var(e_real, ddof=1) if n_total > 1 else 0.0

        return MCMCResult(mean, err, var, avg_acc, n_total)

__all__ = ["MCMCResult", "PostMCMC"]