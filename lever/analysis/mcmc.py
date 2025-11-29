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
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from lever.core import IntCtx, get_local_connections
from lever.engine.features import dets_to_features

if TYPE_CHECKING:
    from lever.config import LeverConfig
    from lever.models import WavefunctionModel

Array = jax.Array


@dataclass(frozen=True)
class MCMCResult:
    """Container for VMC energy statistics."""
    mean: float
    error: float
    variance: float
    acc_rate: float
    n_samples: int


def blocking_analysis(data: np.ndarray) -> tuple[float, float]:
    """
    Estimate mean and standard error via Flyvbjerg-Petersen blocking method.
    
    Algorithm: Recursive block averaging with power-of-2 truncation.
    Error estimate: σ = max(σ_k) over all blocking levels k.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n < 2:
        return float(x[0]) if n == 1 else (np.nan, np.nan)

    # Truncate to nearest power of 2
    n_pow2 = 1 << (n.bit_length() - 1)
    if n_pow2 < 2:
        return np.mean(x), np.nan
    
    curr = x[:n_pow2].copy()
    mean = float(curr.mean())
    best_stderr = np.sqrt(np.var(curr, ddof=1) / curr.size)

    while curr.size >= 4:
        # Block averaging: x'_i = 0.5 * (x_{2i} + x_{2i+1})
        curr = 0.5 * (curr[0::2] + curr[1::2])
        err = np.sqrt(np.var(curr, ddof=1) / curr.size)
        best_stderr = max(best_stderr, err)

    return mean, float(best_stderr)


def apply_excitation_numpy(
    dets: np.ndarray, 
    n_orb: int, 
    rng: np.random.Generator, 
    p_double: float
) -> np.ndarray:
    """
    Apply symmetric single/double excitations via vectorized bitwise operations.
    
    Algorithm:
      - Single excitation: |i⟩ → |a⟩ (one electron hop)
      - Double excitation: two sequential single excitations
      - Updates occupancy masks incrementally for efficiency
    """
    n_w = dets.shape[0]
    new_dets = dets.copy()
    
    # Select move type: single vs double excitation
    is_double = rng.random(n_w) < p_double
    
    # Precompute occupancy masks
    idx_rng = np.arange(n_orb, dtype=np.uint64)
    alpha, beta = new_dets[:, 0], new_dets[:, 1]
    occ_a = ((alpha[:, None] >> idx_rng) & 1).astype(bool)
    occ_b = ((beta[:, None] >> idx_rng) & 1).astype(bool)

    def _do_single_excitation(mask: np.ndarray) -> None:
        """Apply single excitation to masked walkers."""
        if not np.any(mask):
            return

        # Filter valid spin channels
        valid_a = (occ_a[mask].any(axis=1) & (~occ_a[mask]).any(axis=1))
        valid_b = (occ_b[mask].any(axis=1) & (~occ_b[mask]).any(axis=1))
        
        # Select spin channel (alpha/beta)
        n_active = np.sum(mask)
        choose_a = rng.random(n_active) < 0.5
        choose_a = np.where(valid_a & valid_b, choose_a, valid_a)
        
        # Target occupancy patterns
        t_occ = np.where(choose_a[:, None], occ_a[mask], occ_b[mask])
        t_virt = np.where(choose_a[:, None], ~occ_a[mask], ~occ_b[mask])
        
        def sample_idx(m: np.ndarray) -> np.ndarray:
            """Sample occupied/virtual orbital indices."""
            r = rng.random(m.shape, dtype=np.float32)
            return np.argmax(np.where(m, r, -1.0), axis=1).astype(np.uint64)

        idx_i, idx_a = sample_idx(t_occ), sample_idx(t_virt)
        
        # Apply bit flips: |i⟩ → |a⟩
        flip = (np.uint64(1) << idx_i) | (np.uint64(1) << idx_a)
        
        # Map indices back to full array
        active_indices = np.flatnonzero(mask)
        idx_update_a = active_indices[choose_a]
        idx_update_b = active_indices[~choose_a]
        
        # Update bitstrings and occupancy masks
        new_dets[idx_update_a, 0] ^= flip[choose_a]
        new_dets[idx_update_b, 1] ^= flip[~choose_a]
        
        # Incremental mask updates
        if len(idx_update_a) > 0:
            occ_a[idx_update_a, idx_i[choose_a]] = False
            occ_a[idx_update_a, idx_a[choose_a]] = True
        if len(idx_update_b) > 0:
            occ_b[idx_update_b, idx_i[~choose_a]] = False
            occ_b[idx_update_b, idx_a[~choose_a]] = True

    # Apply excitations
    _do_single_excitation(np.ones(n_w, dtype=bool))  # All walkers: single
    _do_single_excitation(is_double)  # Subset: double (second excitation)

    return new_dets


class PostMCMC:
    """
    Hybrid CPU/GPU engine for VMC sampling and energy estimation.
    
    Architecture:
      - CPU: MCMC proposal and constraint checking
      - GPU: Neural network inference for Ψ(x) evaluation
      - Memory: Unique-first evaluation minimizes expensive E_L computations
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

        # Compile JAX evaluators: log_prob = 2 * Re[log(Ψ)]
        @jax.jit
        def _psi(dets: np.ndarray) -> jnp.ndarray:
            return jnp.exp(model.log_psi(params, dets_to_features(dets, self.n_orb)))
        
        @jax.jit
        def _logp(dets: np.ndarray) -> jnp.ndarray:
            return 2.0 * jnp.real(model.log_psi(params, dets_to_features(dets, self.n_orb)))

        self._jit_psi = _psi
        self._jit_logp = _logp

    def _eval_log_prob(self, dets: np.ndarray) -> np.ndarray:
        """Evaluate log-probability on GPU."""
        return np.array(self._jit_logp(dets), copy=True)  # copy=True prevents read-only errors

    def _eval_psi(self, dets: np.ndarray) -> np.ndarray:
        """Batched evaluation of wavefunction amplitudes."""
        n = len(dets)
        if n == 0:
            return np.array([], dtype=np.complex128)
        
        res = []
        for i in range(0, n, self.bs):
            batch = dets[i:i + self.bs]
            res.append(self._jit_psi(batch))
            
        return np.array(jnp.concatenate(res), copy=True)

    def _compute_batch_eloc(self, samples: np.ndarray, hb: bool, eps: float) -> np.ndarray:
        """
        Compute local energy E_L(x) = Σ_{x'} H_{xx'} Ψ(x')/Ψ(x).
        
        Algorithm:
          1. Get Hamiltonian connections for all samples
          2. Evaluate Ψ(x') for all connected states
          3. Contract numerator: Σ H_{xx'} Ψ(x')
          4. Divide by Ψ(x) with numerical safety
        """
        n = len(samples)
        if n == 0:
            return np.array([])

        # Get Hamiltonian connections
        conn = get_local_connections(samples, self.int_ctx, self.n_orb, hb, eps)
        kets = conn["dets"].astype(np.uint64).reshape(-1, 2)
        vals = conn["values"]
        offsets = conn["offsets"]

        # Evaluate wavefunction for all unique determinants
        all_dets = np.vstack((samples, kets))
        uniq, inv = np.unique(all_dets, axis=0, return_inverse=True)
        psi_uniq = self._eval_psi(uniq)
        
        psi_samp = psi_uniq[inv[:n]]
        psi_kets = psi_uniq[inv[n:]]

        # Contract E_L numerator
        num = np.zeros(n, dtype=np.complex128)
        row_idx = np.repeat(np.arange(n), np.diff(offsets))
        np.add.at(num, row_idx, vals * psi_kets)

        # Safe division: E_L(x) = numerator / Ψ(x)
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
          1. Construct allowed space S ∪ C
          2. Run Metropolis-Hastings, rejecting moves outside S ∪ C
          3. Deduplicate resulting chain
          4. Compute E_L for unique states and map back
        
        Mathematical:
          E_VMC = ⟨E_L⟩ = 1/N Σ_{x ~ |Ψ|²} E_L(x)
          E_L(x) = Σ_{x'} H_{xx'} Ψ(x')/Ψ(x)
        """
        if verbose:
            print(f"Starting Post-MCMC (Restricted S∪C) | Walkers={n_walkers} Steps={n_steps}")

        # Build valid space S ∪ C
        if verbose:
            print("  Constructing valid space map...")
        conn = get_local_connections(s_dets, self.int_ctx, self.n_orb, use_heatbath, screen_eps)
        c_dets = conn["dets"].astype(np.uint64).reshape(-1, 2)
        valid_dets = np.unique(np.vstack((s_dets, c_dets)), axis=0)
        
        # Efficient lookup using raw bytes
        void_dt = np.dtype((np.void, 16))
        valid_view = np.ascontiguousarray(valid_dets).view(void_dt).ravel()
        valid_view.sort()  # Sort for binary search in np.isin
        
        if verbose:
            print(f"  Valid Space Size: {len(valid_view)} (S={len(s_dets)})")

        # Initialize walkers from S-space
        rng = np.random.default_rng(seed)
        curr = s_dets[rng.choice(len(s_dets), n_walkers, replace=True)].astype(np.uint64).copy()
        curr_logp = self._eval_log_prob(curr)
        
        samples = []
        total_acc = 0.0
        
        # MCMC sampling loop
        iterator = tqdm(range(n_steps), desc="MCMC", unit="step", disable=not verbose)
        for step in iterator:
            # Propose new states
            prop = apply_excitation_numpy(curr, self.n_orb, rng, p_double)
            
            # Constraint check: x ∈ S ∪ C
            prop_view = np.ascontiguousarray(prop).view(void_dt).ravel()
            is_valid = np.isin(prop_view, valid_view, assume_unique=True)
            prop[~is_valid] = curr[~is_valid]  # Revert invalid proposals
            
            # Metropolis acceptance
            prop_logp = self._eval_log_prob(prop)
            ratio = prop_logp - curr_logp
            accept = np.log(rng.random(n_walkers)) < ratio
            accept[~is_valid] = False  # Force reject invalid moves
            
            # Update state
            curr[accept] = prop[accept]
            curr_logp[accept] = prop_logp[accept]
            
            # Statistics
            total_acc += accept.mean()
            if verbose and step % 100 == 0: 
                iterator.set_postfix(acc=f"{accept.mean():.1%}")
            
            # Collect samples (post burn-in, with thinning)
            if step >= burn_in and (step - burn_in) % thinning == 0:
                samples.append(curr.copy())

        # Statistical analysis
        avg_acc = total_acc / max(n_steps, 1)
        all_samples = np.concatenate(samples) if samples else np.zeros((0, 2), dtype=np.uint64)
        n_total = len(all_samples)
        
        if n_total == 0: 
            return MCMCResult(np.nan, np.nan, np.nan, avg_acc, 0)

        # Unique-first evaluation strategy
        if verbose:
            print(f"  Analysis: {n_total} raw samples. Deduplicating...")
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
            chunk = unique_dets[i:i + self.conn_bs]
            unique_elocs.append(self._compute_batch_eloc(chunk, use_heatbath, screen_eps))
        
        # Map back to full chain and add nuclear energy
        full_elocs = np.concatenate(unique_elocs)[inverse_idx]
        e_real = np.real(full_elocs) + self.int_ctx.get_e_nuc()
        
        # Statistical analysis with blocking method
        mean, err = blocking_analysis(e_real)
        var = np.var(e_real, ddof=1) if n_total > 1 else 0.0

        return MCMCResult(mean, err, var, avg_acc, n_total)


__all__ = ["MCMCResult", "PostMCMC"]
