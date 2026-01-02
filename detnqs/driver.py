# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic optimization drivers for detnqs variational framework.

Computational modes:
  - Variational: E[ψ_S] = <ψ_S|H_SS|ψ_S> / <ψ_S|ψ_S>
  - Effective:   E_eff[ψ_S] via Löwdin downfolding
                 H_eff = H_SS + H_SC · (E_ref - H_CC + ε·I)^(-1) · H_CS
  - Proxy:       E[ψ_T] on T-space with diagonal C-block approx
  - Asymmetric:  E_asym = <ψ|P_S H|ψ> / <ψ|P_S|ψ>

Architecture:
  L0 (C++):    Static integrals, Heat-Bath tables, COO assembly
  L1 (Python): Outer loop - space evolution, Hamiltonian rebuild
  L2 (JAX):    Inner loop - JIT optimization sweep

Memory strategy:
  - Variational/Effective: GPU holds only S-space batch, stream T for scoring
  - Proxy/Asymmetric: GPU holds full T-space batch

File: detnqs/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core
from .operator import functional as func_mod
from .operator import hamiltonian as ham_mod
from .operator import kernel as kern_mod
from .space import DetSpace, Selector
from .state import State
from .system import MolecularSystem


@dataclass
class BaseDriver:
    """
    Base driver for deterministic variational optimization.

    Outer loop workflow:
      1. Build Hamiltonian for current S/C spaces
      2. Execute JIT-compiled inner optimization loop
      3. Compute norm decomposition ||ψ_S||² and ||ψ_C||² on T-space
      4. Trigger callbacks with diagnostic statistics
      5. Update S-space via amplitude-based selection
      6. Check convergence on energy delta

    Inner loop workflow:
      - Evaluate ψ(x; θ) and compute E[θ], ∇_θ E
      - Update parameters: θ ← optimizer(θ, ∇_θ E)
      - Check convergence: δE < tol for patience consecutive steps

    Subclass requirements:
      - mode_tag(): Return mode identifier string
      - device_space(): Return "S" or "T" for GPU memory residency
      - build_hamiltonian(): Construct Hamiltonian and SpMV operator
    """

    system: MolecularSystem
    state: State
    detspace: DetSpace
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    selector: Selector

    max_outer: int = 10
    max_inner: int = 500

    inner_tol: float = 1e-8
    inner_patience: int = 100
    outer_tol: float = 1e-6
    outer_patience: int = 5

    chunk_size: int | None = None

    def device_space(self) -> str:
        """
        Which space is kept as State.batch on device: 'S' or 'T'.
        
        Variational/Effective: 'S' to minimize GPU memory
        Proxy/Asymmetric: 'T' for full-space optimization
        """
        return "T"

    @classmethod
    def build(
        cls,
        system: MolecularSystem,
        detspace: DetSpace,
        model,
        optimizer: optax.GradientTransformation,
        selector: Selector,
        *,
        key: jax.Array | None = None,
        **driver_kwargs,
    ) -> "BaseDriver":
        """
        Initialize driver with model and optimizer.

        Args:
            system: Molecular system from FCIDUMP
            detspace: Initial determinant space
            model: Uninitialized Flax model
            optimizer: Optax optimizer
            selector: Determinant selector
            key: JAX PRNG key
            **driver_kwargs: Additional driver parameters

        Returns:
            Initialized driver instance
        """
        state = State.init(system=system, detspace=detspace, model=model, key=key)
        opt_state = optimizer.init(state.params)

        return cls(
            system=system,
            state=state,
            detspace=detspace,
            optimizer=optimizer,
            opt_state=opt_state,
            selector=selector,
            **driver_kwargs,
        )

    def mode_tag(self) -> str:
        """Return mode identifier for logging."""
        return "base"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """
        Build Hamiltonian and SpMV operator.

        Returns:
            (ham, op): Hamiltonian object and operator function
        """
        raise NotImplementedError

    def _compute_screening_weights(self) -> np.ndarray | None:
        """
        Compute normalized sqrt(p_i) on S-space for Heat-Bath screening.

        Returns:
            sqrt_prob: sqrt(|ψ_i|² / Σ|ψ_j|²) for each S-space determinant
        """
        if self.detspace.size_S == 0:
            return None

        indices = jnp.asarray(self.detspace.S_indices, dtype=jnp.int32)
        sign_s, logabs_s = self.state.forward(indices, chunk_size=self.chunk_size)

        logabs_s = jnp.real(logabs_s)
        logabs_s = logabs_s - jnp.max(logabs_s)

        prob_s = jnp.exp(2.0 * logabs_s)
        prob_s = prob_s / jnp.sum(prob_s)

        return np.asarray(jnp.sqrt(prob_s), dtype=np.float64)

    def _compute_norms(self) -> tuple[float, float]:
        """
        Compute norm decomposition on T-space: ||ψ_S||² and ||ψ_C||².

        Uses streaming H2D if State.batch != T (Variational/Effective modes)
        to avoid VRAM overflow when |C| >> |S|.

        Returns:
            (norm_s, norm_c): S-space and C-space norms
        """
        n_s = self.detspace.size_S
        n_c = self.detspace.size_C

        # Fast path: State.batch already equals T
        if self.state.n_det == self.detspace.size_T:
            _sign_t, logabs_t = self.state.forward(chunk_size=self.chunk_size)
            logabs_t = jnp.real(logabs_t)
            shift = jnp.max(logabs_t) if logabs_t.size > 0 else 0.0
            w = jnp.exp(2.0 * (logabs_t - shift))  # |psi|^2 up to common scale
            w_s = w[:n_s]
            w_c = w[n_s : n_s + n_c] if n_c > 0 else jnp.array([])
            return float(jnp.sum(w_s)), float(jnp.sum(w_c)) if n_c > 0 else 0.0

        # Streaming path: iterate over CPU det blocks, H2D a fixed block at a time
        block_size = int(self.chunk_size or 8192)

        def scores_factory():
            for det_blk in self.detspace.iter_T(block_size):
                _s, la = self.state.forward_dets(det_blk, block_size=block_size, chunk_size=self.chunk_size)
                yield np.asarray(np.real(la), dtype=np.float64), det_blk

        # Pass 1: global shift = max(log|psi|)
        shift = -np.inf
        for la_blk, _ in scores_factory():
            if la_blk.size:
                shift = max(shift, float(np.max(la_blk)))
        if not np.isfinite(shift):
            return 0.0, 0.0

        # Pass 2: accumulate norms using |psi|^2 = exp(2*(log|psi|-shift))
        norm_s = 0.0
        norm_c = 0.0
        seen = 0
        for la_blk, _ in scores_factory():
            w = np.exp(2.0 * (la_blk - shift))
            m = int(w.shape[0])
            if seen < n_s:
                take_s = min(n_s - seen, m)
                norm_s += float(np.sum(w[:take_s]))
                if m > take_s:
                    norm_c += float(np.sum(w[take_s:]))
            else:
                norm_c += float(np.sum(w))
            seen += m

        return norm_s, norm_c

    def evolve_space(self) -> None:
        """
        Update S-space via importance sampling on log|ψ_T|.
        
        Uses streaming H2D if selector.stream=True or State.batch != T
        to avoid materializing full T in VRAM/RAM.
        """
        block_size = int(self.chunk_size or 8192)
        use_stream = bool(getattr(self.selector, "stream", False)) or (self.state.n_det != self.detspace.size_T)

        if not use_stream:
            # Fast path: State.batch already equals T
            _sign_t, logabs_t = self.state.forward(chunk_size=self.chunk_size)
            logabs_t = jnp.real(logabs_t)
            log_amp_host = np.asarray(logabs_t, dtype=np.float64)
            new_space = self.detspace.evolve(self.selector, log_amp_host)
        else:
            # Streaming path: provide factory for multi-pass selectors
            def scores_factory():
                for det_blk in self.detspace.iter_T(block_size):
                    _s, la = self.state.forward_dets(det_blk, block_size=block_size, chunk_size=self.chunk_size)
                    yield np.asarray(np.real(la), dtype=np.float64), det_blk

            new_space = self.detspace.evolve(self.selector, scores_factory)

        self.detspace = new_space
        self.state = self.state.update_space(new_space, device_space=self.device_space())

    def _check_convergence(self, energy_history: list[float]) -> bool:
        """
        Check outer convergence via sliding window.

        Args:
            energy_history: Energy sequence from consecutive outer steps

        Returns:
            True if max(|ΔE|) < tol over patience consecutive steps
        """
        if self.outer_patience <= 0 or len(energy_history) <= self.outer_patience:
            return False

        window = energy_history[-(self.outer_patience + 1) :]
        deltas = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
        max_delta = max(deltas)

        return max_delta < self.outer_tol

    def _build_sweep(self, ham: Any, op: Callable) -> Callable:
        """Build JIT-compiled inner optimization loop."""
        mode = self.mode_tag()
        detspace = self.detspace
        chunk_size = self.chunk_size
        eps = 1e-12

        energy_step = func_mod.make_energy_step(
            ham=ham,
            op=op,
            detspace=detspace,
            mode=mode,
            eps=eps,
            chunk_size=chunk_size,
        )

        def sweep(state, opt_state):
            """Execute inner loop until convergence or max steps."""

            def single_step(state, opt_state):
                energy_val, grad = energy_step(state)
                state, opt_state = state.apply_gradients(
                    gradients=grad,
                    opt_state=opt_state,
                    optimizer=self.optimizer,
                )
                return state, opt_state, energy_val

            single_step = jax.jit(single_step, donate_argnums=(0, 1))

            energy_trace = []
            last_energy = float("inf")
            streak = 0

            for step in range(self.max_inner):
                state, opt_state, energy_val = single_step(state, opt_state)
                energy = float(energy_val)
                energy_trace.append(energy)

                delta = abs(energy - last_energy)
                if delta < self.inner_tol:
                    streak += 1
                else:
                    streak = 0

                if self.inner_tol > 0 and self.inner_patience > 0:
                    if streak >= self.inner_patience:
                        break

                last_energy = energy

            return state, opt_state, np.array(energy_trace)

        return sweep

    def post_inner_sweep(self, energy_trace: np.ndarray) -> None:
        """Hook for mode-specific post-processing after inner loop."""
        pass

    def run(self, callbacks: list[Any] | None = None) -> None:
        """
        Execute outer loop over determinant-space evolution.

        Args:
            callbacks: Observer callbacks with interface:
                    - on_run_start(driver)
                    - on_outer_end(step, stats, driver)
                    - on_run_end(driver)
        
        Note:
            Final detspace retains both S and C for post-analysis.
        """
        callbacks = callbacks or []
        start_time = time.perf_counter()

        for cb in callbacks:
            if hasattr(cb, "on_run_start"):
                cb.on_run_start(self)

        energy_history = []

        for outer in range(self.max_outer):
            # L1: Rebuild Hamiltonian for current space
            ham, op = self.build_hamiltonian()
            inner_sweep = self._build_sweep(ham, op)

            # L2: JIT-compiled inner loop
            self.state, self.opt_state, energy_trace = inner_sweep(
                self.state, self.opt_state
            )

            self.post_inner_sweep(energy_trace)

            # Compute diagnostics
            norm_s, norm_c = self._compute_norms()

            if len(energy_trace) > 0:
                last_e = float(energy_trace[-1])
                elapsed = time.perf_counter() - start_time

                stats = {
                    "outer_step": outer,
                    "energy": last_e,
                    "norm_s": norm_s,
                    "norm_c": norm_c,
                    "size_s": self.detspace.size_S,
                    "size_c": self.detspace.size_C,
                    "timestamp": elapsed,
                    "inner_steps": len(energy_trace),
                    "inner_trace": energy_trace.tolist(),
                }

                for cb in callbacks:
                    cb.on_outer_end(outer, stats, self)

                energy_history.append(last_e)
                
                # Check convergence before space evolution
                if self._check_convergence(energy_history):
                    break

            # Update S-space for next outer step (skip on last iteration)
            if outer < self.max_outer - 1:
                self.evolve_space()

        for cb in callbacks:
            cb.on_run_end(self)


@dataclass
class VariationalDriver(BaseDriver):
    """
    Variational mode: E = <ψ_S|H_SS|ψ_S> / <ψ_S|ψ_S> on S-space.

    Workflow:
      1. Construct H_SS on current S-space
      2. Optimize parameters to minimize E[ψ_S]
      3. Generate complement C via Heat-Bath screening
      4. Update S ← select from T = S ∪ C
      
    Memory: GPU holds only S-space batch, streams T for scoring/norms
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6

    def mode_tag(self) -> str:
        return "variational"

    def device_space(self) -> str:
        return "S"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build H_SS with screened complement C-space."""
        if self.screening == "none":
            C_dets = np.zeros((0, 2), dtype=np.uint64)
        else:
            psi_s = None
            if self.screening == "dynamic":
                psi_s = self._compute_screening_weights()

            S = np.ascontiguousarray(self.detspace.S_dets, dtype=np.uint64)
            C_dets = core.gen_complement_dets(
                ref_dets=S,
                n_orb=self.system.n_orb,
                int_ctx=self.system.int_ctx,
                psi_S=psi_s,
                mode=self.screening,
                eps1=self.screen_eps,
            )

        self.detspace = DetSpace(S_dets=self.detspace.S_dets, C_dets=C_dets)
        self.state = self.state.update_space(self.detspace, device_space=self.device_space())

        ham = ham_mod.build_ss_hamiltonian(self.system, self.detspace)
        op = kern_mod.build_ss_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


@dataclass
class EffectiveDriver(BaseDriver):
    """
    Effective mode: Löwdin downfolded Hamiltonian in S-space.

    Effective Hamiltonian:
      H_eff = H_SS + H_SC · (E_ref - H_CC + ε·I)^(-1) · H_CS

    Regularization:
      - sigma: Constant shift ε·I
      - linear_shift: max(0, E_ref - diag(H_CC))

    Workflow:
      1. Build proxy H and complement C
      2. Compute H_eff via Löwdin partitioning
      3. Optimize E_eff = <ψ_S|H_eff|ψ_S>
      4. Update E_ref ← last energy, S ← select from T
      
    Memory: GPU holds only S-space batch, streams T for scoring/norms
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6
    reg_type: str = "sigma"
    epsilon: float = 1e-12
    e_ref: float = 0.0

    def mode_tag(self) -> str:
        return "effective"

    def device_space(self) -> str:
        return "S"

    def post_inner_sweep(self, energy_trace: np.ndarray) -> None:
        """Update reference energy E_ref for next Hamiltonian rebuild."""
        if len(energy_trace) > 0:
            self.e_ref = float(energy_trace[-1])

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build H_eff via Löwdin downfolding."""
        psi_s = None
        if self.screening == "dynamic":
            psi_s = self._compute_screening_weights()

        proxy_ham, C_dets = ham_mod.build_proxy_hamiltonian(
            system=self.system,
            space=self.detspace,
            screening=self.screening,
            psi_s=psi_s,
            screen_eps=self.screen_eps,
        )

        self.detspace = DetSpace(S_dets=self.detspace.S_dets, C_dets=C_dets)
        self.state = self.state.update_space(self.detspace, device_space=self.device_space())

        ham = ham_mod.build_effective_hamiltonian(
            system=self.system,
            proxy_ham=proxy_ham,
            e_ref=self.e_ref,
            reg_type=self.reg_type,
            epsilon=self.epsilon,
        )

        op = kern_mod.build_ss_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


@dataclass
class ProxyDriver(BaseDriver):
    """
    Proxy mode: Full T-space optimization with diagonal C-block.

    Hamiltonian blocks:
      H = [[H_SS,       H_SC      ],
           [H_CS,  diag(H_CC) + δ·I]]

    Workflow:
      1. Build proxy H on T = S ∪ C with screening
      2. Optimize E = <ψ_T|H|ψ_T> / <ψ_T|ψ_T>
      3. Update S ← select from T
      
    Memory: GPU holds full T-space batch
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6
    diag_shift: float = 0.0

    def mode_tag(self) -> str:
        return "proxy"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build full proxy Hamiltonian H on T-space."""
        psi_s = None
        if self.screening == "dynamic":
            psi_s = self._compute_screening_weights()

        ham, C_dets = ham_mod.build_proxy_hamiltonian(
            system=self.system,
            space=self.detspace,
            screening=self.screening,
            psi_s=psi_s,
            screen_eps=self.screen_eps,
            diag_shift=self.diag_shift,
        )

        self.detspace = DetSpace(S_dets=self.detspace.S_dets, C_dets=C_dets)
        self.state = self.state.update_space(self.detspace, device_space=self.device_space())

        op = kern_mod.build_proxy_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


@dataclass
class AsymmetricDriver(BaseDriver):
    """
    Asymmetric mode: VMC-style truncated estimator on S-space.

    Energy estimator:
        E_asym = <ψ|P_S H|ψ> / <ψ|P_S|ψ>

    where P_S projects onto S-space.

    Workflow:
      - Uses proxy Hamiltonian H_T same as ProxyDriver
      - Inner loop minimizes asymmetric Rayleigh quotient
      - S-space evolution uses full-T amplitudes |ψ_T|
      
    Memory: GPU holds full T-space batch
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6

    def mode_tag(self) -> str:
        return "asymmetric"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build proxy Hamiltonian H_T on T-space."""
        psi_s = None
        if self.screening == "dynamic":
            psi_s = self._compute_screening_weights()

        ham, C_dets = ham_mod.build_proxy_hamiltonian(
            system=self.system,
            space=self.detspace,
            screening=self.screening,
            psi_s=psi_s,
            screen_eps=self.screen_eps,
        )

        self.detspace = DetSpace(S_dets=self.detspace.S_dets, C_dets=C_dets)
        self.state = self.state.update_space(self.detspace, device_space=self.device_space())

        op = kern_mod.build_proxy_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


__all__ = [
    "BaseDriver",
    "VariationalDriver",
    "EffectiveDriver",
    "ProxyDriver",
    "AsymmetricDriver",
]

