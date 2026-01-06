# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic optimization drivers for DetNQS variational framework.

Computational modes:
  - Variational: E[ψ_V] = <ψ_V|H_VV|ψ_V> / <ψ_V|ψ_V>
  - Effective:   E_eff[ψ_V] via Löwdin downfolding
                 H_eff = H_VV + H_VP · (E_ref - H_PP + ε·I)^(-1) · H_PV
  - Proxy:       E[ψ_T] on T-space with diagonal P-block approximation
  - Asymmetric:  E_asym = <ψ|P_V H|ψ> / <ψ|P_V|ψ>

Architecture:
  L0 (C++):    Static integrals, Heat-Bath tables, COO assembly
  L1 (Python): Outer loop - space evolution, Hamiltonian rebuild
  L2 (JAX):    Inner loop - JIT optimization sweep

Memory strategy:
  - Variational/Effective: GPU holds only V-space batch, stream T for scoring
  - Proxy/Asymmetric: GPU holds full T-space batch

File: detnqs/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: January, 2026
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
      1. Build Hamiltonian for current V/P spaces
      2. Execute JIT-compiled inner optimization loop
      3. Compute norm decomposition ||ψ_V||^2 and ||ψ_P||^2 on T-space
      4. Trigger callbacks with diagnostic statistics
      5. Update V-space via amplitude-based selection

    Inner loop workflow:
      - Evaluate ψ(x; θ) and compute E[θ], ∇_θ E
      - Update parameters: θ ← optimizer(θ, ∇_θ E)

    Subclass requirements:
      - mode_tag(): Return mode identifier string
      - device_space(): Return "V" or "T" for GPU memory residency
      - build_hamiltonian(): Construct Hamiltonian and SpMV operator
      - should_skip_inner(outer_step): Return True to skip inner loop
    """

    system: MolecularSystem
    state: State
    detspace: DetSpace
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    selector: Selector

    max_outer: int = 30
    max_inner: int = 1000

    chunk_size: int | None = None
    block_size: int = 4194304
  
    compute_norms: bool = False

    def device_space(self) -> str:
        """Which space is kept as State.batch on device: 'V' or 'T'."""
        return "T"

    def should_skip_inner(self, outer_step: int) -> bool:
        """Return True to skip inner loop at given outer step."""
        return False

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
        Compute normalized sqrt(p_i) on V-space for Heat-Bath screening.

        Returns:
            sqrt_prob: sqrt(|ψ_i|^2 / Σ|ψ_j|^2) for each V-space determinant
        """
        if self.detspace.size_V == 0:
            return None

        indices = jnp.asarray(self.detspace.V_indices, dtype=jnp.int32)
        sign_v, logabs_v = self.state.forward(indices, chunk_size=self.chunk_size)

        logabs_v = jnp.real(logabs_v)
        logabs_v = logabs_v - jnp.max(logabs_v)

        prob_v = jnp.exp(2.0 * logabs_v)
        prob_v = prob_v / jnp.sum(prob_v)

        return np.asarray(jnp.sqrt(prob_v), dtype=np.float64)

    def _compute_norms(self) -> tuple[float, float]:
        """
        Compute norm decomposition on T-space: ||ψ_V||^2 and ||ψ_P||^2.
  
        Returns (0.0, 0.0) if compute_norms=False to skip overhead.
  
        Fast path:
        - If State.batch == T, single device forward and weight computation
  
        Streaming path (State.batch != T):
        - Single-pass stable accumulator over T-space blocks
        - Maintains running max and rescales sums for numerical stability
  
        Returns:
            (norm_v, norm_p): Squared norms of V and P components
        """
        if not self.compute_norms:
            return 0.0, 0.0
  
        n_v = int(self.detspace.size_V)
        n_p = int(self.detspace.size_P)
  
        # Fast path: State.batch already equals T
        if self.state.n_det == self.detspace.size_T:
            _sign_t, logabs_t = self.state.forward(chunk_size=self.chunk_size)
            logabs_t = jnp.real(logabs_t)
            shift = jnp.max(logabs_t) if logabs_t.size > 0 else 0.0
            w = jnp.exp(2.0 * (logabs_t - shift))
            w_v = w[:n_v]
            w_p = w[n_v : n_v + n_p] if n_p > 0 else jnp.array([])
            return float(jnp.sum(w_v)), float(jnp.sum(w_p)) if n_p > 0 else 0.0
  
        # Streaming path: online stable exp-sum accumulator
        block_size = int(self.block_size)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
  
        alpha = -np.inf  # Running max of x = 2*log|ψ|
        sum_v = 0.0
        sum_p = 0.0
        seen = 0  # Number of determinants processed in T order
  
        for det_blk in self.detspace.iter_T(block_size):
            _s, la = self.state.forward_dets(
                det_blk,
                block_size=block_size,
                chunk_size=self.chunk_size,
            )
            la = np.asarray(np.real(la), dtype=np.float64)
            if la.size == 0:
                continue
      
            x = 2.0 * la
            x_max = float(np.max(x))
      
            # Update running max and rescale previous sums
            if x_max > alpha:
                if np.isfinite(alpha):
                    scale = float(np.exp(alpha - x_max))
                    sum_v *= scale
                    sum_p *= scale
                alpha = x_max
      
            # Compute safe weights: exp(x - alpha) with (x - alpha) <= 0
            w = np.exp(x - alpha)
      
            m = int(w.shape[0])
            if seen < n_v:
                take_v = min(n_v - seen, m)
                sum_v += float(np.sum(w[:take_v]))
                if m > take_v:
                    sum_p += float(np.sum(w[take_v:]))
            else:
                sum_p += float(np.sum(w))
      
            seen += m
  
        if not np.isfinite(alpha):
            return 0.0, 0.0
  
        return sum_v, sum_p

    def evolve_space(self) -> None:
        """
        Update V-space via importance sampling on log|ψ_T|.
    
        Uses streaming H2D if selector.stream=True or State.batch != T
        to avoid materializing full T in VRAM/RAM.
        """
        block_size = int(self.block_size)
        use_stream = bool(getattr(self.selector, "stream", False)) or (
            self.state.n_det != self.detspace.size_T
        )

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
                    _s, la = self.state.forward_dets(
                        det_blk,
                        block_size=block_size,
                        chunk_size=self.chunk_size,
                    )
                    yield np.asarray(np.real(la), dtype=np.float64), det_blk

            new_space = self.detspace.evolve(self.selector, scores_factory)

        self.detspace = new_space
        self.state = self.state.update_space(
            new_space, device_space=self.device_space()
        )

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
            """Execute inner loop for max_inner steps."""

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
            for _ in range(self.max_inner):
                state, opt_state, energy_val = single_step(state, opt_state)
                energy_trace.append(float(energy_val))

            return state, opt_state, np.array(energy_trace)

        return sweep

    def post_inner_sweep(self, energy_trace: np.ndarray) -> None:
        """Hook for mode-specific post-processing after inner loop."""
        pass

    def run(self, callbacks: list[Any] | None = None) -> None:
        """
        Execute outer loop over determinant-space evolution.

        Timing breakdown per outer iteration:
        - compile_time: Hamiltonian build + JIT compilation
        - inner_time: Parameter optimization loop
        - overhead_time: Post-processing + space evolution (includes evolve_space)
        - total_time: Cumulative runtime from start

        Args:
            callbacks: Observer callbacks with interface:
                    - on_run_start(driver)
                    - on_outer_end(step, stats, driver)
                    - on_run_end(driver)

        Note:
            All energies in stats are total energies (E_total = E_elec + E_nuc)
            Space sizes (size_v, size_p) reflect the spaces used in current iteration
            Final detspace retains both V and P for post-analysis
        """
        callbacks = callbacks or []
        total_start = time.perf_counter()

        for cb in callbacks:
            if hasattr(cb, "on_run_start"):
                cb.on_run_start(self)

        self._last_outer_step = -1
        self._last_stats = {}

        for outer in range(self.max_outer):
            self._last_outer_step = outer

            # Phase 1: Compile - Hamiltonian build + JIT compilation
            compile_start = time.perf_counter()
            ham, op = self.build_hamiltonian()
          
            skip_inner = self.should_skip_inner(outer)
            if not skip_inner:
                inner_sweep = self._build_sweep(ham, op)
            compile_time = time.perf_counter() - compile_start

            # Phase 2: Inner optimization loop (skip if requested)
            if skip_inner:
                energy_trace_elec = np.array([])
                inner_time = 0.0
            else:
                inner_start = time.perf_counter()
                self.state, self.opt_state, energy_trace_elec = inner_sweep(
                    self.state, self.opt_state
                )
                jax.block_until_ready(self.state.params)
                inner_time = time.perf_counter() - inner_start

            # Phase 3: Overhead - post-processing + diagnostics + space evolution
            overhead_start = time.perf_counter()

            self.post_inner_sweep(energy_trace_elec)
            energy_trace_total = energy_trace_elec + self.system.e_nuc
            norm_v, norm_p = self._compute_norms()

            # Save current iteration's space sizes before evolve_space modifies them
            current_size_v = int(self.detspace.size_V)
            current_size_p = int(self.detspace.size_P)

            # Update V-space for next iteration (unless last step)
            if outer < self.max_outer - 1:
                self.evolve_space()

            overhead_time = time.perf_counter() - overhead_start
            total_time = time.perf_counter() - total_start

            # Assemble statistics using current iteration's space sizes
            last_e = float(energy_trace_total[-1]) if len(energy_trace_total) > 0 else 0.0

            stats = {
                "outer_step": outer,
                "energy": last_e,
                "norm_v": norm_v,
                "norm_p": norm_p,
                "size_v": current_size_v,
                "size_p": current_size_p,
                "inner_steps": len(energy_trace_total),
                "inner_trace": energy_trace_total.tolist(),
                "compile_time": compile_time,
                "inner_time": inner_time,
                "overhead_time": overhead_time,
                "total_time": total_time,
            }

            self._last_stats = stats

            for cb in callbacks:
                cb.on_outer_end(outer, stats, self)

        for cb in callbacks:
            cb.on_run_end(self)


@dataclass
class VariationalDriver(BaseDriver):
    """
    Variational mode: E = <ψ_V|H_VV|ψ_V> / <ψ_V|ψ_V> on V-space.

    Workflow:
      1. [outer=0] Generate complement P via Heat-Bath screening, skip inner loop
      2. [outer≥1] Construct H_VV, optimize parameters, update V
  
    Memory: GPU holds only V-space batch, streams T for scoring/norms
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6

    def mode_tag(self) -> str:
        return "variational"

    def device_space(self) -> str:
        return "V"

    def should_skip_inner(self, outer_step: int) -> bool:
        """Skip inner loop at first iteration (outer=0)."""
        return outer_step == 0

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build H_VV with screened complement P-space."""
        if self.screening == "none":
            P_dets = np.zeros((0, 2), dtype=np.uint64)
        else:
            psi_v = None
            if self.screening == "dynamic":
                psi_v = self._compute_screening_weights()

            V = np.ascontiguousarray(self.detspace.V_dets, dtype=np.uint64)
            P_dets = core.gen_perturbative_dets(
                ref_dets=V,
                n_orb=self.system.n_orb,
                int_ctx=self.system.int_ctx,
                psi_v=psi_v,
                mode=self.screening,
                eps1=self.screen_eps,
            )

        self.detspace = DetSpace(V_dets=self.detspace.V_dets, P_dets=P_dets)
        self.state = self.state.update_space(
            self.detspace, device_space=self.device_space()
        )

        ham = ham_mod.build_vv_hamiltonian(self.system, self.detspace)
        op = kern_mod.build_vv_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


@dataclass
class EffectiveDriver(BaseDriver):
    """
    Effective mode: Löwdin downfolded Hamiltonian in V-space.

    Effective Hamiltonian:
      H_eff = H_VV + H_VP · (E_ref - H_PP + ε·I)^(-1) · H_PV

    Regularization:
      - sigma: Constant shift ε·I
      - linear_shift: max(0, E_ref - diag(H_PP))

    Workflow:
      1. [outer=0] Build proxy H and complement P, skip inner loop
      2. [outer≥1] Compute H_eff, optimize E_eff, update E_ref and V
  
    Memory: GPU holds only V-space batch, streams T for scoring/norms
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6
    reg_type: str = "sigma"
    epsilon: float = 1e-12
    e_ref: float = 0.0  # Electronic energy reference

    def mode_tag(self) -> str:
        return "effective"

    def device_space(self) -> str:
        return "V"

    def should_skip_inner(self, outer_step: int) -> bool:
        """Skip inner loop at first iteration (outer=0)."""
        return outer_step == 0

    def post_inner_sweep(self, energy_trace_elec: np.ndarray) -> None:
        """Update reference electronic energy for next Hamiltonian rebuild."""
        if len(energy_trace_elec) > 0:
            self.e_ref = float(energy_trace_elec[-1])

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build H_eff via Löwdin downfolding."""
        psi_v = None
        if self.screening == "dynamic":
            psi_v = self._compute_screening_weights()

        proxy_ham, P_dets = ham_mod.build_proxy_hamiltonian(
            system=self.system,
            space=self.detspace,
            screening=self.screening,
            psi_v=psi_v,
            screen_eps=self.screen_eps,
        )

        self.detspace = DetSpace(V_dets=self.detspace.V_dets, P_dets=P_dets)
        self.state = self.state.update_space(
            self.detspace, device_space=self.device_space()
        )

        ham = ham_mod.build_effective_hamiltonian(
            system=self.system,
            proxy_ham=proxy_ham,
            e_ref=self.e_ref,
            reg_type=self.reg_type,
            epsilon=self.epsilon,
        )

        op = kern_mod.build_vv_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


@dataclass
class ProxyDriver(BaseDriver):
    """
    Proxy mode: Full T-space optimization with diagonal P-block.

    Hamiltonian blocks:
      H = [[H_VV,       H_VP      ],
           [H_PV,  diag(H_PP) + δ·I]]

    Workflow:
      1. Build proxy H on T = V ∪ P with screening
      2. Optimize E = <ψ_T|H|ψ_T> / <ψ_T|ψ_T>
      3. Update V ← select from T
  
    Memory: GPU holds full T-space batch
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6
    diag_shift: float = 0.0

    def mode_tag(self) -> str:
        return "proxy"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build full proxy Hamiltonian H on T-space."""
        psi_v = None
        if self.screening == "dynamic":
            psi_v = self._compute_screening_weights()

        ham, P_dets = ham_mod.build_proxy_hamiltonian(
            system=self.system,
            space=self.detspace,
            screening=self.screening,
            psi_v=psi_v,
            screen_eps=self.screen_eps,
            diag_shift=self.diag_shift,
        )

        self.detspace = DetSpace(V_dets=self.detspace.V_dets, P_dets=P_dets)
        self.state = self.state.update_space(
            self.detspace, device_space=self.device_space()
        )

        op = kern_mod.build_proxy_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


@dataclass
class AsymmetricDriver(BaseDriver):
    """
    Asymmetric mode: VMC-style truncated estimator on V-space.

    Energy estimator:
        E_asym = <ψ|P_V H|ψ> / <ψ|P_V|ψ>

    where P_V projects onto V-space.

    Workflow:
      - Uses proxy Hamiltonian H_T same as ProxyDriver
      - Inner loop minimizes asymmetric Rayleigh quotient
      - V-space evolution uses full-T amplitudes |ψ_T|
  
    Memory: GPU holds full T-space batch
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6

    def mode_tag(self) -> str:
        return "asymmetric"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """Build proxy Hamiltonian H_T on T-space."""
        psi_v = None
        if self.screening == "dynamic":
            psi_v = self._compute_screening_weights()

        ham, P_dets = ham_mod.build_proxy_hamiltonian(
            system=self.system,
            space=self.detspace,
            screening=self.screening,
            psi_v=psi_v,
            screen_eps=self.screen_eps,
        )

        self.detspace = DetSpace(V_dets=self.detspace.V_dets, P_dets=P_dets)
        self.state = self.state.update_space(
            self.detspace, device_space=self.device_space()
        )

        op = kern_mod.build_proxy_operator(ham, jax_dtype=self.state.psi_dtype)

        return ham, op


__all__ = [
    "BaseDriver",
    "VariationalDriver",
    "EffectiveDriver",
    "ProxyDriver",
    "AsymmetricDriver",
]