# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic optimization drivers for LEVER variational framework.

Computational modes:
  - Variational: E[ψ_S] = <ψ_S|H_SS|ψ_S> / <ψ_S|ψ_S>
  - Effective:   E_eff[ψ_S] via Löwdin downfolding H_eff = H_SS + H_SC·(E_ref - H_CC + ε·I)^(-1)·H_CS
  - Proxy:       E[ψ_T] on T-space with diagonal C-block approximation
  - Asymmetric:  E_asym = <ψ|P_S H|ψ> / <ψ|P_S|ψ>

Architecture layers:
  - L0 (C++):    Static integrals, Heat-Bath tables, COO assembly
  - L1 (Python): Outer loop - space evolution, Hamiltonian rebuild
  - L2 (JAX):    Inner loop - JIT single-step optimization

File: lever/driver.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core
from .operator import functional as func_mod
from .operator import hamiltonian as ham_mod
from .operator import kernel as kern_mod
from .operator import gpu_kernel as gkern_mod
from .space import DetSpace, Selector
from .state import State
from .system import MolecularSystem


Callback = Callable[[dict[str, Any]], None]


@dataclass
class DriverResult:
    """Container for driver execution results."""

    state: State
    detspace: DetSpace
    energies: list[float]
    e_nuc: float

    def total_energies(self) -> list[float]:
        """Total energies: E_tot = E_elec + E_nuc."""
        return [e + self.e_nuc for e in self.energies]


@dataclass
class BaseDriver:
    """
    Base driver for deterministic LEVER optimization.

    Outer loop workflow:
      1. Build Hamiltonian for current S/C spaces
      2. Execute inner optimization loop
      3. Update S-space via amplitude-based selection
      4. Check convergence on energy

    Inner loop workflow:
      - Forward pass: ψ(x; θ)
      - Compute energy and gradient: E[θ], ∇_θ E
      - Update parameters: θ ← optimizer(θ, ∇_θ E)
      - Check convergence: ΔE < tol for patience consecutive steps

    Subclass hooks:
      - mode_tag(): Mode identifier string
      - build_hamiltonian(): Construct Hamiltonian operators
      - post_sweep(): Process results after inner loop
    """

    system: MolecularSystem
    state: State
    detspace: DetSpace
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    selector: Selector

    max_outer: int = 30
    max_inner: int = 500

    inner_tol: float = 1e-8
    inner_patience: int = 100
    outer_tol: float = 1e-6
    outer_patience: int = 5

    chunk_size: int | None = None
    use_gpu_kernel: bool = False

    outer_history: list[float] = field(default_factory=list, init=False)
    callbacks: list[Callback] | None = None
    analysis_recorder: Any = None

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
            Initialized driver ready for run()
        """
        state = State.init(
            system=system, detspace=detspace, model=model, key=key
        )
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
        """Mode identifier for callbacks."""
        return "base"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """
        Build Hamiltonian and SpMV operator.

        Returns:
            (ham, op): Hamiltonian object and operator function
        """
        raise NotImplementedError

    def post_sweep(self, energy_trace: np.ndarray, n_steps: int) -> None:
        """Hook for post-inner-loop processing."""
        pass

    def _compute_screening_weights(self) -> np.ndarray | None:
        """
        Compute normalized screening weights sqrt(p_i) on S-space.

        Returns:
            Weights or None if S-space is empty
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

    def evolve_space(self) -> None:
        """Update S-space via importance sampling on log|ψ_T|."""
        sign_t, logabs_t = self.state.forward(chunk_size=self.chunk_size)
        logabs_t = jnp.real(logabs_t)
        log_amp_host = np.asarray(logabs_t, dtype=np.float64)

        new_space = self.detspace.evolve(self.selector, log_amp_host)
        self.detspace = new_space
        self.state = self.state.update_space(new_space)

    def _run_callbacks(self, info: dict[str, Any]) -> None:
        """Invoke registered callbacks."""
        if self.callbacks:
            for cb in self.callbacks:
                cb(info)

    def _check_outer_convergence(self, value: float) -> tuple[bool, float]:
        """
        Check outer convergence via sliding window.

        Criterion: max|E_i - E_(i-1)| < tol for last (patience+1) points.

        Returns:
            (converged, max_delta)
        """
        self.outer_history.append(value)

        if (
            self.outer_patience <= 0
            or len(self.outer_history) <= self.outer_patience
        ):
            return False, float("inf")

        window = self.outer_history[-(self.outer_patience + 1) :]
        deltas = [abs(window[i] - window[i - 1]) for i in range(1, len(window))]
        max_delta = max(deltas)

        return max_delta < self.outer_tol, max_delta

    def _build_sweep(self, ham: Any, op: Callable) -> Callable:
        """
        Build inner optimization loop with JIT single-step updates.

        All mode-specific decisions resolved at L1. The JIT body sees
        a single energy_step(state) callable.
        """
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
            """Inner loop: optimize parameters until convergence."""
            
            # Define single optimization step
            def single_step(state, opt_state):
                """Single optimization step: energy, gradients, parameter update."""
                energy_val, grad = energy_step(state)
                state, opt_state = state.apply_gradients(
                    gradients=grad,
                    opt_state=opt_state,
                    optimizer=self.optimizer,
                )
                return state, opt_state, energy_val
            
            # JIT compile with buffer donation for memory efficiency
            # donate_argnums=(0, 1) allows JAX to reuse input buffers
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

            return state, opt_state, np.array(energy_trace), len(energy_trace)

        return sweep

    def run(self) -> DriverResult:
        """Execute outer loop over determinant-space evolution."""
        energies: list[float] = []

        for outer in range(self.max_outer):
            ham, op = self.build_hamiltonian()

            inner_sweep = self._build_sweep(ham, op)

            self.state, self.opt_state, energy_trace, n_steps = inner_sweep(
                self.state, self.opt_state
            )

            for inner in range(n_steps):
                e = float(energy_trace[inner])
                energies.append(e)

                self._run_callbacks(
                    {
                        "mode": self.mode_tag(),
                        "outer": outer,
                        "inner": inner,
                        "energy": e,
                    }
                )

            self.post_sweep(energy_trace, n_steps)

            if self.analysis_recorder is not None:
                self.analysis_recorder.summarize_outer(
                    outer,
                    state=self.state,
                    ham=ham,
                    op=op,
                    detspace=self.detspace,
                    mode=self.mode_tag(),
                    chunk_size=self.chunk_size,
                )

            if n_steps > 0:
                last_e = float(energy_trace[n_steps - 1])
                converged, _ = self._check_outer_convergence(last_e)
                if converged:
                    break

            self.evolve_space()

        return DriverResult(
            state=self.state,
            detspace=self.detspace,
            energies=energies,
            e_nuc=self.system.e_nuc,
        )


@dataclass
class VariationalDriver(BaseDriver):
    """
    Variational mode: E = <ψ_S|H_SS|ψ_S> / <ψ_S|ψ_S> on S-space.

    Workflow:
      1. Construct H_SS on current S-space
      2. Optimize parameters to minimize E[ψ_S]
      3. Generate complement C via Heat-Bath screening
      4. Update S ← select from T
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6

    def mode_tag(self) -> str:
        return "variational"

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
        self.state = self.state.update_space(self.detspace)

        ham = ham_mod.build_ss_hamiltonian(self.system, self.detspace)
        psi_dtype = self.state.psi_dtype

        if self.use_gpu_kernel:
            op = gkern_mod.build_ss_operator_gpu(ham, jax_dtype=psi_dtype)
        else:
            op = kern_mod.build_ss_operator(ham, jax_dtype=psi_dtype)

        return ham, op


@dataclass
class EffectiveDriver(BaseDriver):
    """
    Effective mode: Löwdin downfolded Hamiltonian in S-space.

    Effective Hamiltonian:
      H_eff = H_SS + H_SC·(E_ref - H_CC + ε·I)^(-1)·H_CS

    Regularization:
      - sigma: Constant shift ε·I
      - linear_shift: max(0, E_ref - diag(H_CC))

    Workflow:
      1. Build proxy H and complement C
      2. Compute H_eff via Löwdin partitioning
      3. Optimize E_eff = <ψ_S|H_eff|ψ_S>
      4. Update E_ref ← last energy, S ← select from T
    """

    screening: str = "static"
    screen_eps: float = 1e-3
    reg_type: str = "sigma"
    epsilon: float = 1e-12
    e_ref: float = 0.0

    def mode_tag(self) -> str:
        return "effective"

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
        self.state = self.state.update_space(self.detspace)

        ham = ham_mod.build_effective_hamiltonian(
            system=self.system,
            proxy_ham=proxy_ham,
            e_ref=self.e_ref,
            reg_type=self.reg_type,
            epsilon=self.epsilon,
        )

        psi_dtype = self.state.psi_dtype

        if self.use_gpu_kernel:
            op = gkern_mod.build_ss_operator_gpu(ham, jax_dtype=psi_dtype)
        else:
            op = kern_mod.build_ss_operator(ham, jax_dtype=psi_dtype)

        return ham, op

    def post_sweep(self, energy_trace: np.ndarray, n_steps: int) -> None:
        """Update reference energy E_ref ← last energy."""
        if n_steps > 0:
            self.e_ref = float(energy_trace[n_steps - 1])


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
        self.state = self.state.update_space(self.detspace)

        psi_dtype = self.state.psi_dtype

        if self.use_gpu_kernel:
            op = gkern_mod.build_proxy_operator_gpu(ham, jax_dtype=psi_dtype)
        else:
            op = kern_mod.build_proxy_operator(ham, jax_dtype=psi_dtype)

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
    """

    screening: str = "static"
    screen_eps: float = 1e-3

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
        self.state = self.state.update_space(self.detspace)

        psi_dtype = self.state.psi_dtype

        if self.use_gpu_kernel:
            op = gkern_mod.build_proxy_operator_gpu(ham, jax_dtype=psi_dtype)
        else:
            op = kern_mod.build_proxy_operator(ham, jax_dtype=psi_dtype)

        return ham, op


__all__ = [
    "DriverResult",
    "BaseDriver",
    "VariationalDriver",
    "EffectiveDriver",
    "ProxyDriver",
    "AsymmetricDriver",
]