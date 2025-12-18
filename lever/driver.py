# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deterministic optimization drivers for LEVER variational framework.

Computational modes:
  - Variational: Minimize E[ψ_S] = <ψ_S|H_SS|ψ_S> / <ψ_S|ψ_S>
  - Effective:   Minimize E_eff[ψ_S] via Löwdin downfolding
  - Proxy:       Minimize E[ψ_T] on full T-space Hamiltonian
  - Asymmetric:  VMC-style estimator <ψ|P_S H|ψ> / <ψ|P_S|ψ>

Three-layer architecture:
  - L0 (C++):    Static integrals, Heat-Bath tables, COO assembly
  - L1 (Python): Outer loop - space evolution, Hamiltonian rebuild
  - L2 (JAX):    Inner loop - JIT-compiled optimization via lax.while_loop

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
from jax import lax

from . import core
from .operator import functional as func_mod
from .operator import hamiltonian as ham_mod
from .operator import kernel as kern_mod
from .operator import gpu_kernel as gkern_mod
from .space import DetSpace, Selector
from .state import DeterministicState
from .system import MolecularSystem


# ============================================================================
# Type Aliases
# ============================================================================

Callback = Callable[[dict[str, Any]], None]


# ============================================================================
# Result Container
# ============================================================================

@dataclass
class DriverResult:
    """Container for driver execution results."""

    state: DeterministicState
    detspace: DetSpace
    energies: list[float]  # Electronic energies E_elec
    e_nuc: float

    def total_energies(self) -> list[float]:
        """Return total energies: E_tot = E_elec + E_nuc."""
        return [e + self.e_nuc for e in self.energies]


# ============================================================================
# Base Driver
# ============================================================================

@dataclass
class BaseDriver:
    """
    Base class for deterministic LEVER optimization drivers.

    Outer loop (L1):
      1. Build Hamiltonian for current S/C spaces
      2. Execute JIT-compiled inner loop
      3. Update S-space via amplitude-based selection
      4. Check outer convergence

    Inner loop (L2):
      - Forward: ψ(x; θ)
      - Energy & gradient: E[θ], ∇_θ E
      - Update: θ ← optimizer(θ, ∇_θ E)
      - Convergence: ΔE < tol for consecutive patience steps

    Subclass hooks:
      - build_hamiltonian(): Prepare Hamiltonian operators
      - post_sweep(): Process inner-loop results
      - evolve_space(): Update S from T-space amplitudes
    """

    system: MolecularSystem
    state: DeterministicState
    detspace: DetSpace
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    selector: Selector

    # Iteration control
    max_outer: int = 30
    max_inner: int = 1000

    # Convergence criteria
    inner_tol: float = 1e-8
    inner_patience: int = 100
    outer_tol: float = 1e-7
    outer_patience: int = 3

    # Memory optimization
    chunk_size: int | None = None

    # Use GPU sparse kernels (BCOO SpMV) instead of CPU/Numba
    use_gpu_kernel: bool = False

    # Outer convergence tracking
    outer_history: list[float] = field(default_factory=list, init=False)

    # Diagnostic callbacks
    callbacks: list[Callback] | None = None
  
    # Analysis recorder (optional)
    analysis_recorder: Any = None

    # ========================================================================
    # Factory Method
    # ========================================================================

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
        Build driver with automatic state and optimizer initialization.

        Args:
            system: Molecular system from FCIDUMP
            detspace: Initial determinant space
            model: Uninitialized Flax model
            optimizer: Optax optimizer
            selector: Determinant selector
            key: JAX PRNG key (default: fixed seed 0)
            **driver_kwargs: Additional driver parameters

        Returns:
            Initialized driver ready for .run()
        """
        state = DeterministicState.init_from_model(
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

    # ========================================================================
    # Abstract Methods (Subclass Interface)
    # ========================================================================

    def mode_tag(self) -> str:
        """Short identifier for callbacks."""
        return "base"

    def build_hamiltonian(self) -> tuple[Any, Any]:
        """
        Build Hamiltonian and operator for current iteration.

        Returns:
            (ham, op): Hamiltonian object and SpMV operator function
        """
        raise NotImplementedError

    def post_sweep(
        self,
        energy_trace: np.ndarray,
        n_steps: int,
    ) -> None:
        """Hook for post-inner-loop processing."""
        pass

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _compute_screening_weights(self) -> np.ndarray | None:
        """
        Compute normalized probability weights on S-space for screening.

        Returns:
            Weights as sqrt(prob) or None if S-space is empty
        """
        if self.detspace.size_S == 0:
            return None

        indices = jnp.asarray(self.detspace.S_indices, dtype=jnp.int32)

        # New API: forward returns (sign, logabs)
        sign_s, logabs_s = self.state.forward(indices, chunk_size=self.chunk_size)

        # Stable probability: |psi|^2 = exp(2 * logabs)
        # Use max-shift for numerical stability
        logabs_s = jnp.real(logabs_s)
        logabs_s = logabs_s - jnp.max(logabs_s)

        prob_s = jnp.exp(2.0 * logabs_s)
        prob_s = prob_s / jnp.sum(prob_s)

        return np.asarray(jnp.sqrt(prob_s), dtype=np.float64)

    def evolve_space(self) -> None:
        """
        Update S-space via importance-based determinant selection.

        Computes log amplitudes log|ψ_T| and delegates to selector for
        space evolution. Selectors handle normalization as needed.
        """
        # New API: forward returns (sign, logabs)
        sign_t, logabs_t = self.state.forward(chunk_size=self.chunk_size)

        # We pass log|psi| to selector (real)
        logabs_t = jnp.real(logabs_t)

        # Pass log amplitudes to host for selector processing
        log_amp_host = np.asarray(logabs_t, dtype=np.float64)

        new_space = self.detspace.evolve(self.selector, log_amp_host)
        self.detspace = new_space
        self.state = self.state.update_space(new_space)

    # ========================================================================
    # Callback Dispatch
    # ========================================================================

    def _run_callbacks(self, info: dict[str, Any]) -> None:
        """Invoke registered callbacks with diagnostic info."""
        if self.callbacks:
            for cb in self.callbacks:
                cb(info)

    # ========================================================================
    # Convergence Check
    # ========================================================================

    def _check_outer_convergence(self, value: float) -> tuple[bool, float]:
        """
        Check sliding-window convergence on outer loop.

        Criterion: max|E_i - E_{i-1}| < tol for last (patience+1) values.

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

    # ========================================================================
    # Inner Loop Builder
    # ========================================================================

    def _build_sweep(
        self,
        ham: Any,
        op: Callable,
    ) -> Callable:
        """
        Build the jitted inner optimization loop.

        All mode-specific decisions (mode, chunk_size, detspace) resolved
        here at L1. The jitted body sees a single energy_step(state) callable.
        """
        optimizer = self.optimizer
        max_inner = int(self.max_inner)
        inner_tol = float(self.inner_tol)
        inner_patience = int(self.inner_patience)

        # L1: resolve static configuration once per outer iteration
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

        def sweep(
            state: DeterministicState,
            opt_state: optax.OptState,
        ):
            step0 = jnp.int32(0)
            last_energy0 = jnp.array(jnp.inf, dtype=jnp.float64)
            streak0 = jnp.int32(0)
            done0 = jnp.array(False)
            energy_trace0 = jnp.zeros((max_inner,), dtype=jnp.float64)

            def cond_fun(loop_state):
                step, _, _, _, _, done, _ = loop_state
                return jnp.logical_and(step < max_inner, jnp.logical_not(done))

            def body_fun(loop_state):
                (
                    step,
                    state,
                    opt_state,
                    last_energy,
                    streak,
                    done,
                    energy_trace,
                ) = loop_state

                # Single energy + grad call (now returns tuple directly)
                energy_val, grad = energy_step(state)

                # Parameter update
                state, opt_state = state.apply_gradients(
                    gradients=grad,
                    opt_state=opt_state,
                    optimizer=optimizer,
                )

                energy = jnp.array(energy_val, dtype=jnp.float64)

                # Convergence on energy
                delta = jnp.abs(energy - last_energy)
                streak_next = jnp.where(
                    delta < inner_tol, streak + 1, jnp.int32(0)
                )

                has_energy_conv = jnp.logical_and(
                    inner_tol > 0.0, inner_patience > 0
                )
                done_step = jnp.logical_and(
                    has_energy_conv, streak_next >= inner_patience
                )
                done_out = jnp.logical_or(done, done_step)

                energy_trace = energy_trace.at[step].set(energy)

                return (
                    step + 1,
                    state,
                    opt_state,
                    energy,
                    streak_next,
                    done_out,
                    energy_trace,
                )

            loop_state0 = (
                step0,
                state,
                opt_state,
                last_energy0,
                streak0,
                done0,
                energy_trace0,
            )

            (
                step_f,
                state_f,
                opt_state_f,
                _,
                _,
                _,
                energy_trace_f,
            ) = lax.while_loop(cond_fun, body_fun, loop_state0)

            return state_f, opt_state_f, energy_trace_f, step_f

        return jax.jit(sweep, donate_argnums=(0, 1))

    # ========================================================================
    # Main Execution
    # ========================================================================

    def run(self) -> DriverResult:
        """Execute outer loop over determinant-space evolution."""
        energies: list[float] = []

        for outer in range(self.max_outer):
            # Build Hamiltonian for current iteration
            ham, op = self.build_hamiltonian()

            # JIT-compile inner sweep
            inner_sweep = self._build_sweep(ham, op)

            # Execute inner loop
            (
                self.state,
                self.opt_state,
                energy_trace,
                n_steps,
            ) = inner_sweep(self.state, self.opt_state)

            # Transfer to host
            energy_np = np.asarray(energy_trace, dtype=np.float64)
            steps = int(n_steps)

            # Dispatch callbacks for step records
            for inner in range(steps):
                e = float(energy_np[inner])
                energies.append(e)

                self._run_callbacks(
                    {
                        "mode": self.mode_tag(),
                        "outer": outer,
                        "inner": inner,
                        "energy": e,
                    }
                )

            # Subclass post-processing
            self.post_sweep(energy_np, steps)

            # Outer loop diagnostics (once per cycle)
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

            # Outer convergence check
            if steps > 0:
                last_e = float(energy_np[steps - 1])
                converged, _ = self._check_outer_convergence(last_e)
                if converged:
                    break

            # Update S-space for next iteration
            self.evolve_space()

        return DriverResult(
            state=self.state,
            detspace=self.detspace,
            energies=energies,
            e_nuc=self.system.e_nuc,
        )


# ============================================================================
# Variational Driver
# ============================================================================

@dataclass
class VariationalDriver(BaseDriver):
    """
    Variational mode: Minimize E = <ψ_S|H_SS|ψ_S> / <ψ_S|ψ_S> in S-space.

    Workflow:
      1. Construct H_SS on current S-space
      2. Optimize parameters θ to minimize E[ψ_S(θ)]
      3. Generate complement C via Heat-Bath screening
      4. Update S ← T
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


# ============================================================================
# Effective Driver
# ============================================================================

@dataclass
class EffectiveDriver(BaseDriver):
    """
    Effective mode: Minimize on Löwdin downfolded Hamiltonian in S-space.

    Effective Hamiltonian:
      H_eff = H_SS + H_SC·(E_ref - H_CC + ε·I)^(-1)·H_CS

    Regularization strategies:
      - sigma: Constant shift ε·I
      - linear_shift: max(0, E_ref - diag(H_CC))

    Workflow:
      1. Build proxy H and complement C
      2. Compute H_eff via Löwdin partitioning
      3. Optimize E_eff = <ψ_S|H_eff|ψ_S>
      4. Update E_ref ← last inner energy, Update S ← T
    """

    screening: str = "static"
    screen_eps: float = 1e-3
    reg_type: str = "sigma"  # {"sigma", "linear_shift"}
    epsilon: float = 1e-12
    e_ref: float = 0.0  # Reference energy for partitioning

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

    def post_sweep(
        self,
        energy_trace: np.ndarray,
        n_steps: int,
    ) -> None:
        """Update reference energy: E_ref ← last inner energy."""
        if n_steps > 0:
            self.e_ref = float(energy_trace[n_steps - 1])


# ============================================================================
# Proxy Driver
# ============================================================================

@dataclass
class ProxyDriver(BaseDriver):
    """
    Proxy mode: Minimize directly on full T-space Hamiltonian.

    Hamiltonian blocks:
      H = [[H_SS,       H_SC    ],
           [H_CS,  H_CC_diag]]

    Workflow:
      1. Build proxy H on T = S ∪ C with screening
      2. Optimize E = <ψ_T|H|ψ_T> / <ψ_T|ψ_T>
      3. Update S ← T
    """

    screening: str = "dynamic"
    screen_eps: float = 1e-6
    diag_shift: float = 0.0  # Optional diagonal shift on H_CC

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


# ============================================================================
# Asymmetric Driver
# ============================================================================

@dataclass
class AsymmetricDriver(BaseDriver):
    """
    Asymmetric mode: VMC-style truncated estimator on S-space.

    Energy estimator:
        E_asym = <ψ|P_S H|ψ> / <ψ|P_S|ψ>

    where P_S projects onto S-space.

    Implementation:
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