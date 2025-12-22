# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Energy functionals for deterministic NQS on selected CI spaces.

Implements Rayleigh quotient energy functionals:
  - Variational/Effective: E = <psi_S|H_SS|psi_S> / <psi_S|psi_S>
  - Proxy: E = <psi_T|H_T|psi_T> / <psi_T|psi_T>, where T = S union C
  - Asymmetric: E = <psi_S|(H*psi)_S> / <psi_S|psi_S>

Gradients computed via VJP with (sign, logabs) wavefunction representation.

File: lever/operator/functional.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from ..space import DetSpace
from ..state import State
from .hamiltonian import ProxyHamiltonian, SSHamiltonian
from .kernel import ProxyContraction, SSContraction


def _decode_psi(sign: jnp.ndarray, logabs: jnp.ndarray):
    """
    Decode wavefunction from (sign, logabs) representation.
    
    Applies global max-shift to logabs for numerical stability.
    
    Returns:
        psi: sign * exp(logabs - shift)
        amp: exp(logabs - shift)
        logabs_shifted: logabs - shift
    """
    shift = jnp.max(logabs) if logabs.size > 0 else jnp.asarray(0.0, dtype=logabs.dtype)
    logabs_shifted = logabs - shift
    amp = jnp.exp(logabs_shifted)
    psi = sign * amp
    return psi, amp, logabs_shifted


def _cotangents_from_g(
    sign: jnp.ndarray,
    psi: jnp.ndarray,
    amp: jnp.ndarray,
    g: jnp.ndarray,
):
    """
    Compute cotangents for (sign, logabs) outputs given g = dE/d(psi*).
    
    Unified Wirtinger derivative:
      - cot_logabs = 2*Re(conj(psi)*g) for both real and complex
      - cot_sign = amp * g for complex; zero for real (non-differentiable)
    
    Returns:
        (cot_sign, cot_logabs)
    """
    is_complex = jnp.issubdtype(sign.dtype, jnp.complexfloating)
    
    if is_complex:
        cot_sign = amp * g
        cot_logabs = 2.0 * jnp.real(jnp.conj(psi) * g)
    else:
        cot_sign = jnp.zeros_like(sign)
        cot_logabs = 2.0 * psi * g
    
    return cot_sign, cot_logabs


def _energy_step_s(
    state: State,
    ham: SSHamiltonian,
    op: Callable[[jnp.ndarray], SSContraction],
    *,
    eps: float,
    chunk_size: int | None,
) -> tuple[float, Any]:
    """
    Compute energy and gradient on S-space.
    
    Rayleigh quotient: E = <psi_S|H_SS|psi_S> / <psi_S|psi_S>
    Gradient: grad = vjp(2*(H - E)*psi / <psi|psi>)
    """
    n_s = ham.ham_ss.shape[0]
    indices = jnp.arange(n_s, dtype=jnp.int32)
    
    (sign_s, logabs_s), vjp_fn = state.value_and_vjp(indices, chunk_size=chunk_size)
    is_complex = jnp.issubdtype(sign_s.dtype, jnp.complexfloating)
    
    psi_s, amp_s, _ = _decode_psi(sign_s, jnp.real(logabs_s))
    
    contr = op(psi_s)
    h_psi_s = contr.S
    
    num = jnp.vdot(psi_s, h_psi_s)
    den = jnp.vdot(psi_s, psi_s)
    
    if is_complex:
        den_safe = jnp.maximum(den.real, eps).astype(psi_s.dtype)
        e_elec = (num / den_safe).real
    else:
        den_safe = jnp.maximum(den, eps)
        e_elec = num / den_safe
    
    residual = h_psi_s - e_elec * psi_s
    g = residual / den_safe
    
    cot_sign, cot_logabs = _cotangents_from_g(sign_s, psi_s, amp_s, g)
    grad = vjp_fn((cot_sign, cot_logabs))
    
    return e_elec, grad


def _energy_step_proxy(
    state: State,
    ham: ProxyHamiltonian,
    op: Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction],
    detspace: DetSpace,
    *,
    eps: float,
    chunk_size: int | None,
) -> tuple[float, Any]:
    """
    Compute energy and gradient on full T = S union C space.
    
    Rayleigh quotient: E = <psi_T|H_T|psi_T> / <psi_T|psi_T>
    Gradient computed on both S and C components.
    """
    n_s = detspace.size_S
    n_c = detspace.size_C
    
    (sign_t, logabs_t), vjp_fn = state.value_and_vjp(chunk_size=chunk_size)
    is_complex = jnp.issubdtype(sign_t.dtype, jnp.complexfloating)
    
    psi_t, amp_t, _ = _decode_psi(sign_t, jnp.real(logabs_t))
    
    psi_s, psi_c = psi_t[:n_s], psi_t[n_s:n_s + n_c]
    amp_s, amp_c = amp_t[:n_s], amp_t[n_s:n_s + n_c]
    sign_s, sign_c = sign_t[:n_s], sign_t[n_s:n_s + n_c]
    
    contr = op(psi_s, psi_c)
    h_psi_s, h_psi_c = contr.S, contr.C
    
    num = jnp.vdot(psi_s, h_psi_s) + jnp.vdot(psi_c, h_psi_c)
    
    if is_complex:
        den = jnp.vdot(psi_s, psi_s).real + jnp.vdot(psi_c, psi_c).real
        den_safe = jnp.maximum(den, eps).astype(psi_s.dtype)
        e_elec = (num / den_safe).real
    else:
        den = jnp.vdot(psi_s, psi_s) + jnp.vdot(psi_c, psi_c)
        den_safe = jnp.maximum(den, eps)
        e_elec = num / den_safe
    
    g_s = (h_psi_s - e_elec * psi_s) / den_safe
    g_c = (h_psi_c - e_elec * psi_c) / den_safe
    
    cot_sign_s, cot_logabs_s = _cotangents_from_g(sign_s, psi_s, amp_s, g_s)
    cot_sign_c, cot_logabs_c = _cotangents_from_g(sign_c, psi_c, amp_c, g_c)
    
    cot_sign_t = jnp.concatenate([cot_sign_s, cot_sign_c], axis=0)
    cot_logabs_t = jnp.concatenate([cot_logabs_s, cot_logabs_c], axis=0)
    
    grad = vjp_fn((cot_sign_t, cot_logabs_t))
    return e_elec, grad


def _energy_step_asym(
    state: State,
    ham: ProxyHamiltonian,
    op: Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction],
    detspace: DetSpace,
    *,
    eps: float,
    chunk_size: int | None,
) -> tuple[float, Any]:
    """
    Asymmetric energy estimator with S-space metric.
    
    Energy: E = <psi_S|(H*psi)_S> / <psi_S|psi_S>
    Gradient computed only from S-space residual.
    """
    n_s = detspace.size_S
    n_c = detspace.size_C
    
    (sign_t, logabs_t), vjp_fn = state.value_and_vjp(chunk_size=chunk_size)
    is_complex = jnp.issubdtype(sign_t.dtype, jnp.complexfloating)
    
    psi_t, amp_t, _ = _decode_psi(sign_t, jnp.real(logabs_t))
    
    psi_s = psi_t[:n_s]
    psi_c = psi_t[n_s:n_s + n_c]
    amp_s = amp_t[:n_s]
    sign_s = sign_t[:n_s]
    
    contr = op(psi_s, psi_c)
    h_psi_s = contr.S
    
    num = jnp.vdot(psi_s, h_psi_s)
    den = jnp.vdot(psi_s, psi_s)
    
    if is_complex:
        den_safe = jnp.maximum(den.real, eps).astype(psi_s.dtype)
        e_elec = (num / den_safe).real
    else:
        den_safe = jnp.maximum(den, eps)
        e_elec = num / den_safe
    
    g_s = (h_psi_s - e_elec * psi_s) / den_safe
    
    cot_sign_s, cot_logabs_s = _cotangents_from_g(sign_s, psi_s, amp_s, g_s)
    
    cot_sign_c = jnp.zeros((n_c,), dtype=sign_t.dtype)
    cot_logabs_c = jnp.zeros((n_c,), dtype=jnp.real(logabs_t).dtype)
    
    cot_sign_t = jnp.concatenate([cot_sign_s, cot_sign_c], axis=0)
    cot_logabs_t = jnp.concatenate([cot_logabs_s, cot_logabs_c], axis=0)
    
    grad = vjp_fn((cot_sign_t, cot_logabs_t))
    return e_elec, grad


def make_energy_step(
    ham: SSHamiltonian | ProxyHamiltonian,
    op: Callable,
    *,
    detspace: DetSpace | None = None,
    mode: str = "variational",
    eps: float = 1e-12,
    chunk_size: int | None = None,
) -> Callable[[State], tuple[float, Any]]:
    """
    Build energy and gradient function for inner optimization sweep.
    
    Mode selection resolved at L1 (Python layer), returns JIT-compatible
    callable for L2 (inner loop).
    
    Args:
        ham: Hamiltonian operator
        op: Contraction kernel
        detspace: Determinant space (required for ProxyHamiltonian)
        mode: Functional mode ("variational", "proxy", "asymmetric")
        eps: Numerical stability threshold
        chunk_size: Forward pass batch size
    
    Returns:
        step: Function (state) -> (energy, gradient)
    """
    if isinstance(ham, SSHamiltonian):
        def step(state: State) -> tuple[float, Any]:
            return _energy_step_s(state, ham, op, eps=eps, chunk_size=chunk_size)
        return step
    
    if detspace is None:
        raise ValueError("detspace required for ProxyHamiltonian")
    
    if mode == "proxy":
        def step(state: State) -> tuple[float, Any]:
            return _energy_step_proxy(state, ham, op, detspace, eps=eps, chunk_size=chunk_size)
        return step
    
    if mode == "asymmetric":
        def step(state: State) -> tuple[float, Any]:
            return _energy_step_asym(state, ham, op, detspace, eps=eps, chunk_size=chunk_size)
        return step
    
    raise ValueError(f"Unsupported mode: {mode!r}")


__all__ = ["make_energy_step"]
