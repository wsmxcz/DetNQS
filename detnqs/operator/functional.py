# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Energy functionals for deterministic NQS on selected CI spaces.

Implements Rayleigh quotient functionals:
  - Variational/Effective: E = <psi_V|H_VV|psi_V> / <psi_V|psi_V>
  - Proxy: E = <psi_T|H_T|psi_T> / <psi_T|psi_T>, where T = V union P
  - Asymmetric: E = <psi_V|(H*psi)_V> / <psi_V|psi_V>

Gradients via VJP with (sign, logabs) wavefunction representation.

File: detnqs/operator/functional.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp

from ..space import DetSpace
from ..state import State
from .hamiltonian import ProxyHamiltonian, VVHamiltonian
from .kernel import ProxyContraction, VVContraction


def _decode_psi(sign: jnp.ndarray, logabs: jnp.ndarray):
    """
    Decode wavefunction from (sign, logabs) to amplitude.
    
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
    Compute VJP cotangents for (sign, logabs) given g = dE/d(psi*).
    
    Wirtinger derivative:
      cot_logabs = 2*Re(conj(psi)*g)
      cot_sign = amp * g (complex only)
    
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


def _energy_step_variational(
    state: State,
    ham: VVHamiltonian,
    op: Callable[[jnp.ndarray], VVContraction],
    *,
    eps: float,
    chunk_size: int | None,
) -> tuple[float, Any]:
    """
    Variational energy on V-space: E = <psi_V|H_VV|psi_V> / <psi_V|psi_V>.
    
    Gradient: grad = vjp(2*(H - E)*psi / <psi|psi>)
    """
    n_v = ham.ham_vv.shape[0]
    indices = jnp.arange(n_v, dtype=jnp.int32)
    
    (sign_v, logabs_v), vjp_fn = state.value_and_vjp(indices, chunk_size=chunk_size)
    is_complex = jnp.issubdtype(sign_v.dtype, jnp.complexfloating)
    
    psi_v, amp_v, _ = _decode_psi(sign_v, jnp.real(logabs_v))
    
    contr = op(psi_v)
    h_psi_v = contr.V
    
    num = jnp.vdot(psi_v, h_psi_v)
    den = jnp.vdot(psi_v, psi_v)
    
    if is_complex:
        den_safe = jnp.maximum(den.real, eps).astype(psi_v.dtype)
        e_elec = (num / den_safe).real
    else:
        den_safe = jnp.maximum(den, eps)
        e_elec = num / den_safe
    
    residual = h_psi_v - e_elec * psi_v
    g = residual / den_safe
    
    cot_sign, cot_logabs = _cotangents_from_g(sign_v, psi_v, amp_v, g)
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
    Proxy energy on T-space: E = <psi_T|H_T|psi_T> / <psi_T|psi_T>.
    
    Gradient computed on both V and P components.
    """
    n_v = detspace.size_V
    n_p = detspace.size_P
    
    (sign_t, logabs_t), vjp_fn = state.value_and_vjp(chunk_size=chunk_size)
    is_complex = jnp.issubdtype(sign_t.dtype, jnp.complexfloating)
    
    psi_t, amp_t, _ = _decode_psi(sign_t, jnp.real(logabs_t))
    
    psi_v, psi_p = psi_t[:n_v], psi_t[n_v:n_v + n_p]
    amp_v, amp_p = amp_t[:n_v], amp_t[n_v:n_v + n_p]
    sign_v, sign_p = sign_t[:n_v], sign_t[n_v:n_v + n_p]
    
    contr = op(psi_v, psi_p)
    h_psi_v, h_psi_p = contr.V, contr.P
    
    num = jnp.vdot(psi_v, h_psi_v) + jnp.vdot(psi_p, h_psi_p)
    
    if is_complex:
        den = jnp.vdot(psi_v, psi_v).real + jnp.vdot(psi_p, psi_p).real
        den_safe = jnp.maximum(den, eps).astype(psi_v.dtype)
        e_elec = (num / den_safe).real
    else:
        den = jnp.vdot(psi_v, psi_v) + jnp.vdot(psi_p, psi_p)
        den_safe = jnp.maximum(den, eps)
        e_elec = num / den_safe
    
    g_v = (h_psi_v - e_elec * psi_v) / den_safe
    g_p = (h_psi_p - e_elec * psi_p) / den_safe
    
    cot_sign_v, cot_logabs_v = _cotangents_from_g(sign_v, psi_v, amp_v, g_v)
    cot_sign_p, cot_logabs_p = _cotangents_from_g(sign_p, psi_p, amp_p, g_p)
    
    cot_sign_t = jnp.concatenate([cot_sign_v, cot_sign_p], axis=0)
    cot_logabs_t = jnp.concatenate([cot_logabs_v, cot_logabs_p], axis=0)
    
    grad = vjp_fn((cot_sign_t, cot_logabs_t))
    return e_elec, grad


def _energy_step_asymmetric(
    state: State,
    ham: ProxyHamiltonian,
    op: Callable[[jnp.ndarray, jnp.ndarray], ProxyContraction],
    detspace: DetSpace,
    *,
    eps: float,
    chunk_size: int | None,
) -> tuple[float, Any]:
    """
    Asymmetric energy: E = <psi_V|(H*psi)_V> / <psi_V|psi_V>.
    
    Gradient from V-space residual only.
    """
    n_v = detspace.size_V
    n_p = detspace.size_P
    
    (sign_t, logabs_t), vjp_fn = state.value_and_vjp(chunk_size=chunk_size)
    is_complex = jnp.issubdtype(sign_t.dtype, jnp.complexfloating)
    
    psi_t, amp_t, _ = _decode_psi(sign_t, jnp.real(logabs_t))
    
    psi_v = psi_t[:n_v]
    psi_p = psi_t[n_v:n_v + n_p]
    amp_v = amp_t[:n_v]
    sign_v = sign_t[:n_v]
    
    contr = op(psi_v, psi_p)
    h_psi_v = contr.V
    
    num = jnp.vdot(psi_v, h_psi_v)
    den = jnp.vdot(psi_v, psi_v)
    
    if is_complex:
        den_safe = jnp.maximum(den.real, eps).astype(psi_v.dtype)
        e_elec = (num / den_safe).real
    else:
        den_safe = jnp.maximum(den, eps)
        e_elec = num / den_safe
    
    g_v = (h_psi_v - e_elec * psi_v) / den_safe
    
    cot_sign_v, cot_logabs_v = _cotangents_from_g(sign_v, psi_v, amp_v, g_v)
    
    cot_sign_p = jnp.zeros((n_p,), dtype=sign_t.dtype)
    cot_logabs_p = jnp.zeros((n_p,), dtype=jnp.real(logabs_t).dtype)
    
    cot_sign_t = jnp.concatenate([cot_sign_v, cot_sign_p], axis=0)
    cot_logabs_t = jnp.concatenate([cot_logabs_v, cot_logabs_p], axis=0)
    
    grad = vjp_fn((cot_sign_t, cot_logabs_t))
    return e_elec, grad


def make_energy_step(
    ham: VVHamiltonian | ProxyHamiltonian,
    op: Callable,
    *,
    detspace: DetSpace | None = None,
    mode: str = "variational",
    eps: float = 1e-12,
    chunk_size: int | None = None,
) -> Callable[[State], tuple[float, Any]]:
    """
    Build energy and gradient function for inner optimization.
    
    Mode selection at L1 layer, returns JIT-compatible callable for L2.
    
    Args:
        ham: Hamiltonian operator (SS or Proxy)
        op: Contraction kernel
        detspace: Required for ProxyHamiltonian modes
        mode: "variational", "proxy", or "asymmetric"
        eps: Denominator regularization
        chunk_size: Forward pass batch size
    
    Returns:
        step: (state) -> (energy, gradient)
    """
    if isinstance(ham, VVHamiltonian):
        def step(state: State) -> tuple[float, Any]:
            return _energy_step_variational(state, ham, op, eps=eps, chunk_size=chunk_size)
        return step
    
    if detspace is None:
        raise ValueError("detspace required for ProxyHamiltonian")
    
    if mode == "proxy":
        def step(state: State) -> tuple[float, Any]:
            return _energy_step_proxy(state, ham, op, detspace, eps=eps, chunk_size=chunk_size)
        return step
    
    if mode == "asymmetric":
        def step(state: State) -> tuple[float, Any]:
            return _energy_step_asymmetric(state, ham, op, detspace, eps=eps, chunk_size=chunk_size)
        return step
    
    raise ValueError(f"Unsupported mode: {mode!r}")


__all__ = ["make_energy_step"]
