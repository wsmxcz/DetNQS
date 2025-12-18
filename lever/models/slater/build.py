# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Factory for Slater-family ansatz in second quantization.

Provides high-level construction of neural quantum states with various
orbital update mechanisms: additive, gated, low-rank, CPD, and Jastrow.

File: lever/models/slater/build.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
from flax import linen as nn

from ...system import MolecularSystem
from ..utils import logdet_c, logsumexp_c

from .networks import Parametrizer, MLP
from .orbitals import (
    ReferenceSPO,
    AdditiveUpdate,
    GatedUpdate,
    LowRankUpdate,
    CPDUpdate,
    JastrowUpdate,
    SPOMap,
)

UpdateMode = Literal["none", "additive", "gated", "lowrank", "cpd", "hfds"]


class SlaterLogAmplitude(nn.Module):
    """
    Compute wavefunction amplitude as (sign, log|psi|).
  
    For single determinant: psi = J(occ) * det(Phi_occ)
    For multi-determinant: psi = J(occ) * sum_i c_i * det(Phi_occ^i)
  
    Returns complex or real sign and real-valued log-magnitude.
    """
    spo_map: nn.Module
    n_det: int = 1
    coeff_dtype: Any = None

    @nn.compact
    def __call__(self, occ_so: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            occ_so: (..., n_elec) occupied spin-orbital indices
          
        Returns:
            sign: (...) complex or real phase factor
            logabs: (...) log(|psi|)
        """
        occ = occ_so.astype(jnp.int32)
        bundle = self.spo_map(occ)
        phi_occ, j_sign, j_logabs = bundle.phi_occ, bundle.j_sign, bundle.j_logabs

        if self.n_det == 1:
            det_sign, det_logabs = logdet_c(phi_occ)
            sign = det_sign * j_sign
            logabs = det_logabs + j_logabs
            return sign, logabs

        # Multi-determinant: psi = J * sum_i c_i * det_i
        dtype = phi_occ.dtype if self.coeff_dtype is None else self.coeff_dtype
        log_coeff = self.param(
            "log_coeff", 
            nn.initializers.zeros, 
            (self.n_det,), 
            dtype
        )

        det_sign, det_logabs = logdet_c(phi_occ)  # (..., n_det)
        log_w = det_logabs + log_coeff  # Log-weighted amplitudes
      
        # Numerically stable sum via max-shift: sum exp(x_i) = exp(m) * sum exp(x_i - m)
        m = jnp.max(log_w, axis=-1, keepdims=True)
        w = jnp.exp(log_w - m)  # (..., n_det)

        if jnp.issubdtype(det_sign.dtype, jnp.complexfloating):
            a = jnp.sum(det_sign * w, axis=-1)  # Complex amplitude
            abs_a = jnp.abs(a)
            tiny = jnp.finfo(jnp.real(abs_a).dtype).tiny
            sign_sum = a / (abs_a + tiny)
            logabs_sum = jnp.squeeze(m, axis=-1) + jnp.log(abs_a + tiny)
        else:
            s = jnp.sum(det_sign * w, axis=-1)  # Real signed amplitude
            tiny = jnp.finfo(s.dtype).tiny
            sign_sum = jnp.where(s < 0.0, -1.0, 1.0).astype(s.dtype)
            logabs_sum = jnp.squeeze(m, axis=-1) + jnp.log(jnp.abs(s) + tiny)

        sign = sign_sum * j_sign
        logabs = logabs_sum + j_logabs
        return sign, logabs


def make_slater(
    system: MolecularSystem,
    *,
    parametrizer: Parametrizer | None = None,
    update: UpdateMode = "lowrank",
    n_det: int = 1,
    rank: int = 16,
    init_sigma: float = 1e-3,
    param_dtype: Any = jnp.float64,
) -> nn.Module:
    """
    Build Slater-family ansatz: psi = J(occ) * det(phi_ref + delta_phi).

    Args:
        system: Molecular system (orbital count, symmetry, etc.)
        parametrizer: Neural network producing orbital-dependent features
        update: Orbital correction mechanism:
            - "none": Reference orbitals only (HF/CASSCF baseline)
            - "additive": Full delta_phi correction
            - "gated": Multiplicative gate: phi * (1 + g)
            - "lowrank": Low-rank factorization: phi + U @ V^T
            - "cpd": Canonical polyadic decomposition
            - "hfds": Low-rank update + explicit Jastrow
        n_det: Number of determinants (>1 for multi-reference)
        rank: Rank for low-rank/CPD factorization
        init_sigma: Noise scale for reference orbital initialization
        param_dtype: Parameter precision
    
    Returns:
        SlaterLogAmplitude module
    """
    if parametrizer is None:
        parametrizer = MLP(
            n_so=system.n_so,
            dim=128,
            depth=2,
            pool="sum",
            out_scale=1e-2,
            param_dtype=param_dtype,
        )

    ref = ReferenceSPO(
        system=system,
        n_det=n_det,
        init_sigma=init_sigma,
        param_dtype=param_dtype,
    )

    additive_updates = []
    gated_updates = []
    jastrow = None

    if update == "additive":
        additive_updates.append(
            AdditiveUpdate(system, parametrizer, n_det, head="dphi")
        )
    elif update == "gated":
        gated_updates.append(
            GatedUpdate(system, parametrizer, n_det, head="gate")
        )
    elif update == "lowrank":
        additive_updates.append(
            LowRankUpdate(system, parametrizer, rank, n_det, head="uv")
        )
    elif update == "cpd":
        additive_updates.append(
            CPDUpdate(
                system, 
                parametrizer, 
                rank, 
                n_det, 
                head="w", 
                param_dtype=param_dtype
            )
        )
    elif update == "hfds":
        additive_updates.append(
            LowRankUpdate(system, parametrizer, rank, n_det, head="uv")
        )
        jastrow = JastrowUpdate(parametrizer, rank, head="jastrow")
    elif update != "none":
        raise ValueError(f"Unknown update mode: {update!r}")

    spo_map = SPOMap(
        reference=ref,
        additive_updates=tuple(additive_updates),
        gated_updates=tuple(gated_updates),
        jastrow=jastrow,
    )

    return SlaterLogAmplitude(spo_map=spo_map, n_det=n_det)


__all__ = ["SlaterLogAmplitude", "make_slater", "UpdateMode"]