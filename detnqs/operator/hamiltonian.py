# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Hamiltonian construction for determinant spaces.

Provides three Hamiltonian representations:
  - VVHamiltonian: H_VV restricted to variational space V
  - ProxyHamiltonian: Diagonal-approximated H on target space T = V + P
  - Effective Hamiltonian: Löwdin downfolded H_eff on V via perturbative P

Sparse blocks stored in CSR/CSC (SciPy) for efficient CPU SpMV.
Built once per outer iteration (L1), reused across inner steps (L2).

File: lever/operator/hamiltonian.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple

import numpy as np
import scipy.sparse as sp

from .. import core
from ..system import MolecularSystem
from ..space import DetSpace


class DiagonalInfo(NamedTuple):
    """Diagonal elements H_ii for variational and perturbative spaces."""
    V: np.ndarray  # [n_V], float64
    P: np.ndarray  # [n_P], float64 (empty if no P-space)


@dataclass(eq=False)
class VVHamiltonian:
    """
    Hamiltonian restricted to variational space V.
  
    Attributes:
        ham_vv: CSR matrix, shape (n_V, n_V)
        diagonals: Diagonal elements for V (and empty P)
    """
    ham_vv: sp.csr_matrix
    diagonals: DiagonalInfo


@dataclass(eq=False)
class ProxyHamiltonian:
    """
    Proxy Hamiltonian on target space T = V ⊕ P.
  
    Diagonal approximation for P-space: tilde{H}_PP = diag(H_PP).
  
    Attributes:
        ham_vv: H_VV block, CSR, shape (n_V, n_V)
        ham_vp: H_VP block, CSR, shape (n_V, n_P)
        ham_pv: H_PV block, CSC, shape (n_P, n_V), equals (H_VP)^T
        diagonals: Diagonal elements for V and P
    """
    ham_vv: sp.csr_matrix
    ham_vp: sp.csr_matrix
    ham_pv: sp.csc_matrix
    diagonals: DiagonalInfo


def _coo_to_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    """Convert COO triplets to CSR (duplicates summed)."""
    coo = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=shape,
    )
    return coo.tocsr()


def build_vv_hamiltonian(
    system: MolecularSystem,
    space: DetSpace,
) -> VVHamiltonian:
    """
    Build H_VV for fixed variational space V.

    Returns:
        VVHamiltonian with CSR H_VV and diagonal elements.
    """
    V_dets = np.ascontiguousarray(space.V_dets, dtype=np.uint64)
    n_v = int(V_dets.shape[0])

    coo_vv = core.get_ham_vv(
        dets_V=V_dets,
        int_ctx=system.int_ctx,
        n_orb=system.n_orb,
    )

    ham_vv = _coo_to_csr(
        rows=coo_vv["row"],
        cols=coo_vv["col"],
        vals=coo_vv["val"],
        shape=(n_v, n_v),
    )

    h_diag_v = np.asarray(ham_vv.diagonal(), dtype=np.float64)
    diagonals = DiagonalInfo(V=h_diag_v, P=np.zeros((0,), dtype=np.float64))
    return VVHamiltonian(ham_vv=ham_vv, diagonals=diagonals)


def build_proxy_hamiltonian(
    system: MolecularSystem,
    space: DetSpace,
    *,
    screening: Literal["none", "static", "dynamic"] = "static",
    psi_v: np.ndarray | None = None,
    screen_eps: float = 1e-6,
    diag_shift: float = 0.0,
) -> Tuple[ProxyHamiltonian, np.ndarray]:
    """
    Build proxy Hamiltonian H_VV, H_VP, diag(H_PP) with optional screening.

    Screening modes:
      - "none": Full P
      - "static": Heat-bath screening by |H_ij| > eps
      - "dynamic": Amplitude-weighted screening by |psi_V[i] * H_ij| > eps

    Args:
        system: Molecular integrals and orbital count
        space: Current DetSpace with V_dets
        screening: Screening strategy
        psi_v: Amplitudes on V for dynamic screening, shape [n_V]
        screen_eps: Screening threshold
        diag_shift: Shift added to diag(H_PP)

    Returns:
        proxy_ham: ProxyHamiltonian with CSR/CSC blocks
        P_dets: Perturbative configurations, shape [n_P, 2], dtype uint64
    """
    V_dets = np.ascontiguousarray(space.V_dets, dtype=np.uint64)
    n_v = int(V_dets.shape[0])

    if screening == "none":
        P_dets = core.gen_perturbative_dets(V_dets, system.n_orb, mode="none")
        result = core.get_ham_block(
            bra_dets=V_dets,
            ket_dets=P_dets,
            int_ctx=system.int_ctx,
            n_orb=system.n_orb,
        )
    elif screening == "static":
        result = core.get_ham_conn(
            dets_V=V_dets,
            int_ctx=system.int_ctx,
            n_orb=system.n_orb,
            use_heatbath=True,
            eps1=screen_eps,
        )
    elif screening == "dynamic":
        if psi_v is None:
            raise ValueError("Dynamic screening requires psi_v amplitudes on V")
        psi_v_arr = np.ascontiguousarray(psi_v, dtype=np.float64)
        if psi_v_arr.shape[0] != n_v:
            raise ValueError(f"psi_v shape mismatch: {psi_v_arr.shape[0]} != n_V={n_v}")
        result = core.get_ham_conn_amp(
            dets_V=V_dets,
            psi_v=psi_v_arr,
            int_ctx=system.int_ctx,
            n_orb=system.n_orb,
            eps1=screen_eps,
        )
    else:
        raise ValueError(f"Unknown screening mode: {screening!r}")

    coo_vv = result["H_VV"]
    coo_vp = result["H_VP"]
    P_dets = np.asarray(result["det_P"], dtype=np.uint64)
    n_p = int(result["size_P"])

    ham_vv = _coo_to_csr(
        rows=coo_vv["row"],
        cols=coo_vv["col"],
        vals=coo_vv["val"],
        shape=(n_v, n_v),
    )
    ham_vp = _coo_to_csr(
        rows=coo_vp["row"],
        cols=coo_vp["col"],
        vals=coo_vp["val"],
        shape=(n_v, n_p),
    )
    # H_PV = (H_VP)^T in CSC for efficient y_p = H_PV @ psi_v
    ham_pv = ham_vp.transpose().conjugate()

    h_diag_v = np.asarray(ham_vv.diagonal(), dtype=np.float64)
    if n_p > 0:
        h_diag_p = core.get_ham_diag(dets=P_dets, int_ctx=system.int_ctx)
        h_diag_p = np.ascontiguousarray(h_diag_p, dtype=np.float64) + float(diag_shift)
    else:
        h_diag_p = np.zeros((0,), dtype=np.float64)

    diagonals = DiagonalInfo(V=h_diag_v, P=h_diag_p)
    ham = ProxyHamiltonian(
        ham_vv=ham_vv, ham_vp=ham_vp, ham_pv=ham_pv, diagonals=diagonals
    )
    return ham, P_dets


def build_effective_hamiltonian(
    system: MolecularSystem,
    proxy_ham: ProxyHamiltonian,
    e_ref: float,
    *,
    reg_type: Literal["sigma", "linear_shift"] = "sigma",
    epsilon: float = 1e-12,
    upper_only: bool = True,
) -> VVHamiltonian:
    """
    Build effective Hamiltonian via Löwdin partitioning on V.

    Downfolds P-space effects into H_eff on V:
      H_eff = H_VV + H_VP (e_ref - diag(H_PP))^{-1} H_PV

    Regularization avoids division by near-zero denominators.

    Args:
        system: Molecular system (for API consistency, not used)
        proxy_ham: Proxy Hamiltonian with H_VV, H_VP, diag(H_PP)
        e_ref: Reference energy for perturbative denominator
        reg_type: Regularization strategy ("sigma" or "linear_shift")
        epsilon: Regularization parameter
        upper_only: If True, only upper triangle is computed

    Returns:
        VVHamiltonian with effective H_eff on V
    """
    # C++ backend expects COO triplets; convert CSR -> COO
    H_VV_coo = proxy_ham.ham_vv.tocoo()
    H_VP_coo = proxy_ham.ham_vp.tocoo()

    H_VV_dict = {
        "row": np.asarray(H_VV_coo.row, dtype=np.int64),
        "col": np.asarray(H_VV_coo.col, dtype=np.int64),
        "val": np.asarray(H_VV_coo.data, dtype=np.float64),
        "shape": proxy_ham.ham_vv.shape,
    }
    H_VP_dict = {
        "row": np.asarray(H_VP_coo.row, dtype=np.int64),
        "col": np.asarray(H_VP_coo.col, dtype=np.int64),
        "val": np.asarray(H_VP_coo.data, dtype=np.float64),
        "shape": proxy_ham.ham_vp.shape,
    }

    h_pp_diag = np.ascontiguousarray(proxy_ham.diagonals.P, dtype=np.float64)

    result = core.get_ham_eff(
        H_VV=H_VV_dict,
        H_VP=H_VP_dict,
        h_pp_diag=h_pp_diag,
        e_ref=float(e_ref),
        reg_type=reg_type,
        epsilon=float(epsilon),
        upper_only=bool(upper_only),
    )

    ham_eff = _coo_to_csr(
        rows=result["row"],
        cols=result["col"],
        vals=result["val"],
        shape=tuple(result["shape"]),
    )

    h_diag_v = np.asarray(ham_eff.diagonal(), dtype=np.float64)
    diagonals = DiagonalInfo(V=h_diag_v, P=np.zeros((0,), dtype=np.float64))
    return VVHamiltonian(ham_vv=ham_eff, diagonals=diagonals)


__all__ = [
    "DiagonalInfo",
    "VVHamiltonian",
    "ProxyHamiltonian",
    "build_vv_hamiltonian",
    "build_proxy_hamiltonian",
    "build_effective_hamiltonian",
]