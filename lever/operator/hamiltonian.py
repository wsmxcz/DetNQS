# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Hamiltonian constructionterminant spaces.

This version stores sparse blocks in SciPy CSR/CSC formats for fast CPU SpMV.
We keep the C++ backend API unchanged, but convert COO triplets to CSR/CSC once
per outer iteration (L1), and reuse them across inner steps (L2).

File: lever/operator/hamiltonian.py
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
    """Diagonal elements H_ii for S and C spaces."""
    S: np.ndarray  # [n_S], float64
    C: np.ndarray  # [n_C], float64 (empty if no C-space)


@dataclass(eq=False)
class SSHamiltonian:
    """S-space restricted Hamiltonian H_SS (CSR)."""
    ham_ss: sp.csr_matrix
    diagonals: DiagonalInfo


@dataclass(eq=False)
class ProxyHamiltonian:
    """
    Proxy Hamiltonian on T = S ⊕ C.

    Stored blocks:
      - ham_ss: CSR, shape (n_S, n_S)
      - ham_sc: CSR, shape (n_S, n_C)
      - ham_cs: CSC, shape (n_C, n_S) = conj(ham_sc.T)
    """
    ham_ss: sp.csr_matrix
    ham_sc: sp.csr_matrix
    ham_cs: sp.csc_matrix
    diagonals: DiagonalInfo


def _coo_to_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    vals: np.ndarray,
    shape: Tuple[int, int],
) -> sp.csr_matrix:
    """Build a CSR matrix from COO triplets (duplicates are summed)."""
    coo = sp.coo_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=shape,
    )
    return coo.tocsr()


def build_ss_hamiltonian(
    system: MolecularSystem,
    space: DetSpace,
) -> SSHamiltonian:
    """
    Build H_SS for fixed S-space.

    Returns:
        SSHamiltonian with CSR H_SS and diagonal elements.
    """
    S = np.ascontiguousarray(space.S_dets, dtype=np.uint64)
    n_s = int(S.shape[0])

    coo_ss = core.get_ham_ss(
        dets_S=S,
        int_ctx=system.int_ctx,
        n_orb=system.n_orb,
    )

    ham_ss = _coo_to_csr(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(n_s, n_s),
    )

    h_diag_s = np.asarray(ham_ss.diagonal(), dtype=np.float64)
    diagonals = DiagonalInfo(S=h_diag_s, C=np.zeros((0,), dtype=np.float64))
    return SSHamiltonian(ham_ss=ham_ss, diagonals=diagonals)


def build_proxy_hamiltonian(
    system: MolecularSystem,
    space: DetSpace,
    *,
    screening: Literal["none", "static", "dynamic"] = "static",
    psi_s: np.ndarray | None = None,
    screen_eps: float = 1e-6,
    diag_shift: float = 0.0,
) -> Tuple[ProxyHamiltonian, np.ndarray]:
    """
    Build proxy Hamiltonian blocks with screening.

    Returns:
        proxy_ham: ProxyHamiltonian with CSR/CSC blocks
        C_dets: [n_C, 2] uint64
    """
    S = np.ascontiguousarray(space.S_dets, dtype=np.uint64)
    n_s = int(S.shape[0])

    if screening == "none":
        C = core.gen_complement_dets(S, system.n_orb, mode="none")
        result = core.get_ham_block(
            bra_dets=S,
            ket_dets=C,
            int_ctx=system.int_ctx,
            n_orb=system.n_orb,
        )
    elif screening == "static":
        result = core.get_ham_conn(
            dets_S=S,
            int_ctx=system.int_ctx,
            n_orb=system.n_orb,
            use_heatbath=True,
            eps1=screen_eps,
        )
    elif screening == "dynamic":
        if psi_s is None:
            raise ValueError("Dynamic screening requires psi_s amplitudes")
        psi_s_arr = np.ascontiguousarray(psi_s, dtype=np.float64)
        if psi_s_arr.shape[0] != n_s:
            raise ValueError(f"psi_s size mismatch: {psi_s_arr.shape[0]} != n_S={n_s}")
        result = core.get_ham_conn_amp(
            dets_S=S,
            psi_S=psi_s_arr,
            int_ctx=system.int_ctx,
            n_orb=system.n_orb,
            eps1=screen_eps,
        )
    else:
        raise ValueError(f"Unknown screening mode: {screening!r}")

    coo_ss = result["H_SS"]
    coo_sc = result["H_SC"]
    C_dets = np.asarray(result["det_C"], dtype=np.uint64)
    n_c = int(result["size_C"])

    ham_ss = _coo_to_csr(
        rows=coo_ss["row"],
        cols=coo_ss["col"],
        vals=coo_ss["val"],
        shape=(n_s, n_s),
    )
    ham_sc = _coo_to_csr(
        rows=coo_sc["row"],
        cols=coo_sc["col"],
        vals=coo_sc["val"],
        shape=(n_s, n_c),
    )
    # For y_cs = H_CS @ psi_S, CSC is the natural format.
    ham_cs = ham_sc.transpose().conjugate()  # CSC matrix, shape (n_c, n_s)

    h_diag_s = np.asarray(ham_ss.diagonal(), dtype=np.float64)
    if n_c > 0:
        h_diag_c = core.get_ham_diag(dets=C_dets, int_ctx=system.int_ctx)
        h_diag_c = np.ascontiguousarray(h_diag_c, dtype=np.float64) + float(diag_shift)
    else:
        h_diag_c = np.zeros((0,), dtype=np.float64)

    diagonals = DiagonalInfo(S=h_diag_s, C=h_diag_c)
    ham = ProxyHamiltonian(ham_ss=ham_ss, ham_sc=ham_sc, ham_cs=ham_cs, diagonals=diagonals)
    return ham, C_dets


def build_effective_hamiltonian(
    system: MolecularSystem,
    proxy_ham: ProxyHamiltonian,
    e_ref: float,
    *,
    reg_type: Literal["sigma", "linear_shift"] = "sigma",
    epsilon: float = 1e-12,
    upper_only: bool = True,
) -> SSHamiltonian:
    """
    Build effective Hamiltonian via Löwdin partitioning.

    Note: core.get_ham_eff expects COO triplets; we convert CSR -> COO here.
    """
    H_SS_coo = proxy_ham.ham_ss.tocoo()
    H_SC_coo = proxy_ham.ham_sc.tocoo()

    H_SS_dict = {
        "row": np.asarray(H_SS_coo.row, dtype=np.int64),
        "col": np.asarray(H_SS_coo.col, dtype=np.int64),
        "val": np.asarray(H_SS_coo.data, dtype=np.float64),
        "shape": proxy_ham.ham_ss.shape,
    }
    H_SC_dict = {
        "row": np.asarray(H_SC_coo.row, dtype=np.int64),
        "col": np.asarray(H_SC_coo.col, dtype=np.int64),
        "val": np.asarray(H_SC_coo.data, dtype=np.float64),
        "shape": proxy_ham.ham_sc.shape,
    }

    h_cc = np.ascontiguousarray(proxy_ham.diagonals.C, dtype=np.float64)

    result = core.get_ham_eff(
        H_SS=H_SS_dict,
        H_SC=H_SC_dict,
        h_cc_diag=h_cc,
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

    h_diag_s = np.asarray(ham_eff.diagonal(), dtype=np.float64)
    diagonals = DiagonalInfo(S=h_diag_s, C=np.zeros((0,), dtype=np.float64))
    return SSHamiltonian(ham_ss=ham_eff, diagonals=diagonals)


__all__ = [
    "DiagonalInfo",
    "SSHamiltonian",
    "ProxyHamiltonian",
    "build_ss_hamiltonian",
    "build_proxy_hamiltonian",
    "build_effective_hamiltonian",
]