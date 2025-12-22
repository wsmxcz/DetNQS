# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DetBatch: device-ready determinant batch with auxiliary features.

Provides per-determinant tensors (occupation lists, excitation metadata) for
NQS-enhanced selected CI. Models declare required auxiliary fields via BatchSpec,
CPU prepares them once, GPU inner loop operates on pure gather+GEMM.

Ordering convention (for k×k Thouless):
  - Spin-orbital indices: alpha in [0, n_orb), beta in [n_orb, 2*n_orb)
  - Holes annihilated in DESCENDING order, particles created in ASCENDING order
  - Phase accounts for fermionic anti-commutation under this ordering

File: lever/utils/det_utils.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
from flax import struct


# ========================== DetBatch & BatchSpec =============================


@struct.dataclass
class DetBatch:
    """
    Device-ready determinant batch.

    Fields:
      occ: int32 [B, n_e] - occupied spin-orbital indices
      aux: dict[str, Array] - optional per-det features (k, phase, holes/parts, etc.)
    """
    occ: Any  # int32 [B, n_e]
    aux: dict[str, Any] = struct.field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.occ.shape[0])

    def __getitem__(self, idx) -> DetBatch:
        """Slice batch along first axis."""
        return jax.tree_util.tree_map(lambda x: x[idx], self)


@dataclass(frozen=True)
class BatchSpec:
    """
    Model requirements for CPU batch preparation.

    Fields:
      need_k: include excitation rank k
      hp_kmax: if >0, include padded holes/parts arrays + mask
      need_phase: include fermionic sign (+1/-1)
      need_hp_pos: include holes_pos/parts_pos (positions in ref occ/virt)
    """
    need_k: bool = False
    hp_kmax: int = 0
    need_phase: bool = False
    need_hp_pos: bool = False


def get_batch_spec(model: Any) -> BatchSpec:
    """Extract BatchSpec from model; default to occupation-only."""
    spec = getattr(model, "batch_spec", None)
    if spec is None:
        return BatchSpec()
    if isinstance(spec, BatchSpec):
        return spec
    raise TypeError("model.batch_spec must be a BatchSpec instance")


# ====================== Bitstring ↔ Occupation Conversion ====================


_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_u64(x: np.ndarray) -> np.ndarray:
    """Population count for uint64 array."""
    x = np.asarray(x, dtype=np.uint64)
    bytes_view = x.view(np.uint8).reshape(x.shape + (8,))
    return _BYTE_POPCOUNT[bytes_view].sum(axis=-1).astype(np.int16)


def _bit_positions_u64(word: np.uint64, max_bits: int) -> np.ndarray:
    """Extract indices of set bits in uint64, ascending order."""
    w = int(np.uint64(word))
    out: list[int] = []
    while w:
        lsb = w & -w
        idx = lsb.bit_length() - 1
        if idx >= max_bits:
            break
        out.append(idx)
        w ^= lsb
    return np.asarray(out, dtype=np.int32)


def bitstrings_to_occ(
    bitstrings: np.ndarray,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    dtype: Any = np.int32,
) -> np.ndarray:
    """
    Convert (alpha, beta) uint64 bitstrings to occupation lists.

    Args:
      bitstrings: uint64 [B, 2]
      n_orb: spatial orbitals (≤ 64)
      n_alpha, n_beta: electron counts

    Returns:
      occ: int32 [B, n_e], alpha in [0, n_orb), beta in [n_orb, 2*n_orb)
    """
    bitstrings = np.asarray(bitstrings, dtype=np.uint64)
    if bitstrings.ndim != 2 or bitstrings.shape[1] != 2:
        raise ValueError(f"Expected (B, 2), got {bitstrings.shape}")

    b = bitstrings.shape[0]
    n_e = n_alpha + n_beta

    masks = np.uint64(1) << np.arange(n_orb, dtype=np.uint64)
    a_occ = (bitstrings[:, 0:1] & masks) != 0
    b_occ = (bitstrings[:, 1:2] & masks) != 0

    _, cols_a = np.nonzero(a_occ)
    _, cols_b = np.nonzero(b_occ)

    if cols_a.size != b * n_alpha or cols_b.size != b * n_beta:
        raise ValueError("Electron count mismatch in bitstrings")

    alpha_idx = cols_a.reshape(b, n_alpha).astype(dtype, copy=False)
    beta_idx = cols_b.reshape(b, n_beta).astype(dtype, copy=False)

    occ = np.empty((b, n_e), dtype=dtype)
    occ[:, :n_alpha] = alpha_idx
    occ[:, n_alpha:] = beta_idx + np.int32(n_orb)
    return occ


# ======================= Reference Mapping ===================================


def _make_ref_maps(
    ref: np.ndarray,
    *,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build spin-orbital index → position in reference occupied/virtual lists.

    Returns:
      occ_pos_map: int32 [n_so], value in [0, n_e) if occupied, else -1
      virt_pos_map: int32 [n_so], value in [0, n_v) if virtual, else -1
    """
    ref = np.asarray(ref, dtype=np.uint64).reshape(2)
    ref_occ = bitstrings_to_occ(ref[None, :], n_orb=n_orb, n_alpha=n_alpha, n_beta=n_beta)[0]

    n_so = 2 * n_orb
    n_e = n_alpha + n_beta

    occ_pos_map = np.full((n_so,), -1, dtype=np.int32)
    occ_pos_map[ref_occ] = np.arange(n_e, dtype=np.int32)

    is_occ = np.zeros((n_so,), dtype=bool)
    is_occ[ref_occ] = True
    virt_list = np.nonzero(~is_occ)[0].astype(np.int32)

    virt_pos_map = np.full((n_so,), -1, dtype=np.int32)
    virt_pos_map[virt_list] = np.arange(virt_list.size, dtype=np.int32)
    return occ_pos_map, virt_pos_map


# ====================== Excitation Phase (Fermionic Sign) ====================


def _popcount_below_u64(word: np.uint64, idx: int) -> int:
    """Count set bits with positions < idx."""
    if idx <= 0:
        return 0
    if idx >= 64:
        return int(_popcount_u64(np.asarray(word, dtype=np.uint64)))
    mask = (np.uint64(1) << np.uint64(idx)) - np.uint64(1)
    return int(_popcount_u64(np.asarray(np.uint64(word) & mask, dtype=np.uint64)))


def _excitation_phase_spin(ref_word: np.uint64, det_word: np.uint64, n_orb: int) -> int:
    """
    Fermionic phase (+1/-1) for ref → det within one spin sector.

    Algorithm:
      1. Annihilate holes in DESCENDING orbital index
      2. Create particles in ASCENDING orbital index
      Phase flips for each odd number of occupied orbitals passed during reordering.
    """
    ref_word = np.uint64(ref_word)
    det_word = np.uint64(det_word)

    holes = ref_word & ~det_word
    parts = det_word & ~ref_word

    h_idx = _bit_positions_u64(holes, n_orb)
    p_idx = _bit_positions_u64(parts, n_orb)

    phase = 1
    occ = int(ref_word)

    for h in h_idx[::-1]:
        n = _popcount_below_u64(np.uint64(occ), int(h))
        if (n % 2) == 1:
            phase *= -1
        occ ^= (1 << int(h))

    for p in p_idx:
        n = _popcount_below_u64(np.uint64(occ), int(p))
        if (n % 2) == 1:
            phase *= -1
        occ ^= (1 << int(p))

    return int(phase)


def _excitation_phase(dets: np.ndarray, ref: np.ndarray, *, n_orb: int) -> np.ndarray:
    """Compute phase for (alpha, beta) bitstrings relative to reference."""
    dets = np.asarray(dets, dtype=np.uint64)
    ref = np.asarray(ref, dtype=np.uint64).reshape(2)
    ref_a, ref_b = ref[0], ref[1]

    out = np.empty((dets.shape[0],), dtype=np.int8)
    for i in range(dets.shape[0]):
        a, b = dets[i, 0], dets[i, 1]
        s_a = _excitation_phase_spin(ref_a, a, n_orb)
        s_b = _excitation_phase_spin(ref_b, b, n_orb)
        out[i] = np.int8(s_a * s_b)
    return out


# ========================= CPU Batch Preparation =============================


def prepare_batch(
    bitstrings: np.ndarray,
    *,
    n_orb: int,
    n_alpha: int,
    n_beta: int,
    ref: np.ndarray,
    spec: BatchSpec,
) -> DetBatch:
    """
    Prepare DetBatch on CPU with model-specified features.

    Always includes:
      - occ: int32 [B, n_e]

    Optional (per BatchSpec):
      - k: int8 [B] - excitation rank
      - phase: int8 [B] - fermionic sign
      - holes, parts: int32 [B, kmax] - padded with -1
      - hp_mask: bool [B, kmax]
      - holes_pos, parts_pos: int32 [B, kmax] - positions in ref lists, -1 for padding
    """
    dets = np.asarray(bitstrings, dtype=np.uint64)
    if dets.ndim != 2 or dets.shape[1] != 2:
        raise ValueError(f"Expected (B, 2), got {dets.shape}")

    if spec.need_hp_pos and spec.hp_kmax <= 0:
        raise ValueError("need_hp_pos requires hp_kmax > 0")

    ref = np.asarray(ref, dtype=np.uint64).reshape(2)
    ref_a, ref_b = ref

    occ = bitstrings_to_occ(dets, n_orb=n_orb, n_alpha=n_alpha, n_beta=n_beta)
    aux: dict[str, np.ndarray] = {}

    if spec.need_k or spec.hp_kmax > 0:
        alpha, beta = dets[:, 0], dets[:, 1]
        holes_a = ref_a & ~alpha
        holes_b = ref_b & ~beta
        k = _popcount_u64(holes_a) + _popcount_u64(holes_b)
        aux["k"] = np.clip(k, 0, 127).astype(np.int8)

    if spec.need_phase:
        aux["phase"] = _excitation_phase(dets, ref, n_orb=n_orb)

    if spec.hp_kmax > 0:
        kmax = int(spec.hp_kmax)
        b = dets.shape[0]
        n_so = 2 * n_orb

        alpha, beta = dets[:, 0], dets[:, 1]
        holes_a = ref_a & ~alpha
        holes_b = ref_b & ~beta
        parts_a = alpha & ~ref_a
        parts_b = beta & ~ref_b

        holes = np.full((b, kmax), -1, dtype=np.int32)
        parts = np.full((b, kmax), -1, dtype=np.int32)
        mask = np.zeros((b, kmax), dtype=bool)

        for i in range(b):
            h_a = _bit_positions_u64(holes_a[i], n_orb)[::-1]
            h_b = _bit_positions_u64(holes_b[i], n_orb)[::-1] + n_orb
            p_a = _bit_positions_u64(parts_a[i], n_orb)
            p_b = _bit_positions_u64(parts_b[i], n_orb) + n_orb

            h = np.concatenate([h_a, h_b]).astype(np.int32, copy=False)
            p = np.concatenate([p_a, p_b]).astype(np.int32, copy=False)

            m = min(kmax, h.size, p.size)
            if m > 0:
                holes[i, :m] = h[:m]
                parts[i, :m] = p[:m]
                mask[i, :m] = True

        aux["holes"] = holes
        aux["parts"] = parts
        aux["hp_mask"] = mask

        if spec.need_hp_pos:
            occ_map, virt_map = _make_ref_maps(ref, n_orb=n_orb, n_alpha=n_alpha, n_beta=n_beta)

            holes_clip = np.clip(holes, 0, n_so - 1)
            parts_clip = np.clip(parts, 0, n_so - 1)

            holes_pos = occ_map[holes_clip].astype(np.int32)
            parts_pos = virt_map[parts_clip].astype(np.int32)

            holes_pos[holes < 0] = -1
            parts_pos[parts < 0] = -1

            aux["holes_pos"] = holes_pos
            aux["parts_pos"] = parts_pos

    return DetBatch(occ=occ, aux=aux)


__all__ = [
    "DetBatch",
    "BatchSpec",
    "get_batch_spec",
    "prepare_batch",
    "bitstrings_to_occ",
]