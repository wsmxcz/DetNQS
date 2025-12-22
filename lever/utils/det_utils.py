# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinant batch preparation (C++ backend).

Wraps C++ nanobind function `_lever_cpp.prepare_det_batch` to build
device-ready DetBatch with occupation lists and optional auxiliary data
(k-body excitation levels, phases, holes/particles indices).

C++ backend uses std::popcount/countr_zero for bit operations and computes
phase via ordered annihilation (holes DESC) then creation (particles ASC).

File: lever/utils/det_batch.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
from flax import struct

from .. import core


@struct.dataclass
class DetBatch:
    """
    Device-ready determinant batch.
    
    Attributes:
      occ: int32[B, n_e] - occupation list per determinant
      aux: dict - optional fields (k, phase, holes, parts, etc.)
    """
    occ: Any  # int32[B, n_e]
    aux: dict[str, Any] = struct.field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.occ.shape[0])

    def __getitem__(self, idx) -> DetBatch:
        return jax.tree_util.tree_map(lambda x: x[idx], self)


@dataclass(frozen=True)
class BatchSpec:
    """
    Model requirements for batch preparation.
    
    Attributes:
      need_k: compute k-body excitation level
      hp_kmax: max excitation level for holes/particles (0 = disabled)
      need_phase: compute excitation phase
      need_hp_pos: compute positional indices for holes/particles
    """
    need_k: bool = False
    hp_kmax: int = 0
    need_phase: bool = False
    need_hp_pos: bool = False


def get_batch_spec(model: Any) -> BatchSpec:
    """Extract BatchSpec from model, return default if absent."""
    spec = getattr(model, "batch_spec", None)
    if spec is None:
        return BatchSpec()
    if not isinstance(spec, BatchSpec):
        raise TypeError(f"model.batch_spec must be BatchSpec, got {type(spec)}")
    return spec


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
    Prepare DetBatch via C++ backend.
    
    Calls C++ function to convert bitstring determinants into occupation lists
    and compute auxiliary quantities based on BatchSpec.
    
    Args:
      bitstrings: uint64[B, 2] - alpha/beta bitstrings
      n_orb: number of spatial orbitals
      n_alpha: number of alpha electrons
      n_beta: number of beta electrons
      ref: uint64[2] - reference determinant
      spec: BatchSpec - model requirements
      
    Returns:
      DetBatch with occ and optional aux fields
      
    Raises:
      ValueError: invalid input shapes or spec configuration
    """
    dets = np.asarray(bitstrings, dtype=np.uint64)
    if dets.ndim != 2 or dets.shape[1] != 2:
        raise ValueError(f"bitstrings must be (B, 2), got {dets.shape}")

    ref = np.asarray(ref, dtype=np.uint64).reshape(2)
    if ref.shape != (2,):
        raise ValueError(f"ref must be (2,), got {ref.shape}")

    if spec.need_hp_pos and spec.hp_kmax <= 0:
        raise ValueError("need_hp_pos requires hp_kmax > 0")

    need_hp = spec.hp_kmax > 0
    need_k = spec.need_k or need_hp

    out = core.prepare_det_batch(
        dets,
        ref,
        int(n_orb),
        int(n_alpha),
        int(n_beta),
        int(spec.hp_kmax),
        bool(need_k),
        bool(spec.need_phase),
        bool(need_hp),
        bool(spec.need_hp_pos),
    )

    aux: dict[str, np.ndarray] = {}
    
    if need_k:
        aux["k"] = out["k"]
    if spec.need_phase:
        aux["phase"] = out["phase"]
    if need_hp:
        aux["holes"] = out["holes"]
        aux["parts"] = out["parts"]
        aux["hp_mask"] = out["hp_mask"]
        if spec.need_hp_pos:
            aux["holes_pos"] = out["holes_pos"]
            aux["parts_pos"] = out["parts_pos"]

    return DetBatch(occ=out["occ"], aux=aux)


__all__ = ["DetBatch", "BatchSpec", "get_batch_spec", "prepare_batch"]
