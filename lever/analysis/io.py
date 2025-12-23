# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Persistent storage for LEVER results.

Provides:
  - save: Persist DriverResult + Trace + optional analysis
  - load: Restore saved artifacts

File: lever/analysis/io.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..driver import DriverResult
from .trace import Trace


def save(
    path: Path,
    *,
    result: DriverResult,
    trace: Trace | None = None,
    analysis: dict[str, Any] | None = None,
) -> None:
    """
    Save driver result and artifacts.
  
    Directory structure:
        path/
          ├── space.npz      # S_dets, C_dets
          ├── trace.json     # Energy trace + timestamps + norms
          ├── analysis.json  # Optional post-hoc analysis
          └── params.msgpack # Neural network parameters
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
  
    # Save determinant space
    s_dets = np.asarray(result.detspace.S_dets, dtype=np.uint64)
    c_dets = getattr(result.detspace, "C_dets", None)
    if c_dets is not None:
        c_dets = np.asarray(c_dets, dtype=np.uint64)
    else:
        c_dets = np.zeros((0, 2), dtype=np.uint64)
  
    np.savez_compressed(path / "space.npz", S_dets=s_dets, C_dets=c_dets)
  
    # Save trace
    if trace is not None:
        trace_data = {
            "outers": trace.outers.tolist(),
            "energies": trace.energies.tolist(),
            "timestamps": trace.timestamps.tolist(),
            "size_s": trace.size_s.tolist(),
            "size_c": trace.size_c.tolist(),
            "norm_s": trace.norm_s.tolist(),
            "norm_c": trace.norm_c.tolist(),
        }
      
        with (path / "trace.json").open("w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
  
    # Save analysis
    if analysis is not None:
        def _serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _serialize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialize(x) for x in obj]
            return obj
      
        with (path / "analysis.json").open("w", encoding="utf-8") as f:
            json.dump(_serialize(analysis), f, indent=2)
  
    # Save parameters
    try:
        from flax import serialization
      
        with (path / "params.msgpack").open("wb") as f:
            f.write(serialization.to_bytes(result.state.params))
    except ImportError:
        pass


def load(path: Path) -> dict[str, Any]:
    """
    Load saved artifacts.
  
    Returns:
        Dict with: space_data, trace, analysis (all optional)
    """
    path = Path(path)
    out: dict[str, Any] = {}
  
    # Load space
    if (path / "space.npz").exists():
        out["space_data"] = dict(np.load(path / "space.npz"))
  
    # Load trace
    if (path / "trace.json").exists():
        with (path / "trace.json").open("r", encoding="utf-8") as f:
            trace_dict = json.load(f)
      
        out["trace"] = Trace(
            outers=np.array(trace_dict["outers"], dtype=int),
            energies=np.array(trace_dict["energies"]),
            timestamps=np.array(trace_dict["timestamps"]),
            size_s=np.array(trace_dict["size_s"], dtype=int),
            size_c=np.array(trace_dict["size_c"], dtype=int),
            norm_s=np.array(trace_dict["norm_s"]),
            norm_c=np.array(trace_dict["norm_c"]),
        )
  
    # Load analysis
    if (path / "analysis.json").exists():
        with (path / "analysis.json").open("r", encoding="utf-8") as f:
            out["analysis"] = json.load(f)
  
    return out


__all__ = ["save", "load"]
