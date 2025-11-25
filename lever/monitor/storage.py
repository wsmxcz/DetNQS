# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Persistent storage for LEVER run artifacts.

Manages JSON serialization and NumPy artifact storage with consolidated metadata.

File: lever/monitor/storage.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, TextIO, Union
import enum
import json

import numpy as np

from ..dtypes import LeverResult
from ..interface import load_benchmarks

PathLike = Union[str, Path]


def _as_dir(root: PathLike) -> Path:
    """Convert path-like object to normalized directory Path."""
    return Path(root)


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert object to JSON-serializable representation."""
    if is_dataclass(obj):
        return {k: _to_json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(_to_json_safe(k)): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


class RunContext:
    """
    Manages run directory lifecycle with timestamped unique directories.
    
    Creates `{timestamp}_{system_name}` directories and provides file handles.
    Metadata stored directly in config.json.
    """

    def __init__(self, root_dir: str = "runs", system_name: str = "System") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.system_name = system_name
        self.run_id = f"{timestamp}_{system_name}"
        self.root = Path(root_dir) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self._log_file: TextIO = open(self.root / "stdout.log", "a", encoding="utf-8")

    @property
    def log_file(self) -> TextIO:
        return self._log_file

    def save_config(self, cfg: Any, model: Any | None = None) -> None:
        """Save configuration, model metadata, and run identification to config.json."""
        cfg_safe = _to_json_safe(cfg)
        model_summary = None
        
        if model is not None and hasattr(model, "summary"):
            try:
                model_summary = _to_json_safe(model.summary)
            except Exception:
                model_summary = None

        payload: Dict[str, Any] = {
            "run_info": {
                "id": self.run_id,
                "system_name": self.system_name,
                "created_at": self.created_at,
            },
            "config": cfg_safe,
            "model_class": model.__class__.__name__ if model is not None else None,
            "model_summary": model_summary,
        }

        with open(self.root / "config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def close(self) -> None:
        """Close log file handle."""
        try:
            self._log_file.close()
        except Exception:
            pass


# ============================================================================
# Save Operations
# ============================================================================

def save_history(
    root: PathLike,
    result: LeverResult,
    *,
    outer_energies: list[float],
    outer_steps: list[int],
) -> None:
    """Save energy optimization history with cycle boundaries."""
    root_path = _as_dir(root)
    out_path = root_path / "history.json"

    payload = {
        "inner_energy_trace": [float(e) for e in result.full_energy_history],
        "cycle_boundaries": [int(c) for c in result.cycle_boundaries],
        "outer_energies": [float(x) for x in outer_energies],
        "outer_steps": [int(x) for x in outer_steps],
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_summary(
    root: PathLike,
    result: LeverResult,
    *,
    outer_energies: list[float],
) -> None:
    """
    Save comprehensive result summary with benchmark comparisons.
    
    Includes final energies, timing, system configuration, and benchmark energies
    from HDF5 interface when available.
    """
    root_path = _as_dir(root)
    out_path = root_path / "summary.json"

    inner = result.full_energy_history
    final_energy = float(inner[-1]) if inner else 0.0
    cfg = result.config
    sys_cfg = cfg.system

    # Load benchmark energies from HDF5
    benchmarks = {}
    if sys_cfg.meta_path:
        try:
            raw_bench = load_benchmarks(sys_cfg.meta_path)
            mapping = {
                "hf": "e_hf", "mp2": "e_mp2", "cisd": "e_cisd", 
                "ccsd": "e_ccsd", "ccsd_t": "e_ccsd_t", "casci": "e_casci", "fci": "e_fci"
            }
            for k, v in raw_bench.items():
                new_k = mapping.get(k.lower(), k)
                benchmarks[new_k] = float(v)
        except Exception:
            pass

    # Space dimensions
    n_s, n_c = 0, 0
    if result.final_space is not None:
        n_s = int(result.final_space.s_dets.shape[0])
        n_c = int(result.final_space.c_dets.shape[0])

    payload: Dict[str, Any] = {
        "final_energy": final_energy,
        "total_time": float(result.total_time),
        "n_cycles": max(0, len(result.cycle_boundaries) - 1),
        "n_steps": len(inner),
        "energies": {
            "e_lever": final_energy,
            "outer_last": float(outer_energies[-1]) if outer_energies else final_energy,
            **benchmarks,
        },
        "space": {
            "n_s": n_s,
            "n_c": n_c,
            "n_t": n_s + n_c,
        },
        "system": {
            "name": Path(sys_cfg.fcidump_path).stem,
            "n_orbitals": int(sys_cfg.n_orbitals),
            "n_alpha": int(sys_cfg.n_alpha),
            "n_beta": int(sys_cfg.n_beta),
            "fcidump_path": str(sys_cfg.fcidump_path),
            "meta_path": sys_cfg.meta_path,
        },
        "params": {
            "screening_mode": cfg.hamiltonian.screening_mode.value,
            "compute_mode": cfg.compute_mode.value,
        }
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_energies_to_summary(
    root: PathLike,
    extra_energies: Dict[str, float],
) -> None:
    """Update summary.json with additional post-analysis energies."""
    root_path = _as_dir(root)
    path = root_path / "summary.json"
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    energies = dict(data.get("energies", {}))
    energies.update({k: float(v) for k, v in extra_energies.items()})
    data["energies"] = energies

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_space_info(root: PathLike, result: LeverResult) -> None:
    """
    Save final space definition and wavefunction amplitudes.
    
    File: space_info.npz
    Contains:
    - psi_all: Complex amplitudes ψ = [ψ_S, ψ_C]
    - s_dets: S-space determinants (bitstrings)
    - c_dets: C-space determinants (bitstrings)  
    - n_s: Number of S determinants
    """
    if result.final_psi_cache is None:
        return

    root_path = _as_dir(root)
    out_path = root_path / "space_info.npz"

    data_dict = {
        "psi_all": np.array(result.final_psi_cache.psi_all),
        "n_s": int(result.final_psi_cache.n_s),
    }

    if result.final_space is not None:
        data_dict["s_dets"] = np.array(result.final_space.s_dets)
        data_dict["c_dets"] = np.array(result.final_space.c_dets)

    np.savez(out_path, **data_dict)


def save_params(root: PathLike, result: LeverResult) -> None:
    """Save variational parameters using Flax serialization (msgpack)."""
    try:
        from flax import serialization
    except Exception:
        return

    root_path = _as_dir(root)
    out_path = root_path / "params.msgpack"

    data = serialization.to_bytes(result.final_params)
    with out_path.open("wb") as f:
        f.write(data)


def save_run_artifacts(
    root: PathLike,
    result: LeverResult,
    *,
    outer_energies: list[float],
    outer_steps: list[int],
) -> None:
    """Save complete run artifacts: history, summary, space info, and parameters."""
    save_history(root, result, outer_energies=outer_energies, outer_steps=outer_steps)
    save_summary(root, result, outer_energies=outer_energies)
    save_space_info(root, result)
    save_params(root, result)


# ============================================================================
# Load Operations  
# ============================================================================

def load_history(root: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load energy history and cycle boundaries."""
    root_path = _as_dir(root)
    path = root_path / "history.json"
    if not path.exists():
        return np.array([]), np.array([])
        
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    inner = data.get("inner_energy_trace", data.get("full_energy_history", []))
    energies = np.asarray(inner, dtype=float)
    cycles = np.asarray(data.get("cycle_boundaries", []), dtype=int)
    return energies, cycles


def load_summary(root: PathLike) -> Dict[str, Any]:
    """Load summary.json."""
    root_path = _as_dir(root)
    path = root_path / "summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(root: PathLike) -> Dict[str, Any] | None:
    """Load config.json."""
    root_path = _as_dir(root)
    cfg_path = root_path / "config.json"
    if not cfg_path.exists():
        return None
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_space_info(root: PathLike) -> Dict[str, Any] | None:
    """Load space_info.npz with bitstrings and amplitudes."""
    root_path = _as_dir(root)
    path = root_path / "space_info.npz"
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def load_params(root: PathLike, template: Any) -> Any | None:
    """Load variational parameters from params.msgpack using Flax serialization."""
    try:
        from flax import serialization
    except Exception:
        return None

    root_path = _as_dir(root)
    path = root_path / "params.msgpack"
    if not path.exists():
        return None

    with path.open("rb") as f:
        raw = f.read()

    return serialization.from_bytes(template, raw)


__all__ = [
    "RunContext", "save_history", "save_summary", "append_energies_to_summary",
    "save_space_info", "save_params", "save_run_artifacts", "load_history", 
    "load_summary", "load_config", "load_space_info", "load_params"
]
