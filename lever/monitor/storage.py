# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Persistent storage for LEVER run artifacts and metadata.

Provides JSON serialization, file management, and data persistence
for energy histories, wavefunction amplitudes, and configuration.

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

PathLike = Union[str, Path]


@dataclass
class RunMeta:
    """Metadata container for LEVER run identification."""
    run_id: str
    created_at: str
    system_name: str
    notes: str | None = None


def _as_dir(root: PathLike) -> Path:
    """Convert path-like object to normalized directory Path."""
    return Path(root)


def _to_json_safe(obj: Any) -> Any:
    """
    Recursively convert object to JSON-serializable representation.
    
    Conversion rules:
      - dataclass → dict (recursive field conversion)
      - Enum      → .value
      - Path      → string representation  
      - NumPy     → Python native types
      - Container → recursive conversion
    """
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
    Manages run directory lifecycle and artifact persistence.
    
    Creates unique timestamped directories and provides file handles
    for logging and configuration storage.
    """

    def __init__(self, root_dir: str = "runs", system_name: str = "System") -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{system_name}"
        self.root = Path(root_dir) / self.run_id
        self.root.mkdir(parents=True, exist_ok=True)

        self.meta = RunMeta(
            run_id=self.run_id,
            created_at=datetime.now().isoformat(timespec="seconds"),
            system_name=system_name,
        )

        self._log_file: TextIO = open(self.root / "stdout.log", "a", encoding="utf-8")

    @property
    def log_file(self) -> TextIO:
        """Get writable log file handle for this run."""
        return self._log_file

    def save_meta(self) -> None:
        """Persist run metadata as JSON."""
        with open(self.root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, indent=2)

    def save_config(self, cfg: Any, model: Any | None = None) -> None:
        """
        Save configuration and model metadata.
        
        Args:
            cfg: LeverConfig instance
            model: Wavefunction model (optional)
        """
        cfg_safe = _to_json_safe(cfg)

        payload: Dict[str, Any] = {
            "config": cfg_safe,
            "model_class": model.__class__.__name__ if model is not None else None,
            "run_meta": asdict(self.meta),
        }

        with open(self.root / "config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def close(self) -> None:
        """Safely close log file handle."""
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
    """
    Save energy optimization history with cycle tracking.
    
    Stores:
      - inner_energy_trace: All optimizer step energies
      - cycle_boundaries: Inner trace indices marking cycle transitions  
      - outer_energies: Final energy per outer cycle
      - outer_steps: Inner steps per outer cycle
    """
    root_path = _as_dir(root)
    out_path = root_path / "history.json"

    inner = [float(e) for e in result.full_energy_history]
    cycles = [int(c) for c in result.cycle_boundaries]
    outer_e = [float(x) for x in outer_energies]
    outer_n = [int(x) for x in outer_steps]

    payload = {
        "inner_energy_trace": inner,
        "full_energy_history": inner,  # Legacy compatibility
        "cycle_boundaries": cycles,
        "outer_energies": outer_e,
        "outer_steps": outer_n,
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
    Save compact final result summary.
    
    Includes final energy, timing, space dimensions, and electron counts.
    """
    root_path = _as_dir(root)
    out_path = root_path / "summary.json"

    inner = result.full_energy_history
    final_energy = float(inner[-1]) if inner else 0.0
    n_cycles = max(0, len(result.cycle_boundaries) - 1)
    total_time = float(result.total_time)

    cfg = result.config
    sys_cfg = cfg.system

    # Extract final space dimensions
    n_s = 0
    n_c = 0
    if result.final_space is not None:
        space = result.final_space
        n_s = int(space.s_dets.shape[0])
        n_c = int(space.c_dets.shape[0])

    payload: Dict[str, Any] = {
        "final_energy": final_energy,
        "total_time": total_time,
        "n_cycles": n_cycles,
        "n_steps": len(inner),
        "energies": {
            "e_lever": final_energy,
            "outer_last": float(outer_energies[-1]) if outer_energies else final_energy,
        },
        "space": {
            "n_s": n_s,
            "n_c": n_c,
            "n_t": n_s + n_c,
            "n_orbitals": int(sys_cfg.n_orbitals),
            "n_alpha": int(sys_cfg.n_alpha),
            "n_beta": int(sys_cfg.n_beta),
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_energies_to_summary(
    root: PathLike,
    extra_energies: Dict[str, float],
) -> None:
    """
    Merge additional energy diagnostics into existing summary.
    
    Used for post-analysis results like E_VAR and E_FCI.
    """
    root_path = _as_dir(root)
    path = root_path / "summary.json"
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    energies = dict(data.get("energies", {}))
    for k, v in extra_energies.items():
        energies[k] = float(v)

    data["energies"] = energies

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_psi_cache(root: PathLike, result: LeverResult) -> None:
    """
    Save final wavefunction amplitudes ψ = [ψ_S, ψ_C].
    
    Stores complex amplitudes and S-space dimension for reconstruction.
    """
    cache = result.final_psi_cache
    if cache is None:
        return

    root_path = _as_dir(root)
    out_path = root_path / "psi_cache.npz"

    psi_all = np.array(cache.psi_all)

    np.savez(
        out_path,
        psi_all=psi_all,
        n_s=int(cache.n_s),
    )


def save_params(root: PathLike, result: LeverResult) -> None:
    """
    Save variational parameters using Flax serialization.
    
    Uses msgpack format for efficient PyTree storage.
    """
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
    """
    Save all main run artifacts in single call.
    
    Convenience wrapper for history, summary, wavefunction, and parameters.
    """
    save_history(root, result, outer_energies=outer_energies, outer_steps=outer_steps)
    save_summary(root, result, outer_energies=outer_energies)
    save_psi_cache(root, result)
    save_params(root, result)


# ============================================================================
# Load Operations
# ============================================================================

def load_history(root: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Load energy history and cycle boundaries.
    
    Returns:
        Tuple of (inner_energy_trace, cycle_boundaries) as numpy arrays
    """
    root_path = _as_dir(root)
    path = root_path / "history.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    inner = data.get("inner_energy_trace", data.get("full_energy_history", []))
    energies = np.asarray(inner, dtype=float)
    cycles = np.asarray(data.get("cycle_boundaries", []), dtype=int)
    return energies, cycles


def load_summary(root: PathLike) -> Dict[str, Any]:
    """
    Load final result summary.
    
    Returns dict with keys: final_energy, total_time, n_cycles, n_steps, 
    energies, space.
    """
    root_path = _as_dir(root)
    path = root_path / "summary.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "final_energy": float(data.get("final_energy", 0.0)),
        "total_time": float(data.get("total_time", 0.0)),
        "n_cycles": int(data.get("n_cycles", 0)),
        "n_steps": int(data.get("n_steps", 0)),
        "energies": dict(data.get("energies", {})),
        "space": dict(data.get("space", {})),
    }


def load_config(root: PathLike) -> Dict[str, Any] | None:
    """Load run configuration from JSON, returns None if missing."""
    root_path = _as_dir(root)
    cfg_path = root_path / "config.json"
    if not cfg_path.exists():
        return None

    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(root: PathLike) -> Dict[str, Any] | None:
    """Load run metadata from JSON, returns None if missing."""
    root_path = _as_dir(root)
    meta_path = root_path / "meta.json"
    if not meta_path.exists():
        return None

    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_psi_cache(root: PathLike) -> Dict[str, Any] | None:
    """
    Load wavefunction amplitudes from NPZ file.
    
    Returns dict with keys: psi_all (complex array), n_s (int)
    """
    root_path = _as_dir(root)
    path = root_path / "psi_cache.npz"
    if not path.exists():
        return None

    data = np.load(path)
    return {
        "psi_all": data["psi_all"],
        "n_s": int(data["n_s"]),
    }


def load_params(root: PathLike, template: Any) -> Any | None:
    """
    Load variational parameters using Flax deserialization.
    
    Args:
        template: PyTree structure matching original parameters
        
    Returns:
        Reconstructed parameter tree or None if unavailable
    """
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
