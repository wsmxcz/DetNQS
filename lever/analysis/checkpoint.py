# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint management for state persistence and resumption.

Minimal storage design:
  - S_dets (uint64): Selected space determinants
  - params (msgpack): Neural network parameters via Flax serialization
  - opt_state (msgpack): Optimizer state
  - metadata (JSON): Mode, energy, timestamp for reconstruction

C-space is rebuilt from S-dets + system.int_ctx to ensure reproducibility.

File: lever/analysis/checkpoint.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from .callbacks import BaseCallback

import numpy as np
from flax import serialization
import optax

if TYPE_CHECKING:
    from ..driver import BaseDriver


class CheckpointCallback(BaseCallback):
    """Periodic checkpoint callback with automatic cleanup of old files."""

    def __init__(
        self,
        directory: str | Path,
        interval: int = 10,
        keep_last: int = 5,
    ):
        """
        Initialize checkpoint callback.

        Args:
            directory: Output directory for checkpoints
            interval: Save frequency (every N outer steps)
            keep_last: Maximum number of checkpoints to retain
        """
        self.directory = Path(directory)
        self.interval = interval
        self.keep_last = keep_last
      
        self.directory.mkdir(parents=True, exist_ok=True)

    def on_outer_end(self, step: int, stats: dict, driver: BaseDriver) -> None:
        """Save checkpoint if current step is a multiple of interval."""
        if step % self.interval != 0:
            return

        ckpt_path = self.directory / f"step_{step:06d}.npz"
        CheckpointManager.save(ckpt_path, driver, stats)
        self._cleanup_old()

    def _cleanup_old(self) -> None:
        """Remove checkpoints beyond keep_last limit."""
        all_ckpts = sorted(self.directory.glob("step_*.npz"))
      
        if len(all_ckpts) > self.keep_last:
            for old_ckpt in all_ckpts[: -self.keep_last]:
                old_ckpt.unlink()


class CheckpointManager:
    """Low-level checkpoint I/O operations."""

    @staticmethod
    def save(path: Path, driver: BaseDriver, stats: dict | None = None) -> None:
        """
        Atomic checkpoint save via temporary file.
      
        Serializes S-space dets, network params, optimizer state, and metadata.
        Uses atomic swap to prevent corruption during write.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp.npz")

        # Serialize JAX PyTrees to msgpack bytes
        params_bytes = serialization.to_bytes(driver.state.params)
        opt_state_bytes = serialization.to_bytes(driver.opt_state)

        # Build metadata dict
        metadata = {
            "outer_step": stats.get("outer_step", 0) if stats else 0,
            "mode": driver.mode_tag(),
            "fcidump_path": str(driver.system.fcidump_path.absolute()),
            "energy": stats.get("energy") if stats else None,
            "timestamp": datetime.now().isoformat(),
        }

        # Mode-specific extensions (e.g., Effective Hamiltonian reference)
        if hasattr(driver, "e_ref"):
            metadata["energy_ref"] = float(driver.e_ref)

        np.savez_compressed(
            tmp_path,
            S_dets=driver.detspace.S_dets,
            params=np.frombuffer(params_bytes, dtype=np.uint8),
            opt_state=np.frombuffer(opt_state_bytes, dtype=np.uint8),
            metadata=json.dumps(metadata),
        )
      
        # Atomic swap
        shutil.move(str(tmp_path), str(path))

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        """
        Load checkpoint from .npz file.
      
        Returns:
            Dictionary with keys: S_dets, params, opt_state, metadata
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        # Deserialize PyTrees from bytes
        params = serialization.from_bytes(None, data["params"].tobytes())
        opt_state = serialization.from_bytes(None, data["opt_state"].tobytes())
        metadata = json.loads(str(data["metadata"]))

        return {
            "S_dets": data["S_dets"],
            "params": params,
            "opt_state": opt_state,
            "metadata": metadata,
        }

    @staticmethod
    def resume(
        path: Path,
        driver_cls: type[BaseDriver],
        model: Any,
        optimizer: optax.GradientTransformation, # Added: need optimizer to build template
        fcidump_path: str | Path | None = None,
        **override_kwargs: Any,
    ) -> BaseDriver:
        """
        Resume driver from checkpoint with proper PyTree restoration.
        """
        # 1. Load raw bytes and metadata
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))
        
        # 2. Rebuild Molecular System
        from ..system import MolecularSystem
        final_fcidump = Path(fcidump_path or meta["fcidump_path"])
        system = MolecularSystem.from_fcidump(final_fcidump)
        
        # 3. Rebuild Determinant Space
        from ..space import DetSpace
        detspace = DetSpace.initialize(data["S_dets"])
        
        # 4. Create TEMPLATES for deserialization (Crucial Fix)
        from ..state import State
        # Create a dummy state to get the PyTree structure for params
        temp_state = State.init(system=system, detspace=detspace, model=model)
        # Create a dummy opt_state to get the PyTree structure for optimizer
        temp_opt_state = optimizer.init(temp_state.params)

        # 5. Deserialize using the templates
        params = serialization.from_bytes(
            temp_state.params, 
            data["params"].tobytes()
        )
        opt_state = serialization.from_bytes(
            temp_opt_state, 
            data["opt_state"].tobytes()
        )

        # 6. Reconstruct the actual state with loaded params
        state = temp_state.replace(params=params)

        # 7. Instantiate Driver
        driver = driver_cls(
            system=system,
            state=state,
            detspace=detspace,
            optimizer=optimizer,
            opt_state=opt_state,
            **override_kwargs,
        )
        
        if "energy_ref" in meta and hasattr(driver, "e_ref"):
            driver.e_ref = float(meta["energy_ref"])

        return driver


__all__ = ["CheckpointCallback", "CheckpointManager"]
