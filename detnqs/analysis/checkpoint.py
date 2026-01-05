# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Checkpoint management for state persistence and resumption.

Storage design:
  - V_dets (uint64): Variational set V_k determinants
  - params (msgpack): Network parameters via Flax serialization
  - opt_state (msgpack): Optimizer state
  - metadata (JSON): Mode, total energy, timestamp

The perturbative set P_k is deterministically rebuilt from
V_k and system.int_ctx to ensure reproducibility.

File: detnqs/analysis/checkpoint.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optax
from flax import serialization

from .callbacks import BaseCallback

if TYPE_CHECKING:
    from ..driver import BaseDriver


class CheckpointCallback(BaseCallback):
    """Periodic checkpoint with automatic cleanup and guaranteed final save."""

    def __init__(
        self,
        directory: str | Path,
        interval: int = 10,
        keep_last: int = 5,
    ):
        """
        Args:
            directory: Checkpoint output directory
            interval: Save frequency in outer steps
            keep_last: Maximum checkpoints to retain
        """
        self.directory = Path(directory)
        self.interval = interval
        self.keep_last = keep_last
        self.directory.mkdir(parents=True, exist_ok=True)
        self._last_saved_step = -1

    def on_outer_end(self, step: int, stats: dict, driver: BaseDriver) -> None:
        """Save checkpoint at multiples of interval."""
        if step % self.interval != 0:
            return

        ckpt_path = self.directory / f"step_{step:06d}.npz"
        CheckpointManager.save(ckpt_path, driver, stats)
        self._last_saved_step = step
        self._cleanup_old()

    def on_run_end(self, driver: BaseDriver) -> None:
        """Ensure final step is always saved."""
        # Check if driver has valid final step info
        if not hasattr(driver, '_last_outer_step') or driver._last_outer_step < 0:
            return
        
        final_step = driver._last_outer_step
        
        # Only save if not already saved in on_outer_end
        if final_step != self._last_saved_step:
            ckpt_path = self.directory / f"step_{final_step:06d}_final.npz"
            
            # Use saved stats or create minimal version
            stats = getattr(driver, '_last_stats', {"outer_step": final_step})
            
            CheckpointManager.save(ckpt_path, driver, stats)
            print(f"Saved final checkpoint: {ckpt_path.name}")

    def _cleanup_old(self) -> None:
        """Remove checkpoints beyond keep_last limit (excluding final checkpoints)."""
        all_ckpts = sorted([p for p in self.directory.glob("step_*.npz") if "_final" not in p.name])
        if len(all_ckpts) > self.keep_last:
            for old_ckpt in all_ckpts[: -self.keep_last]:
                old_ckpt.unlink()


class CheckpointManager:
    """Low-level checkpoint I/O with atomic write guarantees."""

    @staticmethod
    def save(path: Path, driver: BaseDriver, stats: dict | None = None) -> None:
        """
        Atomic checkpoint save via temporary file.

        Serializes variational set, network params, optimizer state, and metadata.
        All energies stored are total energies (E_total = E_elec + E_nuc).
        Uses atomic swap to prevent corruption during concurrent writes.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp.npz")

        # Serialize JAX PyTrees to msgpack bytes
        params_bytes = serialization.to_bytes(driver.state.params)
        opt_state_bytes = serialization.to_bytes(driver.opt_state)

        # Build metadata (all energies are total energies)
        metadata = {
            "outer_step": stats.get("outer_step", 0) if stats else 0,
            "mode": driver.mode_tag(),
            "fcidump_path": str(driver.system.fcidump_path.absolute()),
            "energy": stats.get("energy") if stats else None,  # Total energy
            "timestamp": datetime.now().isoformat(),
        }

        # Mode-specific fields (store electronic energy for Effective mode reference)
        if hasattr(driver, "e_ref"):
            metadata["e_ref_elec"] = float(driver.e_ref)  # Electronic energy reference

        np.savez_compressed(
            tmp_path,
            V_dets=driver.detspace.V_dets,
            params=np.frombuffer(params_bytes, dtype=np.uint8),
            opt_state=np.frombuffer(opt_state_bytes, dtype=np.uint8),
            metadata=json.dumps(metadata),
        )

        # Atomic swap
        shutil.move(str(tmp_path), str(path))

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        """
        Load checkpoint from .npz archive.

        Returns:
            Dict with keys: V_dets, params, opt_state, metadata
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)

        # Deserialize PyTrees from bytes
        params = serialization.from_bytes(None, data["params"].tobytes())
        opt_state = serialization.from_bytes(None, data["opt_state"].tobytes())
        metadata = json.loads(str(data["metadata"]))

        return {
            "V_dets": data["V_dets"],
            "params": params,
            "opt_state": opt_state,
            "metadata": metadata,
        }

    @staticmethod
    def resume(
        path: Path,
        driver_cls: type[BaseDriver],
        model: Any,
        optimizer: optax.GradientTransformation,
        fcidump_path: str | Path | None = None,
        **override_kwargs: Any,
    ) -> BaseDriver:
        """
        Resume driver from checkpoint with PyTree structure restoration.

        Critical: Flax deserialization requires template PyTrees matching the
        original structure. We create dummy templates from the model and optimizer
        before loading serialized bytes.

        Args:
            path: Checkpoint file path
            driver_cls: Driver class constructor
            model: Network architecture (for param template)
            optimizer: Optimizer instance (for state template)
            fcidump_path: Override FCIDUMP location if moved
            **override_kwargs: Additional driver initialization args

        Returns:
            Fully restored driver instance
        """
        from ..space import DetSpace
        from ..state import State
        from ..system import MolecularSystem

        # Load raw checkpoint data
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["metadata"]))

        # Rebuild molecular system
        final_fcidump = Path(fcidump_path or meta["fcidump_path"])
        system = MolecularSystem.from_fcidump(final_fcidump)

        # Rebuild determinant space from variational set
        detspace = DetSpace.initialize(data["V_dets"])

        # Create PyTree templates for deserialization
        temp_state = State.init(system=system, detspace=detspace, model=model)
        temp_opt_state = optimizer.init(temp_state.params)

        # Deserialize using templates
        params = serialization.from_bytes(
            temp_state.params, data["params"].tobytes()
        )
        opt_state = serialization.from_bytes(
            temp_opt_state, data["opt_state"].tobytes()
        )

        # Reconstruct state with loaded params
        state = temp_state.replace(params=params)

        # Instantiate driver
        driver = driver_cls(
            system=system,
            state=state,
            detspace=detspace,
            optimizer=optimizer,
            opt_state=opt_state,
            **override_kwargs,
        )

        # Restore mode-specific fields (electronic energy reference)
        if "e_ref_elec" in meta and hasattr(driver, "e_ref"):
            driver.e_ref = float(meta["e_ref_elec"])

        return driver


__all__ = ["CheckpointCallback", "CheckpointManager"]