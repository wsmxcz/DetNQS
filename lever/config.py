# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified configuration management using Pydantic.

Acts as the Single Source of Truth (SSOT) for numerical experiments.
Loads physical parameters from SystemMeta and manages optimization hyperparameters.

File: lever/config.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: November, 2025
"""

from __future__ import annotations
import importlib
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

# Import SystemMeta from the concrete metadata module to avoid circular imports
from .interface.metadata import SystemMeta


# ============================================================================
# Enums
# ============================================================================

class ScreenMode(str, Enum):
    """Strategies for screening Connected (C) space determinants."""
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class ComputeMode(str, Enum):
    """Hamiltonian contraction strategies."""
    ASYMMETRIC = "asymmetric"
    PROXY = "proxy"
    EFFECTIVE = "effective"


# ============================================================================
# Sub-Configurations
# ============================================================================

class SystemConfig(BaseModel):
    """
    Runtime view of the physical system (projected from SystemMeta).

    Attributes like 'root_dir' are ephemeral runtime contexts used to
    resolve absolute paths and are not serialized to YAML.
    """
    model_config = ConfigDict(frozen=True, extra="ignore")

    name: str

    # Dimensions (active space)
    n_orb: int
    n_alpha: int
    n_beta: int
    n_core: int
    is_cas: bool

    # Files (relative to root_dir)
    meta_file: str
    fcidump_file: str

    # Context (injected at runtime, excluded from serialization)
    root_dir: Path = Field(exclude=True)

    @classmethod
    def from_meta(cls, meta: SystemMeta, root_dir: Path) -> "SystemConfig":
        """
        Factory: Project SystemMeta into SystemConfig.

        Assumes the metadata JSON is stored as 'system.json' in root_dir.
        FCIDUMP filename is taken from meta.files.fcidump.
        """
        return cls(
            name=meta.system.name,
            n_orb=meta.orbital_space.n_orb,
            n_alpha=meta.orbital_space.n_alpha,
            n_beta=meta.orbital_space.n_beta,
            n_core=meta.orbital_space.n_core,
            is_cas=meta.orbital_space.is_cas,
            meta_file="system.json",
            fcidump_file=meta.files.fcidump,
            root_dir=root_dir,
        )

    @property
    def fcidump_path(self) -> str:
        """Absolute path to FCIDUMP for integral loading."""
        return str(self.root_dir / self.fcidump_file)

    @property
    def meta_path(self) -> str:
        """Absolute path to metadata JSON for initial state loading."""
        return str(self.root_dir / self.meta_file)


class HamiltonianConfig(BaseModel):
    """Parameters for Hamiltonian construction and screening."""
    screening_mode: ScreenMode = ScreenMode.DYNAMIC
    screen_eps: float = 1e-6
    diag_shift: float = 0.5
    reg_eps: float = 1e-4


class LoopConfig(BaseModel):
    """Control parameters for Outer (Evolution) and Inner (Optimization) loops."""
    max_outer: int = 20
    outer_tol: float = 1e-6
    outer_patience: int = 3

    max_inner: int = 500
    inner_tol: float = 1e-8
    inner_patience: int = 100

    chunk_size: Optional[int] = 8192


class RuntimeConfig(BaseModel):
    """
    Runtime environment settings (precision, RNG seed, JAX config).
    """
    seed: int = 42
    enable_x64: bool = True
    compute_mode: ComputeMode = ComputeMode.PROXY
    spin_flip_symmetry: bool = False

    # Advanced stability / logging
    num_eps: float = 1e-12
    normalize_wf: bool = True
    report_interval: int = 50

    def apply(self) -> None:
        """Apply global JAX configuration."""
        jax.config.update("jax_enable_x64", self.enable_x64)

    @property
    def jax_float(self):
        from jax import dtypes
        return dtypes.canonicalize_dtype(float)

    @property
    def jax_complex(self):
        from jax import dtypes
        return dtypes.canonicalize_dtype(complex)

    @property
    def numpy_float(self):
        return np.dtype(self.jax_float).type

    @property
    def numpy_complex(self):
        return np.dtype(self.jax_complex).type


class ObjectSpec(BaseModel):
    """Snapshot of a Python object's configuration (Model/Optimizer/Strategy)."""
    target: str  # e.g., "lever.models.Backflow"
    params: Dict[str, Any] = Field(default_factory=dict)

    def instantiate(self) -> Any:
        """
        Dynamically import and construct the target object.

        Recursively instantiates nested ObjectSpecs in params.
        """
        module_name, attr_name = self.target.rsplit(".", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, attr_name)

        # Recursively instantiate nested specs
        resolved_params: Dict[str, Any] = {}
        for k, v in self.params.items():
            if isinstance(v, dict) and "target" in v and "params" in v:
                # Nested spec: recursively instantiate
                resolved_params[k] = ObjectSpec(**v).instantiate()
            else:
                resolved_params[k] = v

        return factory(**resolved_params)


# ============================================================================
# Main Experiment Config
# ============================================================================

class ExperimentConfig(BaseModel):
    """
    Static snapshot of a numerical experiment.

    Ties together:
      - System (physics and files)
      - Hamiltonian construction
      - Optimization loops
      - Runtime environment
    """
    # Core components
    system: SystemConfig
    hamiltonian: HamiltonianConfig = Field(default_factory=HamiltonianConfig)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    # Snapshots (filled by RunContext.record_experiment)
    model_spec: Optional[ObjectSpec] = None
    optimizer_spec: Optional[ObjectSpec] = None
    strategy_spec: Optional[ObjectSpec] = None

    @classmethod
    def from_meta(
        cls,
        meta: SystemMeta,
        root_dir: Union[str, Path],
        *,
        hamiltonian: Optional[HamiltonianConfig] = None,
        loop: Optional[LoopConfig] = None,
        runtime: Optional[RuntimeConfig] = None,
    ) -> "ExperimentConfig":
        """
        Python-first initialization with explicit arguments.

        Args:
            meta: SystemMeta describing the processed system
            root_dir: Directory containing system.json and FCIDUMP
        """
        root_dir = Path(root_dir)
        sys_cfg = SystemConfig.from_meta(meta, root_dir)

        cfg = cls(
            system=sys_cfg,
            hamiltonian=hamiltonian or HamiltonianConfig(),
            loop=loop or LoopConfig(),
            runtime=runtime or RuntimeConfig(),
        )
        # Apply global settings immediately
        cfg.runtime.apply()
        return cfg

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExperimentConfig":
        """Load from YAML file and apply runtime settings."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Inject root_dir context (relative files are local to config.yaml)
        sys_data = data.pop("system")
        sys_data["root_dir"] = path.parent
        sys_cfg = SystemConfig(**sys_data)

        cfg = cls(system=sys_cfg, **data)
        cfg.runtime.apply()
        return cfg

    def save(self, path: Union[str, Path]) -> None:
        """Save to YAML file (root_dir is excluded by model_config)."""
        path = Path(path)
        data = self.model_dump(mode="json", exclude_none=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def build_components(self) -> tuple[Any, Any, Any]:
        """
        Rebuild model, optimizer, and strategy from recorded specs.

        Returns:
            (model, optimizer, strategy): instantiated components
        """
        model = self.model_spec.instantiate() if self.model_spec else None
        optimizer = self.optimizer_spec.instantiate() if self.optimizer_spec else None
        strategy = self.strategy_spec.instantiate() if self.strategy_spec else None
        return model, optimizer, strategy


__all__ = [
    "ScreenMode",
    "ComputeMode",
    "SystemConfig",
    "HamiltonianConfig",
    "LoopConfig",
    "RuntimeConfig",
    "ExperimentConfig",
    "ObjectSpec",
]