# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

"""
Systematic scan of DetNQS modes and variational space sizes across molecules.

Sweeps over:
  - Molecular systems (multiple FCIDUMP files)
  - Computational modes: Variational, Proxy, Asymmetric
  - V-space sizes: |V| ∈ {64, 128, 256, ..., 8192}

Outputs structured JSONL logs and post-analysis summaries.

File: detnqs/examples/run_scan.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from detnqs import models
from detnqs.analysis import CheckpointCallback, JsonCallback
from detnqs.analysis.metrics import compute_pt2, compute_variational
from detnqs.driver import AsymmetricDriver, ProxyDriver, VariationalDriver
from detnqs.space import DetSpace
from detnqs.space.selector import TopKSelector
from detnqs.system import MolecularSystem


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class ScanConfig:
    """Scan parameters for multi-molecule mode and space size sweep."""
    # Input/Output
    fcidump_paths: tuple[Path, ...]
    output_root: Path = Path("./runs/N2_ccpvdz")
  
    # Scan dimensions
    modes: tuple[str, ...] = ("variational", "proxy", "asymmetric")
    topk_sizes: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
  
    # Model architecture
    mlp_dim: int = 256
    mlp_depth: int = 2
  
    # Optimization
    learning_rate: float = 1e-3
    chunk_size: int = 8192
  
    # Reproducibility
    seed: int = 42


_DRIVER_MAP = {
    "variational": VariationalDriver,
    "proxy": ProxyDriver,
    "asymmetric": AsymmetricDriver,
}


# ============================================================================
# JAX Configuration
# ============================================================================

def configure_jax() -> None:
    """Configure JAX runtime: GPU priority, float64 precision."""
    jax.config.update("jax_platforms", "cuda,cpu")
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_log_compiles", False)
    jax.config.update("jax_debug_nans", False)
  
    warnings.filterwarnings("ignore", message="Explicitly requested dtype.*is not available")
  
    device = jax.devices()[0]
    print(f"JAX backend: {device.platform.upper()}")
    print(f"Device count: {len(jax.devices())}\n")


# ============================================================================
# Helper Functions
# ============================================================================

def _build_model(system: MolecularSystem, config: ScanConfig):
    """Build neural ansatz with MLP parametrizer."""
    parametrizer = models.parametrizers.MLP(
        n_so=system.n_so,
        dim=config.mlp_dim,
        depth=config.mlp_depth,
        param_dtype=jnp.float64,
    )
  
    return models.make_slater2nd(
        system=system,
        parametrizer=parametrizer,
        mapper="full",
        use_fast_kernel=True,
        param_dtype=jnp.float64,
    )


def _run_optimization(driver, output_dir: Path) -> tuple[float, float, float]:
    """
    Execute optimization with callbacks.
    
    Returns:
        (e_total, total_time, inner_time_avg): Final total energy, total runtime, avg inner time
    """
    output_dir.mkdir(parents=True, exist_ok=True)
  
    callbacks = [
        JsonCallback(output_dir / "trace.jsonl"),
        CheckpointCallback(output_dir / "checkpoints", interval=5, keep_last=3),
    ]
  
    t_start = time.perf_counter()
    driver.run(callbacks=callbacks)
    total_runtime = time.perf_counter() - t_start
  
    # Extract final energy and timing from trace
    with open(output_dir / "trace.jsonl", 'r') as f:
        lines = f.readlines()
        last_record = json.loads(lines[-1])
        e_total = last_record['energy']  # Already total energy
        
        # Compute average inner loop time
        inner_times = [json.loads(line)['inner_time'] for line in lines if line.strip()]
        inner_time_avg = sum(inner_times) / len(inner_times) if inner_times else 0.0
  
    return e_total, total_runtime, inner_time_avg


def _post_analysis(mode: str, driver, system: MolecularSystem, e_total: float) -> dict:
    """
    Compute mode-specific post-optimization analysis with timing.
    
    Args:
        mode: Computational mode
        driver: Optimized driver
        system: Molecular system
        e_total: Reference total energy from optimization
        
    Returns:
        Dict with analysis results and timing
    """
    result = {}
  
    if mode == "variational":
        print("  Computing PT2 correction...")
        t_start = time.perf_counter()
        e_pt2 = compute_pt2(
            state=driver.state,
            detspace=driver.detspace,
            system=system,
            e_ref=e_total,
        )
        t_pt2 = time.perf_counter() - t_start
      
        if e_pt2 is not None:
            result["e_pt2"] = e_pt2
            result["e_total_pt2"] = e_total + e_pt2
            result["t_pt2"] = t_pt2
            print(f"  PT2: ΔE = {e_pt2:.8f} Ha, E_total(PT2) = {e_total + e_pt2:.8f} Ha (in {t_pt2:.2f}s)")
  
    elif mode == "proxy":
        print("  Computing variational energies...")
        t_start = time.perf_counter()
        var_result = compute_variational(
            state=driver.state,
            detspace=driver.detspace,
            system=system,
        )
        t_var = time.perf_counter() - t_start
        
        if var_result:
            result.update(var_result)
            result["t_var"] = t_var
            print(f"  E_var(V): {var_result['e_var_v']:.8f} Ha")
            if "e_var_t" in var_result:
                print(f"  E_var(T): {var_result['e_var_t']:.8f} Ha")
            print(f"  (computed in {t_var:.2f}s)")
  
    return result


# ============================================================================
# Single Configuration Execution
# ============================================================================

def run_single_config(
    system: MolecularSystem,
    mode: str,
    topk: int,
    config: ScanConfig,
    output_dir: Path,
) -> dict:
    """
    Execute optimization for single (molecule, mode, topk) configuration.
  
    Args:
        system: Molecular system with integrals
        mode: Computational mode (variational/proxy/asymmetric)
        topk: V-space size for selection
        config: Global scan configuration
        output_dir: Directory for outputs
      
    Returns:
        Summary dict with energies, runtime, and analysis results
    """
    print(f"\n{'='*70}")
    print(f"Config: mode={mode.upper()}, |V|={topk}")
    print(f"Output: {output_dir.relative_to(config.output_root)}")
    print(f"{'='*70}\n")
  
    # Initialize detspace and model
    hf_det = system.hf_determinant()
    detspace = DetSpace.initialize(hf_det)
    model = _build_model(system, config)
  
    # Configure optimizer and selector
    schedule = optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=800, alpha=0.05
    )
    optimizer = optax.adamw(learning_rate=schedule)
    selector = TopKSelector(topk, stream=True)
  
    # Build driver
    driver = _DRIVER_MAP[mode].build(
        system=system,
        detspace=detspace,
        model=model,
        optimizer=optimizer,
        selector=selector,
        chunk_size=config.chunk_size,
    )
  
    # Run optimization
    e_total, total_time, inner_time_avg = _run_optimization(driver, output_dir)
  
    # Build summary
    summary = {
        "mode": mode,
        "topk": topk,
        "e_total": e_total,
        "total_time": total_time,
        "inner_time_avg": inner_time_avg,
    }
  
    # Post-analysis
    analysis = _post_analysis(mode, driver, system, e_total)
    summary.update(analysis)
  
    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
  
    print(f"\n✓ Completed in {total_time:.1f}s")
    print(f"  E_total = {e_total:.8f} Ha")
    
    return summary


# ============================================================================
# Full Scan Execution
# ============================================================================

def run_scan(config: ScanConfig) -> None:
    """Execute systematic scan over all molecules and configurations."""
    configure_jax()
  
    config.output_root.mkdir(parents=True, exist_ok=True)
    all_results = []
    
    global_start = time.perf_counter()
  
    # Loop over molecules
    for fcidump_path in config.fcidump_paths:
        molecule_name = fcidump_path.stem
      
        print(f"\n{'#'*70}")
        print(f"# Molecule: {molecule_name}")
        print(f"{'#'*70}")
      
        # Load system (L0 layer, static per molecule)
        system = MolecularSystem.from_fcidump(fcidump_path)
        print(f"System: N_o={system.n_orb}, N_e={system.n_elec}, MS2={system.ms2}")
        print(f"E_nuc:  {system.e_nuc:.8f} Ha\n")
      
        # Loop over modes and topk sizes
        for mode in config.modes:
            for topk in config.topk_sizes:
                output_dir = config.output_root / molecule_name / mode / f"topk_{topk:04d}"
              
                try:
                    summary = run_single_config(
                        system=system,
                        mode=mode,
                        topk=topk,
                        config=config,
                        output_dir=output_dir,
                    )
                    summary["molecule"] = molecule_name
                    all_results.append(summary)
                  
                except Exception as e:
                    print(f"\n✗ Failed {molecule_name}/{mode}/topk={topk}: {e}")
                    continue
    
    global_runtime = time.perf_counter() - global_start
  
    # Save global summary
    summary_path = config.output_root / "all_results.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
  
    print(f"\n{'='*70}")
    print(f"Scan completed: {len(all_results)} configurations")
    print(f"Total runtime: {global_runtime:.1f}s ({global_runtime/60:.1f}m)")
    print(f"Global summary: {summary_path}")
    print(f"{'='*70}\n")


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    """Execute systematic DetNQS multi-molecule scan."""
    fcidump_dir = Path("../benchmark/FCIDUMP/")
  
    config = ScanConfig(
        fcidump_paths=(
            fcidump_dir / "H2O_631g.FCIDUMP",
            # fcidump_dir / "N2_ccpvdz_2.118B.FCIDUMP",
            # fcidump_dir / "N2_ccpvdz_2.400B.FCIDUMP",
            # fcidump_dir / "N2_ccpvdz_2.700B.FCIDUMP",
            # fcidump_dir / "N2_ccpvdz_3.000B.FCIDUMP",
            # fcidump_dir / "N2_ccpvdz_3.600B.FCIDUMP",
            # fcidump_dir / "N2_ccpvdz_4.200B.FCIDUMP",
            # fcidump_dir / "H2O_ccpvdz_1.0re.FCIDUMP",
            # fcidump_dir / "H2O_ccpvdz_1.5re.FCIDUMP",
            # fcidump_dir / "H2O_ccpvdz_2.0re.FCIDUMP",
            # fcidump_dir / "H2O_ccpvdz_2.5re.FCIDUMP",
            # fcidump_dir / "H2O_ccpvdz_3.0re.FCIDUMP",
            # fcidump_dir / "Cr2_24e30o.FCIDUMP",
            # fcidump_dir / "Cr2_48e42o.FCIDUMP",
        ),
        output_root=Path("./scans/H2O_631g"),
        modes=("variational", "proxy", "asymmetric"),
        # modes=("variational",),
        topk_sizes=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
        # topk_sizes=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144),
        # topk_sizes=(131072, 262144),
    )
  
    run_scan(config)


if __name__ == "__main__":
    main()
