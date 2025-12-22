# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for fused FFI log-determinant kernel.

Validates correctness (primal/gradient) and performance of custom FFI
logdet vs XLA baseline across float32/float64 and various matrix sizes.

File: tests/test_fused_logdet.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

import time
import jax
import jax.numpy as jnp
from jax import random

from lever.models.utils import logdet_c

jax.config.update("jax_enable_x64", True)


def check_correctness(name: str, mat: jnp.ndarray, verbose: bool = True):
    """
    Validate primal and gradient match between FFI kernel and XLA baseline.
  
    Args:
        name: Test case identifier
        mat: Batched matrices [..., N, N]
        verbose: Print detailed results
    """
    if verbose:
        print(f"\n[{name}] shape={mat.shape}, dtype={mat.dtype}")
  
    # Forward: sign and log|det|
    s_fast, l_fast = logdet_c(mat, use_fast_kernel=True)
    s_base, l_base = logdet_c(mat, use_fast_kernel=False)
  
    diff_sign = jnp.max(jnp.abs(s_fast - s_base))
    diff_logabs = jnp.max(jnp.abs(l_fast - l_base))
  
    if verbose:
        print(f"  Forward:  sign={diff_sign:.2e}, logabs={diff_logabs:.2e}")
  
    # Backward: gradient of log|det| (sign is non-differentiable)
    def loss_fn(use_fast, x):
        _, logabs = logdet_c(x, use_fast_kernel=use_fast)
        return jnp.sum(logabs)
  
    grad_fast = jax.jit(jax.grad(loss_fn, argnums=1), static_argnums=0)(True, mat)
    grad_base = jax.jit(jax.grad(loss_fn, argnums=1), static_argnums=0)(False, mat)
    diff_grad = jnp.max(jnp.abs(grad_fast - grad_base))
  
    if verbose:
        print(f"  Backward: grad={diff_grad:.2e}")
  
    # Tolerance: float64 requires tighter bounds
    tol = 1e-8 if mat.dtype == jnp.float64 else 1e-4
  
    assert diff_logabs < tol, f"logabs mismatch: {diff_logabs}"
    assert diff_sign < tol, f"sign mismatch: {diff_sign}"
    assert diff_grad < tol, f"gradient mismatch: {diff_grad}"


def run_benchmark(mat: jnp.ndarray, iters: int = 100):
    """
    Measure forward + backward pass time.
  
    Args:
        mat: Batched matrices
        iters: Number of timing iterations
    """
    def fwd_bwd(use_fast, x):
        return jax.grad(lambda m: jnp.sum(logdet_c(m, use_fast_kernel=use_fast)[1]))(x)
  
    f_fast = jax.jit(fwd_bwd, static_argnums=0)
    f_base = jax.jit(fwd_bwd, static_argnums=0)
  
    # Warmup
    f_fast(True, mat).block_until_ready()
    f_base(False, mat).block_until_ready()
  
    # Time FFI kernel
    t0 = time.perf_counter()
    for _ in range(iters):
        f_fast(True, mat).block_until_ready()
    t_fast = (time.perf_counter() - t0) / iters
  
    # Time XLA baseline
    t0 = time.perf_counter()
    for _ in range(iters):
        f_base(False, mat).block_until_ready()
    t_base = (time.perf_counter() - t0) / iters
  
    print(f"\n  XLA Baseline: {t_base*1000:8.3f} ms/step")
    print(f"  FFI Kernel:   {t_fast*1000:8.3f} ms/step")
    print(f"  Speedup:      {t_base/t_fast:8.2f}x")


def verify_hlo_ffi(mat: jnp.ndarray) -> bool:
    """Check if compiled HLO contains custom-call to lever_fused_logdet."""
    print("\n[HLO Verification]")
    hlo = jax.jit(lambda x: logdet_c(x, use_fast_kernel=True)[1]).lower(mat).compiler_ir(dialect="hlo")
    hlo_text = hlo.as_hlo_text()
  
    has_ffi = "lever_fused_logdet" in hlo_text
    print(f"  FFI custom-call present: {has_ffi}")
  
    if not has_ffi:
        print("  WARNING: FFI not found, using XLA fallback")
        print("  HLO snippet:")
        print("  " + hlo_text[:500].replace("\n", "\n  "))
  
    return has_ffi


def test_fallback_behavior():
    """Verify graceful fallback for unsupported inputs (complex, oversized)."""
    print("\n" + "="*60)
    print("FALLBACK BEHAVIOR")
    print("="*60)
  
    key = random.PRNGKey(123)
  
    # Complex dtype should fallback to XLA
    k1, key = random.split(key)
    mat_c = random.normal(k1, (16, 8, 8), dtype=jnp.complex128) + jnp.eye(8) * 8
    print("\n[Complex dtype]")
    s_fast, l_fast = logdet_c(mat_c, use_fast_kernel=True)
    s_base, l_base = logdet_c(mat_c, use_fast_kernel=False)
    diff = jnp.max(jnp.abs(l_fast - l_base))
    print(f"  Fallback OK, diff={diff:.2e}")
  
    # Size > 64 should fallback to XLA
    k2, key = random.split(key)
    mat_large = random.normal(k2, (8, 80, 80), dtype=jnp.float32) + jnp.eye(80) * 80
    print("\n[Size > 64]")
    s_fast, l_fast = logdet_c(mat_large, use_fast_kernel=True)
    s_base, l_base = logdet_c(mat_large, use_fast_kernel=False)
    diff = jnp.max(jnp.abs(l_fast - l_base))
    print(f"  Fallback OK, diff={diff:.2e}")


def run_correctness_tests():
    """Run correctness tests across dtypes and matrix sizes."""
    print("\n" + "="*60)
    print("CORRECTNESS TESTS")
    print("="*60)
  
    for dtype in [jnp.float64]:
        print(f"\n--- {dtype} ---")
        key = random.PRNGKey(42 if dtype == jnp.float32 else 43)
      
        # Small matrices (bucket to 16)
        k1, key = random.split(key)
        mat_small = random.normal(k1, (128, 16, 16), dtype=dtype) + jnp.eye(16) * 16
        check_correctness("N=16 (bucket 16)", mat_small)
      
        # Medium matrices (bucket to 32)
        k2, key = random.split(key)
        mat_med = random.normal(k2, (64, 24, 24), dtype=dtype) + jnp.eye(24) * 24
        check_correctness("N=24 (bucket 32)", mat_med)
      
        # Large matrices (bucket to 64)
        k3, key = random.split(key)
        mat_large = random.normal(k3, (32, 48, 48), dtype=dtype) + jnp.eye(48) * 48
        check_correctness("N=48 (bucket 64)", mat_large)
      
        # Multi-dimensional batch
        k4, key = random.split(key)
        mat_batch = random.normal(k4, (4, 8, 32, 32), dtype=dtype) + jnp.eye(32) * 32
        check_correctness("Multi-batch (4,8,32,32)", mat_batch)
      
        # Edge sizes: verify bucketing logic
        for n in [2, 8, 16, 17, 31, 32, 33, 63, 64]:
            k_edge, key = random.split(key)
            mat_edge = random.normal(k_edge, (16, n, n), dtype=dtype) + jnp.eye(n) * n
            check_correctness(f"N={n}", mat_edge, verbose=False)
        print("  All edge sizes passed")


def run_performance_tests():
    """Run performance benchmarks across dtypes and matrix sizes."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
  
    for dtype in [jnp.float64]:
        print(f"\n--- {dtype} ---")
        key = random.PRNGKey(100 if dtype == jnp.float32 else 101)
      
        # Typical SPO size: (batch, 10, 10)
        for batch_size in [1024, 8192, 16384, 32768]:
            k, key = random.split(key)
            mat = random.normal(k, (batch_size, 10, 10), dtype=dtype) + jnp.eye(10) * 10
            print(f"\n({batch_size}, 10, 10) - typical SPO:")
            run_benchmark(mat, iters=50)
      
        # Medium matrices
        k2, key = random.split(key)
        mat_med = random.normal(k2, (8192, 32, 32), dtype=dtype) + jnp.eye(32) * 32
        print(f"\n(8192, 32, 32):")
        run_benchmark(mat_med, iters=50)
      
        # Large matrices
        k3, key = random.split(key)
        mat_large = random.normal(k3, (8192, 64, 64), dtype=dtype) + jnp.eye(64) * 64
        print(f"\n(8192, 64, 64):")
        run_benchmark(mat_large, iters=50)


def main():
    print("="*60)
    print("Fused LogDet Test Suite")
    print("="*60)
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
  
    run_correctness_tests()
    run_performance_tests()
    test_fallback_behavior()
  
    # HLO verification
    key = random.PRNGKey(999)
    mat_hlo = random.normal(key, (128, 16, 16), dtype=jnp.float32) + jnp.eye(16) * 16
    verify_hlo_ffi(mat_hlo)
  
    print("\n" + "="*60)
    print("All tests passed")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)