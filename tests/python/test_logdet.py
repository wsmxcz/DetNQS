# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for log determinant computation with custom VJP.

Validates logdet_c against JAX baseline (slogdet) for:
  - Forward correctness (real SPD, complex matrices)
  - VJP/gradient correctness (analytic vs. automatic differentiation)
  - Batch shape handling
  - Optional performance benchmarks

File: test_logdet.py
Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
Date: December, 2025
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import pytest
import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

from lever.models.utils import logdet_c


# ============================================================================
# Helper Functions
# ============================================================================

def _stabilize(mat: jnp.ndarray) -> jnp.ndarray:
    """Add small diagonal shift to avoid numerical singularity."""
    n = mat.shape[-1]
    eps = 1e-8 if mat.dtype in (jnp.complex64, jnp.float32) else 1e-16
    return mat + eps * jnp.eye(n, dtype=mat.dtype)


def logdet_baseline(mat: jnp.ndarray) -> jnp.ndarray:
    """Baseline: log(det(A)) = log|det| + i*arg(det) via slogdet."""
    mat = _stabilize(mat)
    sign, logabs = jnp.linalg.slogdet(mat)
    out_dtype = jnp.result_type(mat.dtype, jnp.complex64)
    return logabs.astype(out_dtype) + jnp.log(sign.astype(out_dtype))


def _make_spd(key: jax.Array, batch: int, n: int, dtype=jnp.float64) -> jnp.ndarray:
    """Generate SPD matrices: A = X X^T + 0.25*I."""
    x = random.normal(key, (batch, n, n), dtype=dtype)
    a = jnp.einsum("bij,bkj->bik", x, x)
    return a + 0.25 * jnp.eye(n, dtype=dtype)[None, :, :]


def _make_complex(key: jax.Array, batch: int, n: int, dtype=jnp.complex128) -> jnp.ndarray:
    """Generate well-conditioned complex matrices: A = (X + iY) + n*I."""
    k1, k2 = random.split(key)
    re = random.normal(k1, (batch, n, n), dtype=jnp.float64)
    im = random.normal(k2, (batch, n, n), dtype=jnp.float64)
    a = (re + 1j * im).astype(dtype)
    return a + (n * jnp.eye(n, dtype=dtype)[None, :, :])


def _loss_real(logdet_fn: Callable, mat: jnp.ndarray) -> jnp.ndarray:
    """Scalar loss: sum(Re(log det(A)))."""
    return jnp.sum(jnp.real(logdet_fn(mat)))


def _grad_analytic(mat: jnp.ndarray) -> jnp.ndarray:
    """Analytic gradient: d/dA sum(Re(log det)) = (A^{-T})."""
    inv = jnp.linalg.inv(_stabilize(mat))
    return jnp.swapaxes(inv, -1, -2)


def _assert_close(a, b, *, rtol=1e-9, atol=1e-9, name=""):
    """Assert arrays are close within tolerance."""
    ok = jnp.allclose(a, b, rtol=rtol, atol=atol)
    assert bool(ok), (
        f"{name} mismatch: max|diff|={jnp.max(jnp.abs(a - b)):.2e}, "
        f"shapes {a.shape}/{b.shape}, dtypes {a.dtype}/{b.dtype}"
    )


# ============================================================================
# Forward Pass Tests
# ============================================================================

@pytest.mark.parametrize("n,batch", [(2, 8), (4, 4), (8, 2)])
def test_fwd_real_spd(n: int, batch: int):
    """Forward: real SPD matrices match baseline."""
    mat = _make_spd(random.PRNGKey(0), batch, n, jnp.float64)
    _assert_close(logdet_c(mat), logdet_baseline(mat), rtol=5e-12, atol=5e-12, name="fwd_real_spd")


@pytest.mark.parametrize("n,batch", [(4, 4), (8, 2)])
def test_fwd_real_neg(n: int, batch: int):
    """Forward: real matrices with det < 0 match baseline."""
    mat = _make_spd(random.PRNGKey(1), batch, n, jnp.float64)
    mat = mat.at[:, 0, :].multiply(-1.0)  # Flip sign -> det < 0
    _assert_close(logdet_c(mat), logdet_baseline(mat), rtol=5e-12, atol=5e-12, name="fwd_real_neg")


@pytest.mark.parametrize("n,batch", [(2, 8), (4, 4), (8, 2)])
def test_fwd_complex(n: int, batch: int):
    """Forward: complex matrices match baseline."""
    mat = _make_complex(random.PRNGKey(2), batch, n, jnp.complex128)
    _assert_close(logdet_c(mat), logdet_baseline(mat), rtol=1e-10, atol=1e-10, name="fwd_complex")


# ============================================================================
# VJP Tests
# ============================================================================

@pytest.mark.parametrize("n,batch", [(2, 4), (4, 2), (8, 1)])
def test_vjp_complex(n: int, batch: int):
    """VJP: complex input with random cotangent matches baseline."""
    mat = _make_complex(random.PRNGKey(10), batch, n, jnp.complex128)

    y1, pb1 = jax.vjp(logdet_c, mat)
    y2, pb2 = jax.vjp(logdet_baseline, mat)

    k_re, k_im = random.split(random.PRNGKey(11))
    ct = (random.normal(k_re, y1.shape, jnp.float64) +
          1j * random.normal(k_im, y1.shape, jnp.float64)).astype(y1.dtype)

    (g1,) = pb1(ct)
    (g2,) = pb2(ct)

    _assert_close(y1, y2, rtol=1e-10, atol=1e-10, name="vjp_complex_primal")
    _assert_close(g1, g2, rtol=1e-8, atol=1e-8, name="vjp_complex_grad")


@pytest.mark.parametrize("n,batch", [(2, 4), (4, 2)])
def test_vjp_real_cplx(n: int, batch: int):
    """VJP: real input with complex cotangent matches baseline."""
    mat = _make_spd(random.PRNGKey(12), batch, n, jnp.float64)
    mat = mat.at[:, 0, :].multiply(-1.0)

    y1, pb1 = jax.vjp(logdet_c, mat)
    y2, pb2 = jax.vjp(logdet_baseline, mat)

    k_re, k_im = random.split(random.PRNGKey(13))
    ct = (random.normal(k_re, y1.shape, jnp.float64) +
          1j * random.normal(k_im, y1.shape, jnp.float64)).astype(y1.dtype)

    (g1,) = pb1(ct)
    (g2,) = pb2(ct)

    _assert_close(y1, y2, rtol=5e-12, atol=5e-12, name="vjp_real_primal")
    _assert_close(g1, g2, rtol=1e-9, atol=1e-9, name="vjp_real_grad")


# ============================================================================
# Gradient Tests
# ============================================================================

@pytest.mark.parametrize("n,batch", [(2, 4), (4, 2), (8, 1)])
def test_grad_real(n: int, batch: int):
    """Gradient: real SPD matches baseline and analytic formula."""
    mat = _make_spd(random.PRNGKey(3), batch, n, jnp.float64)

    g_custom = jax.grad(lambda x: _loss_real(logdet_c, x))(mat)
    g_base = jax.grad(lambda x: _loss_real(logdet_baseline, x))(mat)
    g_ana = _grad_analytic(mat)

    _assert_close(g_custom, g_base, rtol=1e-9, atol=1e-9, name="grad_real_vs_base")
    _assert_close(g_custom, g_ana, rtol=1e-9, atol=1e-9, name="grad_real_vs_ana")


@pytest.mark.parametrize("n,batch", [(2, 4), (4, 2), (8, 1)])
def test_grad_complex(n: int, batch: int):
    """Gradient: complex matrices match baseline and analytic formula."""
    mat = _make_complex(random.PRNGKey(4), batch, n, jnp.complex128)

    g_custom = jax.grad(lambda x: _loss_real(logdet_c, x))(mat)
    g_base = jax.grad(lambda x: _loss_real(logdet_baseline, x))(mat)
    g_ana = _grad_analytic(mat)

    _assert_close(g_custom, g_base, rtol=1e-8, atol=1e-8, name="grad_cplx_vs_base")
    _assert_close(g_custom, g_ana, rtol=1e-8, atol=1e-8, name="grad_cplx_vs_ana")


def test_check_grads():
    """Finite-difference verification on small problems."""
    from jax.test_util import check_grads

    key = random.PRNGKey(5)
    mat_real = _make_spd(key, batch=1, n=3, dtype=jnp.float64)
    mat_cplx = _make_complex(key, batch=1, n=3, dtype=jnp.complex128)

    check_grads(lambda x: _loss_real(logdet_c, x), (mat_real,),
                order=1, modes=("rev",), atol=1e-5, rtol=1e-5)
    check_grads(lambda x: _loss_real(logdet_c, x), (mat_cplx,),
                order=1, modes=("rev",), atol=1e-5, rtol=1e-5)


# ============================================================================
# Batch Shape Tests
# ============================================================================

@pytest.mark.parametrize("shape", [(7, 5, 5), (3, 2, 4, 4)])
def test_batch_shapes(shape: Tuple[int, ...]):
    """Output shape matches leading batch dimensions."""
    *batch_dims, n, _ = shape
    batch = int(jnp.prod(jnp.array(batch_dims)))
    mat = _make_spd(random.PRNGKey(6), batch, n, jnp.float64).reshape(shape)
    y = logdet_c(mat)
    assert y.shape == tuple(batch_dims)


# ============================================================================
# Performance Benchmark
# ============================================================================

@dataclass
class BenchResult:
    name: str
    fwd_ms: float
    bwd_ms: float


def _bench_one(
    name: str,
    logdet_fn: Callable,
    mat: jnp.ndarray,
    *,
    n_warmup: int = 2,
    n_iters: int = 10,
) -> BenchResult:
    """Measure forward/backward time excluding compilation."""
    loss = lambda x: _loss_real(logdet_fn, x)
    f_jit = jax.jit(loss)
    g_jit = jax.jit(jax.grad(loss))

    # Warmup
    for _ in range(n_warmup):
        f_jit(mat).block_until_ready()
        g_jit(mat).block_until_ready()

    # Forward timing
    t0 = time.perf_counter()
    for _ in range(n_iters):
        f_jit(mat).block_until_ready()
    t1 = time.perf_counter()

    # Backward timing
    t2 = time.perf_counter()
    for _ in range(n_iters):
        g_jit(mat).block_until_ready()
    t3 = time.perf_counter()

    return BenchResult(
        name=name,
        fwd_ms=(t1 - t0) * 1e3 / n_iters,
        bwd_ms=(t3 - t2) * 1e3 / n_iters,
    )


def test_perf_timings():
    """Micro-benchmark enabled via LEVER_RUN_PERF=1."""
    if os.environ.get("LEVER_RUN_PERF", "0") != "1":
        pytest.skip("Set LEVER_RUN_PERF=1 to run performance benchmarks.")

    key = random.PRNGKey(7)
    batch, n = 512, 32
    mat = _make_spd(key, batch, n, jnp.float64)

    results = [
        _bench_one("logdet_c (custom)", logdet_c, mat),
        _bench_one("slogdet (baseline)", logdet_baseline, mat),
    ]

    backend = jax.default_backend()
    device = jax.devices()[0].platform
    print(f"\n[logdet benchmark] backend={backend}, device={device}, batch={batch}, n={n}")
    for r in results:
        print(f"  {r.name:22s}  fwd={r.fwd_ms:7.3f} ms  bwd={r.bwd_ms:7.3f} ms")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true", help="Run micro-benchmarks.")
    args = ap.parse_args()
    if args.perf:
        os.environ["LEVER_RUN_PERF"] = "1"
        test_perf_timings()