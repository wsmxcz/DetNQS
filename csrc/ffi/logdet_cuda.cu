// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/*
 * Batched log-determinant computation via cuSolverDx (CUDA kernels).
 *
 * Algorithm:
 *   Forward:  LU decomposition (GETRF) to compute det(A):
 *             log|det(A)| = sum_i log|U_ii|
 *             sign(det(A)) = (-1)^{swaps} * prod_i sign(U_ii)
 *
 *   Backward: Gradient w.r.t. A for L = log|det(A)|:
 *             dL/dA = cotangent * (A^{-T})
 *             Computed via solving A^T X = I using GESV.
 *
 * Supports batched input of shape (..., N, N) with N in [2, 64].
 * Uses identity padding to bucket matrix sizes to {2,4,8,12,16,24,32,48,64}.
 *
 * File: lever/jax/fused_logdet_cuda.cc
 * Author: Zheng (Alex) Che, wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifdef detnqs_WITH_CUSOLVERDX
  #include <cusolverdx.hpp>
  #include <cusolverdx_io.hpp>
#endif

namespace ffi = xla::ffi;

// ========================== Error Handling ==========================

static inline ffi::Error InternalErr(const char* msg) {
  return ffi::Error(ffi::ErrorCode::kInternal, msg);
}

static inline ffi::Error InvalidArg(const char* msg) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, msg);
}

#define CUDA_OK(expr) do {                                             \
  cudaError_t _e = (expr);                                             \
  if (_e != cudaSuccess) return InternalErr(cudaGetErrorString(_e));   \
} while(0)

// ========================== Device Utilities ==========================

template <typename T>
__device__ __forceinline__ double ToDouble(T x) {
  return static_cast<double>(x);
}

template <typename T>
__device__ __forceinline__ double SafeLogAbs(T x) {
  const double ax = fabs(ToDouble(x));
  return (ax > 0.0 && isfinite(ax)) ? log(ax) : -INFINITY;
}

template <typename T>
__device__ __forceinline__ double SignOf(T x) {
  return (ToDouble(x) < 0.0) ? -1.0 : 1.0;
}

// ========================== Host Utilities ==========================

// Bucket matrix size N to {2, 4, 8, 12, 16, 24, 32, 48, 64} for compile-time instantiation.
static inline int BucketN(int N) {
  if (N <= 2)  return 2;
  if (N <= 4)  return 4;
  if (N <= 8)  return 8;
  if (N <= 12) return 12;
  if (N <= 16) return 16;
  if (N <= 24) return 24;
  if (N <= 32) return 32;
  if (N <= 48) return 48;
  return 64;
}


// Parse (..., N, N) buffer into (batch_size, N).
template <ffi::DataType DT>
static ffi::Error ParseBatchedSquare(const ffi::Buffer<DT>& a, int* out_B, int* out_N) {
  auto dims = a.dimensions();
  const int rank = static_cast<int>(dims.size());
  if (rank < 2) return InvalidArg("Expected rank >= 2 for (..., N, N)");

  const int64_t N0 = dims[rank - 1];
  const int64_t N1 = dims[rank - 2];
  if (N0 != N1) return InvalidArg("Last two dims must be equal (square matrix)");

  const int N = static_cast<int>(N0);
  if (N < 2 || N > 64) return InvalidArg("N must be in [2, 64]");

  const int64_t total = a.element_count();
  const int64_t NN = int64_t(N) * int64_t(N);
  if (NN <= 0 || (total % NN) != 0) return InvalidArg("Invalid element count");

  const int64_t B64 = total / NN;
  if (B64 < 1 || B64 > int64_t(std::numeric_limits<int>::max())) {
    return InvalidArg("Batch size out of int32 range");
  }

  *out_B = static_cast<int>(B64);
  *out_N = N;
  return ffi::Error::Success();
}

// Verify that buf has batch prefix matching a.shape[:-drop_trailing].
template <typename TBuf>
static ffi::Error CheckBatchPrefixEquals(const TBuf& buf, const TBuf& a, int drop_trailing) {
  auto bd = buf.dimensions();
  auto ad = a.dimensions();

  const int ar = static_cast<int>(ad.size());
  const int br = static_cast<int>(bd.size());
  const int pr = ar - drop_trailing;
  if (pr < 0) return InvalidArg("Internal: negative prefix rank");
  if (br != pr) return InvalidArg("Rank mismatch");

  for (int i = 0; i < pr; ++i) {
    if (bd[i] != ad[i]) return InvalidArg("Batch shape mismatch");
  }
  return ffi::Error::Success();
}

// ========================== cuSolverDx Configuration ==========================

#ifdef detnqs_WITH_CUSOLVERDX

#ifndef DETNQS_CUSOLVERDX_SM
#error "DETNQS_CUSOLVERDX_SM must be defined (e.g., 900 for sm_90)."
#endif

template <typename Scalar, int Nb>
struct DxSolvers {
  static constexpr int SM = DETNQS_CUSOLVERDX_SM;

  using FwdBase = decltype(
      cusolverdx::Size<Nb, Nb, 1>() +
      cusolverdx::Precision<Scalar>() +
      cusolverdx::Type<cusolverdx::type::real>() +
      cusolverdx::Arrangement<cusolverdx::row_major>() +
      cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>() +
      cusolverdx::SM<SM>() +
      cusolverdx::Block()
  );

  static constexpr unsigned BPB = FwdBase::suggested_batches_per_block;

  // Forward: LU factorization
  using Fwd = decltype(
      cusolverdx::Size<Nb, Nb, 1>() +
      cusolverdx::Precision<Scalar>() +
      cusolverdx::Type<cusolverdx::type::real>() +
      cusolverdx::Arrangement<cusolverdx::row_major>() +
      cusolverdx::BatchesPerBlock<BPB>() +
      cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>() +
      cusolverdx::SM<SM>() +
      cusolverdx::Block()
  );

  // Backward: solve A^T X = I
  // Workaround: place Function before TransposeMode to avoid MathDx 25.12.0 static_assert.
  using Bwd = decltype(
      cusolverdx::Size<Nb, Nb, Nb>() +
      cusolverdx::Precision<Scalar>() +
      cusolverdx::Type<cusolverdx::type::real>() +
      cusolverdx::Arrangement<cusolverdx::row_major>() +
      cusolverdx::BatchesPerBlock<BPB>() +
      cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>() +
      cusolverdx::TransposeMode<cusolverdx::trans>() +
      cusolverdx::SM<SM>() +
      cusolverdx::Block()
  );
};

// ========================== CUDA Kernels ==========================

template <typename Scalar, int Nb>
__global__ void FwdKernelDx(const Scalar* __restrict__ A,
                            Scalar* __restrict__ out_sign,
                            Scalar* __restrict__ out_logabs,
                            int B, int N_actual) {
  using Solver = typename DxSolvers<Scalar, Nb>::Fwd;
  constexpr unsigned BPB = DxSolvers<Scalar, Nb>::BPB;

  using a_t = typename Solver::a_data_type;
  using status_t = typename Solver::status_type;

  extern __shared__ __align__(16) unsigned char smem[];

  a_t* sA = reinterpret_cast<a_t*>(smem);
  constexpr int lda = Solver::lda;
  constexpr int a_elems = Solver::a_size;

  unsigned char* extra = smem + Solver::shared_memory_size;
  auto [ipiv, info] =
      cusolverdx::shared_memory::slice_into_pointers<int, status_t>(
          extra,
          alignof(int), int(BPB * Nb),
          alignof(status_t), int(BPB));

  const int tid = int(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
  const int nthreads = int(blockDim.x * blockDim.y * blockDim.z);
  const int block_batch0 = int(blockIdx.x) * int(BPB);

  // Load all BPB slots (fill invalid batches with identity to avoid garbage data)
  for (unsigned bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + int(bpb);
    const bool valid = (b < B);

    const Scalar* Ab = valid ? (A + size_t(b) * size_t(N_actual) * size_t(N_actual)) : nullptr;
    a_t* sAb = sA + size_t(bpb) * size_t(a_elems);

    for (int idx = tid; idx < a_elems; idx += nthreads) {
      const int r = idx / lda;
      const int c = idx % lda;

      if (valid && r < N_actual && c < N_actual) {
        sAb[idx] = a_t(Ab[r * N_actual + c]);
      } else if (r < Nb && c < Nb) {
        // Fill with identity: valid batch gets block-diag(A, I), invalid batch gets full I
        sAb[idx] = (r == c) ? a_t(1) : a_t(0);
      } else {
        sAb[idx] = a_t(0);
      }
    }

    if (tid == 0) info[bpb] = status_t(0);
  }
  __syncthreads();

  Solver{}.execute(sA, ipiv, info);
  __syncthreads();

  // Write outputs only for valid batches
  for (unsigned bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + int(bpb);
    if (b >= B) continue;  // Skip instead of break

    if (tid == 0) {
      if (info[bpb] != status_t(0)) {
        out_sign[b] = Scalar(0);
        out_logabs[b] = Scalar(-INFINITY);
        continue;
      }

      const a_t* sAb = sA + size_t(bpb) * size_t(a_elems);
      const int* piv = ipiv + size_t(bpb) * size_t(Nb);

      // Detect pivot indexing base (0 or 1)
      int base_p = 1;
      for (int i = 0; i < N_actual; ++i) {
        if (piv[i] == 0) { base_p = 0; break; }
      }

      // Count row swaps for sign: swap occurs when piv[i] != i + base
      int swaps_parity = 0;
      for (int i = 0; i < N_actual; ++i) {
        if (piv[i] != i + base_p) swaps_parity ^= 1;
      }

      double sign = swaps_parity ? -1.0 : 1.0;
      double logabs = 0.0;

      for (int i = 0; i < N_actual; ++i) {
        const a_t diag = sAb[i * lda + i];
        const double ll = SafeLogAbs(diag);
        if (!isfinite(ll)) {
          out_sign[b] = Scalar(0);
          out_logabs[b] = Scalar(-INFINITY);
          sign = 0.0;
          break;
        }
        logabs += ll;
        sign *= SignOf(diag);
      }

      out_sign[b] = Scalar(sign);
      out_logabs[b] = Scalar(logabs);
    }
  }
}

template <typename Scalar, int Nb>
__global__ void BwdKernelDx(const Scalar* __restrict__ A,
                            const Scalar* __restrict__ cot_logabs,
                            Scalar* __restrict__ grad,
                            int B, int N_actual) {
  using Solver = typename DxSolvers<Scalar, Nb>::Bwd;
  constexpr unsigned BPB = DxSolvers<Scalar, Nb>::BPB;

  using a_t = typename Solver::a_data_type;
  using b_t = typename Solver::b_data_type;
  using status_t = typename Solver::status_type;

  static_assert(std::is_same<a_t, b_t>::value, "A/B data type mismatch");

  extern __shared__ __align__(16) unsigned char smem[];

  a_t* sA = reinterpret_cast<a_t*>(smem);
  constexpr int lda = Solver::lda;
  constexpr int ldb = Solver::ldb;
  constexpr int a_elems = Solver::a_size;
  constexpr int b_elems = Solver::b_size;

  b_t* sB = reinterpret_cast<b_t*>(sA + size_t(BPB) * size_t(a_elems));

  unsigned char* extra = smem + Solver::shared_memory_size;
  auto [ipiv, info] =
      cusolverdx::shared_memory::slice_into_pointers<int, status_t>(
          extra,
          alignof(int), int(BPB * Nb),
          alignof(status_t), int(BPB));

  const int tid = int(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
  const int nthreads = int(blockDim.x * blockDim.y * blockDim.z);
  const int block_batch0 = int(blockIdx.x) * int(BPB);

  // Load all BPB slots (fill invalid batches with identity)
  for (unsigned bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + int(bpb);
    const bool valid = (b < B);

    const Scalar* Ab = valid ? (A + size_t(b) * size_t(N_actual) * size_t(N_actual)) : nullptr;
    a_t* sAb = sA + size_t(bpb) * size_t(a_elems);
    b_t* sBb = sB + size_t(bpb) * size_t(b_elems);

    for (int idx = tid; idx < a_elems; idx += nthreads) {
      const int r = idx / lda;
      const int c = idx % lda;

      if (valid && r < N_actual && c < N_actual) {
        sAb[idx] = a_t(Ab[r * N_actual + c]);
      } else if (r < Nb && c < Nb) {
        sAb[idx] = (r == c) ? a_t(1) : a_t(0);
      } else {
        sAb[idx] = a_t(0);
      }
    }

    // RHS always identity
    for (int idx = tid; idx < b_elems; idx += nthreads) {
      const int r = idx / ldb;
      const int c = idx % ldb;
      sBb[idx] = (r == c) ? b_t(1) : b_t(0);
    }

    if (tid == 0) info[bpb] = status_t(0);
  }
  __syncthreads();

  // Solve A^T X = I for all BPB slots
  Solver{}.execute(sA, ipiv, sB, info);
  __syncthreads();

  // Write gradients only for valid batches
  for (unsigned bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + int(bpb);
    if (b >= B) continue;  // Skip instead of break

    const b_t* X = sB + size_t(bpb) * size_t(b_elems);
    const Scalar scale = cot_logabs[b];
    Scalar* Gb = grad + size_t(b) * size_t(N_actual) * size_t(N_actual);

    if (info[bpb] != status_t(0)) {
      for (int idx = tid; idx < N_actual * N_actual; idx += nthreads) {
        Gb[idx] = Scalar(0);
      }
    } else {
      for (int idx = tid; idx < N_actual * N_actual; idx += nthreads) {
        const int i = idx / N_actual;
        const int j = idx % N_actual;
        Gb[idx] = scale * Scalar(X[i * ldb + j]);
      }
    }
  }
}

// ========================== Launch Utilities ==========================

template <typename KernelT>
static ffi::Error EnsureOptInShmem(int shmem_bytes, KernelT kernel) {
  // Directly opt-in without caching (negligible overhead vs. potential bugs)
  cudaError_t e = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shmem_bytes);

  if (e != cudaSuccess) return InternalErr(cudaGetErrorString(e));
  return ffi::Error::Success();
}

template <typename Scalar, int Nb>
static ffi::Error LaunchFwd(cudaStream_t stream,
                            const Scalar* A, Scalar* sign, Scalar* logabs,
                            int B, int N_actual) {
  using Solver = typename DxSolvers<Scalar, Nb>::Fwd;
  using status_t = typename Solver::status_type;
  constexpr unsigned BPB = DxSolvers<Scalar, Nb>::BPB;

  const int blocks = (B + int(BPB) - 1) / int(BPB);
  dim3 block_dim = Solver::suggested_block_dim;

  size_t shmem = cusolverdx::make_shared_storage_calculator()
      .add(alignof(typename Solver::a_data_type), Solver::shared_memory_size)
      .add(alignof(int), sizeof(int), BPB * Nb)
      .add(alignof(status_t), sizeof(status_t), BPB)
      .get();

  if (auto err = EnsureOptInShmem(int(shmem), FwdKernelDx<Scalar, Nb>); err.failure()) {
    return err;
  }

  FwdKernelDx<Scalar, Nb><<<blocks, block_dim, shmem, stream>>>(
      A, sign, logabs, B, N_actual);

  CUDA_OK(cudaPeekAtLastError());
  return ffi::Error::Success();
}

template <typename Scalar, int Nb>
static ffi::Error LaunchBwd(cudaStream_t stream,
                            const Scalar* A, const Scalar* cot, Scalar* grad,
                            int B, int N_actual) {
  using Solver = typename DxSolvers<Scalar, Nb>::Bwd;
  using status_t = typename Solver::status_type;
  constexpr unsigned BPB = DxSolvers<Scalar, Nb>::BPB;

  const int blocks = (B + int(BPB) - 1) / int(BPB);
  dim3 block_dim = Solver::suggested_block_dim;

  size_t shmem = cusolverdx::make_shared_storage_calculator()
      .add(alignof(typename Solver::a_data_type), Solver::shared_memory_size)
      .add(alignof(int), sizeof(int), BPB * Nb)
      .add(alignof(status_t), sizeof(status_t), BPB)
      .get();

  if (auto err = EnsureOptInShmem(int(shmem), BwdKernelDx<Scalar, Nb>); err.failure()) {
    return err;
  }

  BwdKernelDx<Scalar, Nb><<<blocks, block_dim, shmem, stream>>>(
      A, cot, grad, B, N_actual);

  CUDA_OK(cudaPeekAtLastError());
  return ffi::Error::Success();
}

template <typename Scalar>
static ffi::Error DispatchFwd(cudaStream_t stream,
                              const Scalar* A, Scalar* sign, Scalar* logabs,
                              int B, int N_actual) {
  const int Nb = BucketN(N_actual);
  switch (Nb) {
    case 2:  return LaunchFwd<Scalar, 2>(stream, A, sign, logabs, B, N_actual);
    case 4:  return LaunchFwd<Scalar, 4>(stream, A, sign, logabs, B, N_actual);
    case 8:  return LaunchFwd<Scalar, 8>(stream, A, sign, logabs, B, N_actual);
    case 12: return LaunchFwd<Scalar,12>(stream, A, sign, logabs, B, N_actual);
    case 16: return LaunchFwd<Scalar,16>(stream, A, sign, logabs, B, N_actual);
    case 24: return LaunchFwd<Scalar,24>(stream, A, sign, logabs, B, N_actual);
    case 32: return LaunchFwd<Scalar,32>(stream, A, sign, logabs, B, N_actual);
    case 48: return LaunchFwd<Scalar,48>(stream, A, sign, logabs, B, N_actual);
    case 64: return LaunchFwd<Scalar,64>(stream, A, sign, logabs, B, N_actual);
    default: return InvalidArg("Invalid matrix bucket size");
  }
}

template <typename Scalar>
static ffi::Error DispatchBwd(cudaStream_t stream,
                              const Scalar* A, const Scalar* cot, Scalar* grad,
                              int B, int N_actual) {
  const int Nb = BucketN(N_actual);
  switch (Nb) {
    case 2:  return LaunchBwd<Scalar, 2>(stream, A, cot, grad, B, N_actual);
    case 4:  return LaunchBwd<Scalar, 4>(stream, A, cot, grad, B, N_actual);
    case 8:  return LaunchBwd<Scalar, 8>(stream, A, cot, grad, B, N_actual);
    case 12: return LaunchBwd<Scalar,12>(stream, A, cot, grad, B, N_actual);
    case 16: return LaunchBwd<Scalar,16>(stream, A, cot, grad, B, N_actual);
    case 24: return LaunchBwd<Scalar,24>(stream, A, cot, grad, B, N_actual);
    case 32: return LaunchBwd<Scalar,32>(stream, A, cot, grad, B, N_actual);
    case 48: return LaunchBwd<Scalar,48>(stream, A, cot, grad, B, N_actual);
    case 64: return LaunchBwd<Scalar,64>(stream, A, cot, grad, B, N_actual);
    default: return InvalidArg("Invalid matrix bucket size");
  }
}

#endif  // detnqs_WITH_CUSOLVERDX

// ========================== FFI Entry Points ==========================

template <ffi::DataType DT>
static ffi::Error FwdCudaDx(cudaStream_t stream,
                            ffi::Buffer<DT> a,
                            ffi::ResultBuffer<DT> out_sign,
                            ffi::ResultBuffer<DT> out_logabs) {
#ifndef detnqs_WITH_CUSOLVERDX
  return InvalidArg("Built without cuSolverDx support");
#else
  using Scalar = ffi::NativeType<DT>;

  int B = 0, N = 0;
  if (auto err = ParseBatchedSquare<DT>(a, &B, &N); err.failure()) return err;

  if (auto err = CheckBatchPrefixEquals(*out_sign, a, 2); err.failure()) return err;
  if (auto err = CheckBatchPrefixEquals(*out_logabs, a, 2); err.failure()) return err;

  return DispatchFwd<Scalar>(stream, a.typed_data(),
                            out_sign->typed_data(), out_logabs->typed_data(),
                            B, N);
#endif
}

template <ffi::DataType DT>
static ffi::Error BwdCudaDx(cudaStream_t stream,
                            ffi::Buffer<DT> a,
                            ffi::Buffer<DT> cot_sign,
                            ffi::Buffer<DT> cot_logabs,
                            ffi::ResultBuffer<DT> out_grad) {
  (void)cot_sign;  // Sign is non-differentiable
#ifndef detnqs_WITH_CUSOLVERDX
  return InvalidArg("Built without cuSolverDx support");
#else
  using Scalar = ffi::NativeType<DT>;

  int B = 0, N = 0;
  if (auto err = ParseBatchedSquare<DT>(a, &B, &N); err.failure()) return err;

  if (auto err = CheckBatchPrefixEquals(cot_logabs, a, 2); err.failure()) return err;

  auto gd = out_grad->dimensions();
  auto ad = a.dimensions();
  if (gd.size() != ad.size()) return InvalidArg("Gradient rank mismatch");
  for (size_t i = 0; i < ad.size(); ++i) {
    if (gd[i] != ad[i]) return InvalidArg("Gradient shape mismatch");
  }

  return DispatchBwd<Scalar>(stream, a.typed_data(),
                            cot_logabs.typed_data(), out_grad->typed_data(),
                            B, N);
#endif
}

// ========================== XLA FFI Registration ==========================

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    detnqs_fused_logdet_f64_fwd_cuda, FwdCudaDx<ffi::DataType::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    detnqs_fused_logdet_f64_bwd_cuda, BwdCudaDx<ffi::DataType::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    detnqs_fused_logdet_f32_fwd_cuda, FwdCudaDx<ffi::DataType::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    detnqs_fused_logdet_f32_bwd_cuda, BwdCudaDx<ffi::DataType::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());