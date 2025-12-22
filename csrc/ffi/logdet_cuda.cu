// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * Fused log-determinant computation via cuSolverDx batched LU decomposition.
 *
 * Algorithm:
 *   Forward:  LU = PA → det(A) = (-1)^{parity(P)} * prod_{i} U_{ii}
 *            → (sign, log|det|) via diagonal elements and pivot parity
 *   Backward: grad = cot_logabs * (A^{-1})^T, where A^{-1} from GESV(A, I)
 *
 * Supports matrix sizes N in [2, 64] with bucketed compile-time sizes {16, 32, 64}.
 * Padding to bucket size uses block-diagonal structure: blockdiag(A, I).
 *
 * Architecture support: SM 8.0+ (Ampere/Hopper), compiled for {800, 900, 1200}.
 *
 * Note: cusolverdx_io.hpp not guaranteed in all MathDx builds, so we implement
 *       minimal shared memory layout helpers locally.
 *
 * File: csrc/ffi/logdet_cuda.cu
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifdef LEVER_WITH_CUSOLVERDX
  #include <cusolverdx.hpp>
#endif

namespace ffi = xla::ffi;

// ============================================================================
// Error Handling
// ============================================================================

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

#ifdef LEVER_WITH_CUSOLVERDX

// ============================================================================
// Memory Alignment Utilities
// ============================================================================

__host__ __device__ constexpr size_t AlignUp(size_t x, size_t alignment) {
  return (x + (alignment - 1)) & ~(alignment - 1);
}

// ============================================================================
// Device Configuration
// ============================================================================

static inline int GetCurrentSmVersion() {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, dev);
  return prop.major * 100 + prop.minor * 10;
}

// Map runtime SM to compiled variant: 800/900/1200
static inline int SelectSmVariant() {
  const int sm = GetCurrentSmVersion();
  if (sm >= 1200) return 1200;
  if (sm >= 900) return 900;
  return 800;
}

// Snap N to compile-time bucket: 16/32/64
static inline int BucketN(int N) {
  if (N <= 16) return 16;
  if (N <= 32) return 32;
  return 64;
}

// Batches per block: smaller matrices → more batches
template <int Nb>
struct BatchesPerBlock {
  static constexpr int value = (Nb <= 16) ? 4 : (Nb <= 32 ? 2 : 1);
};

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ __forceinline__ int FlatThreadId() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ __forceinline__ int FlatBlockThreads() {
  return blockDim.x * blockDim.y * blockDim.z;
}

template <typename T>
__device__ __forceinline__ double ToDouble(T x) {
  return static_cast<double>(x);
}

template <typename T>
__device__ __forceinline__ T FromDouble(double x) {
  return static_cast<T>(x);
}

// Safe log|x| handling zero/inf
template <typename T>
__device__ __forceinline__ double SafeLogAbs(T x) {
  const double ax = fabs(ToDouble(x));
  return (ax > 0.0 && isfinite(ax)) ? log(ax) : -INFINITY;
}

template <typename T>
__device__ __forceinline__ double SignOf(T x) {
  return (ToDouble(x) < 0.0) ? -1.0 : 1.0;
}

// ============================================================================
// cuSolverDx Configuration Templates
// ============================================================================

template <typename Scalar, int Nb, int SM>
struct DxConfig {
  static constexpr int BPB = BatchesPerBlock<Nb>::value;

  // GETRF: LU decomposition with partial pivoting
  using FwdSolver = decltype(
      cusolverdx::Size<Nb, Nb, 1>() +
      cusolverdx::Precision<Scalar>() +
      cusolverdx::Type<cusolverdx::type::real>() +
      cusolverdx::Arrangement<cusolverdx::row_major>() +
      cusolverdx::BatchesPerBlock<BPB>() +
      cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>() +
      cusolverdx::SM<SM>() +
      cusolverdx::Block()
  );

  // GESV: Linear system solve A*X = B
  using BwdSolver = decltype(
      cusolverdx::Size<Nb, Nb, Nb>() +
      cusolverdx::Precision<Scalar>() +
      cusolverdx::Type<cusolverdx::type::real>() +
      cusolverdx::Arrangement<cusolverdx::row_major>() +
      cusolverdx::BatchesPerBlock<BPB>() +
      cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>() +
      cusolverdx::SM<SM>() +
      cusolverdx::Block()
  );
};

// ============================================================================
// CUDA Kernels
// ============================================================================

/*
 * Forward kernel: Compute (sign, log|det|) via LU decomposition.
 *
 * Strategy:
 *   1. Load A into shared memory with identity padding
 *   2. Perform batched LU decomposition: PA = LU
 *   3. Extract det from diagonal: det(A) = (-1)^{parity} * prod U_{ii}
 */
template <typename Scalar, int Nb, int SM>
__global__ void FwdKernelDx(const Scalar* __restrict__ A,
                            Scalar* __restrict__ out_sign,
                            Scalar* __restrict__ out_logabs,
                            int B, int N_actual) {
  using Solver = typename DxConfig<Scalar, Nb, SM>::FwdSolver;
  constexpr int BPB = DxConfig<Scalar, Nb, SM>::BPB;

  using a_t = typename Solver::a_data_type;
  using status_t = typename Solver::status_type;

  const int tid = FlatThreadId();
  const int nthreads = FlatBlockThreads();
  const int block_batch0 = blockIdx.x * BPB;

  // Shared memory layout: [solver workspace | pivot indices | status flags]
  extern __shared__ __align__(16) unsigned char smem[];
  
  a_t* sA = reinterpret_cast<a_t*>(smem);
  
  size_t offset = Solver::shared_memory_size;
  offset = AlignUp(offset, alignof(int));
  int* ipiv = reinterpret_cast<int*>(smem + offset);
  offset += BPB * Nb * sizeof(int);
  
  offset = AlignUp(offset, alignof(status_t));
  status_t* info = reinterpret_cast<status_t*>(smem + offset);

  constexpr int lda = Solver::lda;
  constexpr int a_elems = Solver::a_size;

  // Load matrices with blockdiag(A, I) padding
  for (int bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + bpb;
    if (b >= B) break;

    const Scalar* Ab = A + size_t(b) * N_actual * N_actual;
    a_t* sAb = sA + size_t(bpb) * a_elems;

    for (int idx = tid; idx < a_elems; idx += nthreads) {
      const int r = idx / lda;
      const int c = idx % lda;

      if (r < N_actual && c < N_actual) {
        sAb[idx] = a_t(Ab[r * N_actual + c]);
      } else if (r < Nb && c < Nb) {
        sAb[idx] = (r == c) ? a_t(1) : a_t(0);
      } else {
        sAb[idx] = a_t(0);
      }
    }

    if (tid == 0) info[bpb] = status_t(0);
  }
  __syncthreads();

  // Execute batched LU
  Solver solver;
  solver.execute(sA, ipiv, info);
  __syncthreads();

  // Extract determinant: det = (-1)^{parity} * prod_{i} U_{ii}
  for (int bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + bpb;
    if (b >= B) break;

    if (tid == 0) {
      if (info[bpb] != status_t(0)) {
        out_sign[b] = Scalar(0);
        out_logabs[b] = FromDouble<Scalar>(-INFINITY);
        continue;
      }

      const a_t* sAb = sA + size_t(bpb) * a_elems;
      const int* piv = ipiv + size_t(bpb) * Nb;

      // Detect pivot indexing base (0 or 1)
      int base_p = 1;
      for (int i = 0; i < N_actual; ++i) {
        if (piv[i] == 0) {
          base_p = 0;
          break;
        }
      }

      // Compute permutation parity
      int parity = 0;
      for (int i = 0; i < N_actual; ++i) {
        if (piv[i] != i + base_p) parity ^= 1;
      }

      double sign = parity ? -1.0 : 1.0;
      double logabs = 0.0;

      // Accumulate log|det| from diagonal
      for (int i = 0; i < N_actual; ++i) {
        const a_t diag = sAb[i * lda + i];
        const double ll = SafeLogAbs(diag);
        
        if (!isfinite(ll)) {
          out_sign[b] = Scalar(0);
          out_logabs[b] = FromDouble<Scalar>(-INFINITY);
          sign = 0.0;
          break;
        }
        
        logabs += ll;
        sign *= SignOf(diag);
      }

      out_sign[b] = FromDouble<Scalar>(sign);
      out_logabs[b] = FromDouble<Scalar>(logabs);
    }
  }
}

/*
 * Backward kernel: Compute gradient grad_A = cot_logabs * (A^{-1})^T.
 *
 * Strategy:
 *   1. Solve A*X = I to get X = A^{-1}
 *   2. Transpose and scale: grad = cot * X^T
 */
template <typename Scalar, int Nb, int SM>
__global__ void BwdKernelDx(const Scalar* __restrict__ A,
                            const Scalar* __restrict__ cot_logabs,
                            Scalar* __restrict__ grad,
                            int B, int N_actual) {
  using Solver = typename DxConfig<Scalar, Nb, SM>::BwdSolver;
  constexpr int BPB = DxConfig<Scalar, Nb, SM>::BPB;

  using a_t = typename Solver::a_data_type;
  using b_t = typename Solver::b_data_type;
  using status_t = typename Solver::status_type;

  const int tid = FlatThreadId();
  const int nthreads = FlatBlockThreads();
  const int block_batch0 = blockIdx.x * BPB;

  extern __shared__ __align__(16) unsigned char smem[];
  
  a_t* sA = reinterpret_cast<a_t*>(smem);
  
  // Place RHS (B matrices) after A in solver workspace
  size_t offB = BPB * Solver::a_size * sizeof(a_t);
  offB = AlignUp(offB, alignof(b_t));
  b_t* sB = reinterpret_cast<b_t*>(reinterpret_cast<unsigned char*>(sA) + offB);

  size_t offset = Solver::shared_memory_size;
  offset = AlignUp(offset, alignof(int));
  int* ipiv = reinterpret_cast<int*>(smem + offset);
  offset += BPB * Nb * sizeof(int);
  
  offset = AlignUp(offset, alignof(status_t));
  status_t* info = reinterpret_cast<status_t*>(smem + offset);

  constexpr int lda = Solver::lda;
  constexpr int ldb = Solver::ldb;
  constexpr int a_elems = Solver::a_size;
  constexpr int b_elems = Solver::b_size;

  // Load A and set RHS = I
  for (int bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + bpb;
    if (b >= B) break;

    const Scalar* Ab = A + size_t(b) * N_actual * N_actual;
    a_t* sAb = sA + size_t(bpb) * a_elems;
    b_t* sBb = sB + size_t(bpb) * b_elems;

    // Load A with identity padding
    for (int idx = tid; idx < a_elems; idx += nthreads) {
      const int r = idx / lda;
      const int c = idx % lda;

      if (r < N_actual && c < N_actual) {
        sAb[idx] = a_t(Ab[r * N_actual + c]);
      } else if (r < Nb && c < Nb) {
        sAb[idx] = (r == c) ? a_t(1) : a_t(0);
      } else {
        sAb[idx] = a_t(0);
      }
    }

    // RHS = I (full Nb x Nb)
    for (int idx = tid; idx < b_elems; idx += nthreads) {
      const int r = idx / ldb;
      const int c = idx % ldb;
      sBb[idx] = (r < Nb && c < Nb && r == c) ? b_t(1) : b_t(0);
    }

    if (tid == 0) info[bpb] = status_t(0);
  }
  __syncthreads();

  // Solve A*X = I
  Solver solver;
  solver.execute(sA, ipiv, sB, info);
  __syncthreads();

  // Compute gradient: grad_A = cot * (A^{-1})^T
  for (int bpb = 0; bpb < BPB; ++bpb) {
    const int b = block_batch0 + bpb;
    if (b >= B) break;

    const b_t* Xinv = sB + size_t(bpb) * b_elems;
    const Scalar scale = cot_logabs[b];
    Scalar* Gb = grad + size_t(b) * N_actual * N_actual;

    if (info[bpb] != status_t(0)) {
      for (int idx = tid; idx < N_actual * N_actual; idx += nthreads) {
        Gb[idx] = Scalar(0);
      }
    } else {
      for (int idx = tid; idx < N_actual * N_actual; idx += nthreads) {
        const int i = idx / N_actual;
        const int j = idx % N_actual;
        Gb[i * N_actual + j] = scale * Scalar(Xinv[j * ldb + i]);
      }
    }
  }
}

// ============================================================================
// Kernel Launch Helpers
// ============================================================================

template <typename Scalar, int Nb, int SM>
static ffi::Error LaunchFwd(cudaStream_t stream,
                            const Scalar* A, Scalar* sign, Scalar* logabs,
                            int B, int N_actual) {
  using Solver = typename DxConfig<Scalar, Nb, SM>::FwdSolver;
  constexpr int BPB = DxConfig<Scalar, Nb, SM>::BPB;

  const int blocks = (B + BPB - 1) / BPB;
  using status_t = typename Solver::status_type;

  // Calculate shared memory requirement
  size_t shmem = Solver::shared_memory_size;
  shmem = AlignUp(shmem, alignof(int)) + BPB * Nb * sizeof(int);
  shmem = AlignUp(shmem, alignof(status_t)) + BPB * sizeof(status_t);

  int dev = 0;
  CUDA_OK(cudaGetDevice(&dev));
  
  int maxOptin = 0;
  CUDA_OK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  
  if (static_cast<int>(shmem) > maxOptin) {
    return InvalidArg("Required shared memory exceeds device limit");
  }

  CUDA_OK(cudaFuncSetAttribute(FwdKernelDx<Scalar, Nb, SM>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shmem)));

  dim3 block_dim = Solver::block_dim;
  FwdKernelDx<Scalar, Nb, SM><<<blocks, block_dim, shmem, stream>>>(
      A, sign, logabs, B, N_actual);
  
  CUDA_OK(cudaPeekAtLastError());
  return ffi::Error::Success();
}

template <typename Scalar, int Nb, int SM>
static ffi::Error LaunchBwd(cudaStream_t stream,
                            const Scalar* A, const Scalar* cot, Scalar* grad,
                            int B, int N_actual) {
  using Solver = typename DxConfig<Scalar, Nb, SM>::BwdSolver;
  constexpr int BPB = DxConfig<Scalar, Nb, SM>::BPB;

  const int blocks = (B + BPB - 1) / BPB;
  using status_t = typename Solver::status_type;

  size_t shmem = Solver::shared_memory_size;
  shmem = AlignUp(shmem, alignof(int)) + BPB * Nb * sizeof(int);
  shmem = AlignUp(shmem, alignof(status_t)) + BPB * sizeof(status_t);

  int dev = 0;
  CUDA_OK(cudaGetDevice(&dev));
  
  int maxOptin = 0;
  CUDA_OK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  
  if (static_cast<int>(shmem) > maxOptin) {
    return InvalidArg("Required shared memory exceeds device limit");
  }

  CUDA_OK(cudaFuncSetAttribute(BwdKernelDx<Scalar, Nb, SM>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(shmem)));

  dim3 block_dim = Solver::block_dim;
  BwdKernelDx<Scalar, Nb, SM><<<blocks, block_dim, shmem, stream>>>(
      A, cot, grad, B, N_actual);
  
  CUDA_OK(cudaPeekAtLastError());
  return ffi::Error::Success();
}

// ============================================================================
// Dispatch Logic: SM Variant × Bucket Size
// ============================================================================

template <typename Scalar, int Nb>
static ffi::Error DispatchSmFwd(int sm, cudaStream_t stream,
                                const Scalar* A, Scalar* sign, Scalar* logabs,
                                int B, int N_actual) {
  switch (sm) {
    case 800:  return LaunchFwd<Scalar, Nb, 800>(stream, A, sign, logabs, B, N_actual);
    case 900:  return LaunchFwd<Scalar, Nb, 900>(stream, A, sign, logabs, B, N_actual);
    case 1200: return LaunchFwd<Scalar, Nb, 1200>(stream, A, sign, logabs, B, N_actual);
    default:   return InvalidArg("Unsupported SM architecture");
  }
}

template <typename Scalar, int Nb>
static ffi::Error DispatchSmBwd(int sm, cudaStream_t stream,
                                const Scalar* A, const Scalar* cot, Scalar* grad,
                                int B, int N_actual) {
  switch (sm) {
    case 800:  return LaunchBwd<Scalar, Nb, 800>(stream, A, cot, grad, B, N_actual);
    case 900:  return LaunchBwd<Scalar, Nb, 900>(stream, A, cot, grad, B, N_actual);
    case 1200: return LaunchBwd<Scalar, Nb, 1200>(stream, A, cot, grad, B, N_actual);
    default:   return InvalidArg("Unsupported SM architecture");
  }
}

template <typename Scalar>
static ffi::Error DispatchFwd(cudaStream_t stream,
                              const Scalar* A, Scalar* sign, Scalar* logabs,
                              int B, int N_actual) {
  const int sm = SelectSmVariant();
  const int Nb = BucketN(N_actual);

  switch (Nb) {
    case 16: return DispatchSmFwd<Scalar, 16>(sm, stream, A, sign, logabs, B, N_actual);
    case 32: return DispatchSmFwd<Scalar, 32>(sm, stream, A, sign, logabs, B, N_actual);
    case 64: return DispatchSmFwd<Scalar, 64>(sm, stream, A, sign, logabs, B, N_actual);
    default: return InvalidArg("Invalid matrix bucket size");
  }
}

template <typename Scalar>
static ffi::Error DispatchBwd(cudaStream_t stream,
                              const Scalar* A, const Scalar* cot, Scalar* grad,
                              int B, int N_actual) {
  const int sm = SelectSmVariant();
  const int Nb = BucketN(N_actual);

  switch (Nb) {
    case 16: return DispatchSmBwd<Scalar, 16>(sm, stream, A, cot, grad, B, N_actual);
    case 32: return DispatchSmBwd<Scalar, 32>(sm, stream, A, cot, grad, B, N_actual);
    case 64: return DispatchSmBwd<Scalar, 64>(sm, stream, A, cot, grad, B, N_actual);
    default: return InvalidArg("Invalid matrix bucket size");
  }
}

#endif  // LEVER_WITH_CUSOLVERDX

// ============================================================================
// XLA FFI Entry Points
// ============================================================================

template <ffi::DataType DT>
static ffi::Error FwdCudaDx(cudaStream_t stream,
                            ffi::Buffer<DT> a,
                            ffi::ResultBuffer<DT> out_sign,
                            ffi::ResultBuffer<DT> out_logabs) {
#ifndef LEVER_WITH_CUSOLVERDX
  return InvalidArg("Built without cuSolverDx support");
#else
  using Scalar = ffi::NativeType<DT>;

  auto dims = a.dimensions();
  if (dims.size() != 3 || dims[1] != dims[2]) {
    return InvalidArg("Expected input shape [B, N, N]");
  }

  const int B = static_cast<int>(dims[0]);
  const int N = static_cast<int>(dims[1]);
  
  if (N < 2 || N > 64) {
    return InvalidArg("Matrix size N must be in range [2, 64]");
  }

  return DispatchFwd<Scalar>(
      stream, a.typed_data(), out_sign->typed_data(), out_logabs->typed_data(), B, N);
#endif
}

template <ffi::DataType DT>
static ffi::Error BwdCudaDx(cudaStream_t stream,
                            ffi::Buffer<DT> a,
                            ffi::Buffer<DT> cot_sign,
                            ffi::Buffer<DT> cot_logabs,
                            ffi::ResultBuffer<DT> out_grad) {
  (void)cot_sign;  // Sign not used in gradient computation
  
#ifndef LEVER_WITH_CUSOLVERDX
  return InvalidArg("Built without cuSolverDx support");
#else
  using Scalar = ffi::NativeType<DT>;

  auto dims = a.dimensions();
  if (dims.size() != 3 || dims[1] != dims[2]) {
    return InvalidArg("Expected input shape [B, N, N]");
  }

  const int B = static_cast<int>(dims[0]);
  const int N = static_cast<int>(dims[1]);
  
  if (N < 2 || N > 64) {
    return InvalidArg("Matrix size N must be in range [2, 64]");
  }

  return DispatchBwd<Scalar>(
      stream, a.typed_data(), cot_logabs.typed_data(), out_grad->typed_data(), B, N);
#endif
}

// ============================================================================
// FFI Registration
// ============================================================================

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f64_fwd_cuda, FwdCudaDx<ffi::DataType::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f64_bwd_cuda, BwdCudaDx<ffi::DataType::F64>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f32_fwd_cuda, FwdCudaDx<ffi::DataType::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f32_bwd_cuda, BwdCudaDx<ffi::DataType::F32>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>());