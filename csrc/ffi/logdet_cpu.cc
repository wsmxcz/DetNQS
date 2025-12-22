// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*
 * CPU implementation of fused log-determinant with automatic differentiation.
 * 
 * Forward Pass:
 *   Computes sign and log|det(A)| via LU decomposition for batched matrices.
 *   For matrix A with shape [B, N, N], outputs:
 *     - sign: [B] tensor of {-1, 0, 1}
 *     - logabs: [B] tensor of log|det(A_b)|
 * 
 * Backward Pass:
 *   Gradient: grad_A = cot_logabs * A^{-T}
 *   where A^{-T} is the transpose of the inverse matrix.
 * 
 * File: csrc/ffi/logdet_cpu.cc
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <cmath>
#include <limits>
#include <string>
#include <Eigen/Dense>
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ============================================================================
// Helper Functions
// ============================================================================

static inline ffi::Error InvalidArg(const std::string& msg) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, msg);
}

template <typename Buf>
static inline ffi::Error ValidateInput3D(Buf& a, int64_t& B, int64_t& N) {
  auto dims = a.dimensions();
  if (dims.size() != 3) return InvalidArg("Input A must be rank-3: [B, N, N]");
  if (dims[1] != dims[2]) return InvalidArg("Last two dimensions must be square");
  B = dims[0];
  N = dims[1];
  return ffi::Error::Success();
}

template <typename Out>
static inline ffi::Error ValidateOutput1D(Out& out, int64_t B, const char* name) {
  auto dims = out->dimensions();
  if (dims.size() != 1 || dims[0] != B) {
    return InvalidArg(std::string(name) + " must have shape [B]");
  }
  return ffi::Error::Success();
}

// ============================================================================
// Forward Pass: Compute sign and log|det(A)|
// ============================================================================

template <ffi::DataType DT>
static ffi::Error LogdetForward(ffi::Buffer<DT> a,
                                ffi::ResultBuffer<DT> out_sign,
                                ffi::ResultBuffer<DT> out_logabs) {
  using T = ffi::NativeType<DT>;
  int64_t B = 0, N = 0;

  if (auto err = ValidateInput3D(a, B, N); err.failure()) return err;
  if (auto err = ValidateOutput1D(out_sign, B, "sign"); err.failure()) return err;
  if (auto err = ValidateOutput1D(out_logabs, B, "logabs"); err.failure()) return err;

  const T* A = a.typed_data();
  T* S = out_sign->typed_data();
  T* L = out_logabs->typed_data();

  #pragma omp parallel for
  for (int64_t b = 0; b < B; ++b) {
    // Copy batch element to Eigen matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(N, N);
    const T* Ab = A + b * N * N;
    for (int64_t i = 0; i < N; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        M(i, j) = Ab[i * N + j];
      }
    }

    // Compute determinant via LU decomposition
    Eigen::PartialPivLU<decltype(M)> lu(M);
    T det = lu.determinant();
    double det_d = static_cast<double>(det);
    double abs_det = std::abs(det_d);

    // Handle singular or invalid matrices
    if (abs_det <= 0.0 || !std::isfinite(abs_det)) {
      S[b] = T(0);
      L[b] = static_cast<T>(-std::numeric_limits<double>::infinity());
    } else {
      S[b] = (det_d < 0.0) ? T(-1) : T(1);
      L[b] = static_cast<T>(std::log(abs_det));
    }
  }

  return ffi::Error::Success();
}

// ============================================================================
// Backward Pass: Compute gradient w.r.t. A
// ============================================================================

template <ffi::DataType DT>
static ffi::Error LogdetBackward(ffi::Buffer<DT> a,
                                 ffi::Buffer<DT> cot_sign,
                                 ffi::Buffer<DT> cot_logabs,
                                 ffi::ResultBuffer<DT> out_grad) {
  using T = ffi::NativeType<DT>;
  int64_t B = 0, N = 0;

  // Sign is non-differentiable, suppress unused warning
  (void)cot_sign;

  if (auto err = ValidateInput3D(a, B, N); err.failure()) return err;

  auto gd = out_grad->dimensions();
  if (gd.size() != 3 || gd[0] != B || gd[1] != N || gd[2] != N) {
    return InvalidArg("Gradient output must have shape [B, N, N]");
  }

  const T* A = a.typed_data();
  const T* CL = cot_logabs.typed_data();
  T* G = out_grad->typed_data();

  #pragma omp parallel for
  for (int64_t b = 0; b < B; ++b) {
    // Copy batch element to Eigen matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(N, N);
    const T* Ab = A + b * N * N;
    for (int64_t i = 0; i < N; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        M(i, j) = Ab[i * N + j];
      }
    }

    // Check matrix validity via determinant
    Eigen::PartialPivLU<decltype(M)> lu(M);
    T det = lu.determinant();
    double det_d = static_cast<double>(det);
    double abs_det = std::abs(det_d);

    T* Gb = G + b * N * N;

    // Zero gradient for singular matrices
    if (abs_det <= 0.0 || !std::isfinite(abs_det)) {
      std::fill_n(Gb, N * N, T(0));
      continue;
    }

    // Gradient formula: grad_A = cot_logabs * A^{-T}
    auto I = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Identity(N, N);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ainv = lu.solve(I);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> grad =
        CL[b] * Ainv.transpose();

    // Copy result back
    for (int64_t i = 0; i < N; ++i) {
      for (int64_t j = 0; j < N; ++j) {
        Gb[i * N + j] = grad(i, j);
      }
    }
  }

  return ffi::Error::Success();
}

// ============================================================================
// XLA FFI Symbol Exports
// ============================================================================

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f64_fwd_cpu, LogdetForward<ffi::DataType::F64>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f64_bwd_cpu, LogdetBackward<ffi::DataType::F64>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f32_fwd_cpu, LogdetForward<ffi::DataType::F32>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    lever_fused_logdet_f32_bwd_cpu, LogdetBackward<ffi::DataType::F32>,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Arg<ffi::Buffer<ffi::DataType::F32>>()
        .Ret<ffi::Buffer<ffi::DataType::F32>>()
);
