// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * FFI module registration for JAX XLA custom calls.
 * 
 * Exposes fused log-determinant operations (forward/backward) to JAX
 * via XLA FFI on CPU and CUDA backends.
 * 
 * File: csrc/ffi/ffi_module.cpp
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <type_traits>

#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

// Wraps XLA FFI handler into Python capsule for registration
template <typename T>
static nb::capsule EncapsulateFfiCall(T* fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                "Function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void*>(fn));
}

// External C linkage for XLA FFI handlers
extern "C" {

// CPU handlers for double precision
XLA_FFI_Error* lever_fused_logdet_f64_fwd_cpu(XLA_FFI_CallFrame*);
XLA_FFI_Error* lever_fused_logdet_f64_bwd_cpu(XLA_FFI_CallFrame*);

// CPU handlers for single precision
XLA_FFI_Error* lever_fused_logdet_f32_fwd_cpu(XLA_FFI_CallFrame*);
XLA_FFI_Error* lever_fused_logdet_f32_bwd_cpu(XLA_FFI_CallFrame*);

#ifdef LEVER_WITH_CUDA
// CUDA handlers for double precision
XLA_FFI_Error* lever_fused_logdet_f64_fwd_cuda(XLA_FFI_CallFrame*);
XLA_FFI_Error* lever_fused_logdet_f64_bwd_cuda(XLA_FFI_CallFrame*);

// CUDA handlers for single precision
XLA_FFI_Error* lever_fused_logdet_f32_fwd_cuda(XLA_FFI_CallFrame*);
XLA_FFI_Error* lever_fused_logdet_f32_bwd_cuda(XLA_FFI_CallFrame*);
#endif

}  // extern "C"

NB_MODULE(_lever_ffi, m) {
  // Register CPU handlers
  m.def("fused_logdet_f64_fwd_cpu",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f64_fwd_cpu); });
  m.def("fused_logdet_f64_bwd_cpu",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f64_bwd_cpu); });
  m.def("fused_logdet_f32_fwd_cpu",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f32_fwd_cpu); });
  m.def("fused_logdet_f32_bwd_cpu",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f32_bwd_cpu); });

#ifdef LEVER_WITH_CUDA
  // Register CUDA handlers
  m.def("fused_logdet_f64_fwd_cuda",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f64_fwd_cuda); });
  m.def("fused_logdet_f64_bwd_cuda",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f64_bwd_cuda); });
  m.def("fused_logdet_f32_fwd_cuda",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f32_fwd_cuda); });
  m.def("fused_logdet_f32_bwd_cuda",
        []() { return EncapsulateFfiCall(lever_fused_logdet_f32_bwd_cuda); });
#else
  // Stub returns for CUDA when not built with CUDA support
  m.def("fused_logdet_f64_fwd_cuda", []() { return nb::none(); });
  m.def("fused_logdet_f64_bwd_cuda", []() { return nb::none(); });
  m.def("fused_logdet_f32_fwd_cuda", []() { return nb::none(); });
  m.def("fused_logdet_f32_bwd_cuda", []() { return nb::none(); });
#endif
}
