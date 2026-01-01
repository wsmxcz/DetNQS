// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

template <typename T>
static nb::capsule EncapsulateFfiCall(T* fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                "Function must be an XLA FFI handler");
  return nb::capsule(reinterpret_cast<void*>(fn));
}

extern "C" {

#ifdef detnqs_WITH_CUDA
XLA_FFI_Error* detnqs_fused_logdet_f64_fwd_cuda(XLA_FFI_CallFrame*);
XLA_FFI_Error* detnqs_fused_logdet_f64_bwd_cuda(XLA_FFI_CallFrame*);
XLA_FFI_Error* detnqs_fused_logdet_f32_fwd_cuda(XLA_FFI_CallFrame*);
XLA_FFI_Error* detnqs_fused_logdet_f32_bwd_cuda(XLA_FFI_CallFrame*);
#endif

}  // extern "C"

NB_MODULE(_detnqs_ffi, m) {
#ifdef detnqs_WITH_CUDA
  // CUDA handlers (CPU FFI removed)
  m.def("fused_logdet_f64_fwd_cuda",
        []() { return EncapsulateFfiCall(detnqs_fused_logdet_f64_fwd_cuda); });
  m.def("fused_logdet_f64_bwd_cuda",
        []() { return EncapsulateFfiCall(detnqs_fused_logdet_f64_bwd_cuda); });
  m.def("fused_logdet_f32_fwd_cuda",
        []() { return EncapsulateFfiCall(detnqs_fused_logdet_f32_fwd_cuda); });
  m.def("fused_logdet_f32_bwd_cuda",
        []() { return EncapsulateFfiCall(detnqs_fused_logdet_f32_bwd_cuda); });
#else
  // Stubs: module importable on CPU-only builds
  m.def("fused_logdet_f64_fwd_cuda", []() { return nb::none(); });
  m.def("fused_logdet_f64_bwd_cuda", []() { return nb::none(); });
  m.def("fused_logdet_f32_fwd_cuda", []() { return nb::none(); });
  m.def("fused_logdet_f32_bwd_cuda", []() { return nb::none(); });
#endif
}