// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bridge.cpp
 * @brief Python-C++ bridge via nanobind - exposes LEVER core algorithms.
 *
 * Provides bindings for:
 *   - Integral management (MO/SO) with optional Heat-Bath screening
 *   - Determinant generation (FCI, excited spaces)
 *   - Hamiltonian assembly (diagonal, blocks, effective H_eff)
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_enum.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/build_ham.hpp>
#include <lever/hamiltonian/ham_eff.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/hamiltonian/local_conn.hpp>
#include <lever/integral/hb_table.hpp>
#include <lever/integral/integral_mo.hpp>
#include <lever/integral/integral_so.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

using lever::Det;
using lever::DetMap;
using lever::FCISpace;
using lever::HamEval;
using lever::HeatBathTable;
using lever::COOMatrix;
using lever::u32;
using lever::u64;

// ============================================================================
// Type Aliases for NumPy Arrays
// ============================================================================
using DetArrayRO   = nb::ndarray<const u64, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using F64VecRO     = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using DetArrayOut  = nb::ndarray<u64, nb::numpy, nb::shape<-1, 2>>;
using F64VecOut    = nb::ndarray<double, nb::numpy, nb::shape<-1>>;
using U32VecOut    = nb::ndarray<u32, nb::numpy, nb::shape<-1>>;
using I32VecOut    = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>>;
using C128VecRO = nb::ndarray<const std::complex<double>, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using SingleDetRO  = nb::ndarray<const u64, nb::shape<2>, nb::c_contig, nb::device::cpu>;

// ============================================================================
// IntCtx: Integral Context with Optional Heat-Bath Cache
// ============================================================================
struct IntCtx {
    explicit IntCtx(const std::string& fcidump_path, int num_orb)
        : mo(num_orb), so(mo), ham(so) {
        mo.load_from_fcidump(fcidump_path);
    }

    [[nodiscard]] double e_nuc() const noexcept { return so.get_e_nuc(); }

    void hb_prepare(double threshold) {
        lever::HBBuildOptions opt{.threshold = threshold};
        hb = std::make_unique<HeatBathTable>(so, opt);
        hb->build();
    }

    void hb_clear() noexcept { hb.reset(); }

    lever::IntegralMO mo;
    lever::IntegralSO so;
    HamEval ham;
    std::unique_ptr<HeatBathTable> hb;
};

// ============================================================================
// Conversion Helpers: Python ↔ C++
// ============================================================================

/**
 * Convert NumPy array (N×2) → std::vector<Det>.
 */
[[nodiscard]] inline std::vector<Det> to_det_vector(DetArrayRO arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 2) {
        throw std::invalid_argument("Expected shape (N, 2) for determinant array");
    }
    
    const size_t N = arr.shape(0);
    std::vector<Det> out;
    out.reserve(N);
    
    const u64* data = arr.data();
    for (size_t i = 0; i < N; ++i) {
        out.emplace_back(Det{data[2*i], data[2*i + 1]});
    }
    
    return out;
}

/**
 * Convert NumPy 1D array → std::vector<double>.
 */
[[nodiscard]] inline std::vector<double> to_double_vector(F64VecRO arr) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument("Expected 1D array for amplitudes");
    }
    return {arr.data(), arr.data() + arr.shape(0)};
}

/**
 * Convert std::vector<Det> → NumPy array (N×2).
 * Memory managed by Python via capsule.
 */
[[nodiscard]] inline DetArrayOut from_det_vector(const std::vector<Det>& dets) {
    const size_t N = dets.size();
    auto* data = new u64[N * 2];
    
    for (size_t i = 0; i < N; ++i) {
        data[2*i]     = dets[i].alpha;
        data[2*i + 1] = dets[i].beta;
    }
    
    nb::capsule owner(data, [](void* p) noexcept { 
        delete[] static_cast<u64*>(p); 
    });
    
    return DetArrayOut(data, {N, 2}, owner);
}

/**
 * Convert std::vector<double> → NumPy 1D array.
 */
[[nodiscard]] inline F64VecOut from_double_vector(const std::vector<double>& xs) {
    const size_t N = xs.size();
    auto* data = new double[N];
    std::memcpy(data, xs.data(), N * sizeof(double));
    
    nb::capsule owner(data, [](void* p) noexcept { 
        delete[] static_cast<double*>(p); 
    });
    
    return F64VecOut(data, {N}, owner);
}

/**
 * Convert COOMatrix → Python dict with 'row', 'col', 'val', 'shape' keys.
 */
[[nodiscard]] inline nb::dict from_coo_matrix(const COOMatrix& coo) {
    const size_t M = coo.nnz();
    
    auto* row_data = new u32[M];
    auto* col_data = new u32[M];
    auto* val_data = new double[M];
    
    std::memcpy(row_data, coo.rows.data(), M * sizeof(u32));
    std::memcpy(col_data, coo.cols.data(), M * sizeof(u32));
    std::memcpy(val_data, coo.vals.data(), M * sizeof(double));
    
    auto make_owner = [](auto* p) {
        return nb::capsule(p, [](void* ptr) noexcept { 
            delete[] static_cast<decltype(p)>(ptr);
        });
    };
    
    nb::dict d;
    d["row"]   = U32VecOut(row_data, {M}, make_owner(row_data));
    d["col"]   = U32VecOut(col_data, {M}, make_owner(col_data));
    d["val"]   = F64VecOut(val_data, {M}, make_owner(val_data));
    d["shape"] = nb::make_tuple(coo.n_rows, coo.n_cols);
    
    return d;
}

/**
 * Convert Python dict → COOMatrix.
 */
[[nodiscard]] inline COOMatrix to_coo_matrix(const nb::dict& d) {
    COOMatrix coo;
    
    auto row_arr = nb::cast<U32VecOut>(d["row"]);
    auto col_arr = nb::cast<U32VecOut>(d["col"]);
    auto val_arr = nb::cast<F64VecOut>(d["val"]);
    
    const size_t nnz = row_arr.shape(0);
    coo.rows.assign(row_arr.data(), row_arr.data() + nnz);
    coo.cols.assign(col_arr.data(), col_arr.data() + nnz);
    coo.vals.assign(val_arr.data(), val_arr.data() + nnz);
    
    if (d.contains("shape")) {
        auto shape = nb::cast<nb::tuple>(d["shape"]);
        coo.n_rows = nb::cast<u32>(shape[0]);
        coo.n_cols = nb::cast<u32>(shape[1]);
    }
    
    return coo;
}

/**
 * Convert std::vector<int> → NumPy 1D array (int32).
 */
[[nodiscard]] inline I32VecOut from_int_vector(const std::vector<int>& xs) {
    const size_t N = xs.size();
    auto* data = new int32_t[N];

    for (size_t i = 0; i < N; ++i) {
        data[i] = static_cast<int32_t>(xs[i]);
    }

    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<int32_t*>(p);
    });

    return I32VecOut(data, {N}, owner);
}

/**
 * Parse regularization type string → enum.
 */
[[nodiscard]] inline lever::Regularization parse_reg_type(const std::string& s) {
    if (s == "linear_shift") return lever::Regularization::LinearShift;
    if (s == "sigma")        return lever::Regularization::Sigma;
    throw std::invalid_argument("Invalid regularization type: " + s);
}

// ============================================================================
// Module Bindings
// ============================================================================

NB_MODULE(_lever_cpp, m) {
    m.doc() = "LEVER C++ core bridge - determinant CI and Hamiltonian assembly";

    // ------------------------------------------------------------------------
    // IntCtx: Integral container with Heat-Bath cache
    // ------------------------------------------------------------------------
    nb::class_<IntCtx>(m, "IntCtx", "Integral context (MO/SO integrals + optional Heat-Bath)")
        .def(nb::init<const std::string&, int>(), 
             "fcidump_path"_a, "num_orb"_a,
             "Load integrals from FCIDUMP file")
        .def("get_e_nuc", &IntCtx::e_nuc,
             "Nuclear repulsion energy")
        .def("hb_prepare", &IntCtx::hb_prepare, 
             "threshold"_a = lever::HEATBATH_THRESH,
             "Build Heat-Bath screening table with ⟨pq||rs⟩ threshold")
        .def("hb_clear", &IntCtx::hb_clear,
             "Release Heat-Bath cache");

    // ------------------------------------------------------------------------
    // Determinant Generation
    // ------------------------------------------------------------------------
    m.def("gen_fci_dets",
          [](int n_orb, int n_alpha, int n_beta) -> DetArrayOut {
              FCISpace fci(n_orb, n_alpha, n_beta);
              const auto& dets_span = fci.dets();
              return from_det_vector({dets_span.begin(), dets_span.end()});
          },
          "n_orb"_a, "n_alpha"_a, "n_beta"_a,
          "Generate full CI determinant basis");

    m.def("gen_excited_dets",
          [](DetArrayRO ref_dets, int n_orb) -> DetArrayOut {
              auto dets = lever::det_space::generate_connected(
                  to_det_vector(ref_dets), n_orb
              );
              return from_det_vector(
                  lever::det_space::canonicalize(std::move(dets))
              );
          },
          "ref_dets"_a, "n_orb"_a,
          "Generate singly/doubly excited determinants from reference set");

    // ------------------------------------------------------------------------
    // Hamiltonian Diagonal
    // ------------------------------------------------------------------------
    m.def("get_ham_diag",
          [](DetArrayRO dets, const IntCtx* ctx) -> F64VecOut {
              return from_double_vector(
                  lever::get_ham_diag(to_det_vector(dets), ctx->ham)
              );
          },
          "dets"_a, "int_ctx"_a,
          "Compute diagonal Hamiltonian elements ⟨D|H|D⟩");

    // ------------------------------------------------------------------------
    // Hamiltonian H_SS Block (for CI solver)
    // ------------------------------------------------------------------------
    m.def("get_ham_ss",
          [](DetArrayRO dets, const IntCtx* ctx, int n_orb) -> nb::dict {
              auto coo = lever::get_ham_ss(
                  to_det_vector(dets), ctx->ham, n_orb
              );
              return from_coo_matrix(coo);
          },
          "dets_S"_a, "int_ctx"_a, "n_orb"_a,
          "Build H_SS block for Davidson/Lanczos diagonalization");

    // ------------------------------------------------------------------------
    // Hamiltonian Blocks H_SS, H_SC (known C-space)
    // ------------------------------------------------------------------------
    m.def("get_ham_block",
          [](DetArrayRO bra_dets, 
             std::optional<DetArrayRO> ket_dets,
             const IntCtx* ctx, 
             int n_orb) -> nb::dict {
              
              auto dets_S = to_det_vector(bra_dets);
              
              std::optional<std::vector<Det>> dets_C;
              if (ket_dets.has_value()) {
                  dets_C = to_det_vector(*ket_dets);
              }
              
              std::optional<std::span<const Det>> c_span;
              if (dets_C.has_value()) {
                  c_span = *dets_C;
              }
              
              auto blocks = lever::get_ham_block(
                  dets_S, c_span, ctx->ham, n_orb
              );
              
              nb::dict out;
              out["H_SS"] = from_coo_matrix(blocks.H_SS);
              out["H_SC"] = from_coo_matrix(blocks.H_SC);
              
              if (blocks.map_C.size() > 0) {
                  out["det_C"] = from_det_vector(blocks.map_C.all_dets());
                  out["size_C"] = nb::int_(blocks.map_C.size());
              }
              
              return out;
          },
          "bra_dets"_a, 
          "ket_dets"_a = nb::none(),
          "int_ctx"_a, 
          "n_orb"_a,
          "Build H_SS and H_SC with explicit C-space determinants");

    // ------------------------------------------------------------------------
    // Hamiltonian Connections (static Heat-Bath screening)
    // ------------------------------------------------------------------------
    m.def("get_ham_conn",
          [](DetArrayRO S, 
             const IntCtx* ctx, 
             int n_orb,
             bool use_heatbath, 
             double eps1) -> nb::dict {
              
              if (use_heatbath && !ctx->hb) {
                  throw std::invalid_argument(
                      "Heat-Bath table not prepared. Call IntCtx.hb_prepare() first."
                  );
              }
              
              auto blocks = lever::get_ham_conn(
                  to_det_vector(S), 
                  ctx->ham, 
                  n_orb,
                  ctx->hb.get(), 
                  eps1, 
                  use_heatbath
              );
              
              nb::dict out;
              out["H_SS"] = from_coo_matrix(blocks.H_SS);
              out["H_SC"] = from_coo_matrix(blocks.H_SC);
              out["det_C"] = from_det_vector(blocks.map_C.all_dets());
              out["size_C"] = nb::int_(blocks.map_C.size());
              
              return out;
          },
          "dets_S"_a, 
          "int_ctx"_a, 
          "n_orb"_a,
          "use_heatbath"_a = false, 
          "eps1"_a = 1e-6,
          "Build H_SS, H_SC with automatic C-space generation via integral screening");

    // ------------------------------------------------------------------------
    // Hamiltonian Connections (dynamic amplitude screening)
    // ------------------------------------------------------------------------
    m.def("get_ham_conn_amp",
          [](DetArrayRO S, 
             F64VecRO psi_S, 
             const IntCtx* ctx, 
             int n_orb,
             double eps1) -> nb::dict {
              
              if (!ctx->hb) {
                  throw std::invalid_argument(
                      "Heat-Bath table required. Call IntCtx.hb_prepare() first."
                  );
              }
              
              auto dets_S_vec = to_det_vector(S);
              auto psi_S_vec = to_double_vector(psi_S);
              
              if (dets_S_vec.size() != psi_S_vec.size()) {
                  throw std::invalid_argument(
                      "dets_S and psi_S must have same length"
                  );
              }
              
              auto blocks = lever::get_ham_conn_amp(
                  dets_S_vec, 
                  psi_S_vec, 
                  ctx->ham, 
                  n_orb,
                  ctx->hb.get(), 
                  eps1
              );
              
              nb::dict out;
              out["H_SS"] = from_coo_matrix(blocks.H_SS);
              out["H_SC"] = from_coo_matrix(blocks.H_SC);
              out["det_C"] = from_det_vector(blocks.map_C.all_dets());
              out["size_C"] = nb::int_(blocks.map_C.size());
              
              return out;
          },
          "dets_S"_a, 
          "psi_S"_a, 
          "int_ctx"_a, 
          "n_orb"_a,
          "eps1"_a = 1e-6,
          "Build H_SS, H_SC with amplitude-weighted Heat-Bath screening");

    // ------------------------------------------------------------------------
    // Effective Hamiltonian H_eff = H_SS + H_SC·D⁻¹·H_CS
    // ------------------------------------------------------------------------
    m.def("get_ham_eff",
          [](const nb::dict& H_SS_dict,
             const nb::dict& H_SC_dict,
             F64VecRO h_cc_diag,
             double e_ref,
             const std::string& reg_type_str,
             double epsilon,
             bool upper_only) -> nb::dict {
              
              const COOMatrix H_SS = to_coo_matrix(H_SS_dict);
              const COOMatrix H_SC = to_coo_matrix(H_SC_dict);
              
              lever::HeffConfig config{
                  .reg_type = parse_reg_type(reg_type_str),
                  .epsilon = epsilon,
                  .upper_only = upper_only
              };
              
              auto h_eff = lever::get_ham_eff(
                  H_SS, 
                  H_SC, 
                  to_double_vector(h_cc_diag), 
                  e_ref, 
                  config
              );
              
              return from_coo_matrix(h_eff);
          },
          "H_SS"_a,
          "H_SC"_a,
          "h_cc_diag"_a,
          "e_ref"_a,
          "reg_type"_a = "sigma",
          "epsilon"_a = 1e-12,
          "upper_only"_a = true,
          "Assemble effective Hamiltonian via downfolding: H_eff = H_SS + H_SC·D⁻¹·H_CS");

    // ------------------------------------------------------------------------
    // Local Hamiltonian connectivity for a single determinant
    // ------------------------------------------------------------------------
    m.def("get_local_conn",
          [](SingleDetRO det,
             const IntCtx* ctx,
             int n_orb,
             bool use_heatbath,
             double eps1) -> nb::dict {
              if (det.ndim() != 1 || det.shape(0) != 2) {
                  throw std::invalid_argument(
                      "get_local_conn: expected det with shape (2,)"
                  );
              }

              Det bra{det.data()[0], det.data()[1]};

              const HeatBathTable* hb = nullptr;
              if (use_heatbath) {
                  if (!ctx->hb) {
                      throw std::invalid_argument(
                          "get_local_conn: Heat-bath requested but IntCtx.hb is null "
                          "(call IntCtx.hb_prepare() first)"
                      );
                  }
                  hb = ctx->hb.get();
              }

              auto row = lever::get_local_conn(
                  bra, ctx->ham, n_orb, hb, eps1, use_heatbath
              );

              nb::dict out;
              out["dets"]   = from_det_vector(row.dets);
              out["values"] = from_double_vector(row.values);
              return out;
          },
          "det"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "use_heatbath"_a = false,
          "eps1"_a = 1e-6,
          "Build local Hamiltonian row for a single determinant.\n\n"
          "Args:\n"
          "  det         : uint64[2] = [alpha_bits, beta_bits]\n"
          "  int_ctx     : integral context\n"
          "  n_orb  : number of spatial orbitals\n"
          "  use_heatbath: enable Heat-Bath screening for doubles and singles\n"
          "  eps1        : Heat-Bath / single threshold\n\n"
          "Returns:\n"
          "  dict with 'dets' (N,2) and 'values' (N,) arrays.");

    // ------------------------------------------------------------------------
    // Local Hamiltonian connectivity for a batch of determinants (CSR-like)
    // ------------------------------------------------------------------------
    m.def("get_local_connections",
          [](DetArrayRO dets,
             const IntCtx* ctx,
             int n_orb,
             bool use_heatbath,
             double eps1) -> nb::dict {
              // Convert input determinants
              auto samples_vec = to_det_vector(dets);

              const HeatBathTable* hb = nullptr;
              if (use_heatbath) {
                  if (!ctx->hb) {
                      throw std::invalid_argument(
                          "get_local_connections: Heat-bath requested but IntCtx.hb is null "
                          "(call IntCtx.hb_prepare() first)"
                      );
                  }
                  hb = ctx->hb.get();
              }

              auto batch = lever::get_local_connections(
                  std::span<const Det>(samples_vec.data(), samples_vec.size()),
                  ctx->ham,
                  n_orb,
                  hb,
                  eps1,
                  use_heatbath
              );

              nb::dict out;
              out["offsets"] = from_int_vector(batch.offsets);
              out["dets"]    = from_det_vector(batch.dets);
              out["values"]  = from_double_vector(batch.values);
              return out;
          },
          "dets"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "use_heatbath"_a = false,
          "eps1"_a = 1e-6,
          "Build local Hamiltonian connections for a batch of determinants.\n\n"
          "Args:\n"
          "  dets        : uint64[N,2] determinant array\n"
          "  int_ctx     : integral context\n"
          "  n_orb  : number of spatial orbitals\n"
          "  use_heatbath: enable Heat-Bath screening for doubles and singles\n"
          "  eps1        : Heat-Bath / single threshold\n\n"
          "Returns:\n"
          "  dict with:\n"
          "    'offsets' : int32[N+1] CSR row offsets\n"
          "    'dets'    : uint64[M,2] concatenated ket determinants\n"
          "    'values'  : float64[M]  ⟨bra_i|H|ket_j⟩ values in CSR order.");

    // ------------------------------------------------------------------------
    // Streaming variational energy: <Psi|H|Psi> on a fixed determinant basis
    // ------------------------------------------------------------------------
    m.def("compute_variational_energy",
          [](DetArrayRO dets,
             C128VecRO coeffs,
             const IntCtx* ctx,
             int n_orb,
             bool use_heatbath,
             double eps1) -> nb::tuple {
              
              auto basis_vec = to_det_vector(dets);
              
              if (coeffs.ndim() != 1) {
                   throw std::invalid_argument("coeffs must be 1D");
              }
              if (static_cast<size_t>(coeffs.shape(0)) != basis_vec.size()) {
                   throw std::invalid_argument("coeffs size mismatch");
              }

              // Zero-copy view of coefficients
              std::span<const std::complex<double>> c_span(
                  coeffs.data(), coeffs.shape(0)
              );

              auto res = lever::compute_variational_energy(
                  std::span<const Det>(basis_vec),
                  c_span,
                  ctx->ham,
                  n_orb,
                  use_heatbath ? ctx->hb.get() : nullptr,
                  eps1,
                  use_heatbath
              );

              return nb::make_tuple(res.e_el, res.norm);
          },
          "dets"_a, "coeffs"_a, "int_ctx"_a, "n_orb"_a,
          "use_heatbath"_a = false, "eps1"_a = 1e-6,
          "Compute <Psi|H|Psi> and <Psi|Psi> on fixed basis (S U C).");
}
