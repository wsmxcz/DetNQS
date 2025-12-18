// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bridge.cpp
 * @brief Python-C++ nanobind bridge for LEVER core operations.
 *
 * Provides Python bindings for:
 *  - Determinant generation (FCI, excitations)
 *  - Hamiltonian construction (H_SS, H_SC, H_eff)
 *  - Connection screening (Heat-Bath, amplitude-weighted)
 *  - Variational energy evaluation
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
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
#include <lever/hamiltonian/ham_utils.hpp>
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

// NumPy array type aliases
using DetArrayRO  = nb::ndarray<const u64, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using F64VecRO    = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using DetArrayOut = nb::ndarray<u64, nb::numpy, nb::shape<-1, 2>>;
using F64VecOut   = nb::ndarray<double, nb::numpy, nb::shape<-1>>;
using U32VecOut   = nb::ndarray<u32, nb::numpy, nb::shape<-1>>;
using I32VecOut   = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>>;
using C128VecRO   = nb::ndarray<const std::complex<double>, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using SingleDetRO = nb::ndarray<const u64, nb::shape<2>, nb::c_contig, nb::device::cpu>;

// Integral context wrapper
struct IntCtx {
    explicit IntCtx(const std::string& fcidump_path, int num_orb)
        : mo(num_orb), so(mo), ham(so) {
        mo.load_from_fcidump(fcidump_path);
    }

    [[nodiscard]] double get_e_nuc() const noexcept { return so.get_e_nuc(); }

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

// Python ↔ C++ conversion utilities
[[nodiscard]] inline std::vector<Det> to_det_vector(DetArrayRO arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 2) {
        throw std::invalid_argument("det array must have shape (N,2)");
    }
    const size_t N = arr.shape(0);
    std::vector<Det> out;
    out.reserve(N);
    const u64* data = arr.data();
    for (size_t i = 0; i < N; ++i) {
        out.emplace_back(Det{data[2 * i], data[2 * i + 1]});
    }
    return out;
}

[[nodiscard]] inline std::vector<double> to_double_vector(F64VecRO arr) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument("expected 1D double array");
    }
    return {arr.data(), arr.data() + arr.shape(0)};
}

[[nodiscard]] inline DetArrayOut from_det_vector(const std::vector<Det>& dets) {
    const size_t N = dets.size();
    auto* data = new u64[N * 2];
    for (size_t i = 0; i < N; ++i) {
        data[2 * i]     = dets[i].alpha;
        data[2 * i + 1] = dets[i].beta;
    }
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<u64*>(p);
    });
    return DetArrayOut(data, {N, 2}, owner);
}

[[nodiscard]] inline F64VecOut from_double_vector(const std::vector<double>& xs) {
    const size_t N = xs.size();
    auto* data = new double[N];
    std::memcpy(data, xs.data(), N * sizeof(double));
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<double*>(p);
    });
    return F64VecOut(data, {N}, owner);
}

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
            delete[] static_cast<std::remove_pointer_t<decltype(p)>*>(ptr);
        });
    };

    nb::dict d;
    d["row"]   = U32VecOut(row_data, {M}, make_owner(row_data));
    d["col"]   = U32VecOut(col_data, {M}, make_owner(col_data));
    d["val"]   = F64VecOut(val_data, {M}, make_owner(val_data));
    d["shape"] = nb::make_tuple(coo.n_rows, coo.n_cols);
    return d;
}

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

[[nodiscard]] inline I32VecOut from_int_vector(const std::vector<int>& xs) {
    const size_t N = xs.size();
    auto* data = new int32_t[N];
    for (size_t i = 0; i < N; ++i) data[i] = static_cast<int32_t>(xs[i]);
    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<int32_t*>(p);
    });
    return I32VecOut(data, {N}, owner);
}

[[nodiscard]] inline lever::Regularization parse_reg_type(const std::string& s) {
    if (s == "linear_shift") return lever::Regularization::LinearShift;
    if (s == "sigma")        return lever::Regularization::Sigma;
    throw std::invalid_argument("invalid reg_type: " + s);
}

[[nodiscard]] inline lever::ScreenMode parse_screen_mode(const std::string& s) {
    if (s == "none")    return lever::ScreenMode::None;
    if (s == "static")  return lever::ScreenMode::Static;
    if (s == "dynamic") return lever::ScreenMode::Dynamic;
    throw std::invalid_argument("invalid screen mode: " + s);
}

// Module definition
NB_MODULE(_lever_cpp, m) {
    m.doc() = "LEVER C++ core bridge";

    // Integral context
    nb::class_<IntCtx>(m, "IntCtx", "Integral context (MO/SO + Heat-Bath)")
        .def(nb::init<const std::string&, int>(),
             "fcidump_path"_a, "num_orb"_a)
        .def("get_e_nuc", &IntCtx::get_e_nuc)
        .def("hb_prepare", &IntCtx::hb_prepare,
             "threshold"_a = lever::HEATBATH_THRESH)
        .def("hb_clear", &IntCtx::hb_clear);

    // Determinant generation
    m.def("gen_fci_dets",
          [](int n_orb, int n_alpha, int n_beta) -> DetArrayOut {
              FCISpace fci(n_orb, n_alpha, n_beta);
              auto span = fci.dets();
              return from_det_vector({span.begin(), span.end()});
          },
          "n_orb"_a, "n_alpha"_a, "n_beta"_a,
          "Generate full CI space.");

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
          "Generate single/double excitations (no screening).");

    // Screened complement generation
    m.def("gen_complement_dets",
        [](DetArrayRO ref_dets,
            int n_orb,
            nb::object int_ctx_obj,
            nb::object psi_S_obj,
            const std::string& mode_str,
            double eps1) -> DetArrayOut {
            auto S = to_det_vector(ref_dets);
            const DetMap map_S = DetMap::from_ordered(
                {S.begin(), S.end()}, true
            );

            const auto mode = parse_screen_mode(mode_str);

            // Mode "none": no integrals needed, pure combinatorial generation
            if (mode == lever::ScreenMode::None) {
                auto comp = lever::det_space::generate_complement(
                    std::span<const Det>(S.data(), S.size()),
                    n_orb,
                    map_S,
                    true  // canonicalize result
                );
                return from_det_vector(comp);
            }

            // Screened modes: IntCtx is required
            if (int_ctx_obj.is_none()) {
                throw std::invalid_argument(
                    "gen_complement: int_ctx required for mode='static' or 'dynamic'"
                );
            }
            const IntCtx* ctx = nb::cast<const IntCtx*>(int_ctx_obj);

            // Check Heat-Bath table availability
            if (!ctx->hb) {
                throw std::invalid_argument(
                    "gen_complement: Heat-Bath table missing; "
                    "call IntCtx.hb_prepare()"
                );
            }
            const HeatBathTable* hb = ctx->hb.get();

            // Dynamic mode: psi_S coefficient vector required
            std::vector<double> psi_vec;
            std::span<const double> psi_span;

            if (mode == lever::ScreenMode::Dynamic) {
                if (psi_S_obj.is_none()) {
                    throw std::invalid_argument(
                        "gen_complement: psi_S required for mode='dynamic'"
                    );
                }
                auto psi_arr = nb::cast<F64VecRO>(psi_S_obj);
                psi_vec = to_double_vector(psi_arr);
                if (psi_vec.size() != S.size()) {
                    throw std::invalid_argument(
                        "gen_complement: psi_S size must match ref_dets"
                    );
                }
                psi_span = std::span<const double>(
                    psi_vec.data(), psi_vec.size()
                );
            }

            auto comp = lever::generate_complement_screened(
                std::span<const Det>(S.data(), S.size()),
                n_orb,
                map_S,
                ctx->ham,
                hb,
                mode,
                psi_span,
                eps1
            );

            return from_det_vector(comp);
        },
        "ref_dets"_a,
        "n_orb"_a,
        "int_ctx"_a = nb::none(),
        "psi_S"_a = nb::none(),
        "mode"_a = "none",
        "eps1"_a = 1e-6,
        "Generate screened complement C-space from reference S.\n"
        "  mode='none': pure combinatorial (no integrals)\n"
        "  mode='static': Heat-Bath screening with fixed eps1\n"
        "  mode='dynamic': adaptive screening using psi_S amplitudes");


    // Diagonal Hamiltonian elements
    m.def("get_ham_diag",
          [](DetArrayRO dets, const IntCtx* ctx) -> F64VecOut {
              return from_double_vector(
                  lever::get_ham_diag(to_det_vector(dets), ctx->ham)
              );
          },
          "dets"_a, "int_ctx"_a,
          "Diagonal ⟨D|H|D⟩.");

    // H_SS block construction
    m.def("get_ham_ss",
          [](DetArrayRO dets, const IntCtx* ctx, int n_orb) -> nb::dict {
              auto coo = lever::get_ham_ss(
                  to_det_vector(dets), ctx->ham, n_orb
              );
              return from_coo_matrix(coo);
          },
          "dets_S"_a, "int_ctx"_a, "n_orb"_a,
          "Build H_SS block.");

    // H_SS/H_SC with explicit C-space
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
                  c_span = std::span<const Det>(
                      dets_C->data(), dets_C->size()
                  );
              }

              auto blocks = lever::get_ham_block(
                  dets_S, c_span, ctx->ham, n_orb
              );

              nb::dict out;
              out["H_SS"] = from_coo_matrix(blocks.H_SS);
              out["H_SC"] = from_coo_matrix(blocks.H_SC);
              if (blocks.map_C.size() > 0) {
                  out["det_C"]  = from_det_vector(blocks.map_C.all_dets());
                  out["size_C"] = nb::int_(blocks.map_C.size());
              }
              return out;
          },
          "bra_dets"_a,
          "ket_dets"_a = nb::none(),
          "int_ctx"_a,
          "n_orb"_a,
          "Build H_SS and H_SC with explicit C-space.");

    // Static Heat-Bath connections
    m.def("get_ham_conn",
          [](DetArrayRO S,
             const IntCtx* ctx,
             int n_orb,
             bool use_heatbath,
             double eps1) -> nb::dict {
              if (use_heatbath && !ctx->hb) {
                  throw std::invalid_argument(
                      "get_ham_conn: call IntCtx.hb_prepare() first"
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
              out["H_SS"]   = from_coo_matrix(blocks.H_SS);
              out["H_SC"]   = from_coo_matrix(blocks.H_SC);
              out["det_C"]  = from_det_vector(blocks.map_C.all_dets());
              out["size_C"] = nb::int_(blocks.map_C.size());
              return out;
          },
          "dets_S"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "use_heatbath"_a = false,
          "eps1"_a = 1e-6,
          "Build H_SS/H_SC with static Heat-Bath screening.");

    // Dynamic amplitude-weighted connections
    m.def("get_ham_conn_amp",
          [](DetArrayRO S,
             F64VecRO psi_S,
             const IntCtx* ctx,
             int n_orb,
             double eps1) -> nb::dict {
              if (!ctx->hb) {
                  throw std::invalid_argument(
                      "get_ham_conn_amp: call IntCtx.hb_prepare() first"
                  );
              }

              auto dets_S_vec = to_det_vector(S);
              auto psi_S_vec  = to_double_vector(psi_S);

              if (dets_S_vec.size() != psi_S_vec.size()) {
                  throw std::invalid_argument(
                      "get_ham_conn_amp: dets_S and psi_S size mismatch"
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
              out["H_SS"]   = from_coo_matrix(blocks.H_SS);
              out["H_SC"]   = from_coo_matrix(blocks.H_SC);
              out["det_C"]  = from_det_vector(blocks.map_C.all_dets());
              out["size_C"] = nb::int_(blocks.map_C.size());
              return out;
          },
          "dets_S"_a,
          "psi_S"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "eps1"_a = 1e-6,
          "Build H_SS/H_SC with dynamic amplitude screening.");

    // Effective Hamiltonian construction
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

              lever::HeffConfig cfg{
                  .reg_type   = parse_reg_type(reg_type_str),
                  .epsilon    = epsilon,
                  .upper_only = upper_only
              };

              auto h_eff = lever::get_ham_eff(
                  H_SS,
                  H_SC,
                  to_double_vector(h_cc_diag),
                  e_ref,
                  cfg
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
          "Assemble effective Hamiltonian H_eff = H_SS - H_SC·(H_CC - E_ref)^{-1}·H_SC^T.");

    // Local Hamiltonian connections for batch processing
    m.def("get_local_connections",
          [](DetArrayRO dets,
             const IntCtx* ctx,
             int n_orb,
             bool use_heatbath,
             double eps1) -> nb::dict {
              auto samples_vec = to_det_vector(dets);

              const HeatBathTable* hb = nullptr;
              if (use_heatbath) {
                  if (!ctx->hb) {
                      throw std::invalid_argument(
                          "get_local_connections: call IntCtx.hb_prepare() first"
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
          "Local Hamiltonian connections for a batch of determinants.");

    // Variational energy evaluation
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
                  throw std::invalid_argument(
                      "compute_variational_energy: coeff size mismatch"
                  );
              }

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
          "dets"_a,
          "coeffs"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "use_heatbath"_a = false,
          "eps1"_a = 1e-6,
          "Compute variational energy ⟨Ψ|H|Ψ⟩ on fixed determinant basis.");
} // NB_MODULE

