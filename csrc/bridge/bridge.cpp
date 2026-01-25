// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bridge.cpp
 * @brief Python-C++ nanobind bridge for deterministic quantum chemistry core.
 *
 * Provides Python bindings for:
 *  - Determinant space generation (FCI, excitations, perturbative complement)
 *  - Hamiltonian matrix construction (H_VV, H_VP, H_eff)
 *  - Heat-Bath screening and amplitude-weighted connection pruning
 *  - Variational energy evaluation and EN-PT2 correction
 *  - Batch feature preparation for neural network forward passes
 *
 * Key notation:
 *  - V: Variational set (primary selected space)
 *  - C: Connected set (all determinants coupled to V via Hamiltonian)
 *  - P: Perturbative set (external complement, P = C \ V)
 *  - T: Target set for deterministic evaluation (T = V union P)
 *
 * Hamiltonian blocks:
 *  - H_VV: Variational block (rows/cols in V)
 *  - H_VP: Coupling block (rows in V, cols in P)
 *  - H_eff: Effective Hamiltonian via downfolding
 *           H_eff = H_VV - H_VP · (H_PP - E_ref)^{-1} · H_VP^T
 *
 * Screening modes:
 *  - None: Combinatorial generation (no integral screening)
 *  - Static: Heat-Bath threshold filtering
 *  - Dynamic: Amplitude-weighted pruning using |psi_v|
 *
 * Author: Zheng (Alex) Che, wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <detnqs/determinant/det.hpp>
#include <detnqs/determinant/det_enum.hpp>
#include <detnqs/determinant/det_space.hpp>
#include <detnqs/determinant/det_batch.hpp>
#include <detnqs/hamiltonian/build_ham.hpp>
#include <detnqs/hamiltonian/ham_eff.hpp>
#include <detnqs/hamiltonian/ham_eval.hpp>
#include <detnqs/hamiltonian/ham_utils.hpp>
#include <detnqs/hamiltonian/local_conn.hpp>
#include <detnqs/integral/hb_table.hpp>
#include <detnqs/integral/integral_mo.hpp>
#include <detnqs/integral/integral_so.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;

using detnqs::Det;
using detnqs::DetMap;
using detnqs::FCISpace;
using detnqs::HamEval;
using detnqs::HeatBathTable;
using detnqs::COOMatrix;
using detnqs::u32;
using detnqs::u64;

// ============================================================================
// NumPy array type aliases for nanobind interface
// ============================================================================

using DetArrayRO  = nb::ndarray<const u64, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using F64VecRO    = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using SingleDetRO = nb::ndarray<const u64, nb::shape<2>, nb::c_contig, nb::device::cpu>;

using DetArrayOut = nb::ndarray<u64, nb::numpy, nb::shape<-1, 2>>;
using F64VecOut   = nb::ndarray<double, nb::numpy, nb::shape<-1>>;
using U32VecOut   = nb::ndarray<u32, nb::numpy, nb::shape<-1>>;
using I32VecOut   = nb::ndarray<int32_t, nb::numpy, nb::shape<-1>>;
using I8VecOut    = nb::ndarray<int8_t, nb::numpy, nb::shape<-1>>;
using BoolMatOut  = nb::ndarray<bool, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;
using I32MatOut   = nb::ndarray<int32_t, nb::numpy, nb::shape<-1, -1>, nb::c_contig>;

// ============================================================================
// Memory management utilities
// ============================================================================

template <typename T>
static inline nb::capsule make_owner(T* p) {
    return nb::capsule(p, [](void* ptr) noexcept {
        delete[] static_cast<T*>(ptr);
    });
}

template <typename T>
static inline T* alloc(std::size_t n) {
    return (n == 0) ? nullptr : new T[n];
}

// ============================================================================
// Integral context wrapper
// ============================================================================

struct IntCtx {
    explicit IntCtx(const std::string& fcidump_path, int num_orb)
        : mo(num_orb), so(mo), ham(so) {
        mo.load_from_fcidump(fcidump_path);
    }

    [[nodiscard]] double get_e_nuc() const noexcept { return so.get_e_nuc(); }

    void hb_prepare(double threshold) {
        detnqs::HBBuildOptions opt{.threshold = threshold};
        hb = std::make_unique<HeatBathTable>(so, opt);
        hb->build();
    }

    void hb_clear() noexcept { hb.reset(); }

    detnqs::IntegralMO mo;
    detnqs::IntegralSO so;
    HamEval ham;
    std::unique_ptr<HeatBathTable> hb;
};

// ============================================================================
// Python-C++ conversion utilities
// ============================================================================

[[nodiscard]] inline std::vector<Det> to_det_vector(DetArrayRO arr) {
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
    return {arr.data(), arr.data() + arr.shape(0)};
}

[[nodiscard]] inline DetArrayOut from_det_vector(const std::vector<Det>& dets) {
    const size_t N = dets.size();
    auto* data = new u64[N * 2];
    for (size_t i = 0; i < N; ++i) {
        data[2 * i]     = dets[i].alpha;
        data[2 * i + 1] = dets[i].beta;
    }
    return DetArrayOut(data, {N, 2}, make_owner(data));
}

[[nodiscard]] inline F64VecOut from_double_vector(const std::vector<double>& xs) {
    const size_t N = xs.size();
    auto* data = new double[N];
    std::memcpy(data, xs.data(), N * sizeof(double));
    return F64VecOut(data, {N}, make_owner(data));
}

[[nodiscard]] inline I32VecOut from_int_vector(const std::vector<int>& xs) {
    const size_t N = xs.size();
    auto* data = new int32_t[N];
    for (size_t i = 0; i < N; ++i) data[i] = static_cast<int32_t>(xs[i]);
    return I32VecOut(data, {N}, make_owner(data));
}

[[nodiscard]] inline nb::dict from_coo_matrix(const COOMatrix& coo) {
    const size_t M = coo.nnz();

    auto* row_data = new u32[M];
    auto* col_data = new u32[M];
    auto* val_data = new double[M];

    std::memcpy(row_data, coo.rows.data(), M * sizeof(u32));
    std::memcpy(col_data, coo.cols.data(), M * sizeof(u32));
    std::memcpy(val_data, coo.vals.data(), M * sizeof(double));

    auto make_capsule = [](auto* p) {
        return nb::capsule(p, [](void* ptr) noexcept {
            delete[] static_cast<std::remove_pointer_t<decltype(p)>*>(ptr);
        });
    };

    nb::dict d;
    d["row"]   = U32VecOut(row_data, {M}, make_capsule(row_data));
    d["col"]   = U32VecOut(col_data, {M}, make_capsule(col_data));
    d["val"]   = F64VecOut(val_data, {M}, make_capsule(val_data));
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

[[nodiscard]] inline detnqs::Regularization parse_reg_type(const std::string& s) {
    if (s == "linear_shift") return detnqs::Regularization::LinearShift;
    if (s == "sigma")        return detnqs::Regularization::Sigma;
    throw std::invalid_argument("invalid reg_type: " + s);
}

[[nodiscard]] inline detnqs::ScreenMode parse_screen_mode(const std::string& s) {
    if (s == "none")    return detnqs::ScreenMode::None;
    if (s == "static")  return detnqs::ScreenMode::Static;
    if (s == "dynamic") return detnqs::ScreenMode::Dynamic;
    throw std::invalid_argument("invalid screen mode: " + s);
}

// ============================================================================
// Module bindings
// ============================================================================

NB_MODULE(_detnqs_cpp, m) {
    m.doc() = "detnqs C++ core bridge";
    nb::set_leak_warnings(false);

    // ------------------------------------------------------------------------
    // Integral context
    // ------------------------------------------------------------------------
    nb::class_<IntCtx>(m, "IntCtx", "Molecular integral context (MO/SO + Heat-Bath table)")
        .def(nb::init<const std::string&, int>(),
             "fcidump_path"_a, "num_orb"_a)
        .def("get_e_nuc", &IntCtx::get_e_nuc)
        .def("hb_prepare", &IntCtx::hb_prepare,
             "threshold"_a = detnqs::HEATBATH_THRESH)
        .def("hb_clear", &IntCtx::hb_clear);

    // ------------------------------------------------------------------------
    // Determinant space generation
    // ------------------------------------------------------------------------
    m.def("gen_fci_dets",
          [](int n_orb, int n_alpha, int n_beta) -> DetArrayOut {
              FCISpace fci(n_orb, n_alpha, n_beta);
              auto span = fci.dets();
              return from_det_vector({span.begin(), span.end()});
          },
          "n_orb"_a, "n_alpha"_a, "n_beta"_a,
          "Generate full CI determinant space.");

    m.def("gen_connected_dets",
          [](DetArrayRO ref_dets, int n_orb) -> DetArrayOut {
              auto dets = detnqs::det_space::generate_connected(
                  to_det_vector(ref_dets), n_orb
              );
              return from_det_vector(
                  detnqs::det_space::canonicalize(std::move(dets))
              );
          },
          "ref_dets"_a, "n_orb"_a,
          "Generate connected set C via single/double excitations from reference V.");

    // ------------------------------------------------------------------------
    // Perturbative complement P-space generation with screening
    // P = C \ V (external determinants coupled to variational space)
    // ------------------------------------------------------------------------
    m.def("gen_perturbative_dets",
        [](DetArrayRO ref_dets,
           int n_orb,
           nb::object int_ctx_obj,
           nb::object psi_V_obj,
           const std::string& mode_str,
           double eps1) -> DetArrayOut {

            auto V = to_det_vector(ref_dets);
            const DetMap map_V = DetMap::from_ordered(
                {V.begin(), V.end()}, true
            );
            const auto mode = parse_screen_mode(mode_str);

            if (mode == detnqs::ScreenMode::None) {
                auto comp = detnqs::det_space::generate_complement(
                    std::span<const Det>(V.data(), V.size()),
                    n_orb,
                    map_V,
                    true
                );
                return from_det_vector(comp);
            }

            const IntCtx* ctx = nb::cast<const IntCtx*>(int_ctx_obj);
            if (!ctx->hb) {
                throw std::invalid_argument("Heat-Bath table not initialized");
            }

            std::vector<double> psi_vec;
            std::span<const double> psi_span;

            if (mode == detnqs::ScreenMode::Dynamic) {
                auto psi_arr = nb::cast<F64VecRO>(psi_V_obj);
                psi_vec = to_double_vector(psi_arr);
                if (psi_vec.size() != V.size()) {
                    throw std::invalid_argument("psi_v size mismatch");
                }
                psi_span = std::span<const double>(psi_vec.data(), psi_vec.size());
            }

            auto comp = detnqs::generate_complement_screened(
                std::span<const Det>(V.data(), V.size()),
                n_orb,
                map_V,
                ctx->ham,
                ctx->hb.get(),
                mode,
                psi_span,
                eps1
            );

            return from_det_vector(comp);
        },
        "ref_dets"_a,
        "n_orb"_a,
        "int_ctx"_a = nb::none(),
        "psi_v"_a = nb::none(),
        "mode"_a = "none",
        "eps1"_a = 1e-6,
        "Generate perturbative complement P = C \\ V with optional screening.\n"
        "  mode='none': combinatorial generation (no integral screening)\n"
        "  mode='static': Heat-Bath threshold eps1 filtering\n"
        "  mode='dynamic': amplitude-weighted pruning using |psi_V_i|");

    // ------------------------------------------------------------------------
    // Batch feature preparation for neural network input
    // Computes occupation indices, excitation rank k, phase, holes/parts
    // ------------------------------------------------------------------------
    m.def("prepare_det_batch",
        [](DetArrayRO dets,
           SingleDetRO ref,
           int n_orb,
           int n_alpha,
           int n_beta,
           int kmax,
           bool need_k,
           bool need_phase,
           bool need_hp,
           bool need_hp_pos) -> nb::dict {

            const std::size_t B = dets.shape(0);
            const int n_e = n_alpha + n_beta;

            detnqs::Det ref_det{ref.data()[0], ref.data()[1]};

            detnqs::det_batch::PrepareOptions opt;
            opt.kmax = kmax;
            opt.need_k = need_k || need_hp;
            opt.need_phase = need_phase;
            opt.need_hp = need_hp;
            opt.need_hp_pos = need_hp_pos;

            auto* occ_data = alloc<int32_t>(B * static_cast<std::size_t>(n_e));

            int8_t*  k_data = nullptr;
            int8_t*  phase_data = nullptr;
            int32_t* holes_data = nullptr;
            int32_t* parts_data = nullptr;
            bool*    mask_data  = nullptr;
            int32_t* holes_pos_data = nullptr;
            int32_t* parts_pos_data = nullptr;

            if (opt.need_k)     k_data     = alloc<int8_t>(B);
            if (opt.need_phase) phase_data = alloc<int8_t>(B);

            if (opt.need_hp) {
                holes_data = alloc<int32_t>(B * static_cast<std::size_t>(kmax));
                parts_data = alloc<int32_t>(B * static_cast<std::size_t>(kmax));
                mask_data  = alloc<bool>(B * static_cast<std::size_t>(kmax));

                if (opt.need_hp_pos) {
                    holes_pos_data = alloc<int32_t>(B * static_cast<std::size_t>(kmax));
                    parts_pos_data = alloc<int32_t>(B * static_cast<std::size_t>(kmax));
                }
            }

            detnqs::det_batch::prepare_det_batch(
                dets.data(),
                B,
                ref_det,
                n_orb,
                n_alpha,
                n_beta,
                opt,
                occ_data,
                k_data,
                phase_data,
                holes_data,
                parts_data,
                mask_data,
                holes_pos_data,
                parts_pos_data
            );

            nb::dict out;
            out["occ"] = I32MatOut(occ_data, {B, static_cast<std::size_t>(n_e)}, make_owner(occ_data));

            if (opt.need_k) {
                out["k"] = I8VecOut(k_data, {B}, make_owner(k_data));
            }
            if (opt.need_phase) {
                out["phase"] = I8VecOut(phase_data, {B}, make_owner(phase_data));
            }
            if (opt.need_hp) {
                out["holes"] = I32MatOut(holes_data, {B, static_cast<std::size_t>(kmax)}, make_owner(holes_data));
                out["parts"] = I32MatOut(parts_data, {B, static_cast<std::size_t>(kmax)}, make_owner(parts_data));
                out["hp_mask"] = BoolMatOut(mask_data, {B, static_cast<std::size_t>(kmax)}, make_owner(mask_data));

                if (opt.need_hp_pos) {
                    out["holes_pos"] = I32MatOut(holes_pos_data, {B, static_cast<std::size_t>(kmax)}, make_owner(holes_pos_data));
                    out["parts_pos"] = I32MatOut(parts_pos_data, {B, static_cast<std::size_t>(kmax)}, make_owner(parts_pos_data));
                }
            }

            return out;
        },
        "dets"_a,
        "ref"_a,
        "n_orb"_a,
        "n_alpha"_a,
        "n_beta"_a,
        "kmax"_a = 0,
        "need_k"_a = false,
        "need_phase"_a = false,
        "need_hp"_a = false,
        "need_hp_pos"_a = false,
        "Prepare batch features (occ, k, phase, holes/parts) for neural network forward pass.");

    // ------------------------------------------------------------------------
    // Hamiltonian matrix construction
    // ------------------------------------------------------------------------
    m.def("get_ham_diag",
          [](DetArrayRO dets, const IntCtx* ctx) -> F64VecOut {
              return from_double_vector(
                  detnqs::get_ham_diag(to_det_vector(dets), ctx->ham)
              );
          },
          "dets"_a, "int_ctx"_a,
          "Compute diagonal elements <x|H|x>.");

    m.def("get_ham_vv",
          [](DetArrayRO dets, const IntCtx* ctx, int n_orb) -> nb::dict {
              auto coo = detnqs::get_ham_vv(
                  to_det_vector(dets), ctx->ham, n_orb
              );
              return from_coo_matrix(coo);
          },
          "dets_V"_a, "int_ctx"_a, "n_orb"_a,
          "Build H_VV block in COO format (variational space only).");

    m.def("get_ham_block",
          [](DetArrayRO bra_dets,
             std::optional<DetArrayRO> ket_dets,
             const IntCtx* ctx,
             int n_orb) -> nb::dict {

              auto dets_V = to_det_vector(bra_dets);

              std::optional<std::vector<Det>> dets_P;
              if (ket_dets.has_value()) {
                  dets_P = to_det_vector(*ket_dets);
              }

              std::optional<std::span<const Det>> p_span;
              if (dets_P.has_value()) {
                  p_span = std::span<const Det>(dets_P->data(), dets_P->size());
              }

              auto blocks = detnqs::get_ham_block(
                  dets_V, p_span, ctx->ham, n_orb
              );

              nb::dict out;
              out["H_VV"] = from_coo_matrix(blocks.H_VV);
              out["H_VP"] = from_coo_matrix(blocks.H_VP);
              if (blocks.map_P.size() > 0) {
                  out["det_P"]  = from_det_vector(blocks.map_P.all_dets());
                  out["size_P"] = nb::int_(blocks.map_P.size());
              }
              return out;
          },
          "bra_dets"_a,
          "ket_dets"_a = nb::none(),
          "int_ctx"_a,
          "n_orb"_a,
          "Build H_VV and H_VP with explicit or auto-generated perturbative space P.");

    // ------------------------------------------------------------------------
    // Hamiltonian construction with static Heat-Bath screening
    // ------------------------------------------------------------------------
    m.def("get_ham_conn",
          [](DetArrayRO V,
             const IntCtx* ctx,
             int n_orb,
             bool use_heatbath,
             double eps1) -> nb::dict {

              if (use_heatbath && !ctx->hb) {
                  throw std::invalid_argument("Heat-Bath table not initialized");
              }

              auto blocks = detnqs::get_ham_conn(
                  to_det_vector(V),
                  ctx->ham,
                  n_orb,
                  ctx->hb.get(),
                  eps1,
                  use_heatbath
              );

              nb::dict out;
              out["H_VV"]   = from_coo_matrix(blocks.H_VV);
              out["H_VP"]   = from_coo_matrix(blocks.H_VP);
              out["det_P"]  = from_det_vector(blocks.map_P.all_dets());
              out["size_P"] = nb::int_(blocks.map_P.size());
              return out;
          },
          "dets_V"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "use_heatbath"_a = false,
          "eps1"_a = 1e-6,
          "Build H_VV/H_VP with static Heat-Bath screening (threshold eps1).");

    // ------------------------------------------------------------------------
    // Hamiltonian construction with dynamic amplitude-weighted screening
    // Prunes connections where |psi_V_i| * |H_ij| < eps1
    // ------------------------------------------------------------------------
    m.def("get_ham_conn_amp",
          [](DetArrayRO V,
             F64VecRO psi_v,
             const IntCtx* ctx,
             int n_orb,
             double eps1) -> nb::dict {

              if (!ctx->hb) {
                  throw std::invalid_argument("Heat-Bath table not initialized");
              }

              auto dets_V_vec = to_det_vector(V);
              auto psi_V_vec  = to_double_vector(psi_v);

              if (dets_V_vec.size() != psi_V_vec.size()) {
                  throw std::invalid_argument("size mismatch: dets_V and psi_v");
              }

              auto blocks = detnqs::get_ham_conn_amp(
                  dets_V_vec,
                  psi_V_vec,
                  ctx->ham,
                  n_orb,
                  ctx->hb.get(),
                  eps1
              );

              nb::dict out;
              out["H_VV"]   = from_coo_matrix(blocks.H_VV);
              out["H_VP"]   = from_coo_matrix(blocks.H_VP);
              out["det_P"]  = from_det_vector(blocks.map_P.all_dets());
              out["size_P"] = nb::int_(blocks.map_P.size());
              return out;
          },
          "dets_V"_a,
          "psi_v"_a,
          "int_ctx"_a,
          "n_orb"_a,
          "eps1"_a = 1e-6,
          "Build H_VV/H_VP with dynamic amplitude-weighted screening (|psi_V_i| * |H_ij| > eps1).");

    // ------------------------------------------------------------------------
    // Effective Hamiltonian via downfolding
    // H_eff = H_VV - H_VP · (H_PP - E_ref)^{-1} · H_VP^T
    // Regularization handles near-singular (H_PP - E_ref) inversions
    // ------------------------------------------------------------------------
    m.def("get_ham_eff",
          [](const nb::dict& H_VV_dict,
             const nb::dict& H_VP_dict,
             F64VecRO h_pp_diag,
             double e_ref,
             const std::string& reg_type_str,
             double epsilon,
             bool upper_only) -> nb::dict {

              const COOMatrix H_VV = to_coo_matrix(H_VV_dict);
              const COOMatrix H_VP = to_coo_matrix(H_VP_dict);

              detnqs::HeffConfig cfg{
                  .reg_type   = parse_reg_type(reg_type_str),
                  .epsilon    = epsilon,
                  .upper_only = upper_only
              };

              auto h_eff = detnqs::get_ham_eff(
                  H_VV,
                  H_VP,
                  to_double_vector(h_pp_diag),
                  e_ref,
                  cfg
              );
              return from_coo_matrix(h_eff);
          },
          "H_VV"_a,
          "H_VP"_a,
          "h_pp_diag"_a,
          "e_ref"_a,
          "reg_type"_a = "sigma",
          "epsilon"_a = 1e-12,
          "upper_only"_a = true,
          "Compute effective Hamiltonian via downfolding:\n"
          "  H_eff = H_VV - H_VP · (H_PP - E_ref)^{-1} · H_VP^T\n"
          "Regularization options: 'linear_shift' or 'sigma'.");

    // ------------------------------------------------------------------------
    // Local connections for batch SpMV operations
    // Returns CSR-like structure: offsets, connected dets, matrix elements
    // ------------------------------------------------------------------------
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
                      throw std::invalid_argument("Heat-Bath table not initialized");
                  }
                  hb = ctx->hb.get();
              }

              auto batch = detnqs::get_local_connections(
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
          "Compute local Hamiltonian connections for batch SpMV.\n"
          "Returns CSR-like structure with offsets, connected determinants, and matrix elements.");

    // ------------------------------------------------------------------------
    // Variational energy evaluation
    // E = <Psi|H|Psi> / <Psi|Psi>
    // Assumes normalized coefficients (Python side handles normalization)
    // ------------------------------------------------------------------------
    m.def("compute_variational_energy",
        [](DetArrayRO dets,
            F64VecRO coeffs,
            const IntCtx* ctx,
            int n_orb,
            bool use_heatbath,
            double eps1) -> double {

            auto basis_vec = to_det_vector(dets);

            if (static_cast<size_t>(coeffs.shape(0)) != basis_vec.size()) {
                throw std::invalid_argument("coefficient size mismatch");
            }

            std::span<const double> c_span(
                coeffs.data(), static_cast<size_t>(coeffs.shape(0))
            );

            return detnqs::compute_variational_energy(
                std::span<const Det>(basis_vec),
                c_span,
                ctx->ham,
                n_orb,
                use_heatbath ? ctx->hb.get() : nullptr,
                eps1,
                use_heatbath
            );
        },
        "dets"_a,
        "coeffs"_a,
        "int_ctx"_a,
        "n_orb"_a,
        "use_heatbath"_a = false,
        "eps1"_a = 1e-6,
        "Compute variational energy <Psi|H|Psi> (coefficients must be normalized).");

    // ------------------------------------------------------------------------
    // EN-PT2 correction: Delta E_PT2
    // Computes second-order perturbative energy correction from external space P
    // E_total = E_ref + Delta E_PT2
    // ------------------------------------------------------------------------
    m.def("compute_pt2",
        [](DetArrayRO dets,
            F64VecRO coeffs,
            const IntCtx* ctx,
            int n_orb,
            double e_ref,
            bool use_heatbath,
            double eps1) -> nb::dict {

            const auto V_vec = to_det_vector(dets);
            if (static_cast<std::size_t>(coeffs.shape(0)) != V_vec.size()) {
                throw std::invalid_argument("coefficient size mismatch");
            }

            if (use_heatbath && !ctx->hb) {
                throw std::invalid_argument("Heat-Bath table not initialized");
            }

            std::span<const double> c_span(
                coeffs.data(), static_cast<std::size_t>(coeffs.shape(0))
            );

            auto res = detnqs::compute_pt2(
                std::span<const Det>(V_vec.data(), V_vec.size()),
                c_span,
                ctx->ham,
                n_orb,
                e_ref,
                use_heatbath ? ctx->hb.get() : nullptr,
                eps1,
                use_heatbath
            );

            // Return decomposed PT2 results
            nb::dict out;
            out["e_pt2_internal"] = res.e_pt2_internal;
            out["e_pt2_external"] = res.e_pt2_external;
            out["n_ext"] = nb::int_(res.n_ext);
            return out;
        },
        "dets_V"_a,
        "coeffs_V"_a,
        "int_ctx"_a,
        "n_orb"_a,
        "e_ref"_a,
        "use_heatbath"_a = false,
        "eps1"_a = 1e-6,
        "Compute decomposed EN-PT2: internal (V-space residual) and external (P-space) contributions.\n"
        "  e_ref: electronic energy from variational optimization (normalized coeffs).\n"
        "Returns dict with 'e_pt2_internal', 'e_pt2_external', 'n_ext'.");

}

