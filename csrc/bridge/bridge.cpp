// csrc/bridge/bridge.cpp
// LEVER nanobind bridge - exposes C++ core to Python

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_enum.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/build_ham.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/integral/hb_table.hpp>
#include <lever/integral/integral_mo.hpp>
#include <lever/integral/integral_so.hpp>
#include <lever/utils/types.hpp>

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
using lever::u32;
using lever::u64;
using lever::f64;

// ============================================================================
// IntCtx: Integral context with optional heat-bath cache
// ============================================================================
struct IntCtx {
    explicit IntCtx(const std::string& fcidump_path, int num_orb)
        : mo(num_orb), so(mo), ham(so) {
        mo.load_from_fcidump(fcidump_path);
    }

    double e_nuc() const noexcept { return so.get_e_nuc(); }

    void hb_prepare(double threshold) {
        lever::HBBuildOptions opt;
        opt.threshold = threshold;
        hb = std::make_unique<HeatBathTable>(so, opt);
        hb->build();
    }

    void hb_clear() { hb.reset(); }

    lever::IntegralMO mo;
    lever::IntegralSO so;
    HamEval ham;
    std::unique_ptr<HeatBathTable> hb;
};

// ============================================================================
// ndarray type aliases
// ============================================================================
// Read-only inputs
using DetArrayRO = nb::ndarray<const u64, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using F64VecRO   = nb::ndarray<const f64, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

// Output arrays (numpy-owned)
using DetArrayOut = nb::ndarray<u64, nb::numpy, nb::shape<-1, 2>>;
using F64VecOut   = nb::ndarray<f64, nb::numpy, nb::shape<-1>>;
using U32VecOut   = nb::ndarray<u32, nb::numpy, nb::shape<-1>>;

// ============================================================================
// Conversion helpers
// ============================================================================
inline std::vector<Det> to_det_vector(DetArrayRO arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 2)
        throw std::invalid_argument("Expected shape (N, 2) for determinant array");
    
    const size_t N = arr.shape(0);
    std::vector<Det> out;
    out.reserve(N);
    
    const u64* data = arr.data();
    for (size_t i = 0; i < N; ++i)
        out.push_back(Det{data[2*i], data[2*i + 1]});
    
    return out;
}

inline std::vector<double> to_double_vector(F64VecRO arr) {
    if (arr.ndim() != 1)
        throw std::invalid_argument("Expected 1D array for amplitudes");
    
    const size_t N = arr.shape(0);
    return std::vector<double>(arr.data(), arr.data() + N);
}

inline DetArrayOut from_det_vector(const std::vector<Det>& dets) {
    const size_t N = dets.size();
    auto* data = new u64[N * 2];
    
    for (size_t i = 0; i < N; ++i) {
        data[2*i]     = dets[i].alpha;
        data[2*i + 1] = dets[i].beta;
    }
    
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<u64*>(p); });
    return DetArrayOut(data, {N, 2}, owner);
}

inline F64VecOut from_double_vector(const std::vector<double>& xs) {
    const size_t N = xs.size();
    auto* data = new double[N];
    std::memcpy(data, xs.data(), N * sizeof(double));
    
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return F64VecOut(data, {N}, owner);
}

inline nb::dict from_conn_vector(const std::vector<lever::Conn>& conns) {
    const size_t M = conns.size();
    
    auto* row_data = new u32[M];
    auto* col_data = new u32[M];
    auto* val_data = new double[M];
    
    for (size_t i = 0; i < M; ++i) {
        row_data[i] = conns[i].row;
        col_data[i] = conns[i].col;
        val_data[i] = conns[i].val;
    }
    
    nb::capsule row_owner(row_data, [](void* p) noexcept { delete[] static_cast<u32*>(p); });
    nb::capsule col_owner(col_data, [](void* p) noexcept { delete[] static_cast<u32*>(p); });
    nb::capsule val_owner(val_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    
    nb::dict d;
    d["row"] = U32VecOut(row_data, {M}, row_owner);
    d["col"] = U32VecOut(col_data, {M}, col_owner);
    d["val"] = F64VecOut(val_data, {M}, val_owner);
    return d;
}

// ============================================================================
// Module definition
// ============================================================================
NB_MODULE(_lever_cpp, m) {
    m.doc() = "LEVER C++ core bridge";

    // IntCtx class
    nb::class_<IntCtx>(m, "IntCtx")
        .def(nb::init<const std::string&, int>(), "fcidump_path"_a, "num_orb"_a)
        .def("get_e_nuc", &IntCtx::e_nuc)
        .def("hb_prepare", &IntCtx::hb_prepare, "threshold"_a = 1e-8)
        .def("hb_clear", &IntCtx::hb_clear);

    // Determinant generation
    m.def("gen_fci_dets",
          [](int n_orb, int n_alpha, int n_beta) {
              FCISpace fci(n_orb, n_alpha, n_beta);
              const auto& dets_span = fci.dets();
              return from_det_vector({dets_span.begin(), dets_span.end()});
          },
          "n_orb"_a, "n_alpha"_a, "n_beta"_a);

    m.def("gen_excited_dets",
          [](DetArrayRO ref_dets, int n_orb) {
              auto dets = lever::det_space::generate_connected(
                  to_det_vector(ref_dets), n_orb
              );
              return from_det_vector(lever::det_space::canonicalize(std::move(dets)));
          },
          "ref_dets"_a, "n_orb"_a);

    // Hamiltonian operations
    m.def("get_ham_diag",
          [](DetArrayRO dets, const IntCtx* ctx) {
              return from_double_vector(
                  lever::get_ham_diag(to_det_vector(dets), ctx->ham)
              );
          },
          "dets"_a, "int_ctx"_a);

    m.def("get_ham_block",
          [](DetArrayRO bra_dets, 
             std::optional<DetArrayRO> ket_dets,
             const IntCtx* ctx, 
             int n_orb, 
             double thresh) {
              
              auto dets_S = to_det_vector(bra_dets);
              
              std::optional<std::vector<Det>> dets_C;
              if (ket_dets.has_value())
                  dets_C = to_det_vector(*ket_dets);
              
              std::optional<std::span<const Det>> c_span;
              if (dets_C.has_value())
                  c_span = *dets_C;
              
              auto res = lever::get_ham_block(dets_S, c_span, ctx->ham, n_orb, thresh);
              
              nb::dict out;
              out["SS"] = from_conn_vector(res.coo_SS);
              out["SC"] = from_conn_vector(res.coo_SC);
              return out;
          },
          "bra_dets"_a, 
          "ket_dets"_a = nb::none(),
          "int_ctx"_a, 
          "n_orbitals"_a, 
          "thresh"_a = 1e-15);

    m.def("get_ham_conn",
          [](DetArrayRO S, 
             const IntCtx* ctx, 
             int n_orb,
             bool use_heatbath, 
             double eps1, 
             double thresh) {
              
              if (use_heatbath && !ctx->hb)
                  throw std::invalid_argument(
                      "Heat-bath table not prepared. Call IntCtx.hb_prepare() first."
                  );
              
              auto res = lever::get_ham_conn(
                  to_det_vector(S), ctx->ham, n_orb,
                  ctx->hb.get(), eps1, use_heatbath, thresh
              );
              
              nb::dict out;
              out["SS"]     = from_conn_vector(res.coo_SS);
              out["SC"]     = from_conn_vector(res.coo_SC);
              out["det_C"]  = from_det_vector(res.map_C.all_dets());
              out["size_C"] = nb::int_(res.map_C.size());
              return out;
          },
          "dets_S"_a, 
          "int_ctx"_a, 
          "n_orbitals"_a,
          "use_heatbath"_a = false, 
          "eps1"_a = 1e-6, 
          "thresh"_a = 1e-15);

    m.def("get_ham_conn_amp",
          [](DetArrayRO S, 
             F64VecRO psi_S, 
             const IntCtx* ctx, 
             int n_orb,
             double eps1, 
             double thresh) {
              
              if (!ctx->hb)
                  throw std::invalid_argument(
                      "Heat-bath table required. Call IntCtx.hb_prepare() first."
                  );
              
              auto dets_S_vec = to_det_vector(S);
              auto psi_S_vec = to_double_vector(psi_S);
              
              if (dets_S_vec.size() != psi_S_vec.size())
                  throw std::invalid_argument("dets_S and psi_S must have same length");
              
              auto res = lever::get_ham_conn_amp(
                  dets_S_vec, psi_S_vec, ctx->ham, n_orb,
                  ctx->hb.get(), eps1, thresh
              );
              
              nb::dict out;
              out["SS"]     = from_conn_vector(res.coo_SS);
              out["SC"]     = from_conn_vector(res.coo_SC);
              out["det_C"]  = from_det_vector(res.map_C.all_dets());
              out["size_C"] = nb::int_(res.map_C.size());
              return out;
          },
          "dets_S"_a, 
          "psi_S"_a, 
          "int_ctx"_a, 
          "n_orbitals"_a,
          "eps1"_a = 1e-6, 
          "thresh"_a = 1e-15);
}
