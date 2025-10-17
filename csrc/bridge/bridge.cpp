// csrc/bridge/bridge.cpp
// LEVER nanobind bridge
// This file exposes C++ core to Python via nanobind (>= 2.8)

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <nanobind/stl/string.h>      
#include <nanobind/stl/vector.h>      
#include <nanobind/stl/pair.h>        
#include <nanobind/stl/unique_ptr.h>  

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_enum.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/ham_conn.hpp>
#include <lever/hamiltonian/ham_block.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/integral/integral_mo.hpp>
#include <lever/integral/integral_so.hpp>
#include <lever/integral/hb_table.hpp>
#include <lever/utils/types.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using lever::Det;
using lever::FCISpace;
using lever::HamEval;
using lever::HeatBathTable;
using lever::SSResult;
using lever::SCResult;
using lever::STResult;
using lever::SSSCResult;
using lever::BuildOpts;
using lever::ExcitationOpts;
using lever::BlockOpts;

using lever::u32;
using lever::u64;
using lever::f64;

// ============================== IntCtx =======================================
// Holds integrals and caches (HB table) with simple RAII semantics.
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
    HamEval           ham;
    std::unique_ptr<HeatBathTable> hb;
};

// ============================ ndarray aliases ================================
// Input: constrain to shape (N, 2), uint64, CPU, C-contiguous.
using DetArrayRO = nb::ndarray<const u64, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;

// Return types: add nb::numpy so Python receives numpy.ndarray directly.
using DetArrayOut = nb::ndarray<u64, nb::numpy, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using F64Vec      = nb::ndarray<f64, nb::numpy, nb::shape<-1>,     nb::c_contig, nb::device::cpu>;
using U32Vec      = nb::ndarray<u32, nb::numpy, nb::shape<-1>,     nb::c_contig, nb::device::cpu>;

// ============================= helpers =======================================
// Allocate owned (via capsule) (N,2) uint64 and fill from vector<Det>.
static DetArrayOut from_det_vector(const std::vector<Det>& dets) {
    const size_t N = dets.size();
    auto* data = new u64[N * 2];
    for (size_t i = 0; i < N; ++i) {
        data[2 * i + 0] = dets[i].alpha;
        data[2 * i + 1] = dets[i].beta;
    }
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<u64*>(p); });
    return DetArrayOut(data, {N, size_t(2)}, owner);
}

// Convert (N,2) uint64 ndarray -> vector<Det>.
static std::vector<Det> to_det_vector(DetArrayRO arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 2)
        throw std::invalid_argument("DetArray must be shape (N,2) uint64.");
    const size_t N = static_cast<size_t>(arr.shape(0));
    std::vector<Det> out;
    out.reserve(N);
    const u64* base = arr.data();
    for (size_t i = 0; i < N; ++i)
        out.push_back(Det{base[2 * i + 0], base[2 * i + 1]});
    return out;
}

// Allocate owned 1D float64.
static F64Vec from_double_vector(const std::vector<double>& xs) {
    const size_t N = xs.size();
    auto* data = new double[N];
    std::memcpy(data, xs.data(), N * sizeof(double));
    nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    return F64Vec(data, {N}, owner);
}

// Build dict {'row': u32[], 'col': u32[], 'val': f64[]} from vector<Conn>.
static nb::dict from_conn_vector(const std::vector<lever::Conn>& conns) {
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
    d["row"] = U32Vec(row_data, {M}, row_owner);
    d["col"] = U32Vec(col_data, {M}, col_owner);
    d["val"] = F64Vec(val_data, {M}, val_owner);
    return d;
}

// ============================ thin wrappers ==================================
static std::vector<Det> gen_fci_dets(int n_orb, int n_alpha, int n_beta) {
    FCISpace fci(n_orb, n_alpha, n_beta);
    const auto& v = fci.dets();
    return std::vector<Det>(v.begin(), v.end());
}

static lever::DetMap gen_excitations_cpp(
    const std::vector<Det>& refs,
    int n_orb,
    const IntCtx* ctx,
    bool use_hb,
    double eps1
) {
    ExcitationOpts opts;
    opts.use_heatbath = use_hb;
    opts.eps1 = eps1;

    // Heat-bath table is required if use_heatbath is true
    if (use_hb && !ctx->hb) {
        throw std::invalid_argument(
            "use_heatbath=True but HB table not prepared. Call IntCtx.hb_prepare()."
        );
    }
    
    return lever::generate_excitations(refs, n_orb, ctx->hb.get(), opts);
}

// NOTE: keep n_orb in signature for Python API parity (unused here).
static std::vector<double> get_ham_diag_cpp(const std::vector<Det>& dets,
                                            const IntCtx* ctx, int /*n_orb*/) {
    return lever::get_ham_diag(dets, ctx->ham);
}

static SSResult get_ham_conns_SS_cpp(const std::vector<Det>& S,
                                     const IntCtx* ctx,
                                     int n_orb,
                                     double thresh) {
    BuildOpts opts; opts.thresh = thresh;
    return lever::get_ham_SS(S, ctx->ham, n_orb, opts);
}

static SCResult get_ham_conns_SC_cpp(const std::vector<Det>& S,
                                     const IntCtx* ctx,
                                     int n_orb,
                                     bool use_hb, double eps1, double thresh) {
    BuildOpts opts; opts.thresh = thresh; opts.eps1 = eps1; opts.use_heatbath = use_hb;
    if (use_hb && !ctx->hb)
        throw std::invalid_argument("use_heatbath=True but HB table not prepared. Call IntCtx.hb_prepare().");
    return lever::get_ham_SC(S, ctx->ham, n_orb, ctx->hb.get(), opts);
}

static STResult get_ham_conns_ST_cpp(const std::vector<Det>& S,
                                     const IntCtx* ctx,
                                     int n_orb,
                                     bool use_hb, double eps1, double thresh) {
    BuildOpts opts; opts.thresh = thresh; opts.eps1 = eps1; opts.use_heatbath = use_hb;
    if (use_hb && !ctx->hb)
        throw std::invalid_argument("use_heatbath=True but HB table not prepared. Call IntCtx.hb_prepare().");
    return lever::get_ham_ST(S, ctx->ham, n_orb, ctx->hb.get(), opts);
}

static SSSCResult get_ham_conns_SSSC_cpp(const std::vector<Det>& S,
                                         const IntCtx* ctx,
                                         int n_orb,
                                         bool use_hb, double eps1, double thresh) {
    BuildOpts opts; opts.thresh = thresh; opts.eps1 = eps1; opts.use_heatbath = use_hb;
    if (use_hb && !ctx->hb)
        throw std::invalid_argument("use_heatbath=True but HB table not prepared. Call IntCtx.hb_prepare().");
    return lever::get_ham_SS_SC(S, ctx->ham, n_orb, ctx->hb.get(), opts);
}

static std::vector<lever::Conn> get_ham_block_cpp(
    const std::vector<Det>& bra,
    const std::vector<Det>& ket,
    const IntCtx* ctx,
    int n_orb,
    double thresh
) {
    BlockOpts opts;
    opts.thresh = thresh;
    return lever::get_ham_block(bra, ket, ctx->ham, n_orb, opts);
}

// ============================== module =======================================
NB_MODULE(_lever_cpp, m) {
    m.doc() = "LEVER C++ core bridge (nanobind)";

    nb::class_<IntCtx>(m, "IntCtx", "Integral context with heat-bath cache")
        .def(nb::init<const std::string&, int>(), "fcidump_path"_a, "num_orb"_a)
        .def("get_e_nuc", &IntCtx::e_nuc, "Get nuclear repulsion energy")
        .def("hb_prepare", &IntCtx::hb_prepare,
             "threshold"_a = 1e-8, "Build and cache heat-bath table")
        .def("hb_clear", &IntCtx::hb_clear, "Clear cached heat-bath table")
        // benign __del__: clear HB cache; nanobind handles destructor anyway
        .def("__del__", [](IntCtx& self) { self.hb_clear(); });

    // Determinant generation
    m.def("gen_fci_dets",
          [](int n_orb, int n_alpha, int n_beta) {
              return from_det_vector(gen_fci_dets(n_orb, n_alpha, n_beta));
          },
          "n_orb"_a, "n_alpha"_a, "n_beta"_a,
          "Generate full FCI space");

    m.def("gen_excited_dets",
          [](DetArrayRO ref_dets, int n_orb, const IntCtx* ctx, bool use_heatbath, double eps1) {
              auto det_map = gen_excitations_cpp(
                  to_det_vector(ref_dets), n_orb, ctx, use_heatbath, eps1
              );
              return from_det_vector(det_map.all_dets());
          },
          "ref_dets"_a, "n_orb"_a, "int_ctx"_a,
          "use_heatbath"_a = false, "eps1"_a = 1e-3,
          "Generate all unique single+double excitations, with optional heat-bath screening.");

    // Hamiltonian operations
    m.def("get_ham_diag",
          [](DetArrayRO dets, const IntCtx* ctx, int n_orbitals) {
              auto diag = get_ham_diag_cpp(to_det_vector(dets), ctx, n_orbitals);
              return from_double_vector(diag);
          },
          "dets"_a, "int_ctx"_a, "n_orbitals"_a,
          "Compute diagonal Hamiltonian elements");

    m.def("get_ham_conns_SS",
          [](DetArrayRO S, const IntCtx* ctx, int n_orbitals, double thresh) {
              auto res = get_ham_conns_SS_cpp(to_det_vector(S), ctx, n_orbitals, thresh);
              return from_conn_vector(res.coo);
          },
          "dets_S"_a, "int_ctx"_a, "n_orbitals"_a, "thresh"_a = 1e-12,
          "Build ⟨S|H|S⟩ block in COO format");

    m.def("get_ham_conns_SC",
          [](DetArrayRO S, const IntCtx* ctx, int n_orbitals,
             bool use_heatbath, double eps1, double thresh) {
              auto res = get_ham_conns_SC_cpp(to_det_vector(S), ctx, n_orbitals,
                                              use_heatbath, eps1, thresh);
              nb::dict out;
              out["conns"]  = from_conn_vector(res.coo);
              out["det_C"]  = from_det_vector(res.map_C.all_dets());
              out["size_C"] = nb::int_(res.map_C.size());
              return out;
          },
          "dets_S"_a, "int_ctx"_a, "n_orbitals"_a,
          "use_heatbath"_a = false, "eps1"_a = 1e-3, "thresh"_a = 1e-12,
          "Build ⟨S|H|C⟩ block with optional heat-bath selection");

    m.def("get_ham_conns_ST",
          [](DetArrayRO S, const IntCtx* ctx, int n_orbitals,
             bool use_heatbath, double eps1, double thresh) {
              auto res = get_ham_conns_ST_cpp(to_det_vector(S), ctx, n_orbitals,
                                              use_heatbath, eps1, thresh);
              nb::dict out;
              out["conns"]  = from_conn_vector(res.coo);
              out["det_T"]  = from_det_vector(res.map_T.all_dets());
              out["size_S"] = nb::int_(res.size_S);
              out["size_T"] = nb::int_(res.map_T.size());
              return out;
          },
          "dets_S"_a, "int_ctx"_a, "n_orbitals"_a,
          "use_heatbath"_a = false, "eps1"_a = 1e-3, "thresh"_a = 1e-12,
          "Build unified ⟨S|H|T⟩ block where T = S ∪ C");

    m.def("get_ham_conns_SSSC",
          [](DetArrayRO S, const IntCtx* ctx, int n_orbitals,
             bool use_heatbath, double eps1, double thresh) {
              auto res = get_ham_conns_SSSC_cpp(to_det_vector(S), ctx, n_orbitals,
                                                use_heatbath, eps1, thresh);
              nb::dict out;
              out["SS"]     = from_conn_vector(res.coo_SS);
              out["SC"]     = from_conn_vector(res.coo_SC);
              out["det_C"]  = from_det_vector(res.map_C.all_dets());
              out["size_C"] = nb::int_(res.map_C.size());
              return out;
          },
          "dets_S"_a, "int_ctx"_a, "n_orbitals"_a,
          "use_heatbath"_a = false, "eps1"_a = 1e-3, "thresh"_a = 1e-12,
          "Build ⟨S|H|S⟩ and ⟨S|H|C⟩ in one pass");

    m.def("get_ham_block",
          [](DetArrayRO bra, DetArrayRO ket, const IntCtx* ctx, int n_orbitals, double thresh) {
              auto coo = get_ham_block_cpp(
                  to_det_vector(bra), to_det_vector(ket), ctx, n_orbitals, thresh
              );
              return from_conn_vector(coo);
          },
          "bra_dets"_a, "ket_dets"_a, "int_ctx"_a, "n_orbitals"_a, "thresh"_a = 1e-12,
          "Compute the Hamiltonian block <bra|H|ket> in COO format.");
}