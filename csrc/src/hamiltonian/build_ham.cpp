// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

#include <lever/hamiltonian/build_ham.hpp>

#include <lever/determinant/det_ops.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lever {

// ============================================================================
// Internal helpers (deterministic merges, HB doubles, occupancy checks)
// ============================================================================

namespace {

struct KeyedConn {
    u32  row;
    Det  key;
    double val;
};

inline void sort_and_merge_coo(std::vector<Conn>& xs) {
    if (xs.empty()) return;
    std::sort(xs.begin(), xs.end(), [](const Conn& a, const Conn& b) {
        return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
    });
    std::vector<Conn> out;
    out.reserve(xs.size());
    for (size_t i = 0; i < xs.size();) {
        u32 r = xs[i].row, c = xs[i].col;
        double v = xs[i].val;
        for (++i; i < xs.size() && xs[i].row == r && xs[i].col == c; ++i) v += xs[i].val;
        if (v != 0.0) out.emplace_back(r, c, v);
    }
    xs.swap(out);
}

inline std::vector<int> get_occupied_sos(const Det& d) {
    std::vector<int> occ_so;
    occ_so.reserve(popcount(d.alpha) + popcount(d.beta));
    for (int mo : extract_bits(d.alpha)) occ_so.push_back(so_from_mo(mo, 0));
    for (int mo : extract_bits(d.beta))  occ_so.push_back(so_from_mo(mo, 1));
    return occ_so;
}

inline bool is_so_occupied(const Det& d, int so_idx, int n_orb) {
    const int mo = mo_from_so(so_idx);
    if (mo < 0 || mo >= n_orb) return true; // treat OOB as "occupied" to skip safely
    const u64 bit = (1ULL << mo);
    return (spin_from_so(so_idx) == 0) ? ((d.alpha & bit) != 0) : ((d.beta & bit) != 0);
}

inline Det apply_double_exc_so(const Det& ket, int i_so, int j_so, int a_so, int b_so) {
    Det out = ket;
    auto flip = [&](int so_idx, bool set_bit) {
        const int mo = mo_from_so(so_idx);
        const int sp = spin_from_so(so_idx);
        const u64 bit = (1ULL << mo);
        u64& mask = (sp == 0) ? out.alpha : out.beta;
        if (set_bit) mask |= bit; else mask &= ~bit;
    };
    flip(i_so, false); flip(j_so, false);
    flip(a_so, true);  flip(b_so, true);
    return out;
}

/** Heat-bath double excitation enumerator (deterministic). */
template<class Visitor>
inline void enumerate_doubles_hb(
    const Det& bra,
    int n_orb,
    const HeatBathTable& hb,
    double cutoff,
    Visitor&& visit
) {
    auto occ_so = get_occupied_sos(bra);
    if (occ_so.size() < 2) return;

    for (size_t p = 1; p < occ_so.size(); ++p) {
        for (size_t q = 0; q < p; ++q) {
            const int i_so = occ_so[p], j_so = occ_so[q];
            auto row = hb.row_view(i_so, j_so).with_cutoff(cutoff);
            for (size_t k = 0; k < row.len; ++k) {
                const int a_so = row.a[k], b_so = row.b[k];
                if (is_so_occupied(bra, a_so, n_orb) || is_so_occupied(bra, b_so, n_orb)) continue;
                visit(apply_double_exc_so(bra, i_so, j_so, a_so, b_so));
            }
        }
    }
}

/** COO reserve heuristic (keeps allocations stable across threads). */
inline size_t est_conn_per_row(int n_orb) {
    // ~ 1 diag + O(n_orb) singles + some doubles; keep small to avoid over-reserving
    return static_cast<size_t>(1 + 2*n_orb + (n_orb*n_orb)/10);
}

// Classification result for a generated ket
struct Classify {
    enum Kind : uint8_t { SS, SC_IDX, SC_KEY, DROP } kind = DROP;
    u32 idx = 0; // valid if SS or SC_IDX
    Det key{};   // valid if SC_KEY
};

// -----------------------------------------------------------------------------
// Policies
// -----------------------------------------------------------------------------

struct PolicyKnownSets {
    const DetMap& map_S;
    const DetMap& map_C;

    // Doubles: full enumeration to preserve variational completeness
    template<class Visitor>
    void enumerate_doubles(const Det& bra, int n_orb, const HeatBathTable*, u32 /*row_i*/, Visitor&& visit) const {
        det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
    }

    // Singles post-filter: accept all (global thresh is applied outside)
    inline bool accept_single(u32 /*row_i*/, double /*h*/, double /*eps1*/) const { return true; }

    inline Classify classify(const Det& ket) const {
        if (auto jS = map_S.get_idx(ket)) return {Classify::SS, *jS, {}};
        if (auto jC = map_C.get_idx(ket)) return {Classify::SC_IDX, *jC, {}};
        return {Classify::DROP, 0, {}};
    }
};

struct PolicyStaticHB {
    const DetMap&       map_S;
    const HeatBathTable* hb;
    double              eps1;
    bool                use_hb;

    template<class Visitor>
    void enumerate_doubles(const Det& bra, int n_orb, const HeatBathTable* hb_ptr, u32 /*row_i*/, Visitor&& visit) const {
        if (use_hb && hb_ptr) {
            enumerate_doubles_hb(bra, n_orb, *hb_ptr, eps1, std::forward<Visitor>(visit));
        } else {
            det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
        }
    }

    inline bool accept_single(u32 /*row_i*/, double /*h*/, double /*eps1_unused*/) const { return true; }

    inline Classify classify(const Det& ket) const {
        if (auto jS = map_S.get_idx(ket)) return {Classify::SS, *jS, {}};
        return {Classify::SC_KEY, 0, ket}; // any non-S goes to C
    }
};

struct PolicyDynamicAmp {
    const DetMap&        map_S;
    const HeatBathTable* hb;
    std::span<const double> psi_S;
    double               eps1;
    double               delta = 1e-12; // avoid div-by-zero

    template<class Visitor>
    void enumerate_doubles(const Det& bra, int n_orb, const HeatBathTable* hb_ptr, u32 row_i, Visitor&& visit) const {
        if (!hb_ptr) {
            // Fallback to full enumeration if table absent (shouldn't happen in normal use)
            det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
            return;
        }
        const double tau = eps1 / std::max(std::abs(psi_S[row_i]), delta);
        enumerate_doubles_hb(bra, n_orb, *hb_ptr, tau, std::forward<Visitor>(visit));
    }

    inline bool accept_single(u32 row_i, double h, double /*eps1_param*/) const {
        return std::abs(h * psi_S[row_i]) >= eps1;
    }

    inline Classify classify(const Det& ket) const {
        if (auto jS = map_S.get_idx(ket)) return {Classify::SS, *jS, {}};
        return {Classify::SC_KEY, 0, ket};
    }
};

// -----------------------------------------------------------------------------
// Single streaming kernel
// -----------------------------------------------------------------------------

template<class KetPolicy>
void stream_build(
    std::span<const Det> S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb,
    const KetPolicy& policy,
    double thresh,
    // thread-sharded outputs:
    std::vector<std::vector<Conn>>& tl_ss,
    std::vector<std::vector<Conn>>& tl_sc_idx,
    std::vector<std::vector<KeyedConn>>& tl_sc_key
) {
    const size_t n = S.size();
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    tl_ss.resize(n_threads);
    tl_sc_idx.resize(n_threads);
    tl_sc_key.resize(n_threads);

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const u32 row = static_cast<u32>(i);
        const Det& bra = S[i];

        auto& ss   = tl_ss[tid];
        auto& sc_i = tl_sc_idx[tid];
        auto& sc_k = tl_sc_key[tid];

        if (ss.capacity() == 0)   ss.reserve(est_conn_per_row(n_orb));
        if (sc_i.capacity() == 0) sc_i.reserve(est_conn_per_row(n_orb) / 2);
        if (sc_k.capacity() == 0) sc_k.reserve(est_conn_per_row(n_orb) / 2);

        // --- Diagonal (always in SS) ---
        {
            const double h = ham.compute_diagonal(bra);
            if (std::abs(h) > thresh) {
                // col index equals row for SS (S is ordered as given)
                ss.emplace_back(row, row, h);
            }
        }

        // --- Singles (generate, evaluate, classify, policy post-filter) ---
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            const Classify c = policy.classify(ket);
            if (c.kind == Classify::DROP) return;

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= thresh) return;
            if (!policy.accept_single(row, h, 0.0)) return; // static HB ignores; dynamic uses |H*psi|

            switch (c.kind) {
                case Classify::SS:      ss.emplace_back(row, c.idx, h); break;
                case Classify::SC_IDX:  sc_i.emplace_back(row, c.idx, h); break;
                case Classify::SC_KEY:  sc_k.push_back({row, ket, h}); break;
                default: break;
            }
        });

        // --- Doubles (policy-driven generator; HB prune happens at generation) ---
        policy.enumerate_doubles(bra, n_orb, hb, row, [&](const Det& ket) {
            const Classify c = policy.classify(ket);
            if (c.kind == Classify::DROP) return;

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= thresh) return;

            switch (c.kind) {
                case Classify::SS:      ss.emplace_back(row, c.idx, h); break;
                case Classify::SC_IDX:  sc_i.emplace_back(row, c.idx, h); break;
                case Classify::SC_KEY:  sc_k.push_back({row, ket, h}); break;
                default: break;
            }
        });
    }
}

/** Merge thread-local buckets and canonicalize. */
inline void merge_and_finalize(
    std::vector<std::vector<Conn>>& tl, std::vector<Conn>& out
) {
    size_t total = 0;
    for (auto& v : tl) total += v.size();
    out.reserve(total);
    for (auto& v : tl) {
        out.insert(out.end(),
                   std::make_move_iterator(v.begin()),
                   std::make_move_iterator(v.end()));
    }
    sort_and_merge_coo(out);
}

/** Finalize SC with deferred indexing. */
inline void finalize_sc_deferred(
    std::vector<std::vector<KeyedConn>>& tl_sc_key,
    std::vector<Conn>& coo_SC,
    DetMap& map_C
) {
    // Gather all keys
    std::vector<Det> keys;
    size_t total = 0;
    for (auto& v : tl_sc_key) total += v.size();
    keys.reserve(total);
    for (auto& v : tl_sc_key) {
        for (auto& x : v) keys.push_back(x.key);
    }
    // Canonical deterministic order (lexicographic on Det)
    if (!keys.empty()) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }
    map_C = DetMap::from_list(std::move(keys));

    // Remap keyed entries to column indices
    std::vector<Conn> tmp;
    tmp.reserve(total);
    for (auto& v : tl_sc_key) {
        for (auto& x : v) {
            auto j = map_C.get_idx(x.key);
            // Should always succeed; guard just in case
            if (j) tmp.emplace_back(x.row, *j, x.val);
        }
    }
    coo_SC.swap(tmp);
    sort_and_merge_coo(coo_SC);
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

std::vector<double> get_ham_diag(std::span<const Det> dets, const HamEval& ham) {
    std::vector<double> diag(dets.size());
    // Parallelize only if the workload is large enough to offset thread creation overhead.
#pragma omp parallel for if(dets.size() > 256)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dets.size()); ++i) {
        diag[i] = ham.compute_diagonal(dets[i]);
    }
    return diag;
}

SSSCResult get_ham_block(
    std::span<const Det> dets_S,
    std::optional<std::span<const Det>> dets_C,
    const HamEval& ham,
    int n_orb,
    double thresh
) {
    SSSCResult out;
    if (dets_S.empty()) return out;

    // Build maps (S must be ordered as given; C too if provided)
    const auto map_S = DetMap::from_ordered({dets_S.begin(), dets_S.end()}, /*verify_unique=*/true);

    DetMap map_C_known;
    bool has_C = dets_C.has_value() && !dets_C->empty();
    if (has_C) {
        map_C_known = DetMap::from_ordered({dets_C->begin(), dets_C->end()}, /*verify_unique=*/true);
    }

    // Policy: Known sets
    PolicyKnownSets policy{map_S, map_C_known};

    // Thread-local sinks
    std::vector<std::vector<Conn>> tl_ss, tl_sc_idx;
    std::vector<std::vector<KeyedConn>> tl_sc_key; // unused in this policy

    stream_build(dets_S, ham, n_orb, /*hb=*/nullptr, policy, thresh,
                 tl_ss, tl_sc_idx, tl_sc_key);

    // Finalize
    merge_and_finalize(tl_ss, out.coo_SS);

    if (has_C) {
        merge_and_finalize(tl_sc_idx, out.coo_SC);
        out.map_C = std::move(map_C_known);
    } else {
        // No C provided → empty SC, empty map_C
        out.coo_SC.clear();
        out.map_C = DetMap{};
    }
    return out;
}

SSSCResult get_ham_conn(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath,
    double thresh
) {
    if (!use_heatbath && !hb_table) {
        // valid: full enumeration fallback
    } else if (use_heatbath && !hb_table) {
        throw std::invalid_argument("get_ham_conn: Heat-bath enabled but hb_table is nullptr");
    }

    SSSCResult out;
    if (dets_S.empty()) return out;

    const auto map_S = DetMap::from_ordered({dets_S.begin(), dets_S.end()}, /*verify_unique=*/true);

    PolicyStaticHB policy{map_S, hb_table, eps1, use_heatbath};

    std::vector<std::vector<Conn>> tl_ss, tl_sc_idx; // sc_idx unused here
    std::vector<std::vector<KeyedConn>> tl_sc_key;

    stream_build(dets_S, ham, n_orb, hb_table, policy, thresh,
                 tl_ss, tl_sc_idx, tl_sc_key);

    merge_and_finalize(tl_ss, out.coo_SS);
    finalize_sc_deferred(tl_sc_key, out.coo_SC, out.map_C);
    return out;
}

SSSCResult get_ham_conn_amp(
    std::span<const Det> dets_S,
    std::span<const double> psi_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    double thresh
) {
    if (!hb_table) {
        throw std::invalid_argument("get_ham_conn_amp: Heat-bath table is required");
    }
    if (psi_S.size() != dets_S.size()) {
        throw std::invalid_argument("get_ham_conn_amp: psi_S size must match |S|");
    }

    SSSCResult out;
    if (dets_S.empty()) return out;

    const auto map_S = DetMap::from_ordered({dets_S.begin(), dets_S.end()}, /*verify_unique=*/true);

    PolicyDynamicAmp policy{map_S, hb_table, psi_S, eps1};

    std::vector<std::vector<Conn>> tl_ss, tl_sc_idx; // sc_idx unused here
    std::vector<std::vector<KeyedConn>> tl_sc_key;

    stream_build(dets_S, ham, n_orb, hb_table, policy, thresh,
                 tl_ss, tl_sc_idx, tl_sc_key);

    merge_and_finalize(tl_ss, out.coo_SS);
    finalize_sc_deferred(tl_sc_key, out.coo_SC, out.map_C);
    return out;
}

} // namespace lever