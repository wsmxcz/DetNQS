// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file build_ham.cpp
 * @brief Hamiltonian matrix construction for variational subspace iterations.
 *
 * Assembles H_VV and H_VP blocks via parallel streaming over |V> determinants.
 * Supports:
 *   - Pre-defined V and P spaces (variational mode)
 *   - Dynamic connection generation with Heat-Bath screening
 *   - Amplitude-weighted perturbative selection
 *
 * Algorithm: For each |i> in V, enumerate singles/doubles excitations,
 * compute matrix elements h_ij = <i|H|j>, and route to appropriate blocks
 * based on whether |j> belongs to V or external space P.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <detnqs/hamiltonian/build_ham.hpp>
#include <detnqs/determinant/det_ops.hpp>
#include <detnqs/determinant/det_space.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace detnqs {

namespace {

// ============================================================================
// Internal structures
// ============================================================================

/**
 * Matrix entry with determinant key for deferred indexing.
 * Used when column space is determined after collecting all connections.
 */
struct KeyedEntry {
    u32    row;
    Det    key;
    double val;
};

/**
 * Classification of generated determinant |j>.
 * 
 * VV:       In variational space V, index known → H_VV
 * VP_IDX:   In external space, index pre-known → H_VP (indexed)
 * VP_KEY:   In external space, index deferred → queue for later mapping
 * DROP:     Outside target set, discard
 */
struct Classify {
    enum Kind : std::uint8_t { VV, VP_IDX, VP_KEY, DROP } kind = DROP;
    u32 idx = 0;
    Det key{};
};

// ============================================================================
// Assembly policies
// ============================================================================

/**
 * Policy: Pre-defined variational and external spaces.
 *
 * Use case: Variational mode where both V_k and external set are fixed.
 * - Full double enumeration (no Heat-Bath)
 * - All singles accepted above threshold
 * - Routing: V → VV, external → VP_IDX, else DROP
 */
struct PolicyKnownSets {
    const DetMap& map_V;
    const DetMap& map_ext;

    template<class Visitor>
    void enumerate_doubles(
        const Det& bra,
        int n_orb,
        const HeatBathTable*,
        u32,
        Visitor&& visit
    ) const {
        det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
    }

    inline bool accept_single(u32, double, double) const {
        return true;
    }

    inline Classify classify(const Det& ket) const {
        if (auto j = map_V.get_idx(ket)) {
            return {Classify::VV, *j, {}};
        }
        if (auto j = map_ext.get_idx(ket)) {
            return {Classify::VP_IDX, *j, {}};
        }
        return {Classify::DROP, 0, {}};
    }
};

/**
 * Policy: Static Heat-Bath screening for connection generation.
 *
 * Use case: Build connected set C_k from V_k with fixed threshold eps1.
 * - Heat-Bath doubles with cutoff tau = eps1
 * - All singles accepted (no amplitude weighting)
 * - Routing: V → VV, new connections → VP_KEY (deferred indexing)
 *
 * Returns P_k = {|j> in C_k | |j> not in V_k} via deferred mapping.
 */
struct PolicyStaticHB {
    const DetMap&        map_V;
    const HeatBathTable* hb;
    double               eps1;
    bool                 use_hb;

    template<class Visitor>
    void enumerate_doubles(
        const Det& bra,
        int n_orb,
        const HeatBathTable* hb_ptr,
        u32,
        Visitor&& visit
    ) const {
        if (use_hb && hb_ptr) {
            for_each_double_hb(bra, n_orb, *hb_ptr, eps1, std::forward<Visitor>(visit));
        } else {
            det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
        }
    }

    inline bool accept_single(u32, double, double) const {
        return true;
    }

    inline Classify classify(const Det& ket) const {
        if (auto j = map_V.get_idx(ket)) {
            return {Classify::VV, *j, {}};
        }
        return {Classify::VP_KEY, 0, ket};
    }
};

/**
 * Policy: Amplitude-weighted dynamic Heat-Bath screening.
 *
 * Use case: Perturbative selection with adaptive thresholds.
 * - Doubles: Heat-Bath cutoff tau_i = eps1 / max(|psi_i|, delta)
 * - Singles: Accept if |h_ij * psi_i| >= eps1
 * - Routing: V → VV, new → VP_KEY
 *
 * Implements energy-driven selection: |<j|H|i> psi_i| ~ eps1.
 */
struct PolicyDynamicAmp {
    const DetMap&            map_V;
    const HeatBathTable*     hb;
    std::span<const double>  psi_V;
    double                   eps1;
    double                   delta = 1e-12;

    template<class Visitor>
    void enumerate_doubles(
        const Det& bra,
        int n_orb,
        const HeatBathTable* hb_ptr,
        u32 row_i,
        Visitor&& visit
    ) const {
        if (!hb_ptr) {
            det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
            return;
        }

        const double amp = std::max(std::abs(psi_V[row_i]), delta);
        const double tau = eps1 / amp;
        for_each_double_hb(bra, n_orb, *hb_ptr, tau, std::forward<Visitor>(visit));
    }

    inline bool accept_single(u32 row_i, double h, double) const {
        return (std::abs(h * psi_V[row_i]) >= eps1);
    }

    inline Classify classify(const Det& ket) const {
        if (auto j = map_V.get_idx(ket)) {
            return {Classify::VV, *j, {}};
        }
        return {Classify::VP_KEY, 0, ket};
    }
};

// ============================================================================
// Core streaming kernel
// ============================================================================

/**
 * Parallel Hamiltonian assembly with pluggable screening policy.
 *
 * Algorithm:
 *   For each |i> in V (parallel, dynamic scheduling):
 *     1. Compute diagonal h_ii → H_VV
 *     2. Enumerate singles: h_ij via deterministic iteration
 *     3. Enumerate doubles: h_ij via policy (Heat-Bath or full)
 *     4. Classify |j>: route to thread-local VV/VP buffers
 *   Merge thread-local COO matrices
 */
template<class Policy>
void stream_build(
    std::span<const Det> V,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb,
    const Policy& policy,
    std::vector<COOMatrix>& tl_vv,
    std::vector<COOMatrix>& tl_vp_idx,
    std::vector<std::vector<KeyedEntry>>& tl_vp_key
) {
    const size_t n = V.size();

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    tl_vv.resize(n_threads);
    tl_vp_idx.resize(n_threads);
    tl_vp_key.resize(n_threads);

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        const u32 row = static_cast<u32>(i);
        const Det& bra = V[i];

        auto& vv   = tl_vv[tid];
        auto& vp_i = tl_vp_idx[tid];
        auto& vp_k = tl_vp_key[tid];

        if (vv.nnz() == 0) {
            const size_t cap = est_conn_cap(n_orb);
            vv.reserve(cap);
            vp_i.reserve(cap / 2);
            vp_k.reserve(cap / 2);
        }

        // Diagonal: h_ii always in H_VV
        {
            const double h = ham.compute_diagonal(bra);
            if (std::abs(h) > MAT_ELEMENT_THRESH) {
                vv.push_back(row, row, h);
            }
        }

        // Single excitations
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            const Classify c = policy.classify(ket);
            if (c.kind == Classify::DROP) return;

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= MAT_ELEMENT_THRESH) return;
            if (!policy.accept_single(row, h, 0.0)) return;

            switch (c.kind) {
                case Classify::VV:
                    vv.push_back(row, c.idx, h);
                    break;
                case Classify::VP_IDX:
                    vp_i.push_back(row, c.idx, h);
                    break;
                case Classify::VP_KEY:
                    vp_k.push_back({row, ket, h});
                    break;
                default:
                    break;
            }
        });

        // Double excitations (policy-controlled enumeration)
        policy.enumerate_doubles(
            bra, n_orb, hb, row,
            [&](const Det& ket) {
                const Classify c = policy.classify(ket);
                if (c.kind == Classify::DROP) return;

                const double h = ham.compute_elem(bra, ket);
                if (std::abs(h) <= MAT_ELEMENT_THRESH) return;

                switch (c.kind) {
                    case Classify::VV:
                        vv.push_back(row, c.idx, h);
                        break;
                    case Classify::VP_IDX:
                        vp_i.push_back(row, c.idx, h);
                        break;
                    case Classify::VP_KEY:
                        vp_k.push_back({row, ket, h});
                        break;
                    default:
                        break;
                }
            }
        );
    }
}

/**
 * Merge thread-local COO buffers into unified matrix.
 * Concatenates and performs sort-merge deduplication.
 */
inline void merge_thread_local(
    std::vector<COOMatrix>& tl,
    COOMatrix& out
) {
    size_t total = 0;
    for (auto& m : tl) total += m.nnz();
    out.reserve(total);

    for (auto& m : tl) {
        out.rows.insert(out.rows.end(), m.rows.begin(), m.rows.end());
        out.cols.insert(out.cols.end(), m.cols.begin(), m.cols.end());
        out.vals.insert(out.vals.end(), m.vals.begin(), m.vals.end());
    }

    sort_and_merge_coo(out);
}

/**
 * Finalize H_VP with deferred perturbative space indexing.
 *
 * Algorithm:
 *   1. Collect all unique determinants from keyed entries
 *   2. Sort lexicographically and build DetMap for P space
 *   3. Remap keyed entries to column indices
 *   4. Sort and merge final COO matrix
 */
inline void finalize_vp_deferred(
    std::vector<std::vector<KeyedEntry>>& tl_vp_key,
    COOMatrix& coo_VP,
    DetMap& map_P
) {
    std::vector<Det> keys;
    size_t total = 0;
    for (auto& v : tl_vp_key) total += v.size();
    keys.reserve(total);

    for (auto& v : tl_vp_key) {
        for (auto& x : v) {
            keys.push_back(x.key);
        }
    }

    if (!keys.empty()) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }

    map_P = DetMap::from_list(std::move(keys));

    coo_VP.reserve(total);
    for (auto& v : tl_vp_key) {
        for (auto& x : v) {
            auto j = map_P.get_idx(x.key);
            if (j) {
                coo_VP.push_back(x.row, *j, x.val);
            }
        }
    }

    sort_and_merge_coo(coo_VP);
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

std::vector<double> get_ham_diag(
    std::span<const Det> dets,
    const HamEval& ham
) {
    std::vector<double> diag(dets.size());

#pragma omp parallel for if(dets.size() > 256)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dets.size()); ++i) {
        diag[i] = ham.compute_diagonal(dets[static_cast<size_t>(i)]);
    }

    return diag;
}

COOMatrix get_ham_vv(
    std::span<const Det> dets_V,
    const HamEval& ham,
    int n_orb
) {
    COOMatrix out;
    if (dets_V.empty()) return out;

    const auto map_V = DetMap::from_ordered({dets_V.begin(), dets_V.end()}, true);
    PolicyKnownSets policy{map_V, DetMap{}};

    std::vector<COOMatrix> tl_vv, tl_vp_idx;
    std::vector<std::vector<KeyedEntry>> tl_vp_key;

    stream_build(dets_V, ham, n_orb, nullptr, policy, tl_vv, tl_vp_idx, tl_vp_key);

    merge_thread_local(tl_vv, out);
    out.n_rows = out.n_cols = static_cast<u32>(dets_V.size());

    return out;
}

HamBlocks get_ham_block(
    std::span<const Det> dets_V,
    std::optional<std::span<const Det>> dets_ext,
    const HamEval& ham,
    int n_orb
) {
    HamBlocks out;
    if (dets_V.empty()) return out;

    const auto map_V = DetMap::from_ordered({dets_V.begin(), dets_V.end()}, true);

    DetMap map_ext;
    const bool has_ext = dets_ext.has_value() && !dets_ext->empty();

    if (has_ext) {
        map_ext = DetMap::from_ordered({dets_ext->begin(), dets_ext->end()}, true);
    }

    PolicyKnownSets policy{map_V, map_ext};

    std::vector<COOMatrix> tl_vv, tl_vp_idx;
    std::vector<std::vector<KeyedEntry>> tl_vp_key;

    stream_build(dets_V, ham, n_orb, nullptr, policy, tl_vv, tl_vp_idx, tl_vp_key);

    merge_thread_local(tl_vv, out.H_VV);
    out.H_VV.n_rows = out.H_VV.n_cols = static_cast<u32>(dets_V.size());

    if (has_ext) {
        merge_thread_local(tl_vp_idx, out.H_VP);
        out.H_VP.n_rows = static_cast<u32>(dets_V.size());
        out.H_VP.n_cols = static_cast<u32>(dets_ext->size());
        out.map_P = std::move(map_ext);
    }

    return out;
}

HamBlocks get_ham_conn(
    std::span<const Det> dets_V,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "get_ham_conn: Heat-Bath enabled but table is nullptr"
        );
    }

    HamBlocks out;
    if (dets_V.empty()) return out;

    const auto map_V = DetMap::from_ordered({dets_V.begin(), dets_V.end()}, true);
    PolicyStaticHB policy{map_V, hb_table, eps1, use_heatbath};

    std::vector<COOMatrix> tl_vv, tl_vp_idx;
    std::vector<std::vector<KeyedEntry>> tl_vp_key;

    stream_build(dets_V, ham, n_orb, hb_table, policy, tl_vv, tl_vp_idx, tl_vp_key);

    merge_thread_local(tl_vv, out.H_VV);
    out.H_VV.n_rows = out.H_VV.n_cols = static_cast<u32>(dets_V.size());

    finalize_vp_deferred(tl_vp_key, out.H_VP, out.map_P);
    out.H_VP.n_rows = static_cast<u32>(dets_V.size());
    out.H_VP.n_cols = static_cast<u32>(out.map_P.size());

    return out;
}

HamBlocks get_ham_conn_amp(
    std::span<const Det> dets_V,
    std::span<const double> psi_V,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1
) {
    if (!hb_table) {
        throw std::invalid_argument(
            "get_ham_conn_amp: Heat-Bath table is required"
        );
    }
    if (psi_V.size() != dets_V.size()) {
        throw std::invalid_argument(
            "get_ham_conn_amp: psi_V size must match |V|"
        );
    }

    HamBlocks out;
    if (dets_V.empty()) return out;

    const auto map_V = DetMap::from_ordered({dets_V.begin(), dets_V.end()}, true);
    PolicyDynamicAmp policy{map_V, hb_table, psi_V, eps1};

    std::vector<COOMatrix> tl_vv, tl_vp_idx;
    std::vector<std::vector<KeyedEntry>> tl_vp_key;

    stream_build(dets_V, ham, n_orb, hb_table, policy, tl_vv, tl_vp_idx, tl_vp_key);

    merge_thread_local(tl_vv, out.H_VV);
    out.H_VV.n_rows = out.H_VV.n_cols = static_cast<u32>(dets_V.size());

    finalize_vp_deferred(tl_vp_key, out.H_VP, out.map_P);
    out.H_VP.n_rows = static_cast<u32>(dets_V.size());
    out.H_VP.n_cols = static_cast<u32>(out.map_P.size());

    return out;
}

} // namespace detnqs
