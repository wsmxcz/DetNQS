// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file build_ham.cpp
 * @brief Hamiltonian matrix assembly for ASCI variational subspace.
 *
 * Implements parallel streaming construction of H_SS and H_SC blocks
 * with optional Heat-bath screening and amplitude-based selection.
 *
 * Core algorithm: For each |S⟩ determinant, enumerate connections
 * via singles/doubles, classify into S/C spaces, and accumulate
 * matrix elements as COO triplets.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <lever/hamiltonian/build_ham.hpp>
#include <lever/determinant/det_ops.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lever {

namespace {

// ============================================================================
// Internal data structures
// ============================================================================

/**
 * Matrix entry with determinant key (for deferred C-space indexing).
 * Used when column index is unknown until all C determinants collected.
 */
struct KeyedEntry {
    u32    row;
    Det    key;
    double val;
};

/**
 * Classification of generated ket determinant.
 * 
 * SS:      In S-space, index known → H_SS
 * SC_IDX:  In C-space, index known → H_SC (pre-indexed)
 * SC_KEY:  In C-space, index TBD   → deferred queue
 * DROP:    Outside S∪C, discard
 */
struct Classify {
    enum Kind : uint8_t { SS, SC_IDX, SC_KEY, DROP } kind = DROP;
    u32 idx = 0;
    Det key{};
};

// ============================================================================
// Determinant manipulation utilities
// ============================================================================

/**
 * Extract occupied spin-orbitals from determinant.
 * Returns ordered list [α₀, α₁, ..., β₀, β₁, ...].
 */
inline std::vector<int> get_occupied_sos(const Det& d) {
    std::vector<int> occ_so;
    occ_so.reserve(popcount(d.alpha) + popcount(d.beta));
    
    for (int mo : extract_bits(d.alpha)) {
        occ_so.push_back(so_from_mo(mo, 0));
    }
    for (int mo : extract_bits(d.beta)) {
        occ_so.push_back(so_from_mo(mo, 1));
    }
    
    return occ_so;
}

/**
 * Check if spin-orbital is occupied in determinant.
 * Returns true if outside orbital range (boundary guard).
 */
inline bool is_so_occupied(const Det& d, int so_idx, int n_orb) {
    const int mo = mo_from_so(so_idx);
    if (mo < 0 || mo >= n_orb) return true;
    
    const u64 bit = (1ULL << mo);
    return (spin_from_so(so_idx) == 0) 
        ? ((d.alpha & bit) != 0) 
        : ((d.beta & bit) != 0);
}

/**
 * Apply double excitation i,j → a,b in spin-orbital basis.
 * Returns new determinant without sign tracking (sign computed separately).
 */
inline Det apply_double_exc_so(
    const Det& ket, 
    int i_so, int j_so, 
    int a_so, int b_so
) {
    Det out = ket;
    
    auto flip = [&](int so_idx, bool set_bit) {
        const int mo = mo_from_so(so_idx);
        const int sp = spin_from_so(so_idx);
        const u64 bit = (1ULL << mo);
        u64& mask = (sp == 0) ? out.alpha : out.beta;
        
        if (set_bit) mask |= bit;
        else         mask &= ~bit;
    };
    
    flip(i_so, false); flip(j_so, false);
    flip(a_so, true);  flip(b_so, true);
    
    return out;
}

/**
 * Heat-bath double excitation generator with integral cutoff.
 *
 * Algorithm: For each occupied pair (i,j), iterate over integral table
 * rows (i,j;a,b) satisfying |⟨ij||ab⟩| ≥ cutoff. Skip if a/b occupied.
 *
 * Deterministic ordering: Outer loop p > q ensures canonical (i > j).
 *
 * @param bra     Source determinant
 * @param n_orb   Number of spatial orbitals
 * @param hb      Pre-sorted integral table
 * @param cutoff  Minimum |⟨ij||ab⟩| threshold
 * @param visit   Callback(Det ket) for each generated excitation
 */
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
            const int i_so = occ_so[p];
            const int j_so = occ_so[q];
            
            auto row = hb.row_view(i_so, j_so).with_cutoff(cutoff);
            
            for (size_t k = 0; k < row.len; ++k) {
                const int a_so = row.a[k];
                const int b_so = row.b[k];
                
                if (is_so_occupied(bra, a_so, n_orb) || 
                    is_so_occupied(bra, b_so, n_orb)) {
                    continue;
                }
                
                visit(apply_double_exc_so(bra, i_so, j_so, a_so, b_so));
            }
        }
    }
}

/**
 * Heuristic initial capacity for thread-local COO buffers.
 * Estimate: diagonal + 2·n_orb singles + ~10% of n_orb² doubles.
 */
inline size_t est_conn_per_row(int n_orb) {
    return static_cast<size_t>(1 + 2*n_orb + (n_orb*n_orb)/10);
}

// ============================================================================
// Assembly policies (strategy pattern for different screening modes)
// ============================================================================

/**
 * Policy: Pre-defined S and C spaces (variational mode).
 *
 * - Full double enumeration (no Heat-bath)
 * - All singles accepted
 * - Classify: S → SS, C → SC_IDX, else DROP
 */
struct PolicyKnownSets {
    const DetMap& map_S;
    const DetMap& map_C;

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
        if (auto jS = map_S.get_idx(ket)) {
            return {Classify::SS, *jS, {}};
        }
        if (auto jC = map_C.get_idx(ket)) {
            return {Classify::SC_IDX, *jC, {}};
        }
        return {Classify::DROP, 0, {}};
    }
};

/**
 * Policy: Static Heat-bath screening (connection generation mode).
 *
 * - Heat-bath doubles with fixed cutoff ε₁
 * - All singles accepted
 * - Classify: S → SS, else → SC_KEY (deferred indexing)
 */
struct PolicyStaticHB {
    const DetMap&        map_S;
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
            enumerate_doubles_hb(bra, n_orb, *hb_ptr, eps1, 
                                std::forward<Visitor>(visit));
        } else {
            det_ops::for_each_double(bra, n_orb, std::forward<Visitor>(visit));
        }
    }

    inline bool accept_single(u32, double, double) const {
        return true;
    }

    inline Classify classify(const Det& ket) const {
        if (auto jS = map_S.get_idx(ket)) {
            return {Classify::SS, *jS, {}};
        }
        return {Classify::SC_KEY, 0, ket};
    }
};

/**
 * Policy: Amplitude-weighted dynamic Heat-bath (perturbative selection).
 *
 * - Heat-bath cutoff τᵢ = ε₁/max(|ψᵢ|, δ) adapts per row
 * - Singles filtered: |hᵢⱼ·ψᵢ| ≥ ε₁
 * - Classify: S → SS, else → SC_KEY
 */
struct PolicyDynamicAmp {
    const DetMap&            map_S;
    const HeatBathTable*     hb;
    std::span<const double>  psi_S;
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
        
        const double tau = eps1 / std::max(std::abs(psi_S[row_i]), delta);
        enumerate_doubles_hb(bra, n_orb, *hb_ptr, tau, std::forward<Visitor>(visit));
    }

    inline bool accept_single(u32 row_i, double h, double) const {
        return std::abs(h * psi_S[row_i]) >= eps1;
    }

    inline Classify classify(const Det& ket) const {
        if (auto jS = map_S.get_idx(ket)) {
            return {Classify::SS, *jS, {}};
        }
        return {Classify::SC_KEY, 0, ket};
    }
};

// ============================================================================
// Parallel streaming kernel
// ============================================================================

/**
 * Core assembly kernel with pluggable screening policy.
 *
 * Algorithm:
 *   1. Parallel loop over S determinants (dynamic scheduling)
 *   2. For each |i⟩ ∈ S:
 *      a. Add diagonal hᵢᵢ → H_SS
 *      b. Enumerate singles/doubles via policy
 *      c. Compute matrix elements hᵢⱼ
 *      d. Classify and route to thread-local buffers
 *   3. Merge thread-local COO matrices
 *
 * Threading: OpenMP with dynamic scheduling for load balancing.
 * Each thread maintains separate buffers to avoid contention.
 *
 * @param S         S-space determinants (ordered)
 * @param ham       Hamiltonian evaluator
 * @param n_orb     Number of spatial orbitals
 * @param hb        Heat-bath table (nullable)
 * @param policy    Assembly policy (type-erased via template)
 * @param tl_ss     Thread-local H_SS buffers (output)
 * @param tl_sc_idx Thread-local H_SC indexed buffers (output)
 * @param tl_sc_key Thread-local H_SC keyed buffers (output)
 */
template<class Policy>
void stream_build(
    std::span<const Det> S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb,
    const Policy& policy,
    std::vector<COOMatrix>& tl_ss,
    std::vector<COOMatrix>& tl_sc_idx,
    std::vector<std::vector<KeyedEntry>>& tl_sc_key
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

        if (ss.nnz() == 0) {
            const size_t cap = est_conn_per_row(n_orb);
            ss.reserve(cap);
            sc_i.reserve(cap / 2);
            sc_k.reserve(cap / 2);
        }

        // Diagonal element always in H_SS
        {
            const double h = ham.compute_diagonal(bra);
            if (std::abs(h) > MAT_ELEMENT_THRESH) {
                ss.push_back(row, row, h);
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
                case Classify::SS:
                    ss.push_back(row, c.idx, h);
                    break;
                case Classify::SC_IDX:
                    sc_i.push_back(row, c.idx, h);
                    break;
                case Classify::SC_KEY:
                    sc_k.push_back({row, ket, h});
                    break;
                default:
                    break;
            }
        });

        // Double excitations (policy-driven enumeration)
        policy.enumerate_doubles(bra, n_orb, hb, row, [&](const Det& ket) {
            const Classify c = policy.classify(ket);
            if (c.kind == Classify::DROP) return;

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= MAT_ELEMENT_THRESH) return;

            switch (c.kind) {
                case Classify::SS:
                    ss.push_back(row, c.idx, h);
                    break;
                case Classify::SC_IDX:
                    sc_i.push_back(row, c.idx, h);
                    break;
                case Classify::SC_KEY:
                    sc_k.push_back({row, ket, h});
                    break;
                default:
                    break;
            }
        });
    }
}

/**
 * Merge thread-local COO buffers into unified matrix.
 * Concatenates all entries and performs sort-merge deduplication.
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
 * Finalize H_SC with deferred C-space indexing.
 *
 * Algorithm:
 *   1. Collect all unique C determinants from keyed entries
 *   2. Sort lexicographically and build DetMap
 *   3. Remap keyed entries to column indices
 *   4. Sort and merge final COO matrix
 *
 * Deterministic: Lexicographic ordering ensures reproducibility.
 *
 * @param tl_sc_key Thread-local keyed entry buffers
 * @param coo_SC    Output H_SC matrix
 * @param map_C     Output C-space determinant map
 */
inline void finalize_sc_deferred(
    std::vector<std::vector<KeyedEntry>>& tl_sc_key,
    COOMatrix& coo_SC,
    DetMap& map_C
) {
    std::vector<Det> keys;
    size_t total = 0;
    for (auto& v : tl_sc_key) total += v.size();
    keys.reserve(total);
    
    for (auto& v : tl_sc_key) {
        for (auto& x : v) {
            keys.push_back(x.key);
        }
    }
    
    if (!keys.empty()) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }
    
    map_C = DetMap::from_list(std::move(keys));
    
    coo_SC.reserve(total);
    
    for (auto& v : tl_sc_key) {
        for (auto& x : v) {
            auto j = map_C.get_idx(x.key);
            if (j) {
                coo_SC.push_back(x.row, *j, x.val);
            }
        }
    }
    
    sort_and_merge_coo(coo_SC);
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
        diag[i] = ham.compute_diagonal(dets[i]);
    }
    
    return diag;
}

COOMatrix get_ham_ss(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb
) {
    COOMatrix out;
    if (dets_S.empty()) return out;
    
    const auto map_S = DetMap::from_ordered(
        {dets_S.begin(), dets_S.end()}, 
        true
    );
    
    PolicyKnownSets policy{map_S, DetMap{}};
    
    std::vector<COOMatrix> tl_ss, tl_sc_idx;
    std::vector<std::vector<KeyedEntry>> tl_sc_key;
    
    stream_build(dets_S, ham, n_orb, nullptr, policy,
                 tl_ss, tl_sc_idx, tl_sc_key);
    
    merge_thread_local(tl_ss, out);
    out.n_rows = out.n_cols = static_cast<u32>(dets_S.size());
    
    return out;
}

HamBlocks get_ham_block(
    std::span<const Det> dets_S,
    std::optional<std::span<const Det>> dets_C,
    const HamEval& ham,
    int n_orb
) {
    HamBlocks out;
    if (dets_S.empty()) return out;

    const auto map_S = DetMap::from_ordered(
        {dets_S.begin(), dets_S.end()}, 
        true
    );

    DetMap map_C_known;
    bool has_C = dets_C.has_value() && !dets_C->empty();
    
    if (has_C) {
        map_C_known = DetMap::from_ordered(
            {dets_C->begin(), dets_C->end()}, 
            true
        );
    }

    PolicyKnownSets policy{map_S, map_C_known};

    std::vector<COOMatrix> tl_ss, tl_sc_idx;
    std::vector<std::vector<KeyedEntry>> tl_sc_key;

    stream_build(dets_S, ham, n_orb, nullptr, policy,
                 tl_ss, tl_sc_idx, tl_sc_key);

    merge_thread_local(tl_ss, out.H_SS);
    out.H_SS.n_rows = out.H_SS.n_cols = static_cast<u32>(dets_S.size());

    if (has_C) {
        merge_thread_local(tl_sc_idx, out.H_SC);
        out.H_SC.n_rows = static_cast<u32>(dets_S.size());
        out.H_SC.n_cols = static_cast<u32>(dets_C->size());
        out.map_C = std::move(map_C_known);
    }

    return out;
}

HamBlocks get_ham_conn(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "get_ham_conn: Heat-bath enabled but table is nullptr"
        );
    }

    HamBlocks out;
    if (dets_S.empty()) return out;

    const auto map_S = DetMap::from_ordered(
        {dets_S.begin(), dets_S.end()}, 
        true
    );

    PolicyStaticHB policy{map_S, hb_table, eps1, use_heatbath};

    std::vector<COOMatrix> tl_ss, tl_sc_idx;
    std::vector<std::vector<KeyedEntry>> tl_sc_key;

    stream_build(dets_S, ham, n_orb, hb_table, policy,
                 tl_ss, tl_sc_idx, tl_sc_key);

    merge_thread_local(tl_ss, out.H_SS);
    out.H_SS.n_rows = out.H_SS.n_cols = static_cast<u32>(dets_S.size());

    finalize_sc_deferred(tl_sc_key, out.H_SC, out.map_C);
    out.H_SC.n_rows = static_cast<u32>(dets_S.size());
    out.H_SC.n_cols = static_cast<u32>(out.map_C.size());

    return out;
}

HamBlocks get_ham_conn_amp(
    std::span<const Det> dets_S,
    std::span<const double> psi_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1
) {
    if (!hb_table) {
        throw std::invalid_argument(
            "get_ham_conn_amp: Heat-bath table is required"
        );
    }
    if (psi_S.size() != dets_S.size()) {
        throw std::invalid_argument(
            "get_ham_conn_amp: psi_S size must match |S|"
        );
    }

    HamBlocks out;
    if (dets_S.empty()) return out;

    const auto map_S = DetMap::from_ordered(
        {dets_S.begin(), dets_S.end()}, 
        true
    );

    PolicyDynamicAmp policy{map_S, hb_table, psi_S, eps1};

    std::vector<COOMatrix> tl_ss, tl_sc_idx;
    std::vector<std::vector<KeyedEntry>> tl_sc_key;

    stream_build(dets_S, ham, n_orb, hb_table, policy,
                 tl_ss, tl_sc_idx, tl_sc_key);

    merge_thread_local(tl_ss, out.H_SS);
    out.H_SS.n_rows = out.H_SS.n_cols = static_cast<u32>(dets_S.size());

    finalize_sc_deferred(tl_sc_key, out.H_SC, out.map_C);
    out.H_SC.n_rows = static_cast<u32>(dets_S.size());
    out.H_SC.n_cols = static_cast<u32>(out.map_C.size());

    return out;
}

} // namespace lever
