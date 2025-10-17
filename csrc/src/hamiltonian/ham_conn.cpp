// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

#include <lever/hamiltonian/ham_conn.hpp>
#include <lever/determinant/det_ops.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lever {

// -----------------------------------------------------------------------------
// Internal Utilities
// -----------------------------------------------------------------------------
namespace {

void sort_and_merge_(std::vector<Conn>& xs) {
    if (xs.empty()) return;
    std::sort(xs.begin(), xs.end(), [](const Conn& a, const Conn& b) {
        return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
    });
    std::vector<Conn> out;
    out.reserve(xs.size());
    for (size_t i = 0; i < xs.size(); ) {
        u32 r = xs[i].row, c = xs[i].col;
        double v = xs[i].val;
        for (++i; i < xs.size() && xs[i].row == r && xs[i].col == c; ++i) v += xs[i].val;
        if (v != 0.0) out.emplace_back(r, c, v);
    }
    xs.swap(out);
}

bool so_is_occ_(const Det& d, int so_idx, int n_orb) {
    const int mo = mo_from_so(so_idx);
    if (mo < 0 || mo >= n_orb) return false;
    const u64 bit = (1ULL << mo);
    return (spin_from_so(so_idx) == 0) ? ((d.alpha & bit) != 0) : ((d.beta & bit) != 0);
}

Det apply_double_exc_so_(const Det& ket, int i_so, int j_so, int a_so, int b_so) {
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

// Extract occupied spin-orbital indices (reusable helper)
std::vector<int> get_occ_sos_(const Det& d) {
    std::vector<int> occ_so;
    occ_so.reserve(popcount(d.alpha) + popcount(d.beta));
    for (int mo : extract_bits(d.alpha)) occ_so.push_back(so_from_mo(mo, 0));
    for (int mo : extract_bits(d.beta))  occ_so.push_back(so_from_mo(mo, 1));
    return occ_so;
}

/**
 * @brief Build external space C from S.
 * 
 * Contract: C ∩ S = ∅ (guaranteed by construction)
 * 
 * Singles: always full enumeration
 * Doubles: heat-bath pruned if enabled, else full enumeration
 * 
 * @throws std::invalid_argument if use_heatbath=true but hb=nullptr
 */
DetMap build_map_C_(std::span<const Det> dets_S, int n_orb, const DetMap& map_S,
                    const HeatBathTable* hb, const BuildOpts& opts) {
    std::unordered_set<Det> set_C;
    set_C.reserve(dets_S.size() * 64);

    // Validate heat-bath configuration upfront
    if (opts.use_heatbath && !hb) {
        throw std::invalid_argument("Heat-bath enabled but table is nullptr");
    }
    const bool use_hb = opts.use_heatbath && hb;

    auto add_to_C = [&](const Det& excited_det) {
        if (!map_S.contains(excited_det)) set_C.insert(excited_det);
    };

    for (const Det& ket : dets_S) {
        // Singles: always full (reuse det_ops API)
        det_ops::for_each_single(ket, n_orb, add_to_C);

        // Doubles: conditional on heat-bath mode
        if (use_hb) {
            auto occ_so = get_occ_sos_(ket);
            if (occ_so.size() < 2) continue;

            for (size_t p = 1; p < occ_so.size(); ++p) {
                for (size_t q = 0; q < p; ++q) {
                    const int i_so = occ_so[p], j_so = occ_so[q];
                    auto row = hb->row_view(i_so, j_so).with_cutoff(opts.eps1);
                    
                    for (size_t k = 0; k < row.len; ++k) {
                        const int a_so = row.a[k], b_so = row.b[k];
                        if (so_is_occ_(ket, a_so, n_orb) || so_is_occ_(ket, b_so, n_orb)) continue;
                        if (mo_from_so(a_so) >= n_orb || mo_from_so(b_so) >= n_orb) continue;
                        add_to_C(apply_double_exc_so_(ket, i_so, j_so, a_so, b_so));
                    }
                }
            }
        } else {
            det_ops::for_each_double(ket, n_orb, add_to_C);
        }
    }

    std::vector<Det> c_vec(set_C.begin(), set_C.end());
    return DetMap::from_list(det_space::canonicalize(std::move(c_vec)));
}

struct InternalBuildResult {
    std::vector<Conn> coo_SS;
    std::vector<Conn> coo_SC;
    DetMap map_C;
};

/**
 * @brief Unified internal builder for SS and/or SC blocks.
 * 
 * Complexity:
 *   - C-space discovery: O(|S| * n_orb^2) for singles, 
 *                        O(|S| * n_hb_avg) for HB doubles,
 *                        O(|S| * n_orb^4) for full doubles
 *   - Matrix evaluation: O(nnz * eval_cost) where eval_cost ~ O(n_elec) for singles,
 *                        O(1) for doubles
 * 
 * Memory:
 *   - Peak: O(|S| + |C| + n_threads * avg_row_nnz)
 *   - C-space: typically |C| ~ 10-1000 * |S| depending on system
 * 
 * Thread-safety: Safe for concurrent calls with different inputs
 * 
 * @param dets_S      Reference space determinants
 * @param ham         Hamiltonian evaluator
 * @param n_orb       Number of spatial orbitals
 * @param hb          Heat-bath table (may be nullptr if not using HB)
 * @param opts        Build options (thresh, eps1, use_heatbath)
 * @param build_ss    Whether to build <S|H|S> block
 * @param build_sc    Whether to build <S|H|C> block
 * 
 * @return InternalBuildResult with requested blocks
 * 
 * Notes:
 * - SS block: includes diagonal, always full enumeration for variational consistency
 * - SC block: C is built from S (singles-all + doubles via HB/full), then matrix elements evaluated
 * - Thread-safe parallel execution via OpenMP
 */
InternalBuildResult build_connections_internal_(
    std::span<const Det> dets_S, const HamEval& ham, int n_orb,
    const HeatBathTable* hb, const BuildOpts& opts,
    bool build_ss, bool build_sc
) {
    InternalBuildResult result;
    if (dets_S.empty()) return result;

    const auto map_S = DetMap::from_ordered({dets_S.begin(), dets_S.end()}, true);

    // PASS 1: Discover C space if needed (defines C once)
    if (build_sc) {
        result.map_C = build_map_C_(dets_S, n_orb, map_S, hb, opts);
    }

    // PASS 2: Parallel matrix element evaluation
    std::vector<std::vector<Conn>> buckets_SS, buckets_SC;
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    if (build_ss) buckets_SS.resize(n_threads);
    if (build_sc) buckets_SC.resize(n_threads);

#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dets_S.size()); ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const auto row_idx = static_cast<u32>(i);
        const Det& bra = dets_S[i];

        // --- Diagonal (SS only) ---
        if (build_ss) {
            const double h_diag = ham.compute_diagonal(bra);
            if (std::abs(h_diag) > opts.thresh) {
                buckets_SS[tid].emplace_back(row_idx, row_idx, h_diag);
            }
        }

        // --- Singles (route to SS or SC) ---
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            if (auto jS = map_S.get_idx(ket)) {
                if (build_ss) {
                    const double h = ham.compute_elem(bra, ket);
                    if (std::abs(h) > opts.thresh) {
                        buckets_SS[tid].emplace_back(row_idx, *jS, h);
                    }
                }
            } else if (build_sc) {
                if (auto jC = result.map_C.get_idx(ket)) {
                    const double h = ham.compute_elem(bra, ket);
                    if (std::abs(h) > opts.thresh) {
                        buckets_SC[tid].emplace_back(row_idx, *jC, h);
                    }
                }
            }
        });

        // --- Doubles to SS (always full for variational consistency) ---
        if (build_ss) {
            det_ops::for_each_double(bra, n_orb, [&](const Det& ket) {
                if (auto jS = map_S.get_idx(ket)) {
                    const double h = ham.compute_elem(bra, ket);
                    if (std::abs(h) > opts.thresh) {
                        buckets_SS[tid].emplace_back(row_idx, *jS, h);
                    }
                }
            });
        }
        
        // --- Doubles to SC (match build_map_C_ strategy) ---
        if (build_sc) {
            const bool use_hb = opts.use_heatbath && hb;
            
            // Helper: evaluate and collect if in C
            auto process_sc_double = [&](const Det& ket) {
                // Exploits C ∩ S = ∅: no need for map_S check
                if (auto jC = result.map_C.get_idx(ket)) {
                    const double h = ham.compute_elem(bra, ket);
                    if (std::abs(h) > opts.thresh) {
                        buckets_SC[tid].emplace_back(row_idx, *jC, h);
                    }
                }
            };

            if (use_hb) {
                auto occ_so = get_occ_sos_(bra);
                if (occ_so.size() < 2) continue;  // ✅ `continue` is correct in parallel-for
                
                for (size_t p = 1; p < occ_so.size(); ++p) {
                    for (size_t q = 0; q < p; ++q) {
                        const int i_so = occ_so[p], j_so = occ_so[q];
                        auto row = hb->row_view(i_so, j_so).with_cutoff(opts.eps1);
                        
                        for (size_t k = 0; k < row.len; ++k) {
                            const int a_so = row.a[k], b_so = row.b[k];
                            if (so_is_occ_(bra, a_so, n_orb) || so_is_occ_(bra, b_so, n_orb)) continue;
                            if (mo_from_so(a_so) >= n_orb || mo_from_so(b_so) >= n_orb) continue;
                            process_sc_double(apply_double_exc_so_(bra, i_so, j_so, a_so, b_so));
                        }
                    }
                }
            } else {
                // Full enumeration: directly query C (no need for map_S check)
                det_ops::for_each_double(bra, n_orb, process_sc_double);
            }
        }
    }

    // PASS 3: Merge thread-local buckets
    if (build_ss) {
        size_t total = std::accumulate(buckets_SS.begin(), buckets_SS.end(), size_t(0),
                                       [](size_t s, const auto& v){ return s + v.size(); });
        result.coo_SS.reserve(total);
        for (auto& b : buckets_SS) {
            result.coo_SS.insert(result.coo_SS.end(), b.begin(), b.end());
        }
        sort_and_merge_(result.coo_SS);
    }
    if (build_sc) {
        size_t total = std::accumulate(buckets_SC.begin(), buckets_SC.end(), size_t(0),
                                       [](size_t s, const auto& v){ return s + v.size(); });
        result.coo_SC.reserve(total);
        for (auto& b : buckets_SC) {
            result.coo_SC.insert(result.coo_SC.end(), b.begin(), b.end());
        }
        sort_and_merge_(result.coo_SC);
    }
    return result;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

std::vector<double> get_ham_diag(std::span<const Det> dets, const HamEval& ham) {
    std::vector<double> diag(dets.size());
#pragma omp parallel for if(dets.size() > 256)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(dets.size()); ++i) {
        diag[i] = ham.compute_diagonal(dets[i]);
    }
    return diag;
}

SSResult get_ham_SS(std::span<const Det> dets_S, const HamEval& ham, 
                    int n_orb, const BuildOpts& opts) {
    auto result = build_connections_internal_(dets_S, ham, n_orb, nullptr, opts, true, false);
    return {std::move(result.coo_SS)};
}

SCResult get_ham_SC(std::span<const Det> dets_S, const HamEval& ham, 
                    int n_orb, const HeatBathTable* hb, const BuildOpts& opts) {
    auto result = build_connections_internal_(dets_S, ham, n_orb, hb, opts, false, true);
    return {std::move(result.coo_SC), std::move(result.map_C)};
}

SSSCResult get_ham_SS_SC(std::span<const Det> dets_S, const HamEval& ham, 
                         int n_orb, const HeatBathTable* hb, const BuildOpts& opts) {
    auto result = build_connections_internal_(dets_S, ham, n_orb, hb, opts, true, true);
    return {std::move(result.coo_SS), std::move(result.coo_SC), std::move(result.map_C)};
}

STResult get_ham_ST(std::span<const Det> dets_S, const HamEval& ham, 
                    int n_orb, const HeatBathTable* hb, const BuildOpts& opts) {
    STResult out;
    if (dets_S.empty()) return out;

    // Build SS and SC blocks internally
    auto internal_res = build_connections_internal_(dets_S, ham, n_orb, hb, opts, true, true);
    
    out.size_S = dets_S.size();
    const u32 col_shift = static_cast<u32>(out.size_S);

    // Construct T = S ++ C with S as prefix
    std::vector<Det> T_dets;
    T_dets.reserve(dets_S.size() + internal_res.map_C.size());
    T_dets.insert(T_dets.end(), dets_S.begin(), dets_S.end());
    const auto& C_dets = internal_res.map_C.all_dets();
    T_dets.insert(T_dets.end(), C_dets.begin(), C_dets.end());
    out.map_T = DetMap::from_ordered(std::move(T_dets), false);

    // Combine blocks: SS (unchanged) + SC (with shifted column indices)
    out.coo.reserve(internal_res.coo_SS.size() + internal_res.coo_SC.size());
    out.coo = std::move(internal_res.coo_SS);  // SS block: cols already in [0, size_S)
    
    // ✅ FIX: Shift SC columns BEFORE insertion
    for (auto& conn : internal_res.coo_SC) {
        conn.col += col_shift;  // Map C-space indices to T-space
    }
    out.coo.insert(out.coo.end(), 
                   std::make_move_iterator(internal_res.coo_SC.begin()), 
                   std::make_move_iterator(internal_res.coo_SC.end()));
    
    sort_and_merge_(out.coo);
    return out;
}

} // namespace lever
