// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

#include <lever/hamiltonian/ham_block.hpp>
#include <lever/hamiltonian/ham_conn.hpp>  // Reuse Conn definition
#include <lever/determinant/det_ops.hpp>
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

namespace {

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

/** Check if a spin-orbital is occupied in determinant. */
bool is_so_occupied(const Det& d, int so_idx, int n_orb) {
    const int mo = mo_from_so(so_idx);
    if (mo < 0 || mo >= n_orb) return false;
    const u64 bit = (1ULL << mo);
    return (spin_from_so(so_idx) == 0) ? ((d.alpha & bit) != 0) : ((d.beta & bit) != 0);
}

/** Apply double excitation (i,j) -> (a,b) in spin-orbital basis. */
Det apply_double_exc_so(const Det& ket, int i_so, int j_so, int a_so, int b_so) {
    Det out = ket;
    auto flip = [&](int so_idx, bool set_bit) {
        const int mo = mo_from_so(so_idx);
        const int sp = spin_from_so(so_idx);
        const u64 bit = (1ULL << mo);
        u64& mask = (sp == 0) ? out.alpha : out.beta;
        if (set_bit) mask |= bit; else mask &= ~bit;
    };
    flip(i_so, false); 
    flip(j_so, false);
    flip(a_so, true);  
    flip(b_so, true);
    return out;
}

/** Extract occupied spin-orbital indices from determinant. */
std::vector<int> get_occupied_sos(const Det& d) {
    std::vector<int> occ_so;
    occ_so.reserve(popcount(d.alpha) + popcount(d.beta));
    for (int mo : extract_bits(d.alpha)) occ_so.push_back(so_from_mo(mo, 0));
    for (int mo : extract_bits(d.beta))  occ_so.push_back(so_from_mo(mo, 1));
    return occ_so;
}

/** Sort and merge COO entries by (row, col), summing duplicate values. */
void sort_and_merge_coo(std::vector<Conn>& xs) {
    if (xs.empty()) return;
    
    std::sort(xs.begin(), xs.end(), [](const Conn& a, const Conn& b) {
        return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
    });
    
    std::vector<Conn> out;
    out.reserve(xs.size());
    
    for (size_t i = 0; i < xs.size(); ) {
        u32 r = xs[i].row, c = xs[i].col;
        double v = xs[i].val;
        for (++i; i < xs.size() && xs[i].row == r && xs[i].col == c; ++i) {
            v += xs[i].val;
        }
        if (v != 0.0) out.emplace_back(r, c, v);
    }
    
    xs.swap(out);
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public API Implementation
// -----------------------------------------------------------------------------

DetMap generate_excitations(
    std::span<const Det> refs,
    int n_orb,
    const HeatBathTable* hb_table,
    const ExcitationOpts& opts
) {
    // Validate heat-bath configuration
    if (opts.use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "Heat-bath screening enabled but hb_table is nullptr"
        );
    }

    // Early exit for empty input
    if (refs.empty()) {
        return DetMap{};
    }

    // Build reference space map for duplicate detection
    const auto ref_map = DetMap::from_list({refs.begin(), refs.end()});

    // Thread-local storage for excited determinants
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<std::unordered_set<Det>> thread_excited(n_threads);

    // Parallel excitation generation
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& local_excited = thread_excited[tid];

        // Estimate capacity based on heat-bath mode
        const size_t estimated_capacity = opts.use_heatbath ? 1000 : 10000;
        local_excited.reserve(estimated_capacity);

        // Helper: add excited determinant if not in reference space
        auto add_if_new = [&](const Det& excited) {
            if (!ref_map.contains(excited)) {
                local_excited.insert(excited);
            }
        };

#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(refs.size()); ++i) {
            const Det& ref = refs[i];

            // --- Single excitations (always full enumeration) ---
            det_ops::for_each_single(ref, n_orb, add_if_new);

            // --- Double excitations (heat-bath screening if enabled) ---
            if (opts.use_heatbath) {
                // Heat-bath screened doubles: only strong integrals
                auto occ_so = get_occupied_sos(ref);
                if (occ_so.size() < 2) continue;

                for (size_t p = 1; p < occ_so.size(); ++p) {
                    for (size_t q = 0; q < p; ++q) {
                        const int i_so = occ_so[p];
                        const int j_so = occ_so[q];
                        
                        // Query heat-bath row with threshold cutoff
                        auto row = hb_table->row_view(i_so, j_so).with_cutoff(opts.eps1);
                        
                        for (size_t k = 0; k < row.len; ++k) {
                            const int a_so = row.a[k];
                            const int b_so = row.b[k];
                            
                            // Skip if target orbitals already occupied or out of range
                            if (is_so_occupied(ref, a_so, n_orb) || 
                                is_so_occupied(ref, b_so, n_orb)) continue;
                            if (mo_from_so(a_so) >= n_orb || 
                                mo_from_so(b_so) >= n_orb) continue;
                            
                            add_if_new(apply_double_exc_so(ref, i_so, j_so, a_so, b_so));
                        }
                    }
                }
            } else {
                // Full double enumeration (no heat-bath screening)
                det_ops::for_each_double(ref, n_orb, add_if_new);
            }
        }
    }

    // Merge thread-local results
    std::unordered_set<Det> excited_space;
    size_t total_size = 0;
    for (const auto& local_set : thread_excited) {
        total_size += local_set.size();
    }
    excited_space.reserve(total_size);
    
    for (auto& local_set : thread_excited) {
        excited_space.insert(
            std::make_move_iterator(local_set.begin()),
            std::make_move_iterator(local_set.end())
        );
    }

    // Convert to canonicalized DetMap
    std::vector<Det> excited_vec(excited_space.begin(), excited_space.end());
    return DetMap::from_list(det_space::canonicalize(std::move(excited_vec)));
}

std::vector<Conn> get_ham_block(
    std::span<const Det> bra,
    std::span<const Det> ket,
    const HamEval& ham,
    int n_orb,
    const BlockOpts& opts
) {
    // Early exit for empty inputs
    if (bra.empty() || ket.empty()) {
        return {};
    }

    // Build ket index map for O(1) lookup, STRICTLY PRESERVING a-priori ket ordering.
    const auto ket_map = DetMap::from_ordered({ket.begin(), ket.end()}, /* verify_unique= */ false);

    // Thread-local COO buckets
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<Conn>> thread_buckets(n_threads);

    // Parallel matrix element evaluation via connection generation
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& local_coo = thread_buckets[tid];
        
        // Heuristic: ~(1 diagonal + 2*n_orb singles + n_orb^2 doubles) per bra
        const size_t est_conn_per_bra = 1 + 2 * n_orb + n_orb * n_orb / 10;
        local_coo.reserve(est_conn_per_bra);

#pragma omp for schedule(dynamic)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(bra.size()); ++i) {
            const u32 row_idx = static_cast<u32>(i);
            const Det& bra_det = bra[i];

            // --- Diagonal element (if bra exists in ket space) ---
            if (auto col_idx = ket_map.get_idx(bra_det)) {
                const double h_diag = ham.compute_diagonal(bra_det);
                if (std::abs(h_diag) > opts.thresh) {
                    local_coo.emplace_back(row_idx, *col_idx, h_diag);
                }
            }

            // --- Single excitations: generate and check existence in ket ---
            det_ops::for_each_single(bra_det, n_orb, [&](const Det& ket_det) {
                if (auto col_idx = ket_map.get_idx(ket_det)) {
                    const double h_elem = ham.compute_elem(bra_det, ket_det);
                    if (std::abs(h_elem) > opts.thresh) {
                        local_coo.emplace_back(row_idx, *col_idx, h_elem);
                    }
                }
            });

            // --- Double excitations: generate and check existence in ket ---
            det_ops::for_each_double(bra_det, n_orb, [&](const Det& ket_det) {
                if (auto col_idx = ket_map.get_idx(ket_det)) {
                    const double h_elem = ham.compute_elem(bra_det, ket_det);
                    if (std::abs(h_elem) > opts.thresh) {
                        local_coo.emplace_back(row_idx, *col_idx, h_elem);
                    }
                }
            });
        }
    }

    // Merge thread-local buckets
    size_t total_nnz = 0;
    for (const auto& bucket : thread_buckets) {
        total_nnz += bucket.size();
    }

    std::vector<Conn> coo;
    coo.reserve(total_nnz);
    for (auto& bucket : thread_buckets) {
        coo.insert(coo.end(),
                   std::make_move_iterator(bucket.begin()),
                   std::make_move_iterator(bucket.end()));
    }

    // Sort and merge duplicates (shouldn't occur but handles edge cases)
    sort_and_merge_coo(coo);

    return coo;
}

} // namespace lever
