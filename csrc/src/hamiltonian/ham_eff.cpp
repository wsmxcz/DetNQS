// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <lever/hamiltonian/ham_eff.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lever {

namespace {

// ============================================================================
// Internal helpers
// ============================================================================

/**
 * Infer S-space dimension from maximum row index in COO blocks.
 */
[[nodiscard]] u32 infer_n_rows_S(
    const std::vector<Conn>& coo_SS,
    const std::vector<Conn>& coo_SC
) noexcept {
    u32 max_row = 0;
    bool has_entries = false;

    for (const auto& x : coo_SS) {
        max_row = std::max(max_row, x.row);
        has_entries = true;
    }
    for (const auto& x : coo_SC) {
        max_row = std::max(max_row, x.row);
        has_entries = true;
    }

    return has_entries ? (max_row + 1) : 0;
}

/**
 * Convert COO to CSC with in-column duplicate merging.
 * Each column is sorted and deduplicated for deterministic results.
 */
[[nodiscard]] CSCMatrix coo_to_csc(
    const std::vector<Conn>& coo,
    u32 n_rows,
    u32 n_cols
) {
    CSCMatrix csc;
    csc.n_rows = n_rows;
    csc.n_cols = n_cols;

    if (coo.empty()) {
        csc.col_ptrs.assign(n_cols + 1, 0);
        return csc;
    }

    // Bucket-sort by column
    std::vector<std::vector<std::pair<u32, double>>> col_buckets(n_cols);
    for (const auto& x : coo) {
        if (x.col >= n_cols) {
            throw std::out_of_range("coo_to_csc: column index out of bounds");
        }
        col_buckets[x.col].emplace_back(x.row, x.val);
    }

    // Build CSC with per-column deduplication
    csc.col_ptrs.resize(n_cols + 1);
    csc.col_ptrs[0] = 0;

    for (u32 j = 0; j < n_cols; ++j) {
        auto& bucket = col_buckets[j];

        if (bucket.empty()) {
            csc.col_ptrs[j + 1] = csc.col_ptrs[j];
            continue;
        }

        // Sort by row
        std::sort(bucket.begin(), bucket.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Merge consecutive duplicates
        for (size_t i = 0; i < bucket.size(); ) {
            const u32 row = bucket[i].first;
            double val = bucket[i].second;

            for (++i; i < bucket.size() && bucket[i].first == row; ++i) {
                val += bucket[i].second;
            }

            if (val != 0.0) {
                csc.row_indices.push_back(row);
                csc.values.push_back(val);
            }
        }

        csc.col_ptrs[j + 1] = csc.row_indices.size();
    }

    return csc;
}

/**
 * Compute regularized diagonal inverse D_jj⁻¹.
 * Returns (d_inv, n_regularized, max_abs_d, max_abs_d_inv).
 */
[[nodiscard]] auto compute_diagonal_inverse(
    std::span<const double> h_cc_diag,
    double e_ref,
    double epsilon,
    Regularization reg_type
) {
    const size_t n_c = h_cc_diag.size();
    std::vector<double> d_inv(n_c);

    double max_abs_d = 0.0;
    double max_abs_d_inv = 0.0;
    size_t n_regularized = 0;

    for (size_t j = 0; j < n_c; ++j) {
        const double d = e_ref - h_cc_diag[j];
        const double abs_d = std::abs(d);
        max_abs_d = std::max(max_abs_d, abs_d);

        if (abs_d < epsilon) {
            ++n_regularized;
        }

        switch (reg_type) {
        case Regularization::LinearShift: {
            // d_eff = d + ε·sign(d), then invert
            const double d_eff = d + std::copysign(epsilon, d);
            d_inv[j] = 1.0 / d_eff;
            break;
        }
        case Regularization::Sigma: {
            // Tikhonov smoothing: d_inv = d/(d² + ε²)
            const double denom = d * d + epsilon * epsilon;
            d_inv[j] = d / denom;
            break;
        }
        }

        max_abs_d_inv = std::max(max_abs_d_inv, std::abs(d_inv[j]));
    }

    return std::make_tuple(d_inv, n_regularized, max_abs_d, max_abs_d_inv);
}

/**
 * Add scaled outer product: ΔH += scale · b_j ⊗ b_j.
 * Applies micro_thresh screening at column and entry levels.
 * In upper_only mode, generates only entries where row ≤ col.
 */
inline void add_scaled_outer_product(
    CSCMatrix::ColView col_view,
    double scale,
    double micro_thresh,
    bool upper_only,
    std::vector<Conn>& out_coo
) {
    const size_t len = col_view.rows.size();
    if (len == 0) return;

    // Column-level screening
    double col_norm_sq = 0.0;
    for (double v : col_view.vals) {
        col_norm_sq += v * v;
    }
    if (std::abs(scale) * col_norm_sq < micro_thresh) {
        return;
    }

    // Reserve capacity
    const size_t reserve_size = upper_only ? (len * (len + 1)) / 2 : len * len;
    out_coo.reserve(out_coo.size() + reserve_size);

    // Generate outer product entries
    for (size_t p = 0; p < len; ++p) {
        const u32 i = col_view.rows[p];
        const double v_i = col_view.vals[p];

        const size_t q_start = upper_only ? p : 0;
        for (size_t q = q_start; q < len; ++q) {
            const u32 k = col_view.rows[q];
            const double v_k = col_view.vals[q];

            const double contrib = scale * v_i * v_k;
            if (std::abs(contrib) > micro_thresh) {
                out_coo.emplace_back(i, k, contrib);
            }
        }
    }
}

/**
 * Sort COO by (row,col) and merge duplicates.
 * Returns maximum absolute value encountered.
 */
inline double sort_and_merge_coo(std::vector<Conn>& coo) {
    if (coo.empty()) return 0.0;

    // Sort by (row, col)
    std::sort(coo.begin(), coo.end(), [](const Conn& a, const Conn& b) {
        return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
    });

    // Merge duplicates
    std::vector<Conn> merged;
    merged.reserve(coo.size());
    double max_abs = 0.0;

    for (size_t i = 0; i < coo.size(); ) {
        const u32 r = coo[i].row;
        const u32 c = coo[i].col;
        double v = coo[i].val;

        for (++i; i < coo.size() && coo[i].row == r && coo[i].col == c; ++i) {
            v += coo[i].val;
        }

        const double abs_v = std::abs(v);
        if (abs_v > 0.0) {
            max_abs = std::max(max_abs, abs_v);
            merged.emplace_back(r, c, v);
        }
    }

    coo.swap(merged);
    return max_abs;
}

/**
 * Merge two sorted COO matrices: C = A + B.
 * Drops entries below threshold.
 */
[[nodiscard]] std::vector<Conn> coo_add(
    const std::vector<Conn>& A,
    const std::vector<Conn>& B,
    double thresh
) {
    std::vector<Conn> C;
    C.reserve(A.size() + B.size());

    size_t i = 0, j = 0;

    while (i < A.size() && j < B.size()) {
        const auto& a = A[i];
        const auto& b = B[j];

        if (a.row < b.row || (a.row == b.row && a.col < b.col)) {
            if (std::abs(a.val) > thresh) C.push_back(a);
            ++i;
        } else if (b.row < a.row || (b.row == a.row && b.col < a.col)) {
            if (std::abs(b.val) > thresh) C.push_back(b);
            ++j;
        } else {
            // Merge at same position
            const double v = a.val + b.val;
            if (std::abs(v) > thresh) {
                C.emplace_back(a.row, a.col, v);
            }
            ++i;
            ++j;
        }
    }

    // Append remaining
    while (i < A.size()) {
        if (std::abs(A[i].val) > thresh) C.push_back(A[i]);
        ++i;
    }
    while (j < B.size()) {
        if (std::abs(B[j].val) > thresh) C.push_back(B[j]);
        ++j;
    }

    return C;
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

HeffResult get_ham_eff(
    const SSSCResult& blocks,
    std::span<const double> h_cc_diag,
    double e_ref,
    const HeffConfig& config
) {
    HeffResult result;

    // Validate dimensions
    const size_t n_c = blocks.map_C.size();
    if (h_cc_diag.size() != n_c) {
        throw std::invalid_argument(
            "get_ham_eff: h_cc_diag size mismatch (expected " +
            std::to_string(n_c) + ", got " + std::to_string(h_cc_diag.size()) + ")"
        );
    }

    // Edge case: no C-space → H_eff = H_SS
    if (n_c == 0) {
        result.coo_heff = blocks.coo_SS;
        result.n_rows_S = infer_n_rows_S(blocks.coo_SS, blocks.coo_SC);
        return result;
    }

    result.n_rows_S = infer_n_rows_S(blocks.coo_SS, blocks.coo_SC);

    // Convert H_SC to CSC (column-wise access for outer products)
    CSCMatrix h_sc_csc = coo_to_csc(
        blocks.coo_SC,
        result.n_rows_S,
        static_cast<u32>(n_c)
    );

    // Compute regularized diagonal inverse
    auto [d_inv, n_reg, max_d, max_d_inv] = compute_diagonal_inverse(
        h_cc_diag, e_ref, config.epsilon, config.reg_type
    );
    result.n_regularized = n_reg;
    result.max_abs_d = max_d;
    result.max_abs_d_inv = max_d_inv;

    // Parallel outer products: ΔH = Σ_j (D_jj⁻¹)·b_j⊗b_j
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    std::vector<std::vector<Conn>> thread_local_coo(n_threads);

#pragma omp parallel for schedule(guided)
    for (std::ptrdiff_t j_signed = 0; j_signed < static_cast<std::ptrdiff_t>(n_c); ++j_signed) {
        const u32 j = static_cast<u32>(j_signed);

        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif

        auto& local_coo = thread_local_coo[tid];
        auto col_view = h_sc_csc.col(j);

        add_scaled_outer_product(
            col_view,
            d_inv[j],
            config.micro_thresh,
            config.upper_only,
            local_coo
        );
    }

    // Merge thread-local results
    std::vector<Conn> delta_h;
    {
        size_t total_entries = 0;
        for (const auto& local : thread_local_coo) {
            total_entries += local.size();
        }
        delta_h.reserve(total_entries);

        for (auto& local : thread_local_coo) {
            delta_h.insert(
                delta_h.end(),
                std::make_move_iterator(local.begin()),
                std::make_move_iterator(local.end())
            );
        }
    }

    result.nnz_correction = delta_h.size();

    // Mirror upper triangle to full matrix if needed
    if (config.upper_only) {
        delta_h = mirror_upper_to_full(delta_h);
    }

    result.max_correction = sort_and_merge_coo(delta_h);

    // Assemble final H_eff = H_SS + ΔH
    result.coo_heff = coo_add(blocks.coo_SS, delta_h, config.thresh);

    return result;
}

std::vector<Conn> mirror_upper_to_full(std::span<const Conn> upper_coo) {
    std::vector<Conn> full_coo;
    full_coo.reserve(2 * upper_coo.size());

    for (const auto& x : upper_coo) {
        full_coo.push_back(x);  // Upper entry (i, j, v)

        if (x.row != x.col) {
            full_coo.emplace_back(x.col, x.row, x.val);  // Lower entry (j, i, v)
        }
    }

    return full_coo;
}

} // namespace lever
