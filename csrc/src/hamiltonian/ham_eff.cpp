// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

#include <lever/hamiltonian/ham_eff.hpp>
#include <algorithm>
#include <cmath>
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
 * Compute regularized diagonal inverse D_jj⁻¹.
 */
[[nodiscard]] std::vector<double> compute_diagonal_inverse(
    std::span<const double> h_cc_diag,
    double e_ref,
    double epsilon,
    Regularization reg_type
) {
    const size_t n_c = h_cc_diag.size();
    std::vector<double> d_inv(n_c);

    for (size_t j = 0; j < n_c; ++j) {
        const double d = e_ref - h_cc_diag[j];

        switch (reg_type) {
        case Regularization::LinearShift: {
            const double d_eff = d + std::copysign(epsilon, d);
            d_inv[j] = 1.0 / d_eff;
            break;
        }
        case Regularization::Sigma: {
            const double denom = d * d + epsilon * epsilon;
            d_inv[j] = d / denom;
            break;
        }
        }
    }

    return d_inv;
}

/**
 * Add scaled outer product: ΔH += scale · b_j ⊗ b_j.
 * Applies micro_thresh screening at column and entry levels.
 */
inline void add_scaled_outer_product(
    CSCMatrix::ColView col_view,
    double scale,
    bool upper_only,
    COOMatrix& out_coo
) {
    const size_t len = col_view.rows.size();
    if (len == 0) return;
    // Column-level screening with MICRO_CONTRIB_THRESH
    double col_norm_sq = 0.0;
    for (double v : col_view.vals) {
        col_norm_sq += v * v;
    }
    if (std::abs(scale) * col_norm_sq < MICRO_CONTRIB_THRESH) {
        return;
    }

    // Reserve capacity
    const size_t reserve_size = upper_only ? (len * (len + 1)) / 2 : len * len;
    const size_t current_size = out_coo.nnz();
    out_coo.reserve(current_size + reserve_size);

    // Generate outer product entries
    for (size_t p = 0; p < len; ++p) {
        const u32 i = col_view.rows[p];
        const double v_i = col_view.vals[p];

        const size_t q_start = upper_only ? p : 0;
        for (size_t q = q_start; q < len; ++q) {
            const u32 k = col_view.rows[q];
            const double v_k = col_view.vals[q];

            const double contrib = scale * v_i * v_k;
            if (std::abs(contrib) > MICRO_CONTRIB_THRESH) {
                out_coo.push_back(i, k, contrib);
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

COOMatrix get_ham_eff(
    const COOMatrix& H_SS,
    const COOMatrix& H_SC,
    std::span<const double> h_cc_diag,
    double e_ref,
    const HeffConfig& config
) {
    const size_t n_c = h_cc_diag.size();

    // Edge case: no C-space → H_eff = H_SS
    if (n_c == 0) {
        return H_SS;
    }

    // Validate dimensions
    if (H_SC.n_cols != static_cast<u32>(n_c)) {
        throw std::invalid_argument(
            "get_ham_eff: H_SC column dimension mismatch with h_cc_diag size"
        );
    }

    const u32 n_s = H_SS.n_rows;

    // Convert H_SC to CSC for column-wise access
    CSCMatrix h_sc_csc = coo_to_csc(H_SC, n_s, static_cast<u32>(n_c));

    // Compute regularized diagonal inverse
    auto d_inv = compute_diagonal_inverse(
        h_cc_diag, e_ref, config.epsilon, config.reg_type
    );

    // Parallel outer products: ΔH = Σ_j (D_jj⁻¹)·b_j⊗b_j
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    std::vector<COOMatrix> thread_local_coo(n_threads);

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
            config.upper_only,
            local_coo
        );
    }

    // Merge thread-local results
    COOMatrix delta_h;
    {
        size_t total_entries = 0;
        for (const auto& local : thread_local_coo) {
            total_entries += local.nnz();
        }
        delta_h.reserve(total_entries);

        for (auto& local : thread_local_coo) {
            delta_h.rows.insert(delta_h.rows.end(), local.rows.begin(), local.rows.end());
            delta_h.cols.insert(delta_h.cols.end(), local.cols.begin(), local.cols.end());
            delta_h.vals.insert(delta_h.vals.end(), local.vals.begin(), local.vals.end());
        }
    }

    delta_h.n_rows = delta_h.n_cols = n_s;

    // Mirror upper triangle to full matrix if needed
    if (config.upper_only) {
        delta_h = mirror_upper_to_full(delta_h);
    }

    sort_and_merge_coo(delta_h);

    // Assemble final H_eff = H_SS + ΔH
    COOMatrix h_eff = coo_add(H_SS, delta_h, MAT_ELEMENT_THRESH);
    h_eff.n_rows = h_eff.n_cols = n_s;

    return h_eff;
}

} // namespace lever
