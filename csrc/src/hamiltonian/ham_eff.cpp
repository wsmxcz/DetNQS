// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eff.cpp
 * @brief Effective Hamiltonian assembly implementation.
 *
 * Algorithm: Column-wise sparse outer products
 * H_eff = H_SS + H_SC·D⁻¹·H_CS where D_jj = E_ref - H_CC[j,j]
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

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
// Diagonal inverse computation
// ============================================================================

/**
 * Compute regularized D_jj⁻¹ with specified strategy.
 *
 * LinearShift: d_inv = 1/(d + ε·sign(d))
 * Sigma:       d_inv = d/(d² + ε²)
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
        case Regularization::LinearShift:
            d_inv[j] = 1.0 / (d + std::copysign(epsilon, d));
            break;

        case Regularization::Sigma:
            d_inv[j] = d / (d * d + epsilon * epsilon);
            break;
        }
    }

    return d_inv;
}

/**
 * Add scaled outer product: ΔH += scale·b_j⊗b_jᵀ.
 *
 * Screens contributions at column and entry levels using MICRO_CONTRIB_THRESH.
 * Optionally builds upper triangle only for symmetric matrices.
 */
inline void add_scaled_outer_product(
    CSCMatrix::ColView col_view,
    double scale,
    bool upper_only,
    COOMatrix& out_coo
) {
    const size_t len = col_view.rows.size();
    if (len == 0) return;

    // Column-level screening: |scale|·‖b_j‖² < threshold
    double col_norm_sq = 0.0;
    for (double v : col_view.vals) {
        col_norm_sq += v * v;
    }
    if (std::abs(scale) * col_norm_sq < MICRO_CONTRIB_THRESH) {
        return;
    }

    // Reserve capacity for outer product entries
    const size_t reserve_size = upper_only ? (len * (len + 1)) / 2 : len * len;
    out_coo.reserve(out_coo.nnz() + reserve_size);

    // Generate outer product: (b_j)_i · (b_j)_k for all (i,k) pairs
    for (size_t p = 0; p < len; ++p) {
        const u32 i = col_view.rows[p];
        const double v_i = col_view.vals[p];

        const size_t q_start = upper_only ? p : 0;
        for (size_t q = q_start; q < len; ++q) {
            const u32 k = col_view.rows[q];
            const double contrib = scale * v_i * col_view.vals[q];

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

    // Validate input dimensions
    if (H_SC.n_cols != static_cast<u32>(n_c)) {
        throw std::invalid_argument(
            "get_ham_eff: H_SC column count mismatch with h_cc_diag size"
        );
    }

    const u32 n_s = H_SS.n_rows;

    // Convert to CSC for efficient column access
    CSCMatrix h_sc_csc = coo_to_csc(H_SC, n_s, static_cast<u32>(n_c));

    // Compute D_jj⁻¹ for all C-space determinants
    auto d_inv = compute_diagonal_inverse(
        h_cc_diag, e_ref, config.epsilon, config.reg_type
    );

    // Parallel accumulation: ΔH = Σ_j D_jj⁻¹·b_j⊗b_jᵀ
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    std::vector<COOMatrix> thread_coo(n_threads);

#pragma omp parallel for schedule(guided)
    for (std::ptrdiff_t j_idx = 0; j_idx < static_cast<std::ptrdiff_t>(n_c); ++j_idx) {
        const u32 j = static_cast<u32>(j_idx);

#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif

        add_scaled_outer_product(
            h_sc_csc.col(j),
            d_inv[j],
            config.upper_only,
            thread_coo[tid]
        );
    }

    // Merge thread-local contributions
    COOMatrix delta_h;
    {
        size_t total_nnz = 0;
        for (const auto& local : thread_coo) {
            total_nnz += local.nnz();
        }
        delta_h.reserve(total_nnz);

        for (auto& local : thread_coo) {
            delta_h.rows.insert(delta_h.rows.end(), local.rows.begin(), local.rows.end());
            delta_h.cols.insert(delta_h.cols.end(), local.cols.begin(), local.cols.end());
            delta_h.vals.insert(delta_h.vals.end(), local.vals.begin(), local.vals.end());
        }
    }

    delta_h.n_rows = delta_h.n_cols = n_s;

    // Complete symmetric matrix if upper triangle only
    if (config.upper_only) {
        delta_h = mirror_upper_to_full(delta_h);
    }

    sort_and_merge_coo(delta_h);

    // Final assembly: H_eff = H_SS + ΔH
    COOMatrix h_eff = coo_add(H_SS, delta_h, MAT_ELEMENT_THRESH);
    h_eff.n_rows = h_eff.n_cols = n_s;

    return h_eff;
}

} // namespace lever
