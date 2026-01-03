// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eff.cpp
 * @brief Effective Hamiltonian via Gustavson-style SpGEMM.
 *
 * Computes H_add = H_VP · D^{-1} · H_PV using CSR(H_VP) × CSC(H_VP)
 * to avoid temporary outer-product storage. Returns H_eff = H_VV + H_add.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <detnqs/hamiltonian/ham_eff.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace detnqs {

namespace {

/**
 * Compute regularized inverse D^{-1} for denominators d = E_ref - H_PP[j,j].
 * 
 * Regularization schemes:
 *   - LinearShift: D^{-1}_j = 1 / (d_j + sign(d_j) * eps)
 *   - Sigma:       D^{-1}_j = d_j / (d_j^2 + eps^2)
 */
[[nodiscard]] std::vector<double> compute_dinv(
    std::span<const double> h_pp_diag,
    double e_ref,
    double eps,
    Regularization reg
) {
    const size_t n = h_pp_diag.size();
    std::vector<double> dinv(n);
    
    for (size_t j = 0; j < n; ++j) {
        const double d = e_ref - h_pp_diag[j];
        switch (reg) {
        case Regularization::LinearShift:
            dinv[j] = 1.0 / (d + std::copysign(eps, d));
            break;
        case Regularization::Sigma:
            dinv[j] = d / (d * d + eps * eps);
            break;
        }
    }
    return dinv;
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================

COOMatrix get_ham_eff(
    const COOMatrix& H_VV_in,
    const COOMatrix& H_VP_in,
    std::span<const double> h_pp_diag,
    double e_ref,
    const HeffConfig& cfg
) {
    // Handle empty perturbative space
    const size_t n_p = h_pp_diag.size();
    if (n_p == 0) {
        COOMatrix H = H_VV_in;
        (void)sort_and_merge_coo(H);
        return H;
    }

    // Validate dimensions
    if (H_VP_in.n_cols != static_cast<u32>(n_p)) {
        throw std::invalid_argument("H_VP columns != h_pp_diag size");
    }
    const u32 n_v = H_VV_in.n_rows ? H_VV_in.n_rows : H_VP_in.n_rows;
    if (H_VP_in.n_rows != n_v) {
        throw std::invalid_argument("H_VP rows != H_VV rows");
    }

    // Convert to CSR/CSC (includes deduplication & sorting)
    CSRMatrix VP_csr = coo_to_csr(H_VP_in, n_v, static_cast<u32>(n_p));
    CSCMatrix VP_csc = coo_to_csc(H_VP_in, n_v, static_cast<u32>(n_p));

    // Precompute D^{-1} with regularization
    std::vector<double> dinv = compute_dinv(h_pp_diag, e_ref, cfg.epsilon, cfg.reg_type);

    // Column screening: skip j if |D^{-1}_j| * ||col_j||^2 < threshold
    std::vector<double> col_norm2(n_p, 0.0);
    for (u32 j = 0; j < VP_csc.n_cols; ++j) {
        auto col = VP_csc.col(j);
        double s = 0.0;
        for (double v : col.vals) s += v * v;
        col_norm2[j] = s;
    }
    std::vector<uint8_t> skip(n_p, 0);
    for (size_t j = 0; j < n_p; ++j) {
        if (std::abs(dinv[j]) * col_norm2[j] < MICRO_CONTRIB_THRESH) {
            skip[j] = 1;
        }
    }

    // ========================================================================
    // Phase 1: SYMBOLIC - Determine sparsity pattern
    // ========================================================================
    std::vector<size_t> row_nnz(n_v, 0);
    std::vector<std::vector<u32>> row_cols(n_v);

#pragma omp parallel for schedule(guided)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n_v); ++ii) {
        const u32 i = static_cast<u32>(ii);
        auto r = VP_csr.row(i);

        // Sparse accumulator (SPA) with stamp-based marking
        std::vector<int> mark(n_v, -1);
        std::vector<u32> cols;
        cols.reserve(128);

        for (size_t p = 0; p < r.cols.size(); ++p) {
            const u32 k = r.cols[p];
            if (skip[k]) continue;

            auto colk = VP_csc.col(k);
            for (size_t q = 0; q < colk.rows.size(); ++q) {
                const u32 j = colk.rows[q];
                if (mark[j] != static_cast<int>(i)) {
                    mark[j] = static_cast<int>(i);
                    cols.push_back(j);
                }
            }
        }
        
        if (!cols.empty()) {
            std::sort(cols.begin(), cols.end());
        }
        row_nnz[i] = cols.size();
        row_cols[i] = std::move(cols);
    }

    // Build CSR structure for H_add
    CSRMatrix ADD;
    ADD.n_rows = n_v;
    ADD.n_cols = n_v;
    ADD.row_ptrs.resize(n_v + 1);
    
    size_t total_nnz = 0;
    for (u32 i = 0; i < n_v; ++i) {
        ADD.row_ptrs[i] = total_nnz;
        total_nnz += row_nnz[i];
    }
    ADD.row_ptrs[n_v] = total_nnz;
    
    ADD.col_indices.resize(total_nnz);
    ADD.values.assign(total_nnz, 0.0);

    // Copy patterns into col_indices
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n_v); ++ii) {
        const u32 i = static_cast<u32>(ii);
        const size_t s = ADD.row_ptrs[i];
        std::copy(row_cols[i].begin(), row_cols[i].end(), 
                  ADD.col_indices.begin() + static_cast<std::ptrdiff_t>(s));
    }

    // ========================================================================
    // Phase 2: NUMERIC - Accumulate values
    // ========================================================================
#pragma omp parallel for schedule(guided)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n_v); ++ii) {
        const u32 i = static_cast<u32>(ii);
        const size_t row_start = ADD.row_ptrs[i];
        const size_t row_end   = ADD.row_ptrs[i + 1];
        const size_t row_len   = row_end - row_start;

        if (row_len == 0) continue;

        // Build position map for this row
        std::vector<int> pos(n_v, -1);
        std::vector<u32> touched;
        touched.reserve(row_len);

        for (size_t t = 0; t < row_len; ++t) {
            const u32 j = ADD.col_indices[row_start + t];
            pos[j] = static_cast<int>(row_start + t);
            touched.push_back(j);
        }

        // Accumulate: H_add[i,j] += sum_k H_VP[i,k] * D^{-1}[k] * H_VP[j,k]
        auto r = VP_csr.row(i);
        for (size_t p = 0; p < r.cols.size(); ++p) {
            const u32 k = r.cols[p];
            if (skip[k]) continue;

            const double a = r.vals[p];     // H_VP(i,k)
            if (a == 0.0) continue;

            const double s = dinv[k];       // D^{-1}(k,k)
            if (s == 0.0) continue;

            const double as = a * s;
            auto colk = VP_csc.col(k);
            for (size_t q = 0; q < colk.rows.size(); ++q) {
                const u32 j = colk.rows[q];
                const double b = colk.vals[q];  // H_VP(j,k)
                const int at = pos[j];
                if (at >= 0) {
                    ADD.values[static_cast<size_t>(at)] += as * b;
                }
            }
        }

        // Reset position map
        for (u32 j : touched) pos[j] = -1;
    }

    // ========================================================================
    // Finalize: H_eff = H_VV + H_add
    // ========================================================================
    CSRMatrix VV = coo_to_csr(H_VV_in, n_v, n_v);
    CSRMatrix EFF = csr_add(VV, ADD, MAT_ELEMENT_THRESH);

    COOMatrix H_eff = csr_to_coo(EFF);
    H_eff.n_rows = n_v;
    H_eff.n_cols = n_v;
    return H_eff;
}

} // namespace detnqs
