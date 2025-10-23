// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Effective Hamiltonian assembly via sparse outer products.
 *
 * Implements H_eff = H_SS + H_SC·D⁻¹·H_CS where D_jj = E_ref - H_CC[j,j].
 * Uses column-wise scaled outer products for efficient SpGEMM computation.
 *
 * File: lever/hamiltonian/ham_eff.hpp
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: January, 2025
 */

#pragma once

#include <lever/hamiltonian/build_ham.hpp>
#include <cstdint>
#include <span>
#include <vector>

namespace lever {

using u32 = std::uint32_t;

/**
 * Compressed Sparse Column format for column-wise access patterns.
 * Each column stores sorted, unique row indices with corresponding values.
 */
struct CSCMatrix {
    std::vector<u32>    row_indices;  // Row index per non-zero
    std::vector<double> values;       // Value per non-zero
    std::vector<size_t> col_ptrs;     // Column start indices (size = n_cols + 1)
    u32 n_rows = 0;
    u32 n_cols = 0;

    struct ColView {
        std::span<const u32>    rows;
        std::span<const double> vals;
    };

    [[nodiscard]] ColView col(u32 j) const noexcept {
        const size_t start = col_ptrs[j];
        const size_t end   = col_ptrs[j + 1];
        return {
            {row_indices.data() + start, end - start},
            {values.data() + start, end - start}
        };
    }
};

/**
 * Regularization strategies for small denominators D_jj = E_ref - H_CC[j,j].
 *
 * LinearShift: d_inv = 1/(d + ε·sign(d))  [Sharp transition at zero]
 * Sigma:       d_inv = d/(d² + ε²)         [Smooth Tikhonov-style, recommended]
 */
enum class Regularization : uint8_t {
    LinearShift,
    Sigma
};

/**
 * Assembly configuration parameters.
 */
struct HeffConfig {
    Regularization reg_type = Regularization::Sigma;
    double epsilon = 1e-12;      // Regularization parameter
    double thresh = 1e-15;       // Drop entries |H_ij| < thresh
    double micro_thresh = 1e-18; // Skip outer product terms < micro_thresh
    bool upper_only = true;     // Compute upper triangle only, then mirror
};

/**
 * Assembly result with diagnostic information.
 */
struct HeffResult {
    std::vector<Conn> coo_heff;   // Sorted, merged effective Hamiltonian
    u32 n_rows_S = 0;             // S-space dimension

    // Diagnostics for numerical stability monitoring
    size_t nnz_correction = 0;    // Non-zeros in ΔH before merge
    size_t n_regularized = 0;     // Columns with |D_jj| < epsilon
    double max_abs_d = 0.0;       // Largest |D_jj| encountered
    double max_abs_d_inv = 0.0;   // Largest |D_jj⁻¹| after regularization
    double max_correction = 0.0;  // Largest |ΔH_ij| magnitude
};

/**
 * Assemble effective Hamiltonian from SS/SC blocks.
 *
 * Computes: H_eff = H_SS + ΔH where ΔH = H_SC·D⁻¹·H_CS
 * Algorithm: Column-wise outer products ΔH = Σ_j (1/D_jj)·b_j⊗b_j
 *
 * Input requirements:
 *   - blocks.coo_SS: Full symmetric matrix, sorted by (row,col)
 *   - blocks.coo_SC: Sorted by (row,col)
 *   - h_cc_diag[j]: Diagonal elements matching blocks.map_C order
 *
 * Threading: OpenMP parallel over C-space columns (guided scheduling).
 *
 * @param blocks      Pre-built H_SS and H_SC matrices
 * @param h_cc_diag   H_CC diagonal elements
 * @param e_ref       Reference energy for denominator D_jj = E_ref - H_CC[j,j]
 * @param config      Assembly configuration
 * @return            Assembled H_eff with diagnostic metadata
 */
[[nodiscard]] HeffResult get_ham_eff(
    const SSSCResult& blocks,
    std::span<const double> h_cc_diag,
    double e_ref,
    const HeffConfig& config = {}
);

/**
 * Mirror upper triangle to full symmetric matrix.
 * Creates A_full = A_upper + A_upper^T by duplicating off-diagonal entries.
 *
 * Note: Output is NOT sorted - call sort_and_merge_coo() afterward if needed.
 */
[[nodiscard]] std::vector<Conn> mirror_upper_to_full(
    std::span<const Conn> upper_coo
);

} // namespace lever
