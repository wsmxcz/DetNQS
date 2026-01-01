// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eff.hpp
 * @brief Effective Hamiltonian assembly via Gustavson SpGEMM.
 *
 * Computes H_eff = H_SS + H_SC·D⁻¹·H_CS where D_jj = E_ref - H_CC[j,j].
 * Uses CSR×CSC row-wise accumulation for efficient sparse multiplication.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <detnqs/hamiltonian/ham_utils.hpp>
#include <cstdint>
#include <span>

namespace detnqs {

/**
 * Denominator regularization strategies.
 *
 * LinearShift: 1/(d + ε·sgn(d))  - Sharp cutoff
 * Sigma:       d/(d² + ε²)       - Smooth Tikhonov regularization
 */
enum class Regularization : uint8_t {
    LinearShift,
    Sigma
};

/**
 * Effective Hamiltonian assembly configuration.
 */
struct HeffConfig {
    Regularization reg_type = Regularization::Sigma;
    double epsilon = 1e-12;       // Regularization parameter ε
    bool upper_only = false;      // Reserved for compatibility (unused)
};

/**
 * Assemble effective Hamiltonian H_eff = H_SS + H_SC·D⁻¹·H_CS.
 *
 * Algorithm: Two-phase Gustavson SpGEMM
 *   Phase 1 - Symbolic: Pattern discovery via sparse accumulator (SPA)
 *   Phase 2 - Numeric:  Row-wise accumulation with column screening
 *
 * Input requirements:
 *   - H_SS:      ⟨S|H|S⟩ block (can be unsorted COO)
 *   - H_SC:      ⟨S|H|C⟩ block (can be unsorted COO)
 *   - h_cc_diag: Diagonal elements H_CC[j,j]
 *   - e_ref:     Reference energy for D_jj = E_ref - H_CC[j,j]
 *
 * Threading: OpenMP parallelized over S-space rows with guided scheduling.
 *
 * @return H_eff in COO format (full matrix, deduped, sorted)
 */
[[nodiscard]] COOMatrix get_ham_eff(
    const COOMatrix& H_SS,
    const COOMatrix& H_SC,
    std::span<const double> h_cc_diag,
    double e_ref,
    const HeffConfig& config = {}
);

} // namespace detnqs
