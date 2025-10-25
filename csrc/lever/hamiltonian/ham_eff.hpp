// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eff.hpp
 * @brief Effective Hamiltonian assembly via sparse outer products.
 *
 * Computes H_eff = H_SS + H_SC·D⁻¹·H_CS where D_jj = E_ref - H_CC[j,j].
 * Uses column-wise scaled outer products for efficient SpGEMM.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <lever/hamiltonian/ham_utils.hpp>
#include <cstdint>
#include <span>

namespace lever {

/**
 * Denominator regularization strategies.
 *
 * LinearShift: d_inv = 1/(d + ε·sign(d))  - Sharp cutoff
 * Sigma:       d_inv = d/(d² + ε²)        - Smooth Tikhonov (default)
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
    bool upper_only = true;       // Build upper triangle only (exploit symmetry)
};

/**
 * Assemble effective Hamiltonian H_eff = H_SS + H_SC·D⁻¹·H_CS.
 *
 * Algorithm: Column-wise outer products ΔH = Σⱼ (1/D_jj)·b_j⊗b_jᵀ
 * where b_j is the j-th column of H_SC.
 *
 * Input requirements:
 *   - H_SS: Full symmetric matrix, sorted by (row, col)
 *   - H_SC: Sorted by (row, col)
 *   - h_cc_diag[j]: Diagonal elements matching H_SC column indices
 *
 * Threading: OpenMP parallel over C-space columns with guided scheduling.
 *
 * @param H_SS      ⟨S|H|S⟩ block (full symmetric)
 * @param H_SC      ⟨S|H|C⟩ block
 * @param h_cc_diag H_CC diagonal elements
 * @param e_ref     Reference energy for D_jj = E_ref - H_CC[j,j]
 * @param config    Assembly configuration
 * @return          Assembled H_eff (full symmetric, sorted, merged)
 */
[[nodiscard]] COOMatrix get_ham_eff(
    const COOMatrix& H_SS,
    const COOMatrix& H_SC,
    std::span<const double> h_cc_diag,
    double e_ref,
    const HeffConfig& config = {}
);

} // namespace lever
