// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eff.hpp
 * @brief Effective Hamiltonian assembly via sparse outer products.
 *
 * Computes H_eff = H_SS + H_SC·D⁻¹·H_CS where D_jj = E_ref - H_CC[j,j].
 * Uses column-wise scaled outer products for efficient SpGEMM.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: January, 2025
 */

#pragma once

#include <lever/hamiltonian/ham_utils.hpp>
#include <cstdint>
#include <span>

namespace lever {

/**
 * Regularization strategies for small denominators.
 *
 * LinearShift: d_inv = 1/(d + ε·sign(d))  - Sharp transition at zero
 * Sigma:       d_inv = d/(d² + ε²)         - Smooth Tikhonov-style (recommended)
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
    double epsilon = 1e-12;       // Regularization parameter
    bool upper_only = true;       // Compute upper triangle, then mirror
};

/**
 * Assemble effective Hamiltonian from SS/SC blocks.
 *
 * Computes: H_eff = H_SS + ΔH where ΔH = H_SC·D⁻¹·H_CS
 * Algorithm: Column-wise outer products ΔH = Σ_j (1/D_jj)·b_j⊗b_j
 *
 * Input requirements:
 *   - H_SS: Full symmetric matrix, sorted by (row,col)
 *   - H_SC: Sorted by (row,col)
 *   - h_cc_diag[j]: Diagonal elements in same order as H_SC columns
 *
 * Threading: OpenMP parallel over C-space columns (guided scheduling)
 *
 * @param H_SS      <S|H|S> block
 * @param H_SC      <S|H|C> block
 * @param h_cc_diag H_CC diagonal elements
 * @param e_ref     Reference energy for D_jj = E_ref - H_CC[j,j]
 * @param config    Assembly configuration
 * @return          Assembled H_eff (full, sorted, merged)
 */
[[nodiscard]] COOMatrix get_ham_eff(
    const COOMatrix& H_SS,
    const COOMatrix& H_SC,
    std::span<const double> h_cc_diag,
    double e_ref,
    const HeffConfig& config = {}
);

} // namespace lever
