// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eff.hpp
 * @brief Effective Hamiltonian assembly for EFFECTIVE mode.
 *
 * Computes H_eff = H_VV + H_VP·D^{-1}·H_PV, where D_jj = E_ref - H_PP[j,j].
 * The perturbative space P contribution is down-folded into the variational 
 * space V via Gustavson's CSR×CSC SpGEMM with row-wise accumulation.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#pragma once

#include <detnqs/hamiltonian/ham_utils.hpp>
#include <cstdint>
#include <span>

namespace detnqs {

/**
 * Denominator regularization for D_jj = E_ref - H_PP[j,j].
 *
 * LinearShift: 1/(d + ε·sgn(d))  - Sharp cutoff near zero
 * Sigma:       d/(d² + ε²)       - Smooth Tikhonov-style stabilization
 */
enum class Regularization : uint8_t {
    LinearShift,
    Sigma
};

/**
 * Configuration for effective Hamiltonian assembly.
 */
struct HeffConfig {
    Regularization reg_type = Regularization::Sigma;
    double epsilon = 1e-12;       // Regularization parameter ε
    bool upper_only = false;      // Compatibility placeholder (unused)
};

/**
 * Assemble H_eff = H_VV + H_VP·D^{-1}·H_PV for the variational subspace.
 *
 * Algorithm: Two-phase Gustavson SpGEMM
 *   Phase 1 (Symbolic): Discover nonzero pattern via sparse accumulator
 *   Phase 2 (Numeric):  Compute values using CSR×CSC row-wise accumulation
 *
 * Input requirements:
 *   - H_VV:      Hamiltonian block ⟨V|H|V⟩ (unsorted COO acceptable)
 *   - H_VP:      Hamiltonian block ⟨V|H|P⟩ (unsorted COO acceptable)
 *   - h_pp_diag: Diagonal elements H_PP[j,j] of perturbative block
 *   - e_ref:     Reference energy for denominator construction
 *
 * Threading: OpenMP parallelized over V-space rows (guided scheduling).
 *
 * @return H_eff in COO format (full matrix, deduplicated, sorted by row/col)
 */
[[nodiscard]] COOMatrix get_ham_eff(
    const COOMatrix& H_VV,
    const COOMatrix& H_VP,
    std::span<const double> h_pp_diag,
    double e_ref,
    const HeffConfig& config = {}
);

} // namespace detnqs
