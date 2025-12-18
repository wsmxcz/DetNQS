// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.hpp
 * @brief Local Hamiltonian connectivity and variational energy computation.
 *
 * Computes ⟨D|H|D'⟩ rows for individual determinants or batches.
 * Supports Heat-Bath screening: |⟨ij||ab⟩| ≥ ε₁.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#pragma once

#include <complex>
#include <span>
#include <vector>

#include <lever/determinant/det.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/integral/hb_table.hpp>

namespace lever {

/** Connectivity row for a single determinant. */
struct LocalConnRow {
    std::vector<Det> dets;
    std::vector<double> values;
};

/** Batch connectivity in CSR-like format. */
struct LocalConnBatch {
    std::vector<int> offsets;
    std::vector<Det> dets;
    std::vector<double> values;
};

/** Result of ⟨Ψ|H|Ψ⟩. */
struct VariationalResult {
    double e_el;
    double norm;
};

/**
 * Compute Hamiltonian row for a single determinant.
 *
 * Enumerates diagonal, singles, and doubles connections.
 * Doubles screening: |⟨ij||ab⟩| ≥ ε₁ via Heat-Bath table.
 *
 * @param bra Determinant for row index
 * @param ham Hamiltonian evaluator
 * @param n_orb Number of spatial orbitals
 * @param hb_table Heat-Bath table (optional)
 * @param eps1 Heat-Bath cutoff threshold
 * @param use_heatbath Enable Heat-Bath pruning
 * @return LocalConnRow{connected dets, matrix elements}
 */
[[nodiscard]] LocalConnRow get_local_conn(
    const Det& bra,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

/**
 * Compute Hamiltonian rows for a batch of determinants (parallel).
 *
 * OpenMP parallel with dynamic scheduling.
 *
 * @param samples Determinant batch
 * @param ham Hamiltonian evaluator
 * @param n_orb Number of spatial orbitals
 * @param hb_table Heat-Bath table (optional)
 * @param eps1 Heat-Bath cutoff threshold
 * @param use_heatbath Enable Heat-Bath pruning
 * @return LocalConnBatch{offsets, dets, values} in CSR-like format
 */
[[nodiscard]] LocalConnBatch get_local_connections(
    std::span<const Det> samples,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

/**
 * Compute variational energy ⟨Ψ|H|Ψ⟩ on a fixed determinant basis.
 *
 * Streaming algorithm: Σᵢ cᵢ* Σⱼ Hᵢⱼ cⱼ.
 * Uses DetMap for O(1) coefficient lookup.
 *
 * @param basis Determinant basis
 * @param coeffs Complex coefficients aligned with basis
 * @param ham Hamiltonian evaluator
 * @param n_orb Number of spatial orbitals
 * @param hb_table Heat-Bath table (optional)
 * @param eps1 Heat-Bath cutoff threshold
 * @param use_heatbath Enable Heat-Bath pruning
 * @return VariationalResult{e_el, norm}
 */
[[nodiscard]] VariationalResult compute_variational_energy(
    std::span<const Det> basis,
    std::span<const std::complex<double>> coeffs,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

} // namespace lever
