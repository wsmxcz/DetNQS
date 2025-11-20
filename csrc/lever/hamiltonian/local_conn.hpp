// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.hpp
 * @brief Local Hamiltonian connectivity generators and streaming energy evaluation.
 *
 * Provides tools to generate Hamiltonian rows on-the-fly and compute
 * variational energies without full matrix construction.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <complex>
#include <span>
#include <vector>

#include <lever/determinant/det.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/integral/hb_table.hpp>

namespace lever {

/**
 * Connectivity row for a single determinant |bra⟩.
 * Contains list of connected |ket⟩ and matrix elements H_bra,ket.
 */
struct LocalConnRow {
    std::vector<Det> dets;
    std::vector<double> values;
};

/**
 * Batch connectivity in CSR-like format (offsets, flat arrays).
 */
struct LocalConnBatch {
    std::vector<int> offsets;
    std::vector<Det> dets;
    std::vector<double> values;
};

/**
 * Result of exact variational energy calculation.
 */
struct VariationalResult {
    double e_el;  ///< Electronic energy <Psi|H|Psi>
    double norm;  ///< Squared norm <Psi|Psi>
};

// ─── Connectivity Generators ──────────────────────────────────────────

/**
 * Generate Hamiltonian row for a single determinant.
 *
 * @param bra           Reference determinant
 * @param ham           Hamiltonian evaluator
 * @param n_orb         Number of orbitals
 * @param hb_table      Heat-bath table (optional)
 * @param eps1          Screening threshold
 * @param use_heatbath  Enable HB screening for doubles
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
 * Generate Hamiltonian connections for a batch of determinants.
 * Parallelized with OpenMP.
 */
[[nodiscard]] LocalConnBatch get_local_connections(
    std::span<const Det> samples,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

// ─── Streaming Energy Evaluation ──────────────────────────────────────

/**
 * Compute <Psi|H|Psi> and <Psi|Psi> on a fixed basis.
 *
 * Streaming algorithm:
 * 1. Build DetMap for O(1) index lookup.
 * 2. Iterate bra in basis (parallel).
 * 3. Generate connected kets on-the-fly.
 * 4. Accumulate E += conj(c_bra) * H_bra,ket * c_ket.
 *
 * @param basis       Basis determinants (S U C)
 * @param coeffs      Coefficients aligned with basis
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