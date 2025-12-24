// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.hpp
 * @brief Local Hamiltonian connectivity and variational energy computation.
 *
 * Real-only kernels:
 *  - coeffs are double (no complex)
 *  - PT2 is computed for a normalized wavefunction (handled internally)
 */

#pragma once

#include <cstddef>
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

[[nodiscard]] LocalConnRow get_local_conn(
    const Det& bra,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

[[nodiscard]] LocalConnBatch get_local_connections(
    std::span<const Det> samples,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

/** Result of EN-PT2 correction computed from S-space wavefunction. */
struct Pt2Result {
    double e_pt2;       // PT2 correction (Psi_S is normalized in Python)
    std::size_t n_ext;  // number of unique external dets accumulated
};

/**
 * Compute variational energy numerator <Psi|H|Psi> on a fixed basis (real-only).
 *
 * Note: coeffs must be domain-normalized in Python.
 */
[[nodiscard]] double compute_variational_energy(
    std::span<const Det> basis,
    std::span<const double> coeffs,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

/**
 * Compute EN-PT2 correction only (real-only).
 *
 * Note:
 *   - coeffs_S must be domain-normalized in Python.
 *   - e_ref is the optimized variational electronic energy from Python.
 */
[[nodiscard]] Pt2Result compute_pt2(
    std::span<const Det> S,
    std::span<const double> coeffs_S,
    const HamEval& ham,
    int n_orb,
    double e_ref,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

} // namespace lever