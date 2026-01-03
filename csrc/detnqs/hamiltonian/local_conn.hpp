// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.hpp
 * @brief Local Hamiltonian connectivity and variational energy computation.
 * 
 * Provides deterministic Hamiltonian action on Fock states and energy
 * evaluation for real-valued wavefunctions. Core functionality includes:
 *  - Local connectivity generation via single/double excitations
 *  - Variational energy <Psi_V | H | Psi_V> on fixed basis
 *  - EN-PT2 correction from external (perturbative) space
 *
 * Author: Zheng (Alex) Che, wsmxcz@gmail.com
 * Date: December, 2025
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include <detnqs/determinant/det.hpp>
#include <detnqs/hamiltonian/ham_eval.hpp>
#include <detnqs/integral/hb_table.hpp>

namespace detnqs {

/** Single-determinant connectivity: |x'> coupled to |x> via H. */
struct LocalConnRow {
    std::vector<Det> dets;      // Connected configurations |x'>
    std::vector<double> values; // Matrix elements <x'|H|x>
};

/** Batch connectivity in CSR-like format for efficient SpMV. */
struct LocalConnBatch {
    std::vector<int> offsets;   // Row offsets (length = n_dets + 1)
    std::vector<Det> dets;      // All connected configurations
    std::vector<double> values; // Corresponding matrix elements
};

/**
 * Generate local Hamiltonian connectivity for a single determinant.
 *
 * @param bra        Input Fock state |x>
 * @param ham        Hamiltonian evaluator (1-/2-body integrals)
 * @param n_orb      Number of spin-orbitals
 * @param hb_table   Optional heat-bath screening table
 * @param eps1       Screening threshold for matrix elements
 * @param use_heatbath Enable heat-bath-driven excitation generation
 * @return LocalConnRow containing all |x'> where <x'|H|x> != 0
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
 * Batch version of get_local_conn for multiple determinants.
 *
 * @param samples  Input Fock basis subset
 * @return LocalConnBatch in CSR format for matrix-free SpMV
 */
[[nodiscard]] LocalConnBatch get_local_connections(
    std::span<const Det> samples,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

/** EN-PT2 correction computed from variational-space wavefunction. */
struct Pt2Result {
    double e_pt2;       // Second-order energy correction Delta E_PT2
    std::size_t n_ext;  // Number of unique external configurations in P_k
};

/**
 * Compute variational energy <Psi_V | H | Psi_V> on fixed basis.
 *
 * @param basis   Variational set V_k (Fock states)
 * @param coeffs  Wavefunction amplitudes psi_theta(x), normalized over V_k
 * @return Energy numerator (unnormalized if coeffs not normalized)
 *
 * Note: Normalization is handled in Python driver.
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
 * Compute EN-PT2 correction from perturbative space P_k.
 *
 * @param V_k       Variational set (current basis)
 * @param coeffs_V  Wavefunction amplitudes on V_k (normalized)
 * @param e_ref     Reference variational energy (from optimization)
 * @return Pt2Result containing Delta E_PT2 and external space size
 *
 * The correction is computed via:
 *   Delta E_PT2 = sum_{x in P_k} |<x|H|Psi_V>|^2 / (E_ref - <x|H|x>)
 *
 * Note: coeffs_V must be normalized in Python before calling this function.
 */
[[nodiscard]] Pt2Result compute_pt2(
    std::span<const Det> V_k,
    std::span<const double> coeffs_V,
    const HamEval& ham,
    int n_orb,
    double e_ref,
    const HeatBathTable* hb_table = nullptr,
    double eps1 = 1e-6,
    bool use_heatbath = false
);

} // namespace detnqs
