// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file build_ham.hpp
 * @brief Streaming Hamiltonian builders for ⟨S|H|S⟩ and ⟨S|H|C⟩ blocks.
 *
 * Construction policies:
 *  - KnownSets:  Pre-defined S and C spaces
 *  - StaticHB:   Heat-Bath screening with fixed ε₁ cutoff
 *  - DynamicAmp: Amplitude-weighted screening τᵢ = ε₁/|ψ_S[i]|
 *
 * Output: Deterministic sorted COO with merged duplicates.
 * Threading: OpenMP parallelism with thread-local sinks.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/hamiltonian/ham_utils.hpp>
#include <lever/integral/hb_table.hpp>

#include <optional>
#include <span>
#include <vector>

namespace lever {

/**
 * Compute diagonal elements ⟨D|H|D⟩ for all determinants.
 *
 * @param dets Determinant list
 * @param ham  Hamiltonian evaluator
 * @return Vector of diagonal elements aligned with input
 */
[[nodiscard]] std::vector<double> get_ham_diag(
    std::span<const Det> dets,
    const HamEval& ham
);

/**
 * Build H_SS block via full S-space enumeration.
 *
 * Enumerates: H_SS = diag + singles + doubles within S.
 *
 * @param dets_S S determinants
 * @param ham    Hamiltonian evaluator
 * @param n_orb  Number of spatial orbitals
 * @return COO matrix for H_SS
 */
[[nodiscard]] COOMatrix get_ham_ss(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb
);

/**
 * Build H_SS and H_SC with pre-defined determinant spaces.
 *
 * Policy: KnownSets - compute ⟨S|H|S⟩ and ⟨S|H|C⟩ for given C.
 *
 * @param dets_S S determinants
 * @param dets_C Optional C determinants; nullopt → empty H_SC
 * @param ham    Hamiltonian evaluator
 * @param n_orb  Number of spatial orbitals
 * @return HamBlocks{H_SS, H_SC, C_index_map}
 */
[[nodiscard]] HamBlocks get_ham_block(
    std::span<const Det> dets_S,
    std::optional<std::span<const Det>> dets_C,
    const HamEval& ham,
    int n_orb
);

/**
 * Build H_SS and discover C via static Heat-Bath screening.
 *
 * Policy: StaticHB
 *  - Doubles: Keep if |⟨ij||ab⟩| ≥ ε₁ via Heat-Bath table
 *  - Singles: Post-evaluate and keep above numerical threshold
 *  - C determinants indexed lexicographically
 *
 * @param dets_S       S determinants
 * @param ham          Hamiltonian evaluator
 * @param n_orb        Number of spatial orbitals
 * @param hb_table     Heat-Bath table (required if use_heatbath=true)
 * @param eps1         Heat-Bath cutoff threshold
 * @param use_heatbath Enable Heat-Bath pruning
 * @return HamBlocks{H_SS, H_SC, C_index_map}
 */
[[nodiscard]] HamBlocks get_ham_conn(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1 = 1e-6,
    bool use_heatbath = true
);

/**
 * Build H_SS and discover C via dynamic amplitude-weighted screening.
 *
 * Policy: DynamicAmp - per-row adaptive cutoff
 *  - Doubles: Keep if |⟨ij||ab⟩| ≥ τᵢ where τᵢ = ε₁/max(|ψ_S[i]|, δ)
 *  - Singles: Keep if |H_ik·ψ_S[i]| ≥ ε₁
 *  - Favors important configurations with large amplitudes
 *
 * @param dets_S   S determinants
 * @param psi_S    Amplitudes aligned with dets_S
 * @param ham      Hamiltonian evaluator
 * @param n_orb    Number of spatial orbitals
 * @param hb_table Heat-Bath table (required)
 * @param eps1     Amplitude criterion threshold
 * @return HamBlocks{H_SS, H_SC, C_index_map}
 */
[[nodiscard]] HamBlocks get_ham_conn_amp(
    std::span<const Det> dets_S,
    std::span<const double> psi_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1 = 1e-6
);

} // namespace lever