// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file build_ham.hpp
 * @brief Unified streaming Hamiltonian builders (<S|H|S> and <S|H|C>).
 *
 * Core features:
 * - Single streaming kernel + policy pattern (KnownSets/StaticHB/DynamicAmp)
 * - Deferred indexing for discovered C determinants (lexicographic order)
 * - Deterministic output: sorted COO entries with merged duplicates
 * - Singles always enumerated; doubles with optional Heat-bath pruning
 * - OpenMP parallelism over bra rows with thread-local sinks
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: January, 2025
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
 * Compute diagonal matrix elements <D|H|D>.
 *
 * @param dets Determinant list (any order)
 * @param ham  Hamiltonian evaluator
 * @return Vector of <D|H|D> in same order as input
 */
[[nodiscard]] std::vector<double> get_ham_diag(
    std::span<const Det> dets,
    const HamEval& ham
);

/**
 * Build H_SS only (for CI evaluations without SC block).
 *
 * Behavior:
 *  - Diagonal + full singles/doubles within S
 *  - No C-space discovery or SC interactions
 *
 * @param dets_S  S determinants
 * @param ham     Hamiltonian evaluator
 * @param n_orb   Number of spatial orbitals
 * @return COO matrix for H_SS
 */
[[nodiscard]] COOMatrix get_ham_ss(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb
);


/**
 * Build H_SS and H_SC with pre-defined S and C spaces.
 *
 * Behavior:
 *  - H_SS: Diagonal + all singles/doubles within S (full enumeration)
 *  - H_SC: Only connections from S to provided C (no discovery)
 *  - Preserves input ordering for S and C spaces
 *
 * @param dets_S  S determinants (row/column space for SS; row space for SC)
 * @param dets_C  Optional C determinants; if nullopt, H_SC is empty
 * @param ham     Hamiltonian evaluator
 * @param n_orb   Number of spatial orbitals
 * @param thresh  Drop entries with |H_ij| <= thresh
 * @return HamBlocks containing H_SS, H_SC, and C mapping
 */
[[nodiscard]] HamBlocks get_ham_block(
    std::span<const Det> dets_S,
    std::optional<std::span<const Det>> dets_C,
    const HamEval& ham,
    int n_orb
);

/**
 * Build H_SS and discover C via static Heat-bath screening.
 *
 * Behavior:
 *  - H_SS: Diagonal + full singles/doubles within S
 *  - H_SC: Doubles via Heat-bath with eps1 cutoff; singles post-evaluated
 *  - C discovered on-the-fly with deterministic lexical ordering
 *
 * @param dets_S      S determinants
 * @param ham         Hamiltonian evaluator
 * @param n_orb       Number of spatial orbitals
 * @param hb_table    Heat-bath table (required if use_heatbath=true)
 * @param eps1        Heat-bath cutoff |<ij||ab>| >= eps1
 * @param use_heatbath Enable/disable Heat-bath pruning
 * @param thresh      Drop entries with |H_ij| <= thresh
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
 * Build H_SS and discover C via dynamic amplitude screening.
 *
 * Behavior:
 *  - H_SS: Diagonal + full singles/doubles within S
 *  - H_SC: Per-row Heat-bath cutoff tau_i = eps1 / max(|psi_S[i]|, delta)
 *          Singles post-filtered by |H_ik * psi_S[i]| >= eps1
 *  - Deterministic deferred indexing for discovered C
 *
 * @param dets_S   S determinants
 * @param psi_S    Amplitudes aligned with dets_S
 * @param ham      Hamiltonian evaluator
 * @param n_orb    Number of spatial orbitals
 * @param hb_table Heat-bath table (required)
 * @param eps1     Dynamic criterion: keep if |H * psi| >= eps1
 * @param thresh   Drop entries with |H_ij| <= thresh
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
