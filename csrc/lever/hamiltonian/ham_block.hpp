// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_block.hpp
 * @brief Hamiltonian matrix block builders for arbitrary determinant batches.
 *
 * Design philosophy:
 * - Explicit excitation space generation (vs. ham_conn's streaming approach).
 * - Two-phase workflow: (1) generate full excited space, (2) compute matrix blocks.
 * - Use case: ML-driven selection where external logic filters excited space before matrix eval.
 * - Thread-parallel via OpenMP.
 * - Diagonal elements always included when bra == ket.
 *
 * Differences from ham_conn:
 * - ham_conn: Streaming construction, S and C spaces coupled, optimized for FCIQMC workflow.
 * - ham_block: Decoupled excitation generation + block evaluation, optimized for ML pipelines.
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/integral/hb_table.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace lever {

using u32 = std::uint32_t;

// Forward declaration from ham_conn.hpp for COO entries
struct Conn;

/** Options for excitation generation. */
struct ExcitationOpts {
    double eps1         = 1e-6;   ///< Heat-bath cutoff for doubles (|<ij||ab>| >= eps1)
    bool   use_heatbath = false;  ///< Enable HB screening for double excitations
};

/** Options for block matrix evaluation. */
struct BlockOpts {
    double thresh = 1e-15;  ///< Drop entries with |H_ij| <= thresh (post-evaluation)
};

/**
 * @brief Generate all single and double excitations from a reference space.
 *
 * Behavior:
 *   - Singles: Always generated via full enumeration
 *   - Doubles: Full enumeration if use_heatbath=false, screened by eps1 if use_heatbath=true
 *
 * Complexity:
 *   - Singles: O(|refs| * n_orb * n_elec)
 *   - Doubles (full): O(|refs| * n_orb^4)
 *   - Doubles (HB): O(|refs| * n_elec^2 * n_hb_avg) where n_hb_avg ~ 10-100
 *
 * Memory: O(|excited_space|), typically 10-1000x larger than |refs|
 *
 * @param refs        Reference determinants (any order)
 * @param n_orb       Number of spatial orbitals
 * @param hb_table    Heat-bath table (required if use_heatbath=true, else nullptr)
 * @param opts        Generation options (heat-bath mode and threshold)
 *
 * @return DetMap of all generated excitations (canonicalized, duplicates removed)
 *
 * @throws std::invalid_argument if use_heatbath=true but hb_table==nullptr
 *
 * Notes:
 * - Output excludes input refs (pure excitation space).
 * - Thread-safe for concurrent calls with different inputs.
 */
[[nodiscard]] DetMap generate_excitations(
    std::span<const Det> refs,
    int n_orb,
    const HeatBathTable* hb_table = nullptr,
    const ExcitationOpts& opts = {}
);

/**
 * @brief Compute Hamiltonian matrix block <bra|H|ket> in COO format.
 *
 * Algorithm: For each bra determinant, generate all connected determinants
 * (diagonal + singles + doubles) and check if they exist in ket space.
 * Only evaluates matrix elements for actual connections (exploits sparsity).
 *
 * Complexity: O(|bra| * (n_orb^2 + n_orb^4) * ket_lookup)
 *             vs. naive O(|bra| * |ket|) when using connection-based approach
 *
 * Memory: O(nnz + n_threads * avg_row_nnz)
 *
 * @param bra         Row-space determinants (row indices in COO)
 * @param ket         Column-space determinants (column indices in COO)
 * @param ham         Hamiltonian evaluator (Slater-Condon rules)
 * @param n_orb       Number of spatial orbitals (needed for excitation generation)
 * @param opts        Evaluation options (sparsity threshold)
 *
 * @return COO matrix entries (sorted by row, col; duplicates merged)
 *
 * Notes:
 * - Diagonal elements included when bra[i] exists in ket.
 * - Exploits Hamiltonian sparsity: only single/double excitations have non-zero elements.
 * - Parallel evaluation across bra determinants.
 * - Thread-safe for concurrent calls.
 */
[[nodiscard]] std::vector<Conn> get_ham_block(
    std::span<const Det> bra,
    std::span<const Det> ket,
    const HamEval& ham,
    int n_orb,
    const BlockOpts& opts = {}
);

} // namespace lever
