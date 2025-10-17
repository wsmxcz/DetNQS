// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_conn.hpp
 * @brief Hamiltonian connectivity builders (COO) for S, C, and T = S ∪ C.
 *
 * Design:
 * - Deterministic COO output: sorted by (row, col), duplicates merged.
 * - Thread-parallel via OpenMP (optional).
 * - <S|H|S> / <S|H|T> include diagonals by default (full COO blocks).
 * - <S|H|C> has no对角元（行在 S，列在 C）。
 *
 * Heat-bath integration (doubles only):
 * - Singles: always enumerate fully (no HB pruning).
 * - Doubles to C: if `use_heatbath && hb_table`, query HB rows and keep pairs
 *   with weight >= eps1; otherwise fall back to full enumeration.
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

/** Non-zero Hamiltonian entry in COO format. */
struct Conn {
    u32    row = 0;   ///< row index (S)
    u32    col = 0;   ///< col index (S, C, or T depending on builder)
    double val = 0.0; ///< H(row, col)

    Conn() = default;
    Conn(u32 r, u32 c, double v) : row(r), col(c), val(v) {}
};

/** Build options (hierarchy: HB build threshold << eps1 << final |H_ij| thresh). */
struct BuildOpts {
    double thresh       = 1e-15;  ///< Drop entries with |H_ij| <= thresh (post-eval)
    double eps1         = 1e-6;   ///< Heat-bath cutoff for doubles (|<ij||ab>| >= eps1)
    bool   use_heatbath = true;   ///< Enable HB pruning for doubles to C (if table is provided)
};

/** Result of ⟨S|H|S⟩ (diagonals included). */
struct SSResult {
    std::vector<Conn> coo;  ///< sorted, merged
};

/** Result of ⟨S|H|C⟩ (no diagonals). */
struct SCResult {
    std::vector<Conn> coo;  ///< sorted, merged
    DetMap            map_C; ///< deterministic C mapping (C ∩ S = ∅)
};

/** Result of ⟨S|H|T⟩ where T = S ++ C (S prefix preserved). */
struct STResult {
    std::vector<Conn> coo;  ///< sorted, merged
    DetMap            map_T; ///< T determinants with S as prefix
    std::size_t       size_S{}; ///< boundary: [0,size_S) is S, [size_S,|T|) is C
};

/** Optional: single-pass build of SS & SC blocks together. */
struct SSSCResult {
    std::vector<Conn> coo_SS; ///< includes diagonals
    std::vector<Conn> coo_SC; ///< no diagonals
    DetMap            map_C;
};

// -----------------------------------------------------------------------------
// Diagonal-only API
// -----------------------------------------------------------------------------

/**
 * @brief Compute diagonal matrix elements <D|H|D> for a list of determinants.
 * @param dets Determinants (any order).
 * @param ham  Hamiltonian evaluator (Slater–Condon).
 * @return Vector of <D|H|D>, same order as input.
 */
[[nodiscard]] std::vector<double> get_ham_diag(
    std::span<const Det> dets,
    const HamEval& ham
);

// -----------------------------------------------------------------------------
// Connectivity builders (COO)
// -----------------------------------------------------------------------------

/**
 * @brief Construct ⟨S|H|S⟩ COO (diagonals included；variational block).
 *        Always uses full enumeration for singles & doubles within S.
 */
[[nodiscard]] SSResult get_ham_SS(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const BuildOpts& opts = {}
);

/**
 * @brief Construct ⟨S|H|C⟩ COO with C built from S.
 *        Singles: full；Doubles: HB-pruned if enabled (else full).
 *        (No diagonals because rows∈S, cols∈C).
 *
 * @throws std::invalid_argument if use_heatbath=true but hb_table==nullptr
 */
[[nodiscard]] SCResult get_ham_SC(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,  // may be nullptr if not using HB
    const BuildOpts& opts = {}
);

/**
 * @brief Construct ⟨S|H|T⟩ COO with T = S ++ C (S prefix preserved).
 *        SS block (within T) includes diagonals；SC block follows SC rules.
 *
 * @throws std::invalid_argument if use_heatbath=true but hb_table==nullptr
 */
[[nodiscard]] STResult get_ham_ST(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,  // may be nullptr if not using HB
    const BuildOpts& opts = {}
);

/**
 * @brief Single-pass constructor for ⟨S|H|S⟩ and ⟨S|H|C⟩ together.
 *        SS includes diagonals；SC as in get_ham_SC.
 */
[[nodiscard]] SSSCResult get_ham_SS_SC(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    const BuildOpts& opts = {}
);

} // namespace lever