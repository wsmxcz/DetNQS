// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file build_ham.hpp
 * @brief Unified streaming Hamiltonian builders (<S|H|S> and <S|H|C>) with policies.
 *
 * Core ideas:
 * - Single streaming kernel + policies (KnownSets / StaticHB / DynamicAmp).
 * - Deferred indexing for discovered C (deterministic, lexicographic).
 * - Deterministic output: COO entries sorted by (row, col); duplicates merged.
 * - Singles always enumerated; doubles optionally Heat-bath pruned.
 * - Optional post-filter for singles (used by DynamicAmp: |H_ik * psi_i| >= eps1).
 *
 * Threading:
 * - OpenMP parallelism over bra rows (optional).
 * - Thread-local sinks; merged & canonicalized at finalize.
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/integral/hb_table.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace lever {

using u32 = std::uint32_t;

/** Non-zero Hamiltonian entry in COO format. */
struct Conn {
    u32    row = 0;   ///< row index in S
    u32    col = 0;   ///< col index in S or C (depending on block)
    double val = 0.0; ///< H(row, col)

    Conn() = default;
    Conn(u32 r, u32 c, double v) : row(r), col(c), val(v) {}
};

/** Unified result for SS and SC blocks. */
struct SSSCResult {
    std::vector<Conn> coo_SS; // <S|H|S> (diagonals included)
    std::vector<Conn> coo_SC; // <S|H|C> (no diagonal by construction)
    DetMap            map_C;  // C-space mapping (deterministic)
};

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


/**
 * @brief Build <S|H|S> and (optionally) <S|H|C> with S and C provided.
 *
 * Behavior:
 *  - SS: Diagonal + all single/double connections that lie within S (full enumeration).
 *  - SC: Only connections from S to the provided C set (no discovery).
 *  - No entries outside S or C are kept.
 *
 * Determinism:
 *  - Keeps the exact input order of S and C when building maps.
 *
 * @param dets_S  S determinants (row/column space for SS; row space for SC)
 * @param dets_C  Optional C determinants; if std::nullopt, SC will be empty
 * @param ham     Hamiltonian evaluator (Slater–Condon)
 * @param n_orb   Number of spatial orbitals
 * @param thresh  Drop entries with |H_ij| <= thresh
 */
[[nodiscard]] SSSCResult get_ham_block(
    std::span<const Det> dets_S,
    std::optional<std::span<const Det>> dets_C,
    const HamEval& ham,
    int n_orb,
    double thresh = 1e-15
);

/**
 * @brief Build <S|H|S> and discover C via static Heat-bath screening on doubles.
 *
 * Behavior:
 *  - SS: Diagonal + full singles/doubles within S (variational completeness).
 *  - SC: Singles are post-evaluated (no HB pre-prune), doubles from HB rows with eps1.
 *  - C is discovered on the fly and deterministically indexed via lexical order.
 *
 * @param dets_S      S determinants (rows)
 * @param ham         Hamiltonian evaluator
 * @param n_orb       Number of spatial orbitals
 * @param hb_table    Heat-bath table (required if use_heatbath=true)
 * @param eps1        Heat-bath cutoff for doubles (|<ij||ab>| >= eps1)
 * @param use_heatbath If false, doubles fall back to full enumeration
 * @param thresh      Drop entries with |H_ij| <= thresh
 */
[[nodiscard]] SSSCResult get_ham_conn(
    std::span<const Det> dets_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1 = 1e-6,
    bool use_heatbath = true,
    double thresh = 1e-15
);

/**
 * @brief Build <S|H|S> and discover C via dynamic amplitude screening.
 *
 * Behavior:
 *  - SS: Diagonal + full singles/doubles within S.
 *  - SC: Doubles via per-row HB cutoff tau_i = eps1 / max(|psi_S[i]|, delta).
 *        Singles are post-filtered by |H_ik * psi_S[i]| >= eps1.
 *  - Deterministic deferred indexing for discovered C.
 *
 * @param dets_S   S determinants (rows)
 * @param psi_S    Amplitudes aligned with dets_S (|S|)
 * @param ham      Hamiltonian evaluator
 * @param n_orb    Number of spatial orbitals
 * @param hb_table Heat-bath table (required)
 * @param eps1     Dynamic criterion: keep if |H * psi| >= eps1
 * @param thresh   Drop entries with |H_ij| <= thresh (after evaluation)
 */
[[nodiscard]] SSSCResult get_ham_conn_amp(
    std::span<const Det> dets_S,
    std::span<const double> psi_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1 = 1e-6,
    double thresh = 1e-15
);

} // namespace lever