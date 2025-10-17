// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eval.hpp
 * @brief Hamiltonian matrix element evaluator using Slater–Condon rules.
 *
 * Design notes:
 * - Thread-safe, stateless w.r.t. determinants (holds a const ref to integrrals).
 * - Works in spin-orbital basis; even indices = alpha, odd indices = beta.
 * - Excitation analysis and Condon–Shortley phase are delegated to det_ops.
 * - Degree>2 excitations vanish by Slater–Condon orthogonality.
 *
 * References (formulas & notation):
 * - Slater–Condon rules overview: diagonal / single / double elements. 
 * - Antisymmetrized two-electron integrals for same-spin doubles.
 * - Physicist↔Chemist notation relation: (pq|rs) = <pr|qs>.
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_ops.hpp>
#include <lever/utils/bit_utils.hpp>

#include <lever/integral/integral_so.hpp> // Spin-orbital integrals: h1e_so, h2e_phys, h2e_anti

#include <vector>
#include <cstdint>

namespace lever {

/**
 * @class HamEval
 * @brief Thread-safe evaluator for Hamiltonian matrix elements <bra|H|ket>.
 *
 * Slater–Condon rules:
 *  - degree 0 (diagonal): sum_p h_pp + sum_{p<q} <pq||pq>
 *  - degree 1 (single):   h_ap + sum_{k in occ(ket)} <ak||pk>
 *  - degree 2 (double):   same-spin: <ab||ij>; mixed-spin: <ab|ij> (Physicist)
 *  - degree > 2:          0
 *
 * Notes:
 *  - The constant core energy term (if any, e.g. FCIDUMP e_core) is typically
 *    added outside the many-electron matrix builder; this class computes purely
 *    electronic contributions from one-/two-electron integrals.
 */
class HamEval {
public:
    /// Construct from a spin-orbital integral provider (read-only reference).
    explicit HamEval(const IntegralSO& so_ints) noexcept;

    /// Analyze excitation pattern and phase between determinants (delegates).
    [[nodiscard]] ExcInfo analyze_exc(const Det& bra, const Det& ket) const noexcept;

    /// Compute full matrix element <bra|H|ket>.
    [[nodiscard]] double compute_elem(const Det& bra, const Det& ket) const noexcept;

    /// Compute diagonal element <D|H|D> (fast path).
    [[nodiscard]] double compute_diagonal(const Det& det) const noexcept;

private:
    /// Compute single-excitation matrix element, assumes degree==1; uses ket occupancy.
    [[nodiscard]] double compute_single(const ExcInfo& info, const Det& ket) const noexcept;

    /// Compute double-excitation matrix element, assumes degree==2.
    [[nodiscard]] double compute_double(const ExcInfo& info) const noexcept;

    /// Spin-orbital index from MO index and spin (0: alpha, 1: beta).
    [[nodiscard]] static constexpr int so_from_mo(int mo_idx, int spin) noexcept {
        return (mo_idx << 1) | (spin & 1);
    }

    const IntegralSO& so_ints_; ///< spin-orbital integrals (thread-safe accessors)
};

} // namespace lever