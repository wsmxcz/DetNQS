// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eval.hpp
 * @brief Hamiltonian matrix element evaluator via Slater-Condon rules.
 *
 * Thread-safe evaluator for ⟨bra|H|ket⟩ in spin-orbital basis
 * (even indices = α, odd = β). Excitation analysis delegated to det_ops.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <detnqs/determinant/det.hpp>
#include <detnqs/determinant/det_ops.hpp>
#include <detnqs/utils/bit_utils.hpp>
#include <detnqs/integral/integral_so.hpp>

#include <cstdint>

namespace detnqs {

/**
 * @class HamEval
 * @brief Evaluates ⟨bra|H|ket⟩ via Slater-Condon rules.
 *
 * Slater-Condon rules (physicist notation):
 *   degree 0: ∑_p h_pp + ∑_{p<q} ⟨pq||pq⟩
 *   degree 1: h_ap + ∑_{k∈occ(ket)} ⟨ak||pk⟩
 *   degree 2: same-spin: ⟨ab||ij⟩; mixed-spin: ⟨ab|ij⟩
 *   degree > 2: 0
 *
 * Note: Core energy (e.g., FCIDUMP e_core) added externally.
 */
class HamEval {
public:
    /// Construct from spin-orbital integral provider.
    explicit HamEval(const IntegralSO& so_ints) noexcept;

    /// Analyze excitation pattern and phase.
    [[nodiscard]] ExcInfo analyze_exc(const Det& bra, const Det& ket) const noexcept;

    /// Compute matrix element ⟨bra|H|ket⟩.
    [[nodiscard]] double compute_elem(const Det& bra, const Det& ket) const noexcept;

    /// Compute diagonal element ⟨D|H|D⟩ (optimized path).
    [[nodiscard]] double compute_diagonal(const Det& det) const noexcept;

private:
    /// Single excitation contribution (degree=1).
    [[nodiscard]] double compute_single(const ExcInfo& info, const Det& ket) const noexcept;

    /// Double excitation contribution (degree=2).
    [[nodiscard]] double compute_double(const ExcInfo& info) const noexcept;

    /// Convert MO index + spin to spin-orbital index: 2·mo + σ.
    [[nodiscard]] static constexpr int so_from_mo(int mo_idx, int spin) noexcept {
        return (mo_idx << 1) | (spin & 1);
    }

    const IntegralSO& so_ints_; ///< Spin-orbital integrals
};

} // namespace detnqs
