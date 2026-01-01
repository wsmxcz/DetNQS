// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eval.cpp
 * @brief Slater-Condon rules for Hamiltonian matrix elements ⟨Φ_i|H|Φ_j⟩.
 *
 * Implements the Slater-Condon rules for computing Hamiltonian matrix elements
 * between Slater determinants:
 *   - Degree 0 (diagonal): E = Σ_p h_pp + Σ_{p<q} ⟨pq||pq⟩
 *   - Degree 1 (single):   H_ai = h_ai + Σ_k ⟨ak||ik⟩
 *   - Degree 2 (double):   H_abij = ⟨ab||ij⟩
 *   - Degree > 2:          H = 0 (orthogonality)
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <detnqs/hamiltonian/ham_eval.hpp>
#include <detnqs/utils/bit_utils.hpp>

#include <algorithm>
#include <vector>

namespace detnqs {

HamEval::HamEval(const IntegralSO& so_ints) noexcept : so_ints_(so_ints) {}

ExcInfo HamEval::analyze_exc(const Det& bra, const Det& ket) const noexcept {
    return det_ops::analyze_excitation(bra, ket);
}

double HamEval::compute_elem(const Det& bra, const Det& ket) const noexcept {
    const auto info = det_ops::analyze_excitation(bra, ket);

    // Slater-Condon orthogonality: degree > 2 ⇒ H = 0
    if (info.degree > MAX_EXCITATION_DEGREE) [[unlikely]] {
        return 0.0;
    }

    switch (info.degree) {
        case 0:
            return compute_diagonal(ket);
        case 1:
            return info.phase * compute_single(info, ket);
        case 2:
            return info.phase * compute_double(info);
        default:
            return 0.0;
    }
}

double HamEval::compute_diagonal(const Det& det) const noexcept {
    // Extract occupied spin-orbitals: α → 2p, β → 2p+1
    std::vector<int> occ_sos;
    occ_sos.reserve(popcount(det.alpha) + popcount(det.beta));

    // Alpha orbitals (even indices)
    for (u64 am = det.alpha; am; am = clear_lsb(am)) {
        occ_sos.push_back(so_from_mo(ctz(am), 0));
    }

    // Beta orbitals (odd indices)
    for (u64 bm = det.beta; bm; bm = clear_lsb(bm)) {
        occ_sos.push_back(so_from_mo(ctz(bm), 1));
    }

    // One-body term: Σ_p h_pp
    double e1 = 0.0;
    for (int p : occ_sos) {
        e1 += so_ints_.get_h1e_so(p, p);
    }

    // Two-body term: Σ_{p<q} ⟨pq||pq⟩
    double e2 = 0.0;
    const auto n = occ_sos.size();
    for (std::size_t i = 1; i < n; ++i) {
        const int p = occ_sos[i];
        for (std::size_t j = 0; j < i; ++j) {
            const int q = occ_sos[j];
            e2 += so_ints_.get_h2e_anti(p, q, p, q);
        }
    }

    return e1 + e2;
}

double HamEval::compute_single(const ExcInfo& info, const Det& ket) const noexcept {
    // Single excitation i → a (one spin channel active)
    const bool is_alpha = (info.n_alpha_exc == 1);
    const int i = is_alpha
        ? so_from_mo(info.holes_alpha[0], 0)
        : so_from_mo(info.holes_beta[0], 1);
    const int a = is_alpha
        ? so_from_mo(info.particles_alpha[0], 0)
        : so_from_mo(info.particles_beta[0], 1);

    // H_ai = h_ai + Σ_{k∈occ, k≠i} ⟨ak||ik⟩
    double elem = so_ints_.get_h1e_so(a, i);

    // Sum over occupied orbitals in ket (both spins)
    auto add_two_body = [&](u64 mask, int spin) {
        for (u64 m = mask; m; m = clear_lsb(m)) {
            const int k = so_from_mo(ctz(m), spin);
            if (k != i) {
                elem += so_ints_.get_h2e_anti(a, k, i, k);
            }
        }
    };

    add_two_body(ket.alpha, 0);
    add_two_body(ket.beta, 1);

    return elem;
}

double HamEval::compute_double(const ExcInfo& info) const noexcept {
    // Double excitation: i,j → a,b
    // Same-spin (α-α or β-β): H_abij = ⟨ab||ij⟩ (antisymmetrized)
    // Mixed-spin (α-β):        H_abij = ⟨ab|ij⟩ (no antisymmetrization)

    if (info.n_alpha_exc == 2) {
        // α-α excitation
        const int i = so_from_mo(info.holes_alpha[0], 0);
        const int j = so_from_mo(info.holes_alpha[1], 0);
        const int a = so_from_mo(info.particles_alpha[0], 0);
        const int b = so_from_mo(info.particles_alpha[1], 0);
        return so_ints_.get_h2e_anti(a, b, i, j);
    }

    if (info.n_beta_exc == 2) {
        // β-β excitation
        const int i = so_from_mo(info.holes_beta[0], 1);
        const int j = so_from_mo(info.holes_beta[1], 1);
        const int a = so_from_mo(info.particles_beta[0], 1);
        const int b = so_from_mo(info.particles_beta[1], 1);
        return so_ints_.get_h2e_anti(a, b, i, j);
    }

    // α-β mixed-spin excitation
    const int i = so_from_mo(info.holes_alpha[0], 0);
    const int a = so_from_mo(info.particles_alpha[0], 0);
    const int j = so_from_mo(info.holes_beta[0], 1);
    const int b = so_from_mo(info.particles_beta[0], 1);

    return so_ints_.get_h2e_phys(a, b, i, j);
}

} // namespace detnqs
