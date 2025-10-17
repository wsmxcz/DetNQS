// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_eval.cpp
 * @brief Implementation of Slater–Condon Hamiltonian evaluator.
 */

#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <vector>
#include <cmath>

namespace lever {

HamEval::HamEval(const IntegralSO& so_ints) noexcept : so_ints_(so_ints) {}

ExcInfo HamEval::analyze_exc(const Det& bra, const Det& ket) const noexcept {
    // Delegate to det_ops to keep phase convention consistent across the codebase.
    return det_ops::analyze_excitation(bra, ket);
}

double HamEval::compute_elem(const Det& bra, const Det& ket) const noexcept {
    const auto info = det_ops::analyze_excitation(bra, ket);

    // Slater–Condon orthogonality: >2 excitations => zero element.
    if (info.degree > MAX_EXCITATION_DEGREE) [[unlikely]] {
        return 0.0;
    }

    switch (info.degree) {
        case 0:
            // Diagonal depends only on ket (or bra, identical on diagonal).
            return compute_diagonal(ket);
        case 1:
            // Single: apply Condon–Shortley phase from bra<-ket permutation.
            return info.phase * compute_single(info, ket);
        case 2:
            // Double: same as above.
            return info.phase * compute_double(info);
        default:
            return 0.0;
    }
}

double HamEval::compute_diagonal(const Det& det) const noexcept {
    // Collect occupied spin-orbitals from the determinant.
    std::vector<int> occ_sos;
    occ_sos.reserve(popcount(det.alpha) + popcount(det.beta));

    // Alpha spin-orbitals (even indices)
    u64 am = det.alpha;
    while (am) {
        const int mo = ctz(am);
        occ_sos.push_back(so_from_mo(mo, 0));
        am = clear_lsb(am);
    }

    // Beta spin-orbitals (odd indices)
    u64 bm = det.beta;
    while (bm) {
        const int mo = ctz(bm);
        occ_sos.push_back(so_from_mo(mo, 1));
        bm = clear_lsb(bm);
    }

    // One-body sum: sum_p <p|h|p>
    double e1 = 0.0;
    for (int p : occ_sos) {
        e1 += so_ints_.get_h1e_so(p, p);
    }

    // Two-body (antisymmetrized) sum: sum_{p<q} <pq||pq>
    double e2 = 0.0;
    for (std::size_t i = 1; i < occ_sos.size(); ++i) {
        const int p = occ_sos[i];
        for (std::size_t j = 0; j < i; ++j) {
            const int q = occ_sos[j];
            e2 += so_ints_.get_h2e_anti(p, q, p, q);
        }
    }

    return e1 + e2;
}

double HamEval::compute_single(const ExcInfo& info, const Det& ket) const noexcept {
    // Determine hole/particle spin-orbital indices from ExcInfo.
    // Only one spin channel contributes for degree==1.
    const bool is_alpha = (info.n_alpha_exc == 1);
    const int hole_so = is_alpha
        ? so_from_mo(info.holes_alpha[0], 0)
        : so_from_mo(info.holes_beta [0], 1);
    const int part_so = is_alpha
        ? so_from_mo(info.particles_alpha[0], 0)
        : so_from_mo(info.particles_beta [0], 1);

    // One-body contribution: <a|h|i>
    double elem = so_ints_.get_h1e_so(part_so, hole_so);

    // Two-body Fock-like term: sum_{k in occ(ket), k != i} <ak||ik>
    // We iterate occupied alpha then beta in ket, constructing k's spin-orbital index.
    u64 am = ket.alpha;
    while (am) {
        const int mo = ctz(am);
        const int k_so = so_from_mo(mo, 0);
        if (k_so != hole_so) {
            elem += so_ints_.get_h2e_anti(part_so, k_so, hole_so, k_so);
        }
        am = clear_lsb(am);
    }

    u64 bm = ket.beta;
    while (bm) {
        const int mo = ctz(bm);
        const int k_so = so_from_mo(mo, 1);
        if (k_so != hole_so) {
            elem += so_ints_.get_h2e_anti(part_so, k_so, hole_so, k_so);
        }
        bm = clear_lsb(bm);
    }

    return elem;
}

double HamEval::compute_double(const ExcInfo& info) const noexcept {
    // Three cases: (aa), (bb), (ab). Use antisymmetrized integrals for same-spin,
    // and Physicist’s notation for mixed-spin (no antisymmetrization across spins).
    if (info.n_alpha_exc == 2) {
        // Same-spin alpha: i,j -> a,b  (all alpha)
        const int i = so_from_mo(info.holes_alpha[0],     0);
        const int j = so_from_mo(info.holes_alpha[1],     0);
        const int a = so_from_mo(info.particles_alpha[0], 0);
        const int b = so_from_mo(info.particles_alpha[1], 0);
        return so_ints_.get_h2e_anti(a, b, i, j);
    }
    if (info.n_beta_exc == 2) {
        // Same-spin beta: i,j -> a,b  (all beta)
        const int i = so_from_mo(info.holes_beta[0],      1);
        const int j = so_from_mo(info.holes_beta[1],      1);
        const int a = so_from_mo(info.particles_beta[0],  1);
        const int b = so_from_mo(info.particles_beta[1],  1);
        return so_ints_.get_h2e_anti(a, b, i, j);
    }

    // Mixed-spin (ab): i(alpha), j(beta) -> a(alpha), b(beta)
    const int i = so_from_mo(info.holes_alpha[0],     0);
    const int a = so_from_mo(info.particles_alpha[0], 0);
    const int j = so_from_mo(info.holes_beta[0],      1);
    const int b = so_from_mo(info.particles_beta[0],  1);

    // For mixed-spin, <a b | i j> in Physicist’s notation equals (a i | b j) in Chemists’,
    // and does not require antisymmetrization across spins.
    return so_ints_.get_h2e_phys(a, b, i, j);
}

} // namespace lever