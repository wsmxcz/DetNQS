// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_ops.hpp
 * @brief Pure-function operations on Slater determinants (analysis & generators).
 *
 * Hole/particle conventions (Scemama & Giner):
 * - holes:    occupied in |ket⟩ but not in ⟨bra|    (ket & ~bra)
 * - particles: occupied in ⟨bra| but not in |ket⟩   (bra & ~ket)
 *
 * Phase computation (Algorithm 4, Scemama & Giner):
 * - Per spin: count occupied orbitals strictly between each (hole, particle) pair
 *   using occupation from ⟨bra|
 * - Same-spin doubles: add +1 if intervals cross
 * - No cross-spin phase contribution
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/utils/constants.hpp>
#include <lever/utils/types.hpp>
#include <lever/utils/bit_utils.hpp>

#include <array>
#include <cassert>
#include <utility>
#include <vector>

namespace lever {

/**
 * Excitation analysis data from |ket⟩ to ⟨bra|.
 *
 * degree = n_α_exc + n_β_exc ∈ {0,1,2}; >2 implies zero by Slater–Condon.
 * Orbital indices stored in ascending order (LSB→MSB).
 */
struct ExcInfo {
    u8  degree = 255;   ///< Excitation degree; 255 = uninitialized
    f64 phase  = 1.0;   ///< ±1 for degree≤2; undefined if degree>2

    std::array<int, MAX_EXCITATION_DEGREE> holes_alpha{};
    std::array<int, MAX_EXCITATION_DEGREE> particles_alpha{};
    std::array<int, MAX_EXCITATION_DEGREE> holes_beta{};
    std::array<int, MAX_EXCITATION_DEGREE> particles_beta{};

    u8 n_alpha_exc = 0; ///< α-spin excitations (0,1,2)
    u8 n_beta_exc  = 0; ///< β-spin excitations (0,1,2)
};

namespace det_ops {

// ============================================================================
// Excitation analysis & phase
// ============================================================================

/// Analyze excitation pattern and phase for ⟨bra|...|ket⟩.
[[nodiscard]] ExcInfo analyze_excitation(const Det& bra, const Det& ket) noexcept;

/// Compute phase only (±1 for degree≤2, 0 if degree>2).
[[nodiscard]] f64 phase(const Det& bra, const Det& ket) noexcept;

// ============================================================================
// Streaming generators (deterministic order, zero allocation)
// ============================================================================

/// Generate all single α-excitations from |ket⟩.
template<typename Visitor>
void for_each_single_alpha(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all single β-excitations from |ket⟩.
template<typename Visitor>
void for_each_single_beta(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all double α-α excitations from |ket⟩.
template<typename Visitor>
void for_each_double_aa(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all double β-β excitations from |ket⟩.
template<typename Visitor>
void for_each_double_bb(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all double α-β excitations from |ket⟩.
template<typename Visitor>
void for_each_double_ab(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all single excitations (α + β).
template<typename Visitor>
void for_each_single(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all double excitations (αα + ββ + αβ).
template<typename Visitor>
void for_each_double(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

/// Generate all connected determinants (singles + doubles).
template<typename Visitor>
void for_each_connected(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

// ============================================================================
// Batch collection (compatibility helpers)
// ============================================================================

[[nodiscard]] std::vector<Det> collect_singles(const Det& ket, int n_orb);
[[nodiscard]] std::vector<Det> collect_doubles(const Det& ket, int n_orb);
[[nodiscard]] std::vector<Det> collect_connected(const Det& ket, int n_orb);

// ============================================================================
// Template implementations
// ============================================================================

template<typename Visitor>
void for_each_single_alpha(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.alpha & ~make_mask<u64>(n_orb)) == 0);

    const u64 occ  = ket.alpha;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (!occ || !virt) return;

    for (u64 h = occ; h; h = clear_lsb(h)) {
        const u64 h_bit = isolate_lsb(h);
        for (u64 p = virt; p; p = clear_lsb(p)) {
            const u64 p_bit = isolate_lsb(p);
            std::forward<Visitor>(visit)(Det{ket.alpha ^ h_bit ^ p_bit, ket.beta});
        }
    }
}

template<typename Visitor>
void for_each_single_beta(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.beta & ~make_mask<u64>(n_orb)) == 0);

    const u64 occ  = ket.beta;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (!occ || !virt) return;

    for (u64 h = occ; h; h = clear_lsb(h)) {
        const u64 h_bit = isolate_lsb(h);
        for (u64 p = virt; p; p = clear_lsb(p)) {
            const u64 p_bit = isolate_lsb(p);
            std::forward<Visitor>(visit)(Det{ket.alpha, ket.beta ^ h_bit ^ p_bit});
        }
    }
}

template<typename Visitor>
void for_each_double_aa(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.alpha & ~make_mask<u64>(n_orb)) == 0);

    const u64 occ  = ket.alpha;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (popcount(occ) < 2 || popcount(virt) < 2) return;

    for (u64 h1 = occ; h1; h1 = clear_lsb(h1)) {
        const u64 h1_bit = isolate_lsb(h1);
        for (u64 h2 = clear_lsb(h1); h2; h2 = clear_lsb(h2)) {
            const u64 h2_bit = isolate_lsb(h2);
            const u64 holes = h1_bit | h2_bit;

            for (u64 p1 = virt; p1; p1 = clear_lsb(p1)) {
                const u64 p1_bit = isolate_lsb(p1);
                for (u64 p2 = clear_lsb(p1); p2; p2 = clear_lsb(p2)) {
                    const u64 p2_bit = isolate_lsb(p2);
                    const u64 parts = p1_bit | p2_bit;
                    std::forward<Visitor>(visit)(Det{ket.alpha ^ holes ^ parts, ket.beta});
                }
            }
        }
    }
}

template<typename Visitor>
void for_each_double_bb(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.beta & ~make_mask<u64>(n_orb)) == 0);

    const u64 occ  = ket.beta;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (popcount(occ) < 2 || popcount(virt) < 2) return;

    for (u64 h1 = occ; h1; h1 = clear_lsb(h1)) {
        const u64 h1_bit = isolate_lsb(h1);
        for (u64 h2 = clear_lsb(h1); h2; h2 = clear_lsb(h2)) {
            const u64 h2_bit = isolate_lsb(h2);
            const u64 holes = h1_bit | h2_bit;

            for (u64 p1 = virt; p1; p1 = clear_lsb(p1)) {
                const u64 p1_bit = isolate_lsb(p1);
                for (u64 p2 = clear_lsb(p1); p2; p2 = clear_lsb(p2)) {
                    const u64 p2_bit = isolate_lsb(p2);
                    const u64 parts = p1_bit | p2_bit;
                    std::forward<Visitor>(visit)(Det{ket.alpha, ket.beta ^ holes ^ parts});
                }
            }
        }
    }
}

template<typename Visitor>
void for_each_double_ab(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.alpha & ~make_mask<u64>(n_orb)) == 0);
    assert((ket.beta  & ~make_mask<u64>(n_orb)) == 0);

    const u64 occ_a  = ket.alpha;
    const u64 occ_b  = ket.beta;
    const u64 virt_a = ~occ_a & make_mask<u64>(n_orb);
    const u64 virt_b = ~occ_b & make_mask<u64>(n_orb);
    if (!occ_a || !virt_a || !occ_b || !virt_b) return;

    for (u64 h_a = occ_a; h_a; h_a = clear_lsb(h_a)) {
        const u64 ha_bit = isolate_lsb(h_a);
        for (u64 p_a = virt_a; p_a; p_a = clear_lsb(p_a)) {
            const u64 pa_bit = isolate_lsb(p_a);
            const u64 alpha_new = ket.alpha ^ ha_bit ^ pa_bit;

            for (u64 h_b = occ_b; h_b; h_b = clear_lsb(h_b)) {
                const u64 hb_bit = isolate_lsb(h_b);
                for (u64 p_b = virt_b; p_b; p_b = clear_lsb(p_b)) {
                    const u64 pb_bit = isolate_lsb(p_b);
                    const u64 beta_new = ket.beta ^ hb_bit ^ pb_bit;
                    std::forward<Visitor>(visit)(Det{alpha_new, beta_new});
                }
            }
        }
    }
}

template<typename Visitor>
void for_each_single(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    for_each_single_alpha(ket, n_orb, visit);
    for_each_single_beta (ket, n_orb, visit);
}

template<typename Visitor>
void for_each_double(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    for_each_double_aa(ket, n_orb, visit);
    for_each_double_bb(ket, n_orb, visit);
    for_each_double_ab(ket, n_orb, visit);
}

template<typename Visitor>
void for_each_connected(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    for_each_single(ket, n_orb, visit);
    for_each_double(ket, n_orb, visit);
}

} // namespace det_ops
} // namespace lever
