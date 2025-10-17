// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_ops.hpp
 * @brief Pure-function operations on Slater determinants (analysis & generators).
 *
 * Conventions (unified with Scemama & Giner and common QC practice):
 * - holes:    occupied in ket but unoccupied in bra      (ket & ~bra)
 * - particles:occupied in bra but unoccupied in ket      (bra & ~ket)
 *
 * Phase follows Scemama & Giner (Algorithm 4):
 * - Per spin, add the number of occupied orbitals strictly between each
 *   (hole, particle) pair (occupation taken from the *bra* determinant).
 * - For same-spin double excitations, if the two intervals cross, add +1.
 * - No cross-spin phase term.
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
 * @brief Excitation analysis data from |ket> to |bra>.
 *
 * degree = n_alpha_exc + n_beta_exc (0, 1, 2; >2 means zero by Slater–Condon).
 * holes/particles arrays store *orbital indices* in ascending order (LSB→MSB).
 *
 * holes_spin:     indices where ket has 1 and bra has 0
 * particles_spin: indices where bra has 1 and ket has 0
 */
struct ExcInfo {
    u8  degree = 255;   ///< 0,1,2; >2 means zero matrix element
    f64 phase  = 1.0;   ///< +1 or -1 for degree<=2; 0.0 if degree>2 (undefined)

    std::array<int, MAX_EXCITATION_DEGREE> holes_alpha{};
    std::array<int, MAX_EXCITATION_DEGREE> particles_alpha{};
    std::array<int, MAX_EXCITATION_DEGREE> holes_beta{};
    std::array<int, MAX_EXCITATION_DEGREE> particles_beta{};

    u8 n_alpha_exc = 0; ///< # of excitations in alpha channel (0,1,2)
    u8 n_beta_exc  = 0; ///< # of excitations in beta  channel (0,1,2)
};

namespace det_ops {

// ============================================================================
// Excitation analysis & phase
// ============================================================================

/// Analyze excitation pattern and phase for <bra| ... |ket>.
[[nodiscard]] ExcInfo analyze_excitation(const Det& bra, const Det& ket) noexcept;

/// Phase-only helper when indices are not required.
/// Returns ±1 for degree<=2, and 0.0 if degree>2 (phase undefined / element zero).
[[nodiscard]] f64 phase(const Det& bra, const Det& ket) noexcept;

// ============================================================================
// Streaming generators (no allocation; deterministic order)
// ============================================================================

template<typename Visitor>
void for_each_single_alpha(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

template<typename Visitor>
void for_each_single_beta(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

template<typename Visitor>
void for_each_double_aa(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

template<typename Visitor>
void for_each_double_bb(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

template<typename Visitor>
void for_each_double_ab(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

// Composite helpers (order: s_alpha, s_beta, d_aa, d_bb, d_ab)
template<typename Visitor>
void for_each_single(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

template<typename Visitor>
void for_each_double(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>())));

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
    assert((ket.alpha & ~make_mask<u64>(n_orb)) == 0 && "alpha occ beyond n_orb");
    const u64 occ = ket.alpha;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (!occ || !virt) return;

    for (u64 h = occ; h; h = clear_lsb(h)) {
        const u64 hb = isolate_lsb(h);
        for (u64 p = virt; p; p = clear_lsb(p)) {
            const u64 pb = isolate_lsb(p);
            std::forward<Visitor>(visit)(Det{ket.alpha ^ hb ^ pb, ket.beta});
        }
    }
}

template<typename Visitor>
void for_each_single_beta(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.beta & ~make_mask<u64>(n_orb)) == 0 && "beta occ beyond n_orb");
    const u64 occ = ket.beta;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (!occ || !virt) return;

    for (u64 h = occ; h; h = clear_lsb(h)) {
        const u64 hb = isolate_lsb(h);
        for (u64 p = virt; p; p = clear_lsb(p)) {
            const u64 pb = isolate_lsb(p);
            std::forward<Visitor>(visit)(Det{ket.alpha, ket.beta ^ hb ^ pb});
        }
    }
}

template<typename Visitor>
void for_each_double_aa(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.alpha & ~make_mask<u64>(n_orb)) == 0 && "alpha occ beyond n_orb");
    const u64 occ = ket.alpha;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (popcount(occ) < 2 || popcount(virt) < 2) return;

    for (u64 h1 = occ; h1; h1 = clear_lsb(h1)) {
        const u64 hb1 = isolate_lsb(h1);
        for (u64 h2 = clear_lsb(h1); h2; h2 = clear_lsb(h2)) {
            const u64 hb2 = isolate_lsb(h2);
            const u64 hole_flip = hb1 | hb2;

            for (u64 p1 = virt; p1; p1 = clear_lsb(p1)) {
                const u64 pb1 = isolate_lsb(p1);
                for (u64 p2 = clear_lsb(p1); p2; p2 = clear_lsb(p2)) {
                    const u64 pb2 = isolate_lsb(p2);
                    const u64 part_flip = pb1 | pb2;
                    std::forward<Visitor>(visit)(Det{ket.alpha ^ hole_flip ^ part_flip, ket.beta});
                }
            }
        }
    }
}

template<typename Visitor>
void for_each_double_bb(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.beta & ~make_mask<u64>(n_orb)) == 0 && "beta occ beyond n_orb");
    const u64 occ = ket.beta;
    const u64 virt = ~occ & make_mask<u64>(n_orb);
    if (popcount(occ) < 2 || popcount(virt) < 2) return;

    for (u64 h1 = occ; h1; h1 = clear_lsb(h1)) {
        const u64 hb1 = isolate_lsb(h1);
        for (u64 h2 = clear_lsb(h1); h2; h2 = clear_lsb(h2)) {
            const u64 hb2 = isolate_lsb(h2);
            const u64 hole_flip = hb1 | hb2;

            for (u64 p1 = virt; p1; p1 = clear_lsb(p1)) {
                const u64 pb1 = isolate_lsb(p1);
                for (u64 p2 = clear_lsb(p1); p2; p2 = clear_lsb(p2)) {
                    const u64 pb2 = isolate_lsb(p2);
                    const u64 part_flip = pb1 | pb2;
                    std::forward<Visitor>(visit)(Det{ket.alpha, ket.beta ^ hole_flip ^ part_flip});
                }
            }
        }
    }
}

template<typename Visitor>
void for_each_double_ab(const Det& ket, int n_orb, Visitor&& visit)
    noexcept(noexcept(std::forward<Visitor>(visit)(std::declval<const Det&>()))) {
    assert(n_orb >= 0 && n_orb <= MAX_ORBITALS_U64);
    assert((ket.alpha & ~make_mask<u64>(n_orb)) == 0 && "alpha occ beyond n_orb");
    assert((ket.beta  & ~make_mask<u64>(n_orb)) == 0 && "beta  occ beyond n_orb");

    const u64 occ_a = ket.alpha, occ_b = ket.beta;
    const u64 virt_a = ~occ_a & make_mask<u64>(n_orb);
    const u64 virt_b = ~occ_b & make_mask<u64>(n_orb);
    if (!occ_a || !virt_a || !occ_b || !virt_b) return;

    for (u64 ha = occ_a; ha; ha = clear_lsb(ha)) {
        const u64 hba = isolate_lsb(ha);
        for (u64 pa = virt_a; pa; pa = clear_lsb(pa)) {
            const u64 pba = isolate_lsb(pa);
            const u64 next_a = ket.alpha ^ hba ^ pba;

            for (u64 hb = occ_b; hb; hb = clear_lsb(hb)) {
                const u64 hbb = isolate_lsb(hb);
                for (u64 pb = virt_b; pb; pb = clear_lsb(pb)) {
                    const u64 pbb = isolate_lsb(pb);
                    const u64 next_b = ket.beta ^ hbb ^ pbb;
                    std::forward<Visitor>(visit)(Det{next_a, next_b});
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
    for_each_single (ket, n_orb, visit);
    for_each_double (ket, n_orb, visit);
}

} // namespace det_ops
} // namespace lever