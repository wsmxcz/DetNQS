// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_enum.cpp
 * @brief Implementation of FCI, CAS, and RAS space generators.
 */

#include <lever/determinant/det_enum.hpp>
#include <lever/utils/constants.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace lever {

namespace { // ==== Internal helpers =================================================

/**
 * @brief Enumerate all n-bit masks with exactly k bits set (Gosper's hack).
 * @param n_bits Bit width (0..64).
 * @param k_set_bits Number of 1-bits (0..n_bits).
 * @return Vector of masks in increasing order (unspecified if n_bits==64).
 *
 * Implementation avoids undefined behavior for shifts by 64 and uses a
 * robust termination condition for both n_bits<64 (overflow out of range)
 * and n_bits==64 (wrap-around detection).
 */
std::vector<u64> generate_spin_configurations(int n_bits, int k_set_bits) {
    if (k_set_bits < 0 || k_set_bits > n_bits) return {};
    if (k_set_bits == 0) return {0ULL};
    if (k_set_bits == n_bits) {
        return { make_mask<u64>(n_bits) };
    }

    std::vector<u64> configs;
    u64 x = make_mask<u64>(k_set_bits);

    const bool full64 = (n_bits == 64);
    const u64 limitBit = full64 ? 0ULL : (u64{1} << n_bits);

    while (true) {
        configs.push_back(x);
        const u64 c = isolate_lsb(x);
        const u64 r = x + c;
        // Termination:
        // - n_bits < 64: next pattern would set bit n_bits (overflow out of domain)
        // - n_bits == 64: addition wraps around to smaller value
        if ((!full64 && (r & limitBit)) || (full64 && r < x)) break;
        x = (((x ^ r) >> 2) / c) | r;
    }
    return configs;
}

/**
 * @brief Build a contiguous bitmask [offset, offset+count) within u64.
 */
constexpr u64 block_mask(int offset, int count) noexcept {
    return (count <= 0) ? 0ULL : (make_mask<u64>(count) << offset);
}

} // anonymous namespace

// ==== FCISpace ====================================================================

FCISpace::FCISpace(int n_orb, int n_alpha, int n_beta) {
    if (n_orb <= 0 || n_orb > MAX_ORBITALS_U64) {
        throw std::invalid_argument("FCISpace: n_orb must be in [1,64]. Got " + std::to_string(n_orb));
    }
    if (n_alpha < 0 || n_alpha > n_orb || n_beta < 0 || n_beta > n_orb) {
        throw std::invalid_argument("FCISpace: invalid electron counts.");
    }

    const auto alpha_cfgs = generate_spin_configurations(n_orb, n_alpha);

    if (n_alpha == n_beta) {
        dets_.reserve(alpha_cfgs.size() * alpha_cfgs.size());
        for (u64 beta_occ : alpha_cfgs) {
            for (u64 alpha_occ : alpha_cfgs) {
                dets_.emplace_back(Det{alpha_occ, beta_occ});
            }
        }
    } else {
        const auto beta_cfgs = generate_spin_configurations(n_orb, n_beta);
        dets_.reserve(alpha_cfgs.size() * beta_cfgs.size());
        for (u64 beta_occ : beta_cfgs) {
            for (u64 alpha_occ : alpha_cfgs) {
                dets_.emplace_back(Det{alpha_occ, beta_occ});
            }
        }
    }
}

const std::vector<Det>& FCISpace::dets() const noexcept { return dets_; }
size_t FCISpace::size() const noexcept { return dets_.size(); }

// ==== CASSpace ====================================================================

CASSpace::CASSpace(int n_core_orb, int n_active_orb, int n_virtual_orb,
                   int n_alpha_active, int n_beta_active)
    : FCISpace() {
    // Basic validation
    if (n_core_orb < 0 || n_active_orb < 0 || n_virtual_orb < 0) {
        throw std::invalid_argument("CASSpace: negative counts are not allowed.");
    }
    const int n_orb_total = n_core_orb + n_active_orb + n_virtual_orb;
    if (n_orb_total <= 0 || n_orb_total > MAX_ORBITALS_U64) {
        throw std::invalid_argument("CASSpace: total orbitals must be in [1,64].");
    }
    if (n_alpha_active < 0 || n_alpha_active > n_active_orb ||
        n_beta_active  < 0 || n_beta_active  > n_active_orb) {
        throw std::invalid_argument("CASSpace: active electron counts out of range.");
    }
    if (n_alpha_active + n_beta_active > 2 * n_active_orb) {
        throw std::invalid_argument("CASSpace: total active electrons exceed capacity.");
    }

    // Core mask (doubly occupied)
    const u64 core_mask = block_mask(0, n_core_orb);

    // Enumerate active-space FCI per spin
    const auto alpha_active_cfgs = generate_spin_configurations(n_active_orb, n_alpha_active);
    const auto beta_active_cfgs  = generate_spin_configurations(n_active_orb, n_beta_active);

    dets_.reserve(alpha_active_cfgs.size() * beta_active_cfgs.size());
    for (u64 beta_active : beta_active_cfgs) {
        for (u64 alpha_active : alpha_active_cfgs) {
            const u64 a = core_mask | (alpha_active << n_core_orb);
            const u64 b = core_mask | (beta_active  << n_core_orb);
            dets_.emplace_back(Det{a, b});
        }
    }
}

// ==== RASSpace ====================================================================

namespace { // RAS internal helpers

/**
 * @brief Generate per-spin masks for RAS1/2/3 (core is always set).
 * @param p       Orbital partition (core, ras1, ras2, ras3, virtual).
 * @param n_elec  # of spin electrons to distribute in RAS1/2/3 (excluding core).
 * @param max_holes1 Per-spin limit for holes in RAS1 (use -1 for no limit).
 * @param max_elecs3 Per-spin limit for electrons in RAS3 (use -1 for no limit).
 *
 * NOTE: In the final RAS spec, RAS1/RAS3 constraints apply to TOTAL (alpha+beta).
 * We call this with (-1,-1) and filter combined masks afterward.
 */
std::vector<u64> generate_total_spin_masks(const RASOrbitalPartition& p,
                                           int n_elec,
                                           int max_holes1,
                                           int max_elecs3) {
    // Bounds for per-spin distributions within each subspace
    const int min_e1 = (max_holes1 == -1) ? 0 : std::max(0, p.n_ras1 - max_holes1);
    const int max_e1 = p.n_ras1;
    const int max_e3 = (max_elecs3 == -1) ? p.n_ras3 : std::min(p.n_ras3, max_elecs3);

    std::vector<u64> total_masks;
    total_masks.reserve(1024);

    const int off_core = 0;
    const int off_r1   = off_core + p.n_core;
    const int off_r2   = off_r1   + p.n_ras1;
    const int off_r3   = off_r2   + p.n_ras2;
    const u64 core_mask = block_mask(off_core, p.n_core);

    for (int n1 = min_e1; n1 <= max_e1; ++n1) {
        for (int n3 = 0; n3 <= max_e3; ++n3) {
            const int n2 = n_elec - n1 - n3;
            if (n2 < 0 || n2 > p.n_ras2) continue;

            const auto cfg1 = generate_spin_configurations(p.n_ras1, n1);
            const auto cfg2 = generate_spin_configurations(p.n_ras2, n2);
            const auto cfg3 = generate_spin_configurations(p.n_ras3, n3);

            for (u64 c1 : cfg1) {
                for (u64 c2 : cfg2) {
                    for (u64 c3 : cfg3) {
                        const u64 m1 = (c1 << off_r1);
                        const u64 m2 = (c2 << off_r2);
                        const u64 m3 = (c3 << off_r3);
                        total_masks.push_back(core_mask | m1 | m2 | m3);
                    }
                }
            }
        }
    }
    return total_masks;
}

} // anonymous namespace

RASSpace::RASSpace(const RASOrbitalPartition& p,
                   const RASElectronConstraint& e)
    : FCISpace() {
    // Basic partition validation
    if (p.n_core   < 0 || p.n_ras1 < 0 || p.n_ras2 < 0 ||
        p.n_ras3   < 0 || p.n_virtual < 0) {
        throw std::invalid_argument("RASSpace: negative counts are not allowed.");
    }
    const int n_orb_total = p.n_core + p.n_ras1 + p.n_ras2 + p.n_ras3 + p.n_virtual;
    if (n_orb_total <= 0 || n_orb_total > MAX_ORBITALS_U64) {
        throw std::invalid_argument("RASSpace: total orbitals must be in [1,64].");
    }

    // Electron count validation (per spin)
    const int non_virtual_per_spin = p.n_core + p.n_ras1 + p.n_ras2 + p.n_ras3;
    if (e.n_alpha_total < p.n_core || e.n_beta_total < p.n_core) {
        throw std::invalid_argument("RASSpace: each spin total must be >= n_core (core is doubly occupied).");
    }
    if (e.n_alpha_total > non_virtual_per_spin || e.n_beta_total > non_virtual_per_spin) {
        throw std::invalid_argument("RASSpace: each spin total exceeds non-virtual capacity.");
    }

    // Convert total per-spin electrons to RAS(1/2/3) electrons (excluding core)
    const int n_alpha_ras = e.n_alpha_total - p.n_core;
    const int n_beta_ras  = e.n_beta_total  - p.n_core;

    // Generate per-spin masks without applying RAS1/RAS3 limits (use -1, -1).
    // Total RAS constraints will be enforced after combining alpha+beta masks.
    const auto alpha_masks = generate_total_spin_masks(p, n_alpha_ras, -1, -1);
    const auto beta_masks  = generate_total_spin_masks(p, n_beta_ras,  -1, -1);

    // Precompute section masks for combined filtering
    const int off_core = 0;
    const int off_r1   = off_core + p.n_core;
    const int off_r2   = off_r1   + p.n_ras1;
    const int off_r3   = off_r2   + p.n_ras2;

    const u64 ras1Mask = block_mask(off_r1, p.n_ras1);
    const u64 ras3Mask = block_mask(off_r3, p.n_ras3);

    const int ras1_cap_total = 2 * p.n_ras1; // total electrons if doubly occupied
    dets_.reserve(alpha_masks.size() * beta_masks.size());

    for (u64 a : alpha_masks) {
        for (u64 b : beta_masks) {
            // Combined RAS1/RAS3 constraints (TOTAL = alpha + beta)
            const int occ_ras1 = popcount(a & ras1Mask) + popcount(b & ras1Mask);
            const int holes1   = ras1_cap_total - occ_ras1;
            const int elecs3   = popcount(a & ras3Mask) + popcount(b & ras3Mask);

            if ((e.max_holes_ras1 >= 0 && holes1 > e.max_holes_ras1) ||
                (e.max_elecs_ras3 >= 0 && elecs3 > e.max_elecs_ras3)) {
                continue; // violates TOTAL constraints
            }
            dets_.emplace_back(Det{a, b});
        }
    }
}

} // namespace lever