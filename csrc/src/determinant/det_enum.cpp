// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_enum.cpp
 * @brief FCI/CAS/RAS determinant space generators via Gosper's hack.
 *
 * Generates all n-bit configurations with exactly k bits set using
 * combinatorial enumeration. Supports restricted active spaces with
 * hole/particle constraints.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <lever/determinant/det_enum.hpp>
#include <lever/utils/constants.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <format>
#include <stdexcept>

namespace lever {

namespace {

/**
 * Generate all n-bit masks with k bits set (Gosper's hack).
 *
 * Algorithm: Next combination via x' = ((x⊕r)≫2)/c | r where
 * c = lsb(x), r = x+c. Terminates on overflow or wrap-around.
 *
 * @param n_bits Bit width ∈ [0,64]
 * @param k_bits Number of ones ∈ [0,n_bits]
 * @return Sorted configurations (ascending order)
 */
std::vector<u64> generate_spin_configs(int n_bits, int k_bits) {
    if (k_bits < 0 || k_bits > n_bits) return {};
    if (k_bits == 0) return {0ULL};
    if (k_bits == n_bits) return {make_mask<u64>(n_bits)};

    std::vector<u64> configs;
    configs.reserve(64); // Heuristic initial capacity

    u64 x = make_mask<u64>(k_bits);
    const bool is_full_width = (n_bits == 64);
    const u64 overflow_bit = is_full_width ? 0ULL : (u64{1} << n_bits);

    while (true) {
        configs.push_back(x);
        const u64 c = isolate_lsb(x);
        const u64 r = x + c;
        // Termination: overflow (n<64) or wrap-around (n=64)
        if ((!is_full_width && (r & overflow_bit)) || (is_full_width && r < x))
            break;
        x = (((x ^ r) >> 2) / c) | r;
    }
    return configs;
}

/**
 * Build contiguous bitmask [offset, offset+count).
 */
constexpr u64 orbital_block_mask(int offset, int count) noexcept {
    return (count <= 0) ? 0ULL : (make_mask<u64>(count) << offset);
}

} // anonymous namespace

// ============================================================================
// FCISpace: Full Configuration Interaction
// ============================================================================

FCISpace::FCISpace(int n_orb, int n_α, int n_β) {
    if (n_orb <= 0 || n_orb > MAX_ORBITALS_U64) {
        throw std::invalid_argument(
            std::format("FCISpace: n_orb={} out of range [1,{}]", n_orb, MAX_ORBITALS_U64));
    }
    if (n_α < 0 || n_α > n_orb || n_β < 0 || n_β > n_orb) {
        throw std::invalid_argument(
            std::format("FCISpace: invalid electrons n_α={}, n_β={} for {} orbitals", n_α, n_β, n_orb));
    }

    const auto α_configs = generate_spin_configs(n_orb, n_α);
    const auto β_configs = (n_α == n_β) ? α_configs : generate_spin_configs(n_orb, n_β);

    dets_.reserve(α_configs.size() * β_configs.size());
    for (u64 β_occ : β_configs) {
        for (u64 α_occ : α_configs) {
            dets_.emplace_back(Det{α_occ, β_occ});
        }
    }
}

const std::vector<Det>& FCISpace::dets() const noexcept { return dets_; }
size_t FCISpace::size() const noexcept { return dets_.size(); }

// ============================================================================
// CASSpace: Complete Active Space (frozen core + active FCI)
// ============================================================================

CASSpace::CASSpace(int n_core, int n_active, int n_virtual,
                   int n_α_active, int n_β_active)
    : FCISpace() {
    if (n_core < 0 || n_active < 0 || n_virtual < 0) {
        throw std::invalid_argument("CASSpace: negative orbital counts not allowed");
    }

    const int n_total = n_core + n_active + n_virtual;
    if (n_total <= 0 || n_total > MAX_ORBITALS_U64) {
        throw std::invalid_argument(
            std::format("CASSpace: total orbitals {} out of range [1,{}]", n_total, MAX_ORBITALS_U64));
    }

    if (n_α_active < 0 || n_α_active > n_active || n_β_active < 0 || n_β_active > n_active) {
        throw std::invalid_argument(
            std::format("CASSpace: active electrons ({},{}) exceed capacity {}", 
                        n_α_active, n_β_active, n_active));
    }

    // Core orbitals doubly occupied
    const u64 core_mask = orbital_block_mask(0, n_core);

    // Enumerate active-space FCI per spin
    const auto α_active = generate_spin_configs(n_active, n_α_active);
    const auto β_active = generate_spin_configs(n_active, n_β_active);

    dets_.reserve(α_active.size() * β_active.size());
    for (u64 β_cfg : β_active) {
        for (u64 α_cfg : α_active) {
            dets_.emplace_back(Det{
                core_mask | (α_cfg << n_core),
                core_mask | (β_cfg << n_core)
            });
        }
    }
}

// ============================================================================
// RASSpace: Restricted Active Space with hole/particle constraints
// ============================================================================

namespace {

/**
 * Generate per-spin masks for RAS(1/2/3) with optional constraints.
 *
 * Enumerates distributions n₁+n₂+n₃ = n_elec subject to:
 *   - min_e₁ ≤ n₁ ≤ n_ras1  (hole constraint via min_e₁)
 *   - 0 ≤ n₂ ≤ n_ras2
 *   - 0 ≤ n₃ ≤ max_e₃      (particle constraint)
 *
 * @param p Orbital partition (core always set in output)
 * @param n_elec RAS electrons (excluding core)
 * @param max_holes1 Per-spin hole limit in RAS1 (-1 = no limit)
 * @param max_elecs3 Per-spin particle limit in RAS3 (-1 = no limit)
 */
std::vector<u64> generate_ras_spin_masks(
    const RASOrbitalPartition& p, int n_elec, int max_holes1, int max_elecs3)
{
    const int min_e1 = (max_holes1 == -1) ? 0 : std::max(0, p.n_ras1 - max_holes1);
    const int max_e1 = p.n_ras1;
    const int max_e3 = (max_elecs3 == -1) ? p.n_ras3 : std::min(p.n_ras3, max_elecs3);

    std::vector<u64> masks;
    masks.reserve(1024);

    const int off_core = 0;
    const int off_r1 = off_core + p.n_core;
    const int off_r2 = off_r1 + p.n_ras1;
    const int off_r3 = off_r2 + p.n_ras2;
    const u64 core_mask = orbital_block_mask(off_core, p.n_core);

    for (int n1 = min_e1; n1 <= max_e1; ++n1) {
        for (int n3 = 0; n3 <= max_e3; ++n3) {
            const int n2 = n_elec - n1 - n3;
            if (n2 < 0 || n2 > p.n_ras2) continue;

            for (u64 c1 : generate_spin_configs(p.n_ras1, n1)) {
                for (u64 c2 : generate_spin_configs(p.n_ras2, n2)) {
                    for (u64 c3 : generate_spin_configs(p.n_ras3, n3)) {
                        masks.push_back(core_mask | (c1 << off_r1) | (c2 << off_r2) | (c3 << off_r3));
                    }
                }
            }
        }
    }
    return masks;
}

} // anonymous namespace

RASSpace::RASSpace(const RASOrbitalPartition& p, const RASElectronConstraint& e)
    : FCISpace() {
    // Validate partition
    if (p.n_core < 0 || p.n_ras1 < 0 || p.n_ras2 < 0 || p.n_ras3 < 0 || p.n_virtual < 0) {
        throw std::invalid_argument("RASSpace: negative orbital counts not allowed");
    }

    const int n_total = p.n_core + p.n_ras1 + p.n_ras2 + p.n_ras3 + p.n_virtual;
    if (n_total <= 0 || n_total > MAX_ORBITALS_U64) {
        throw std::invalid_argument(
            std::format("RASSpace: total orbitals {} out of range [1,{}]", n_total, MAX_ORBITALS_U64));
    }

    // Validate electron counts (core doubly occupied)
    const int n_non_virtual = p.n_core + p.n_ras1 + p.n_ras2 + p.n_ras3;
    if (e.n_alpha_total < p.n_core || e.n_beta_total < p.n_core) {
        throw std::invalid_argument("RASSpace: spin totals must be ≥ n_core");
    }
    if (e.n_alpha_total > n_non_virtual || e.n_beta_total > n_non_virtual) {
        throw std::invalid_argument("RASSpace: spin totals exceed non-virtual capacity");
    }

    // RAS(1/2/3) electrons excluding core
    const int n_α_ras = e.n_alpha_total - p.n_core;
    const int n_β_ras = e.n_beta_total - p.n_core;

    // Generate per-spin masks without RAS1/3 limits (filtered later by total constraints)
    const auto α_masks = generate_ras_spin_masks(p, n_α_ras, -1, -1);
    const auto β_masks = generate_ras_spin_masks(p, n_β_ras, -1, -1);

    // Precompute RAS1/3 section masks for combined filtering
    const int off_r1 = p.n_core;
    const int off_r3 = p.n_core + p.n_ras1 + p.n_ras2;
    const u64 ras1_mask = orbital_block_mask(off_r1, p.n_ras1);
    const u64 ras3_mask = orbital_block_mask(off_r3, p.n_ras3);
    const int ras1_capacity = 2 * p.n_ras1; // max electrons if doubly occupied

    dets_.reserve(α_masks.size() * β_masks.size());

    for (u64 α : α_masks) {
        for (u64 β : β_masks) {
            // Apply total (α+β) RAS1/3 constraints
            const int n_ras1_total = popcount(α & ras1_mask) + popcount(β & ras1_mask);
            const int holes1 = ras1_capacity - n_ras1_total;
            const int elecs3 = popcount(α & ras3_mask) + popcount(β & ras3_mask);

            if ((e.max_holes_ras1 >= 0 && holes1 > e.max_holes_ras1) ||
                (e.max_elecs_ras3 >= 0 && elecs3 > e.max_elecs_ras3)) {
                continue;
            }
            dets_.emplace_back(Det{α, β});
        }
    }
}

} // namespace lever
