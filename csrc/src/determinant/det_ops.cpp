// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_ops.cpp
 * @brief Determinant operations: excitation analysis and connectivity collectors.
 *
 * Implements Slater-Condon rules for matrix elements ⟨bra|op|ket⟩.
 * Phase calculation follows Scemama & Giner.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <detnqs/determinant/det_ops.hpp>
#include <detnqs/utils/bit_utils.hpp>

#include <limits>

namespace detnqs::det_ops {

// ============================================================================
// Internal helpers
// ============================================================================

namespace {

/**
 * Spin-channel excitation data: holes (occupied in ket) and particles (occupied in bra).
 */
struct SpinDiff {
    std::array<int, MAX_EXCITATION_DEGREE> holes{};    // Indices of annihilated orbitals
    std::array<int, MAX_EXCITATION_DEGREE> particles{};// Indices of created orbitals
    int n_exc = 0;                                     // Excitation rank (0,1,2 or >2)
};

/**
 * Extract holes and particles for one spin channel.
 * 
 * Holes:     ket ∖ bra  (occupied in ket but not in bra)
 * Particles: bra ∖ ket  (occupied in bra but not in ket)
 * 
 * Indices stored in ascending order (LSB → MSB).
 */
[[nodiscard]] SpinDiff extract_spin_diff(u64 bra_occ, u64 ket_occ) noexcept {
    SpinDiff diff;

    const u64 holes_mask = ket_occ & ~bra_occ;
    const u64 parts_mask = bra_occ & ~ket_occ;

    const int n_holes = popcount(holes_mask);
    const int n_parts = popcount(parts_mask);

#ifndef NDEBUG
    assert(n_holes == n_parts); // Electron number conservation per spin
#endif

    if (n_holes != n_parts) {
        diff.n_exc = MAX_EXCITATION_DEGREE + 1;
        return diff;
    }

    diff.n_exc = n_holes;
    if (diff.n_exc > MAX_EXCITATION_DEGREE) return diff;

    // Extract orbital indices
    int idx = 0;
    for (u64 mask = holes_mask; mask; mask = clear_lsb(mask)) {
        diff.holes[idx++] = ctz(mask);
    }
    idx = 0;
    for (u64 mask = parts_mask; mask; mask = clear_lsb(mask)) {
        diff.particles[idx++] = ctz(mask);
    }

    return diff;
}

/**
 * Count occupied orbitals in bra strictly between positions (lo, hi).
 * 
 * Used for phase calculation (Alg. 4): counts permutations needed
 * to move creation operators past occupied sites.
 */
[[nodiscard]] inline int count_between(u64 bra_occ, int pos_a, int pos_b) noexcept {
    if (pos_a == pos_b) return 0;
    
    const int lo = std::min(pos_a, pos_b);
    const int hi = std::max(pos_a, pos_b);
    
    const u64 mask = make_mask<u64>(hi) & ~make_mask<u64>(lo + 1); // (lo, hi)
    return popcount(bra_occ & mask);
}

/**
 * Phase factor for one spin channel via Scemama-Giner algorithm.
 * 
 * For excitation i→a: count occupied orbitals in (i, a).
 * For double excitation: add +1 if intervals cross: (i₁,a₁) ∩ (i₂,a₂) ≠ ∅.
 */
[[nodiscard]] inline int parity_one_spin(
    u64 bra_occ,
    int n_exc,
    const std::array<int, MAX_EXCITATION_DEGREE>& holes,
    const std::array<int, MAX_EXCITATION_DEGREE>& particles
) noexcept {
    int n_perm = 0;
    
    // Sum permutations for each hole→particle pair
    for (int i = 0; i < n_exc; ++i) {
        n_perm += count_between(bra_occ, holes[i], particles[i]);
    }
    
    // Crossing correction for double excitations
    if (n_exc == 2) {
        const int a = std::min(holes[0], particles[0]);
        const int b = std::max(holes[0], particles[0]);
        const int c = std::min(holes[1], particles[1]);
        const int d = std::max(holes[1], particles[1]);
        
        const bool crossed = (c > a && c < b && d > b) || 
                           (a > c && a < d && b > d);
        if (crossed) ++n_perm;
    }
    
    return n_perm; // Caller takes mod 2
}

/**
 * Binomial coefficient C(n,k) with overflow saturation.
 */
[[nodiscard]] size_t n_choose_k(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (k > n - k) k = n - k; // Symmetry optimization
    
    size_t result = 1;
    for (int i = 1; i <= k; ++i) {
        const size_t numerator = static_cast<size_t>(n - i + 1);
        
        // Overflow guard
        constexpr size_t max_val = std::numeric_limits<size_t>::max();
        if (result > max_val / numerator) {
            return max_val / 2; // Saturate at large value
        }
        
        result *= numerator;
        result /= static_cast<size_t>(i);
    }
    return result;
}

} // namespace

// ============================================================================
// Excitation analysis & phase computation
// ============================================================================

ExcInfo analyze_excitation(const Det& bra, const Det& ket) noexcept {
    ExcInfo info;

    // Extract per-spin excitation data
    const SpinDiff alpha_diff = extract_spin_diff(bra.alpha, ket.alpha);
    const SpinDiff beta_diff  = extract_spin_diff(bra.beta,  ket.beta);

    info.n_alpha_exc = static_cast<u8>(alpha_diff.n_exc);
    info.n_beta_exc  = static_cast<u8>(beta_diff.n_exc);
    info.degree      = static_cast<u8>(info.n_alpha_exc + info.n_beta_exc);

    // High-rank excitations yield zero matrix elements (Slater-Condon rules)
    if (info.degree > MAX_EXCITATION_DEGREE) {
        info.phase = 0.0;
        return info;
    }

    // Copy orbital indices (already sorted ascending)
    for (int i = 0; i < alpha_diff.n_exc; ++i) {
        info.holes_alpha[i]    = alpha_diff.holes[i];
        info.particles_alpha[i] = alpha_diff.particles[i];
    }
    for (int i = 0; i < beta_diff.n_exc; ++i) {
        info.holes_beta[i]     = beta_diff.holes[i];
        info.particles_beta[i]  = beta_diff.particles[i];
    }

    // Compute phase: (-1)^n_perm with n_perm = n_α + n_β (mod 2)
    int n_perm = 0;
    n_perm += parity_one_spin(bra.alpha, alpha_diff.n_exc, 
                             info.holes_alpha, info.particles_alpha);
    n_perm += parity_one_spin(bra.beta,  beta_diff.n_exc,
                             info.holes_beta,  info.particles_beta);
    
    info.phase = (n_perm & 1) ? -1.0 : 1.0;

    return info;
}

f64 phase(const Det& bra, const Det& ket) noexcept {
    // Lightweight version: only compute phase without full excitation info
    const SpinDiff alpha_diff = extract_spin_diff(bra.alpha, ket.alpha);
    const SpinDiff beta_diff  = extract_spin_diff(bra.beta,  ket.beta);
    
    const int total_exc = alpha_diff.n_exc + beta_diff.n_exc;
    if (total_exc > MAX_EXCITATION_DEGREE) return 0.0;

    int n_perm = 0;
    n_perm += parity_one_spin(bra.alpha, alpha_diff.n_exc, 
                             alpha_diff.holes, alpha_diff.particles);
    n_perm += parity_one_spin(bra.beta,  beta_diff.n_exc,
                             beta_diff.holes,  beta_diff.particles);
    
    return (n_perm & 1) ? -1.0 : 1.0;
}

// ============================================================================
// Determinant collectors
// ============================================================================

std::vector<Det> collect_singles(const Det& ket, int n_orb) {
    std::vector<Det> result;
    
    const int n_alpha = popcount(ket.alpha);
    const int n_beta  = popcount(ket.beta);
    const int v_alpha = std::max(0, n_orb - n_alpha); // Virtual α orbitals
    const int v_beta  = std::max(0, n_orb - n_beta);  // Virtual β orbitals
    
    const size_t capacity = static_cast<size_t>(n_alpha * v_alpha + 
                                                n_beta  * v_beta);
    result.reserve(capacity);

    for_each_single(ket, n_orb, [&](const Det& det) {
        result.push_back(det);
    });
    
    return result;
}

std::vector<Det> collect_doubles(const Det& ket, int n_orb) {
    std::vector<Det> result;
    
    const int n_alpha = popcount(ket.alpha);
    const int n_beta  = popcount(ket.beta);
    const int v_alpha = std::max(0, n_orb - n_alpha);
    const int v_beta  = std::max(0, n_orb - n_beta);
    
    // Estimate capacity: C(n_α,2)·C(v_α,2) + C(n_β,2)·C(v_β,2) + n_α·v_α·n_β·v_β
    const size_t doubles_aa = n_choose_k(n_alpha, 2) * n_choose_k(v_alpha, 2);
    const size_t doubles_bb = n_choose_k(n_beta,  2) * n_choose_k(v_beta,  2);
    const size_t doubles_ab = static_cast<size_t>(n_alpha * v_alpha) *
                             static_cast<size_t>(n_beta  * v_beta);
    
    result.reserve(doubles_aa + doubles_bb + doubles_ab);

    for_each_double(ket, n_orb, [&](const Det& det) {
        result.push_back(det);
    });
    
    return result;
}

std::vector<Det> collect_connected(const Det& ket, int n_orb) {
    std::vector<Det> result;
    
    const int n_alpha = popcount(ket.alpha);
    const int n_beta  = popcount(ket.beta);
    const int v_alpha = std::max(0, n_orb - n_alpha);
    const int v_beta  = std::max(0, n_orb - n_beta);
    
    // Singles + doubles capacity
    const size_t singles = static_cast<size_t>(n_alpha * v_alpha + n_beta * v_beta);
    const size_t doubles_aa = n_choose_k(n_alpha, 2) * n_choose_k(v_alpha, 2);
    const size_t doubles_bb = n_choose_k(n_beta,  2) * n_choose_k(v_beta,  2);
    const size_t doubles_ab = static_cast<size_t>(n_alpha * v_alpha) *
                             static_cast<size_t>(n_beta  * v_beta);
    
    result.reserve(singles + doubles_aa + doubles_bb + doubles_ab);

    for_each_connected(ket, n_orb, [&](const Det& det) {
        result.push_back(det);
    });
    
    return result;
}

} // namespace detnqs::det_ops
