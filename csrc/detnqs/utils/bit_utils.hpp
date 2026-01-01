// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bit_utils.hpp
 * @brief High-performance bit manipulation for Slater determinants.
 *
 * Provides efficient operations for orbital occupancy manipulation,
 * fermion permutation parity, and second quantization operators.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include "types.hpp"
#include <bit>
#include <concepts>
#include <vector>

namespace detnqs {

// ============================================================================
// Bit Counting and Scanning
// ============================================================================

/**
 * Count set bits (population count).
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr int popcount(T x) noexcept {
    return static_cast<int>(std::popcount(x));
}

/**
 * Count trailing zeros (lowest set bit position).
 * Returns sizeof(T)*8 if x == 0.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr int ctz(T x) noexcept {
    return static_cast<int>(std::countr_zero(x));
}

/**
 * Count leading zeros.
 * Returns sizeof(T)*8 if x == 0.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr int clz(T x) noexcept {
    return static_cast<int>(std::countl_zero(x));
}

// ============================================================================
// Bit Manipulation
// ============================================================================

/**
 * Clear lowest set bit: x & (x - 1).
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T clear_lsb(T x) noexcept {
    return x & (x - 1);
}

/**
 * Isolate lowest set bit: x & -x.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T isolate_lsb(T x) noexcept {
    return x & (~x + 1);
}

/**
 * Test if bit at position pos is set.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr bool test_bit(T x, int pos) noexcept {
    return (x >> pos) & 1;
}

/**
 * Set bit at position pos.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T set_bit(T x, int pos) noexcept {
    return x | (T{1} << pos);
}

/**
 * Clear bit at position pos.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T clear_bit(T x, int pos) noexcept {
    return x & ~(T{1} << pos);
}

/**
 * Flip bit at position pos.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T flip_bit(T x, int pos) noexcept {
    return x ^ (T{1} << pos);
}

// ============================================================================
// Bitmask Utilities
// ============================================================================

/**
 * Create mask with n lowest bits set.
 * Example: make_mask(3) = 0b111
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr T make_mask(int n) noexcept {
    if (n <= 0) return 0;
    if (n >= static_cast<int>(sizeof(T) * 8)) return ~T{0};
    return (T{1} << n) - 1;
}

/**
 * Extract occupied orbital indices from bitmask.
 * Returns sorted indices of set bits.
 */
template<std::unsigned_integral T>
[[nodiscard]] std::vector<int> extract_bits(T mask) {
    std::vector<int> indices;
    indices.reserve(popcount(mask));
    
    while (mask) {
        indices.push_back(ctz(mask));
        mask = clear_lsb(mask);
    }
    
    return indices;
}

/**
 * Extract virtual orbital indices in [0, n_orb).
 * Returns unoccupied orbital positions.
 */
template<std::unsigned_integral T>
[[nodiscard]] std::vector<int> extract_virtuals(T occ_mask, int n_orb) {
    const T universe = (n_orb >= static_cast<int>(sizeof(T) * 8)) 
                       ? ~T{0} 
                       : make_mask<T>(n_orb);
    
    return extract_bits(~occ_mask & universe);
}

// ============================================================================
// Parity and Permutation Analysis
// ============================================================================

/**
 * Count set bits in range [0, pos).
 * Used for fermion permutation parity calculation.
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr int popcount_below(T x, int pos) noexcept {
    if (pos <= 0) return 0;
    return popcount(x & make_mask<T>(pos));
}

/**
 * Compute fermion permutation parity.
 *
 * Calculates sign factor (-1)^p where p counts operator crossings
 * in second quantization: â†_i₁...â†_iₙ â_j₁...â_jₘ.
 *
 * Algorithm: For each particle position i_k, count holes j_l with j_l < i_k.
 * Total crossings determine exchange parity.
 *
 * @param particles  Bitmask of creation operator positions
 * @param holes      Bitmask of annihilation operator positions
 * @return           0 for even parity (+1), 1 for odd parity (-1)
 */
template<std::unsigned_integral T>
[[nodiscard]] constexpr int parity(T particles, T holes) noexcept {
    int count = 0;
    
    T temp = particles;
    while (temp) {
        const int p_idx = ctz(temp);
        count += popcount_below(holes, p_idx);
        temp = clear_lsb(temp);
    }
    
    return count & 1;
}

} // namespace detnqs
