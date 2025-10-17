// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file bit_utils.hpp
 * @brief High-performance bit manipulation utilities
 */

#pragma once

#include "types.hpp"
#include <bit>
#include <concepts>
#include <vector>

namespace lever {

// ============================================================================
// Bit Counting and Scanning
// ============================================================================

/// Count number of set bits (population count)
template<std::unsigned_integral T>
[[nodiscard]] constexpr int popcount(T x) noexcept {
    return static_cast<int>(std::popcount(x));
}

/// Count trailing zeros (position of lowest set bit)
/// Returns sizeof(T)*8 if x == 0
template<std::unsigned_integral T>
[[nodiscard]] constexpr int ctz(T x) noexcept {
    return static_cast<int>(std::countr_zero(x));
}

/// Count leading zeros
/// Returns sizeof(T)*8 if x == 0
template<std::unsigned_integral T>
[[nodiscard]] constexpr int clz(T x) noexcept {
    return static_cast<int>(std::countl_zero(x));
}

// ============================================================================
// Bit Manipulation
// ============================================================================

/// Clear the lowest set bit: x & (x - 1)
template<std::unsigned_integral T>
[[nodiscard]] constexpr T clear_lsb(T x) noexcept {
    return x & (x - 1);
}

/// Isolate the lowest set bit: x & -x
template<std::unsigned_integral T>
[[nodiscard]] constexpr T isolate_lsb(T x) noexcept {
    return x & (~x + 1);
}

/// Test if bit at position pos is set
template<std::unsigned_integral T>
[[nodiscard]] constexpr bool test_bit(T x, int pos) noexcept {
    return (x >> pos) & 1;
}

/// Set bit at position pos
template<std::unsigned_integral T>
[[nodiscard]] constexpr T set_bit(T x, int pos) noexcept {
    return x | (T{1} << pos);
}

/// Clear bit at position pos
template<std::unsigned_integral T>
[[nodiscard]] constexpr T clear_bit(T x, int pos) noexcept {
    return x & ~(T{1} << pos);
}

/// Flip bit at position pos
template<std::unsigned_integral T>
[[nodiscard]] constexpr T flip_bit(T x, int pos) noexcept {
    return x ^ (T{1} << pos);
}

// ============================================================================
// Bitmask Utilities
// ============================================================================

/// Create bitmask with n lowest bits set
/// Example: make_mask(3) = 0b111
template<std::unsigned_integral T>
[[nodiscard]] constexpr T make_mask(int n) noexcept {
    if (n <= 0) return 0;
    if (n >= static_cast<int>(sizeof(T) * 8)) return ~T{0};
    return (T{1} << n) - 1;
}

/// Extract occupied orbital indices from bitmask
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

/// Extract virtual (unoccupied) orbital indices within range [0, n_orb)
template<std::unsigned_integral T>
[[nodiscard]] std::vector<int> extract_virtuals(T occ_mask, int n_orb) {
    // Handle n_orb == bitwidth edge case
    const T universe = (n_orb >= static_cast<int>(sizeof(T) * 8)) 
                       ? ~T{0} 
                       : make_mask<T>(n_orb);
    
    return extract_bits(~occ_mask & universe);
}

// ============================================================================
// Bit Parity and Permutation Counting
// ============================================================================

/// Count number of set bits in range [0, pos)
/// Used for permutation parity calculation
template<std::unsigned_integral T>
[[nodiscard]] constexpr int popcount_below(T x, int pos) noexcept {
    if (pos <= 0) return 0;
    return popcount(x & make_mask<T>(pos));
}

/// Compute parity of a permutation (0: even, 1: odd)
/// Counts crossings between two bitmasks
template<std::unsigned_integral T>
[[nodiscard]] constexpr int parity(T particles, T holes) noexcept {
    int count = 0;
    
    // Count particles crossing holes
    T temp = particles;
    while (temp) {
        const int p_idx = ctz(temp);
        count += popcount_below(holes, p_idx);
        temp = clear_lsb(temp);
    }
    
    return count & 1;
}

} // namespace lever
