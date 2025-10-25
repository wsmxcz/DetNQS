// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det.hpp
 * @brief Slater determinant representation via bitstring encoding.
 *
 * Represents |Ψ⟩ = |α⟩⊗|β⟩ using 64-bit occupation patterns (bit i → orbital i).
 * Optimized for fast bitwise operations, hashing, and trivial copying.
 * Supports up to 64 spatial orbitals.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <lever/utils/types.hpp>
#include <compare>
#include <concepts>
#include <functional>

namespace lever {

/**
 * Slater determinant as α/β occupation bitstrings.
 *
 * POD type for efficient storage and manipulation. Bit i (LSB=0) corresponds
 * to spatial orbital i. Trivially copyable and hashable.
 */
struct Det {
    u64 alpha;  ///< α-spin occupation (bit i = 1 if orbital i occupied)
    u64 beta;   ///< β-spin occupation

    /// Three-way comparison (enables ==, !=, <, <=, >, >=)
    constexpr auto operator<=>(const Det&) const = default;
};

// ============================================================================
// Configuration Concept (Future: CSF support)
// ============================================================================

/**
 * Generalized electron configuration concept.
 *
 * Placeholder for future extensions to Configuration State Functions (CSF).
 * Currently satisfied by Det only.
 */
template<typename T>
concept ConfigLike = requires(const T& cfg) {
    { cfg.alpha } -> std::convertible_to<u64>;
    { cfg.beta }  -> std::convertible_to<u64>;
};

static_assert(ConfigLike<Det>);

} // namespace lever

// ============================================================================
// Standard Library Hash Specialization
// ============================================================================

namespace std {

/**
 * Hash functor for lever::Det (enables use in unordered containers).
 *
 * Combines α/β hashes via Boost-style hash_combine with golden ratio constant.
 */
template<>
struct hash<lever::Det> {
    [[nodiscard]]
    std::size_t operator()(const lever::Det& d) const noexcept {
        const std::size_t h_α = std::hash<lever::u64>{}(d.alpha);
        const std::size_t h_β = std::hash<lever::u64>{}(d.beta);
        
        // h_combined = h_α ⊕ (h_β + φ_64 + (h_α << 6) + (h_α >> 2))
        // where φ_64 = 2⁶⁴/φ (golden ratio constant)
        return h_α ^ (h_β + 0x9e3779b97f4a7c15ULL + (h_α << 6) + (h_α >> 2));
    }
};

} // namespace std
