// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det.hpp
 * @brief Slater determinant (Fock state) representation via bitstring encoding.
 *
 * Represents |x⟩ = |alpha⟩|beta⟩ using 64-bit occupation patterns (bit i → orbital i).
 * Now supports up to 64 spatial orbitals.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Last Modified: January, 2026
 */

#pragma once

#include <detnqs/utils/types.hpp>
#include <compare>
#include <concepts>
#include <functional>

namespace detnqs {

/**
 * Slater determinant as alpha/beta occupation bitstrings.
 *
 * POD type for efficient storage and manipulation. Bit i (LSB=0) corresponds
 * to spatial orbital i. Trivially copyable and hashable.
 */
struct Det {
    u64 alpha;  ///< alpha-spin occupation
    u64 beta;   ///< beta-spin occupation

    /// Three-way comparison (enables ==, !=, <, <=, >, >=)
    constexpr auto operator<=>(const Det&) const = default;
};

// ============================================================================
// Configuration Concept (Future: CSF support)
// ============================================================================

/**
 * Placeholder for future extensions to Configuration State Functions (CSF).
 * Currently satisfied by Det only.
 */
template<typename T>
concept ConfigLike = requires(const T& cfg) {
    { cfg.alpha } -> std::convertible_to<u64>;
    { cfg.beta }  -> std::convertible_to<u64>;
};

static_assert(ConfigLike<Det>);

} // namespace detnqs

// ============================================================================
// Standard Library Hash Specialization
// ============================================================================

namespace std {

/**
 * Hash functor for detnqs::Det (enables use in unordered containers).
 *
 * Combines alpha/beta hashes via Boost-style hash_combine with golden ratio constant.
 */
template<>
struct hash<detnqs::Det> {
    [[nodiscard]]
    std::size_t operator()(const detnqs::Det& d) const noexcept {
        const std::size_t h_α = std::hash<detnqs::u64>{}(d.alpha);
        const std::size_t h_β = std::hash<detnqs::u64>{}(d.beta);
        
        // h_combined = h_alpha \oplus (h_beta + phi_64 + (h_alpha << 6) + (h_alpha >> 2))
        // where phi_64 = 2^{64}/phi (golden ratio constant)
        return h_α ^ (h_β + 0x9e3779b97f4a7c15ULL + (h_α << 6) + (h_α >> 2));
    }
};

} // namespace std
