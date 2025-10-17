// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det.hpp
 * @brief Slater determinant representation using bitstring encoding.
 */

#pragma once

#include <lever/utils/types.hpp>
#include <compare>
#include <concepts> // Required for std::convertible_to
#include <functional>

namespace lever {

/**
 * @brief Slater determinant represented as alpha/beta occupation bitstrings.
 * 
 * Bit i (LSB=0) corresponds to spatial orbital i. This POD (Plain Old Data)
 * type is optimized for fast bitwise operations, efficient hashing, and
 * trivial copying. It supports up to 64 spatial orbitals.
 */
struct Det {
    u64 alpha;  ///< Alpha spin occupation bitstring
    u64 beta;   ///< Beta spin occupation bitstring

    /// C++20 three-way comparison (enables all relational operators)
    constexpr auto operator<=>(const Det&) const = default;
};

// ============================================================================
// Future Extension: Configuration-like Concept
// ============================================================================

/**
 * @brief Concept for generalized electron configurations (e.g., Det, CSF).
 * 
 * This is a placeholder for future extensions to support other configuration
 * types like Configuration State Functions (CSF).
 */
template<typename T>
concept ConfigLike = requires(const T& cfg) {
    { cfg.alpha } -> std::convertible_to<u64>;
    { cfg.beta }  -> std::convertible_to<u64>;
};

// Ensure Det satisfies the concept for compile-time validation
static_assert(ConfigLike<Det>);

} // namespace lever

// ============================================================================
// Standard Library Hash Specialization
// ============================================================================

// The specialization of std::hash must be within the std namespace.
namespace std {

/**
 * @brief Hash function for lever::Det.
 * 
 * Enables Det to be used as a key in unordered containers like
 * std::unordered_map. Uses a Boost-style hash_combine algorithm.
 */
template<>
struct hash<lever::Det> {
    [[nodiscard]]
    std::size_t operator()(const lever::Det& d) const noexcept {
        const std::size_t h1 = std::hash<lever::u64>{}(d.alpha);
        const std::size_t h2 = std::hash<lever::u64>{}(d.beta);
        
        // Boost-style hash_combine with a 64-bit golden ratio constant
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

} // namespace std