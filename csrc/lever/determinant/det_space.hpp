// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_space.hpp
 * @brief Determinant space operations: bidirectional indexing, set algebra, and excitation generators.
 *
 * Thread-safety: Read-only after construction; concurrent writes require external synchronization.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/utils/constants.hpp>
#include <lever/utils/types.hpp>

#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace lever {

/**
 * Immutable bidirectional mapping: Det ↔ dense index [0, N).
 *
 * Construction strategies:
 *   - from_list: Canonical ordering (sort + unique)
 *   - from_ordered: Preserve input order with optional uniqueness check
 */
class DetMap {
public:
    DetMap() = default;

    /// Canonical map: sorted and deduplicated.
    [[nodiscard]] static DetMap from_list(std::vector<Det> dets);

    /**
     * Preserve input order.
     *
     * @param verify_unique  true: throw on duplicates; false: keep first occurrence
     */
    [[nodiscard]] static DetMap from_ordered(std::vector<Det> dets, bool verify_unique = true);

    /// Query index by determinant.
    [[nodiscard]] std::optional<u32> get_idx(const Det& d) const;

    /// O(1) membership test.
    [[nodiscard]] bool contains(const Det& d) const noexcept;

    /// Index → determinant (bounds-checked).
    [[nodiscard]] const Det& get_det(u32 i) const;

    [[nodiscard]] size_t size() const noexcept;
    [[nodiscard]] const std::vector<Det>& all_dets() const noexcept;

private:
    explicit DetMap(std::vector<Det> ordered_unique);

    std::vector<Det> dets_;
    std::unordered_map<Det, u32> det2idx_;
};

/**
 * Stateless operations on determinant collections.
 *
 * Determinism guarantees:
 *   - sorted=true  → canonical output (sorted, unique)
 *   - sorted=false → unspecified order (performance optimized)
 */
namespace det_space {

// ─── Set Operations ───────────────────────────────────────────────────

/// Sort and deduplicate (canonical form).
[[nodiscard]] std::vector<Det> canonicalize(std::vector<Det> dets);

/// Hash-based A∪B. Optional canonicalization.
[[nodiscard]] std::vector<Det> set_union_hash(
    std::span<const Det> a,
    std::span<const Det> b,
    bool sorted = false
);

/// Merge presorted sets → sorted union.
[[nodiscard]] std::vector<Det> merge_sorted(
    std::span<const Det> a,
    std::span<const Det> b
);

/// Order-preserving union: A then (B \ A).
[[nodiscard]] std::vector<Det> stable_union(
    std::span<const Det> a,
    std::span<const Det> b
);

// ─── Excitation Generators ────────────────────────────────────────────

/// Generate all unique single excitations S_1 = {T̂_i^a |ψ⟩}.
[[nodiscard]] std::vector<Det> generate_singles(
    std::span<const Det> kets,
    int n_orb,
    bool sorted = false
);

/// Generate all unique double excitations S_2 = {T̂_ij^ab |ψ⟩}.
[[nodiscard]] std::vector<Det> generate_doubles(
    std::span<const Det> kets,
    int n_orb,
    bool sorted = false
);

/// Combined singles and doubles: S_1 ∪ S_2.
[[nodiscard]] std::vector<Det> generate_connected(
    std::span<const Det> kets,
    int n_orb,
    bool sorted = false
);

/// Complementary space: (S_1 ∪ S_2) \ exclude.
[[nodiscard]] std::vector<Det> generate_complement(
    std::span<const Det> kets,
    int n_orb,
    const DetMap& exclude,
    bool sorted = false
);

} // namespace det_space
} // namespace lever
