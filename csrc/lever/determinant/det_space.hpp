// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_space.hpp
 * @brief Determinant collections: indexing, set operations, and generators.
 *
 * Thread-safety: All classes and free functions here are safe for concurrent
 * reads. External synchronization is required for concurrent writes or
 * mixed read/write access.
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <lever/utils/constants.hpp>
#include <lever/utils/types.hpp>

#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <version>   // for feature test macros like __cpp_lib_unordered_map_contains

namespace lever {

/**
 * @class DetMap
 * @brief Immutable bidirectional mapping between Det and dense indices [0, N).
 *
 * Construction builds a stable index for the given ordered list (or canonical
 * ordering for from_list). After construction, the object is read-only and
 * safe for concurrent reads.
 */
class DetMap {
public:
    DetMap() = default;

    /// Build a canonical map: sort + unique the given determinants.
    [[nodiscard]] static DetMap from_list(std::vector<Det> dets);

    /**
     * @brief Build a map preserving the input order.
     * @param dets Input determinants.
     * @param verify_unique If true, throws std::invalid_argument on duplicates.
     *                      If false, duplicates are dropped, keeping the first.
     */
    [[nodiscard]] static DetMap from_ordered(std::vector<Det> dets, bool verify_unique = true);

    /// Return the dense index of a determinant, or std::nullopt if not found.
    [[nodiscard]] std::optional<u32> get_idx(const Det& d) const;

    /// Membership test. Potentially faster than get_idx().
    [[nodiscard]] bool contains(const Det& d) const noexcept;

    /// Access determinant by index. Throws std::out_of_range if invalid.
    [[nodiscard]] const Det& get_det(u32 i) const;

    /// Number of determinants.
    [[nodiscard]] size_t size() const noexcept;

    /// Immutable view of the underlying determinant list (index order).
    [[nodiscard]] const std::vector<Det>& all_dets() const noexcept;

private:
    explicit DetMap(std::vector<Det> ordered_unique);

    std::vector<Det> dets_;
    std::unordered_map<Det, u32> det2idx_;
};

/**
 * @namespace det_space
 * @brief Stateless utilities operating on collections of determinants.
 *
 * Notes on determinism:
 * - Functions with `sorted=true` guarantee canonical (sorted, unique) output.
 * - Functions with `sorted=false` may return in unspecified order (fast path).
 */
namespace det_space {

// --- Basic Set Operations ---

/// Sort and deduplicate a vector of determinants (canonical order).
[[nodiscard]] std::vector<Det> canonicalize(std::vector<Det> dets);

/// Hash-based union. If `sorted=true`, the output is canonicalized.
[[nodiscard]] std::vector<Det> set_union_hash(std::span<const Det> a,
                                              std::span<const Det> b,
                                              bool sorted = false);

/// Merge two *sorted & unique* sets into a *sorted & unique* union.
[[nodiscard]] std::vector<Det> merge_sorted(std::span<const Det> a,
                                            std::span<const Det> b);

/**
 * @brief Stable union: preserve the order of `a`, then append new elements from `b`.
 * Duplicates in both `a` and `b` are removed (first occurrence wins).
 */
[[nodiscard]] std::vector<Det> stable_union(std::span<const Det> a,
                                            std::span<const Det> b);

// --- Batch Generators ---

/// Unique single excitations from a set of kets. Canonical if `sorted=true`.
[[nodiscard]] std::vector<Det> generate_singles(std::span<const Det> kets,
                                                int n_orb,
                                                bool sorted = false);

/// Unique double excitations from a set of kets. Canonical if `sorted=true`.
[[nodiscard]] std::vector<Det> generate_doubles(std::span<const Det> kets,
                                                int n_orb,
                                                bool sorted = false);

/// Unique connected (singles+doubles) determinants. Canonical if `sorted=true`.
[[nodiscard]] std::vector<Det> generate_connected(std::span<const Det> kets,
                                                  int n_orb,
                                                  bool sorted = false);

/**
 * @brief Build the complementary space: all connected dets not in `exclude`.
 * The typical usage is to generate C from S, excluding S itself.
 */
[[nodiscard]] std::vector<Det> generate_complement(std::span<const Det> kets,
                                                   int n_orb,
                                                   const DetMap& exclude,
                                                   bool sorted = false);

} // namespace det_space
} // namespace lever