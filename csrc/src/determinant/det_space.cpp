// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_space.cpp
 * @brief Determinant collection operations and utilities.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <lever/determinant/det_space.hpp>
#include <lever/determinant/det_ops.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace lever {

DetMap DetMap::from_list(std::vector<Det> dets) {
    std::sort(dets.begin(), dets.end());
    dets.erase(std::unique(dets.begin(), dets.end()), dets.end());
    return DetMap(std::move(dets));
}

DetMap DetMap::from_ordered(std::vector<Det> dets, bool verify_unique) {
    if (verify_unique) {
        // Strict verification: throw on duplicates
        std::unordered_set<Det> seen;
        seen.reserve(dets.size());
        for (const auto& d : dets) {
            if (!seen.insert(d).second) {
                throw std::invalid_argument(
                    "DetMap::from_ordered: duplicate determinant found");
            }
        }
    } else {
        // Tolerant mode: remove duplicates preserving first occurrence
        std::unordered_set<Det> seen;
        seen.reserve(dets.size());
        auto new_end = std::remove_if(dets.begin(), dets.end(),
            [&seen](const Det& d) { return !seen.insert(d).second; });
        dets.erase(new_end, dets.end());
    }
    return DetMap(std::move(dets));
}

DetMap::DetMap(std::vector<Det> ordered_unique)
    : dets_(std::move(ordered_unique)) {
    if (dets_.size() > std::numeric_limits<u32>::max()) {
        throw std::length_error("DetMap size exceeds u32 index capacity");
    }
    det2idx_.reserve(dets_.size());
    for (size_t i = 0; i < dets_.size(); ++i) {
        det2idx_.emplace(dets_[i], static_cast<u32>(i));
    }
}

std::optional<u32> DetMap::get_idx(const Det& d) const {
    auto it = det2idx_.find(d);
    return (it != det2idx_.end()) ? std::optional{it->second} : std::nullopt;
}

bool DetMap::contains(const Det& d) const noexcept {
#if defined(__cpp_lib_unordered_map_contains) && \
    (__cpp_lib_unordered_map_contains >= 201811L)
    return det2idx_.contains(d);
#else
    return det2idx_.count(d) > 0;
#endif
}

const Det& DetMap::get_det(u32 i) const {
    if (i >= dets_.size()) {
        throw std::out_of_range(
            "DetMap::get_det: index " + std::to_string(i) +
            " out of range [0, " + std::to_string(dets_.size()) + ")");
    }
    return dets_[i];
}

size_t DetMap::size() const noexcept { return dets_.size(); }
const std::vector<Det>& DetMap::all_dets() const noexcept { return dets_; }

namespace det_space {

/**
 * Sort and deduplicate determinant list.
 * Complexity: O(n log n)
 */
std::vector<Det> canonicalize(std::vector<Det> dets) {
    std::sort(dets.begin(), dets.end());
    dets.erase(std::unique(dets.begin(), dets.end()), dets.end());
    return dets;
}

/**
 * Compute set union a ∪ b using hash table.
 * Complexity: O(|a| + |b|) average, O(|a||b|) worst-case
 */
std::vector<Det> set_union_hash(std::span<const Det> a,
                                std::span<const Det> b,
                                bool sorted) {
    std::unordered_set<Det> unique_set;
    unique_set.reserve(a.size() + b.size());
    unique_set.insert(a.begin(), a.end());
    unique_set.insert(b.begin(), b.end());

    std::vector<Det> out(unique_set.begin(), unique_set.end());
    if (sorted) {
        std::sort(out.begin(), out.end());
    }
    return out;
}

/**
 * Merge two sorted unique determinant lists.
 * Requires: a and b are sorted and contain no duplicates.
 * Complexity: O(|a| + |b|)
 */
std::vector<Det> merge_sorted(std::span<const Det> a, std::span<const Det> b) {
#ifndef NDEBUG
    auto is_sorted_unique = [](std::span<const Det> v) {
        return std::is_sorted(v.begin(), v.end()) &&
               std::adjacent_find(v.begin(), v.end()) == v.end();
    };
    assert(is_sorted_unique(a) && is_sorted_unique(b));
#endif

    std::vector<Det> out;
    out.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(out));
    return out;
}

/**
 * Compute a ∪ b preserving order from a, then b.
 * Complexity: O(|a| + |b|) average
 */
std::vector<Det> stable_union(std::span<const Det> a, std::span<const Det> b) {
    std::vector<Det> out;
    out.reserve(a.size() + b.size());
    out.assign(a.begin(), a.end());

    std::unordered_set<Det> seen(a.begin(), a.end());
    seen.reserve(a.size() + b.size());

    for (const auto& det : b) {
        if (seen.insert(det).second) {
            out.push_back(det);
        }
    }
    return out;
}

/**
 * Generic neighbor generator using visitor pattern.
 *
 * Template Gen: callable as gen(det, n_orb, visitor) where
 * visitor receives each generated determinant via visitor(det).
 * Avoids std::function overhead for hot inner loops.
 */
template<class Gen>
static std::vector<Det> generate_from_kets(std::span<const Det> kets,
                                           int n_orb,
                                           bool sorted,
                                           Gen&& gen) {
    if (kets.empty()) return {};

    std::unordered_set<Det> unique_set;
    unique_set.reserve(kets.size() * 8);  // Heuristic for singles/doubles

    for (const auto& ket : kets) {
        std::forward<Gen>(gen)(ket, n_orb, [&](const Det& d) {
            unique_set.insert(d);
        });
    }

    std::vector<Det> out(unique_set.begin(), unique_set.end());
    if (sorted) {
        std::sort(out.begin(), out.end());
    }
    return out;
}

/**
 * Generate all single excitations from kets: |I⟩ → a^†_p a_q |I⟩.
 */
std::vector<Det> generate_singles(std::span<const Det> kets,
                                  int n_orb,
                                  bool sorted) {
    return generate_from_kets(kets, n_orb, sorted,
        [](const Det& ket, int norb, auto&& visit) {
            det_ops::for_each_single(ket, norb,
                                    std::forward<decltype(visit)>(visit));
        });
}

/**
 * Generate all double excitations from kets: |I⟩ → a^†_p a^†_q a_s a_r |I⟩.
 */
std::vector<Det> generate_doubles(std::span<const Det> kets,
                                  int n_orb,
                                  bool sorted) {
    return generate_from_kets(kets, n_orb, sorted,
        [](const Det& ket, int norb, auto&& visit) {
            det_ops::for_each_double(ket, norb,
                                    std::forward<decltype(visit)>(visit));
        });
}

/**
 * Generate all singly and doubly connected determinants from kets.
 */
std::vector<Det> generate_connected(std::span<const Det> kets,
                                    int n_orb,
                                    bool sorted) {
    return generate_from_kets(kets, n_orb, sorted,
        [](const Det& ket, int norb, auto&& visit) {
            det_ops::for_each_connected(ket, norb,
                                       std::forward<decltype(visit)>(visit));
        });
}

/**
 * Generate connected determinants excluding those in the exclude set.
 * Useful for building C-space as complement of S-space.
 */
std::vector<Det> generate_complement(std::span<const Det> kets,
                                     int n_orb,
                                     const DetMap& exclude,
                                     bool sorted) {
    if (kets.empty()) return {};

    std::unordered_set<Det> unique_set;
    unique_set.reserve(kets.size() * 16);

    for (const auto& ket : kets) {
        det_ops::for_each_connected(ket, n_orb, [&](const Det& d) {
            if (!exclude.contains(d)) {
                unique_set.insert(d);
            }
        });
    }

    std::vector<Det> out(unique_set.begin(), unique_set.end());
    if (sorted) {
        std::sort(out.begin(), out.end());
    }
    return out;
}

} // namespace det_space
} // namespace lever
