// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_space.cpp
 * @brief Implementation of determinant collections and utilities.
 */

#include <lever/determinant/det_space.hpp>
#include <lever/determinant/det_ops.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace lever {

// ============================================================================
// DetMap Implementation
// ============================================================================

DetMap DetMap::from_list(std::vector<Det> dets) {
    std::sort(dets.begin(), dets.end());
    dets.erase(std::unique(dets.begin(), dets.end()), dets.end());
    return DetMap(std::move(dets));
}

DetMap DetMap::from_ordered(std::vector<Det> dets, bool verify_unique) {
    if (verify_unique) {
        std::unordered_set<Det> seen;
        seen.reserve(dets.size());
        for (const auto& d : dets) {
            if (!seen.insert(d).second) {
                throw std::invalid_argument("DetMap::from_ordered: duplicate determinant found");
            }
        }
        return DetMap(std::move(dets));
    } else {
        // Remove duplicates keeping the first occurrence (stable).
        std::unordered_set<Det> seen;
        seen.reserve(dets.size());
        auto new_end = std::remove_if(dets.begin(), dets.end(),
            [&seen](const Det& d) { return !seen.insert(d).second; });
        dets.erase(new_end, dets.end());
        return DetMap(std::move(dets));
    }
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
    if (auto it = det2idx_.find(d); it != det2idx_.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool DetMap::contains(const Det& d) const noexcept {
#if defined(__cpp_lib_unordered_map_contains) && (__cpp_lib_unordered_map_contains >= 201811L)
    return det2idx_.contains(d);
#else
    return det2idx_.count(d) > 0;
#endif
}

const Det& DetMap::get_det(u32 i) const {
    if (i >= dets_.size()) {
        throw std::out_of_range("DetMap::get_det: index " + std::to_string(i) +
                                " is out of range for size " + std::to_string(dets_.size()));
    }
    return dets_[i];
}

size_t DetMap::size() const noexcept { return dets_.size(); }
const std::vector<Det>& DetMap::all_dets() const noexcept { return dets_; }

// ============================================================================
// det_space Implementation
// ============================================================================

namespace det_space {

std::vector<Det> canonicalize(std::vector<Det> dets) {
    std::sort(dets.begin(), dets.end());
    dets.erase(std::unique(dets.begin(), dets.end()), dets.end());
    return dets;
}

std::vector<Det> set_union_hash(std::span<const Det> a,
                                std::span<const Det> b,
                                bool sorted) {
    std::unordered_set<Det> s;
    s.reserve(a.size() + b.size());
    s.insert(a.begin(), a.end());
    s.insert(b.begin(), b.end());

    std::vector<Det> out;
    out.reserve(s.size());
    out.assign(s.begin(), s.end());
    if (sorted) {
        std::sort(out.begin(), out.end());
    }
    return out;
}

std::vector<Det> merge_sorted(std::span<const Det> a, std::span<const Det> b) {
#ifndef NDEBUG
    // Preconditions: inputs must be sorted & unique.
    auto is_sorted_unique = [](std::span<const Det> v) {
        return std::is_sorted(v.begin(), v.end()) &&
               std::adjacent_find(v.begin(), v.end()) == v.end();
    };
    assert(is_sorted_unique(a) && is_sorted_unique(b));
#endif

    std::vector<Det> out;
    out.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(),
                   b.begin(), b.end(),
                   std::back_inserter(out));
    return out;
}

std::vector<Det> stable_union(std::span<const Det> a, std::span<const Det> b) {
    std::vector<Det> out;
    out.reserve(a.size() + b.size());
    out.assign(a.begin(), a.end());

    std::unordered_set<Det> seen(a.begin(), a.end());
    seen.reserve(a.size() + b.size());

    // Append only if not seen in `a` or previously appended from `b`.
    for (const auto& x : b) {
        if (seen.insert(x).second) {
            out.push_back(x);
        }
    }
    return out;
}

// --------------------------------------------------------------------------
// Internal helper: template-based generator to avoid std::function overhead.
// Gen must be callable as: gen(const Det&, int n_orb, Visitor&&)
// where Visitor is a callable with signature: void(const Det&).
// --------------------------------------------------------------------------
template<class Gen>
static std::vector<Det> generate_from_kets(std::span<const Det> kets,
                                           int n_orb,
                                           bool sorted,
                                           Gen&& gen) {
    if (kets.empty()) return {};

    std::unordered_set<Det> s;
    // Heuristic reserve: assume each ket produces a handful of neighbors.
    s.reserve(kets.size() * 8);

    for (const auto& k : kets) {
        std::forward<Gen>(gen)(k, n_orb, [&](const Det& d) { s.insert(d); });
    }

    std::vector<Det> out;
    out.reserve(s.size());
    out.assign(s.begin(), s.end());
    if (sorted) {
        std::sort(out.begin(), out.end());
    }
    return out;
}

std::vector<Det> generate_singles(std::span<const Det> kets, int n_orb, bool sorted) {
    return generate_from_kets(kets, n_orb, sorted,
        [](const Det& k, int norb, auto&& visit) {
            det_ops::for_each_single(k, norb, std::forward<decltype(visit)>(visit));
        });
}

std::vector<Det> generate_doubles(std::span<const Det> kets, int n_orb, bool sorted) {
    return generate_from_kets(kets, n_orb, sorted,
        [](const Det& k, int norb, auto&& visit) {
            det_ops::for_each_double(k, norb, std::forward<decltype(visit)>(visit));
        });
}

std::vector<Det> generate_connected(std::span<const Det> kets, int n_orb, bool sorted) {
    return generate_from_kets(kets, n_orb, sorted,
        [](const Det& k, int norb, auto&& visit) {
            det_ops::for_each_connected(k, norb, std::forward<decltype(visit)>(visit));
        });
}

std::vector<Det> generate_complement(std::span<const Det> kets,
                                     int n_orb,
                                     const DetMap& exclude,
                                     bool sorted) {
    if (kets.empty()) return {};

    std::unordered_set<Det> s;
    s.reserve(kets.size() * 16); // doubles can be larger than singles

    for (const auto& k : kets) {
        det_ops::for_each_connected(k, n_orb, [&](const Det& d) {
            if (!exclude.contains(d)) {
                s.insert(d);
            }
        });
    }

    std::vector<Det> out;
    out.reserve(s.size());
    out.assign(s.begin(), s.end());
    if (sorted) {
        std::sort(out.begin(), out.end());
    }
    return out;
}

} // namespace det_space
} // namespace lever