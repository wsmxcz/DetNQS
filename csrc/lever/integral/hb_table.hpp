// Copyright 2025 The Nebula-QC Authors
// SPDX-License-Identifier: Apache-2.0
//
// Heat-bath table for double excitations with optimized query interface.
//
// Design Philosophy:
//   - Build once with minimal numerical threshold (filters only floating-point noise)
//   - Query dynamically with variable eps1 (actual heat-bath selection threshold)
//   - Leverage descending-weight ordering for efficient binary-search cutoff
//
// Storage:
//   - CSR-like layout: each row {i>j} stores virtual pairs (a>b) with |⟨ij||ab⟩| >= threshold
//   - Rows are sorted by weight descending for fast eps1-based cutoff
//   - Spin-preserving: (a,b) matches (i,j) spin pattern up to permutation
//
// Performance:
//   - Build: O(n_so^4) worst-case, heavily pruned by threshold and spin rules
//   - Query: O(log avg_row_size) per cutoff via binary search

#pragma once

#include <lever/integral/integral_so.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace lever {

/**
 * @brief Build-time options for the heat-bath table.
 * 
 * The threshold here is purely for filtering numerical noise (e.g., 1e-14),
 * NOT for heat-bath selection logic. Actual selection happens at query time
 * via the eps1 parameter in with_cutoff().
 */
struct HBBuildOptions {
    double threshold = 1e-14;  ///< Filter integrals with |⟨ij||ab⟩| < threshold (noise only)
};

/**
 * @brief Heat-bath table: pre-indexed significant double excitations.
 *
 * For each occupied spin-orbital pair {i>j}, stores all virtual pairs (a>b)
 * with |⟨ij||ab⟩| >= threshold, sorted by weight descending. This enables:
 *   1. Avoid enumerating O(n_orb^4) candidates (compute savings)
 *   2. Fast dynamic filtering via eps1 (flexibility)
 *   3. Deterministic ordering (reproducibility)
 *
 * Usage:
 *   HeatBathTable hb(so_ints, {.threshold = 1e-14});
 *   hb.build();
 *   
 *   auto row = hb.row_view(i_so, j_so);
 *   auto filtered = row.with_cutoff(current_eps1);  // O(log N) binary search
 *   for (size_t k = 0; k < filtered.len; ++k) {
 *       int a = filtered.a[k], b = filtered.b[k];
 *       // guaranteed: filtered.w[k] >= current_eps1
 *   }
 */
class HeatBathTable {
public:
    /// Single entry: virtual pair (a,b) with weight w = |⟨ij||ab⟩|
    struct Entry { 
        int a; 
        int b; 
        double w; 
    };

    /**
     * @brief Construct table builder (does not build yet; call build() explicitly).
     * @param so_ints Spin-orbital integral provider
     * @param opt Build options (threshold for noise filtering)
     */
    explicit HeatBathTable(const IntegralSO& so_ints, 
                          HBBuildOptions opt = HBBuildOptions());

    /**
     * @brief Build the entire table.
     * 
     * Complexity: O(n_so^4) in worst case, but:
     *   - Pruned by threshold (keeps only |⟨ij||ab⟩| >= threshold)
     *   - Pruned by spin rules (aa/bb/ab patterns only)
     *   - Each row sorted descending for fast query
     * 
     * This should be called once at initialization. Subsequent queries with
     * varying eps1 do NOT require rebuild.
     */
    void build();

    // --- Accessors ---

    [[nodiscard]] int n_so() const noexcept { return n_so_; }
    [[nodiscard]] std::size_t num_rows() const noexcept { return num_rows_; }

    /**
     * @brief Canonical row ID for unordered pair {i,j}.
     * @return Row index if i!=j, else max (invalid)
     */
    [[nodiscard]] std::size_t row_id(int i, int j) const noexcept;

    /// Number of entries in row {i,j} (before any eps1 cutoff)
    [[nodiscard]] std::size_t row_size(int i, int j) const noexcept;

    /// Sum of weights in row {i,j} (before any eps1 cutoff)
    [[nodiscard]] double row_weight_sum(int i, int j) const noexcept;

    /**
     * @brief Lightweight zero-copy view into a row's storage.
     * 
     * The view exposes raw pointers to contiguous CSR data. It supports
     * dynamic cutoff via with_cutoff(eps1) without copying data.
     */
    struct RowView {
        const int*    a   = nullptr;  ///< Virtual orbital indices a
        const int*    b   = nullptr;  ///< Virtual orbital indices b
        const double* w   = nullptr;  ///< Weights (descending order)
        std::size_t   len = 0;        ///< Number of entries

        [[nodiscard]] bool empty() const noexcept { return len == 0; }

        /**
         * @brief Return a truncated view with only entries >= eps1.
         * 
         * Leverages descending weight order: uses binary search to find
         * cutoff position, then returns prefix.
         * 
         * Complexity: O(log len)
         * 
         * @param eps1 Minimum weight threshold (heat-bath selection parameter)
         * @return New RowView with len' <= len, where all w[k] >= eps1
         */
        [[nodiscard]] RowView with_cutoff(double eps1) const noexcept {
            if (eps1 <= 0.0 || len == 0) return *this;

            // Binary search: find first index where w[idx] < eps1
            // Since w is descending, this gives us the cutoff point
            std::size_t left = 0, right = len;
            while (left < right) {
                std::size_t mid = left + (right - left) / 2;
                if (w[mid] >= eps1) {
                    left = mid + 1;  // Search right half
                } else {
                    right = mid;     // Search left half
                }
            }
            
            return RowView{a, b, w, left};
        }
    };

    /**
     * @brief Get full row view for unordered pair {i,j}.
     * @return RowView with all entries (before eps1 cutoff). Empty if i==j or no entries.
     */
    [[nodiscard]] RowView row_view(int i, int j) const noexcept;

    /// Estimate total memory usage in bytes
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    const IntegralSO& so_;
    const int n_so_;
    std::size_t num_rows_;  ///< n_so * (n_so - 1) / 2
    HBBuildOptions opt_;

    // CSR-like storage
    std::vector<std::size_t> row_offsets_;  ///< [num_rows_ + 1], cumulative nnz
    std::vector<int>         a_;            ///< Flat array of 'a' indices
    std::vector<int>         b_;            ///< Flat array of 'b' indices
    std::vector<double>      w_;            ///< Flat array of weights (descending per row)
    std::vector<double>      row_sum_;      ///< [num_rows_], sum of weights per row

    /// Canonical index for unordered pair {i>j}
    static inline std::size_t unordered_pair_index(int i, int j) noexcept {
        if (i < j) std::swap(i, j);
        return static_cast<std::size_t>(i) * (i - 1) / 2 + j;
    }

    /// Build single row {i>j}, appending to buffer
    void build_row_(int i, int j, std::vector<Entry>& buffer, std::size_t& nnz_total);

    /// Finalize CSR layout from temporary row buffers
    void finalize_layout_(const std::vector<std::vector<Entry>>& rows, std::size_t nnz_total);
};

} // namespace lever
