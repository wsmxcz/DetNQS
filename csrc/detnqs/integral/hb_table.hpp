// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file heat_bath_table.hpp
 * @brief Heat-bath table for efficient double excitation screening.
 *
 * Design:
 *   - Build once with minimal threshold (filter numerical noise only)
 *   - Query dynamically with variable ε₁ (heat-bath selection threshold)
 *   - CSR layout: row {i>j} stores (a>b) pairs with |⟨ij||ab⟩| ≥ threshold
 *   - Descending weight order enables O(log N) binary-search cutoff
 *
 * Complexity:
 *   - Build: O(n_so⁴) worst-case, pruned by threshold and spin symmetry
 *   - Query: O(log avg_row_size) per cutoff
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <detnqs/integral/integral_so.hpp>

#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>

namespace detnqs {

/**
 * Build configuration for heat-bath table.
 *
 * Threshold here filters floating-point noise (e.g., 1e-14), NOT for
 * heat-bath selection. Actual selection uses ε₁ at query time.
 */
struct HBBuildOptions {
    double threshold = 1e-14;  ///< Noise floor for |⟨ij||ab⟩|
};

/**
 * Pre-indexed heat-bath table for double excitations.
 *
 * For each occupied pair {i>j}, stores virtual pairs (a>b) satisfying
 * |⟨ij||ab⟩| ≥ threshold, sorted by weight descending. Enables:
 *   - Avoid O(n_orb⁴) enumeration
 *   - Dynamic filtering via ε₁
 *   - Deterministic ordering
 *
 * Usage:
 * @code
 *   HeatBathTable hb(so_ints, {.threshold = 1e-14});
 *   hb.build();
 *
 *   auto row = hb.row_view(i_so, j_so);
 *   auto filtered = row.with_cutoff(eps1);  // O(log N)
 *   for (size_t k = 0; k < filtered.len; ++k) {
 *       int a = filtered.a[k], b = filtered.b[k];
 *       // Guaranteed: filtered.w[k] ≥ eps1
 *   }
 * @endcode
 */
class HeatBathTable {
public:
    /// Entry: virtual pair (a,b) with weight w = |⟨ij||ab⟩|
    struct Entry {
        int a;
        int b;
        double w;
    };

    /**
     * Construct table builder (call build() explicitly).
     *
     * @param so_ints Spin-orbital integral provider
     * @param opt     Build options
     */
    explicit HeatBathTable(const IntegralSO& so_ints,
                          HBBuildOptions opt = {});

    /**
     * Build full table.
     *
     * Prunes by threshold and spin rules, sorts each row descending.
     * Call once at initialization; subsequent queries with varying ε₁
     * do not require rebuild.
     */
    void build();

    // --- Accessors ---

    [[nodiscard]] int n_so() const noexcept { return n_so_; }
    [[nodiscard]] std::size_t num_rows() const noexcept { return num_rows_; }

    /**
     * Canonical row ID for unordered pair {i,j}.
     * @return Row index if i≠j, else SIZE_MAX
     */
    [[nodiscard]] std::size_t row_id(int i, int j) const noexcept;

    /// Entries count in row {i,j} (before ε₁ cutoff)
    [[nodiscard]] std::size_t row_size(int i, int j) const noexcept;

    /// Weight sum in row {i,j} (before ε₁ cutoff)
    [[nodiscard]] double row_weight_sum(int i, int j) const noexcept;

    /**
     * Zero-copy view into row storage.
     *
     * Exposes contiguous CSR data. Supports dynamic cutoff via
     * with_cutoff(ε₁) without copying.
     */
    struct RowView {
        const int*    a   = nullptr;  ///< Virtual index a
        const int*    b   = nullptr;  ///< Virtual index b
        const double* w   = nullptr;  ///< Weight (descending)
        std::size_t   len = 0;

        [[nodiscard]] bool empty() const noexcept { return len == 0; }

        /**
         * Return prefix where all w[k] ≥ ε₁.
         *
         * Binary search on descending weights: O(log len).
         *
         * @param eps1 Heat-bath selection threshold
         * @return     Truncated view with len' ≤ len
         */
        [[nodiscard]] RowView with_cutoff(double eps1) const noexcept {
            if (eps1 <= 0.0 || len == 0) return *this;

            // Find first index where w[idx] < eps1
            std::size_t left = 0, right = len;
            while (left < right) {
                std::size_t mid = left + (right - left) / 2;
                if (w[mid] >= eps1) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            return RowView{a, b, w, left};
        }
    };

    /**
     * Get full row view for {i,j}.
     * @return RowView with all entries (before ε₁ cutoff). Empty if i=j.
     */
    [[nodiscard]] RowView row_view(int i, int j) const noexcept;

    /// Memory footprint in bytes
    [[nodiscard]] std::size_t memory_bytes() const noexcept;

private:
    const IntegralSO& so_;
    const int n_so_;
    std::size_t num_rows_;  ///< n_so·(n_so - 1)/2
    HBBuildOptions opt_;

    // CSR layout
    std::vector<std::size_t> row_offsets_;  ///< [num_rows_ + 1] cumulative nnz
    std::vector<int>         a_;            ///< Flat 'a' indices
    std::vector<int>         b_;            ///< Flat 'b' indices
    std::vector<double>      w_;            ///< Flat weights (descending per row)
    std::vector<double>      row_sum_;      ///< [num_rows_] weight sums

    /// Canonical index for {i>j}
    static inline std::size_t unordered_pair_index(int i, int j) noexcept {
        if (i < j) std::swap(i, j);
        return static_cast<std::size_t>(i) * (i - 1) / 2 + j;
    }

    /// Build row {i>j}, append to buffer
    void build_row_(int i, int j, std::vector<Entry>& buffer, std::size_t& nnz_total);

    /// Finalize CSR from row buffers
    void finalize_layout_(const std::vector<std::vector<Entry>>& rows, std::size_t nnz_total);
};

} // namespace detnqs
