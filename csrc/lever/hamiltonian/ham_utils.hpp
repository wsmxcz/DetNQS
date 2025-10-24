// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.hpp
 * @brief Common data structures and utilities for Hamiltonian assembly.
 *
 * Provides unified COO/CSC formats and conversion/manipulation routines
 * used across build_ham, ham_eff, and Python bridge modules.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: January, 2025
 */

#pragma once
#include <lever/determinant/det_space.hpp>
#include <cstdint>
#include <span>
#include <vector>
namespace lever {
using u32 = std::uint32_t;
// ============================================================================
// Unified Threshold Constants
// ============================================================================
/** Drop matrix elements with |H_ij| <= this value */
inline constexpr double MAT_ELEMENT_THRESH = 1e-15;
/** Heat-bath integral screening cutoff */
inline constexpr double HEATBATH_THRESH = 1e-15;
/** Skip outer product terms with |contribution| < this value */
inline constexpr double MICRO_CONTRIB_THRESH = 1e-18;

/**
 * Coordinate (COO) sparse matrix format.
 * Entries may be unsorted and contain duplicates initially.
 */
struct COOMatrix {
    std::vector<u32>    rows;
    std::vector<u32>    cols;
    std::vector<double> vals;
    u32 n_rows = 0;  // Row dimension (optional metadata)
    u32 n_cols = 0;  // Column dimension (optional metadata)

    [[nodiscard]] size_t nnz() const noexcept { return vals.size(); }
    [[nodiscard]] bool empty() const noexcept { return vals.empty(); }

    void reserve(size_t n) {
        rows.reserve(n);
        cols.reserve(n);
        vals.reserve(n);
    }

    void push_back(u32 r, u32 c, double v) {
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
    }

    void clear() {
        rows.clear();
        cols.clear();
        vals.clear();
    }
};

/**
 * Compressed Sparse Column (CSC) format for efficient column access.
 * Each column stores sorted, unique row indices with values.
 */
struct CSCMatrix {
    std::vector<u32>    row_indices;
    std::vector<double> values;
    std::vector<size_t> col_ptrs;  // Size = n_cols + 1
    u32 n_rows = 0;
    u32 n_cols = 0;

    struct ColView {
        std::span<const u32>    rows;
        std::span<const double> vals;
    };

    [[nodiscard]] ColView col(u32 j) const noexcept {
        const size_t start = col_ptrs[j];
        const size_t end   = col_ptrs[j + 1];
        return {
            {row_indices.data() + start, end - start},
            {values.data() + start, end - start}
        };
    }
};

/**
 * Hamiltonian block assembly result (separated S and C blocks).
 */
struct HamBlocks {
    COOMatrix H_SS;   // <S|H|S> block
    COOMatrix H_SC;   // <S|H|C> block
    DetMap    map_C;  // C-space determinant mapping
};

// ============================================================================
// COO Matrix Operations
// ============================================================================

/**
 * Sort COO entries by (row, col) and merge duplicates.
 * Drops zero entries after merging.
 *
 * @param coo Input/output matrix (modified in-place)
 * @return Maximum absolute value after merge
 */
[[nodiscard]] double sort_and_merge_coo(COOMatrix& coo);

/**
 * Add two sorted COO matrices: C = A + B.
 * Assumes A and B are already sorted by (row, col).
 *
 * @param A First matrix (sorted)
 * @param B Second matrix (sorted)
 * @param thresh Drop entries with |val| <= thresh
 * @return Merged matrix C
 */
[[nodiscard]] COOMatrix coo_add(
    const COOMatrix& A,
    const COOMatrix& B,
    double thresh = 0.0
);

/**
 * Mirror upper triangle to full symmetric matrix.
 * Creates A_full = A_upper + A_upper^T by duplicating off-diagonal entries.
 *
 * @param upper Input upper triangular matrix
 * @return Full symmetric matrix (unsorted)
 */
[[nodiscard]] COOMatrix mirror_upper_to_full(const COOMatrix& upper);

/**
 * Infer row dimension from maximum row index.
 */
[[nodiscard]] u32 infer_n_rows(const COOMatrix& coo) noexcept;

// ============================================================================
// Format Conversions
// ============================================================================

/**
 * Convert COO to CSC format with in-column duplicate merging.
 * Each column is sorted and deduplicated for deterministic output.
 *
 * @param coo Input COO matrix
 * @param n_rows Row dimension
 * @param n_cols Column dimension
 * @return CSC matrix
 */
[[nodiscard]] CSCMatrix coo_to_csc(
    const COOMatrix& coo,
    u32 n_rows,
    u32 n_cols
);

} // namespace lever
