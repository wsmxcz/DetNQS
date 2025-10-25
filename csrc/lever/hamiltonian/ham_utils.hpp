// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.hpp
 * @brief Sparse matrix structures and utilities for Hamiltonian assembly.
 *
 * Provides COO/CSC formats with conversion and manipulation routines.
 * Used across Hamiltonian construction and Python interface modules.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once
#include <lever/determinant/det_space.hpp>
#include <cstdint>
#include <span>
#include <vector>

namespace lever {

using u32 = std::uint32_t;

// ============================================================================
// Numerical Thresholds
// ============================================================================

/** Drop matrix elements with |H_ij| ≤ threshold */
inline constexpr double MAT_ELEMENT_THRESH = 1e-15;

/** Heat-bath integral screening cutoff */
inline constexpr double HEATBATH_THRESH = 1e-15;

/** Skip outer product contributions with |value| < threshold */
inline constexpr double MICRO_CONTRIB_THRESH = 1e-18;

// ============================================================================
// Sparse Matrix Formats
// ============================================================================

/**
 * Coordinate (COO) sparse matrix: triplet (row, col, val) storage.
 * May contain unsorted entries and duplicates before consolidation.
 */
struct COOMatrix {
    std::vector<u32>    rows;
    std::vector<u32>    cols;
    std::vector<double> vals;
    u32 n_rows = 0;  // Row dimension metadata
    u32 n_cols = 0;  // Column dimension metadata

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
 * Compressed Sparse Column (CSC) format: column-wise compressed storage.
 * Each column stores sorted, unique row indices with corresponding values.
 */
struct CSCMatrix {
    std::vector<u32>    row_indices;  // Concatenated row indices
    std::vector<double> values;       // Corresponding matrix values
    std::vector<size_t> col_ptrs;     // Column pointers: size = n_cols + 1
    u32 n_rows = 0;
    u32 n_cols = 0;

    /** Column slice: lightweight view into row indices and values */
    struct ColView {
        std::span<const u32>    rows;
        std::span<const double> vals;
    };

    /** Access column j as a view: O(1) operation */
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
 * Hamiltonian assembly result: separated S and C space blocks.
 * 
 * H_SS: ⟨S|H|S⟩ block (S-space interactions)
 * H_SC: ⟨S|H|C⟩ block (S-C coupling)
 * map_C: Determinant indexing for C-space
 */
struct HamBlocks {
    COOMatrix H_SS;
    COOMatrix H_SC;
    DetMap    map_C;
};

// ============================================================================
// COO Matrix Operations
// ============================================================================

/**
 * Sort COO matrix by (row, col) and merge duplicate entries.
 * Algorithm: lexicographic sort → accumulate duplicates → drop zeros.
 *
 * @param coo Matrix modified in-place
 * @return max|value| after merge
 */
[[nodiscard]] double sort_and_merge_coo(COOMatrix& coo);

/**
 * Add two sorted COO matrices: C = A + B.
 * Algorithm: two-pointer merge with duplicate accumulation.
 *
 * @param A First sorted matrix
 * @param B Second sorted matrix
 * @param thresh Drop entries with |val| ≤ thresh
 * @return Merged matrix C
 */
[[nodiscard]] COOMatrix coo_add(
    const COOMatrix& A,
    const COOMatrix& B,
    double thresh = 0.0
);

/**
 * Mirror upper triangle to full symmetric matrix.
 * Algorithm: duplicate off-diagonal (i,j) as (j,i) for i < j.
 *
 * @param upper Upper triangular input
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
 * Convert COO to CSC format with column-wise sorting and deduplication.
 * Algorithm: bucket by column → sort rows within each column → merge duplicates.
 *
 * @param coo Input COO matrix
 * @param n_rows Target row dimension
 * @param n_cols Target column dimension
 * @return CSC matrix with sorted, unique entries per column
 */
[[nodiscard]] CSCMatrix coo_to_csc(
    const COOMatrix& coo,
    u32 n_rows,
    u32 n_cols
);

} // namespace lever
