// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.hpp
 * @brief Sparse matrix utilities (COO/CSC/CSR) for Hamiltonian assembly.
 *
 * Provides coordinate (COO), compressed sparse column (CSC), and compressed
 * sparse row (CSR) formats with conversion, merge, and arithmetic operations.
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

/** Matrix element pruning: drop |H_ij| ≤ MAT_ELEMENT_THRESH */
inline constexpr double MAT_ELEMENT_THRESH = 1e-15;

/** Heat-bath integral screening cutoff */
inline constexpr double HEATBATH_THRESH = 1e-15;

/** Micro contribution filter: skip terms with negligible impact */
inline constexpr double MICRO_CONTRIB_THRESH = 1e-12;

// ============================================================================
// Sparse Matrix Formats
// ============================================================================

/**
 * Coordinate (COO) format: unsorted triplets (row, col, val).
 * May contain duplicates; use sort_and_merge_coo() to canonicalize.
 */
struct COOMatrix {
    std::vector<u32>    rows;
    std::vector<u32>    cols;
    std::vector<double> vals;
    u32 n_rows = 0;
    u32 n_cols = 0;

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
 * Compressed Sparse Column (CSC): column-major storage.
 * Each column has sorted, unique row indices.
 */
struct CSCMatrix {
    std::vector<u32>    row_indices;  ///< Concatenated row IDs
    std::vector<double> values;       ///< Corresponding values
    std::vector<size_t> col_ptrs;     ///< Column pointers (size = n_cols + 1)
    u32 n_rows = 0;
    u32 n_cols = 0;

    struct ColView {
        std::span<const u32>    rows;
        std::span<const double> vals;
    };

    /** Access j-th column as read-only view */
    [[nodiscard]] ColView col(u32 j) const noexcept {
        const size_t start = col_ptrs[j];
        const size_t end = col_ptrs[j + 1];
        return {
            {row_indices.data() + start, end - start},
            {values.data() + start, end - start}
        };
    }
};

/**
 * Compressed Sparse Row (CSR): row-major storage.
 * Each row has sorted, unique column indices.
 */
struct CSRMatrix {
    std::vector<size_t> row_ptrs;     ///< Row pointers (size = n_rows + 1)
    std::vector<u32>    col_indices;  ///< Concatenated column IDs
    std::vector<double> values;       ///< Corresponding values
    u32 n_rows = 0;
    u32 n_cols = 0;

    struct RowView {
        std::span<const u32>    cols;
        std::span<const double> vals;
    };

    /** Access i-th row as read-only view */
    [[nodiscard]] RowView row(u32 i) const noexcept {
        const size_t start = row_ptrs[i];
        const size_t end = row_ptrs[i + 1];
        return {
            {col_indices.data() + start, end - start},
            {values.data() + start, end - start}
        };
    }
};

/**
 * Hamiltonian block decomposition: H = [H_SS  H_SC]
 *                                       [H_CS  H_CC]
 *
 * H_SS: S-space internal interactions ⟨S|H|S⟩
 * H_SC: S-C space coupling ⟨S|H|C⟩
 * map_C: C-space determinant indexing
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
 * Canonicalize COO matrix: sort by (row, col), merge duplicates, drop zeros.
 *
 * Algorithm: Indirect sort via index permutation, then single-pass merge.
 *
 * @param coo  Input/output matrix
 * @return     Maximum absolute value: max_ij |H_ij|
 */
[[nodiscard]] double sort_and_merge_coo(COOMatrix& coo);

/**
 * Matrix addition C = A + B for sorted COO matrices.
 *
 * Algorithm: Two-pointer merge (O(nnz_A + nnz_B) time).
 *
 * @param A      First matrix (sorted)
 * @param B      Second matrix (sorted)
 * @param thresh Drop elements with |val| ≤ thresh
 * @return       Sum matrix (sorted, unique entries)
 */
[[nodiscard]] COOMatrix coo_add(
    const COOMatrix& A,
    const COOMatrix& B,
    double thresh = 0.0
);

/**
 * Mirror upper triangle to symmetric full matrix.
 *
 * For each (i,j) with i < j, adds (j,i). Unsorted output.
 *
 * @param upper  Upper triangular matrix
 * @return       Full symmetric matrix (unsorted)
 */
[[nodiscard]] COOMatrix mirror_upper_to_full(const COOMatrix& upper);

/**
 * Infer row dimension from data: n_rows = max(rows) + 1.
 *
 * @param coo  Input matrix
 * @return     Inferred number of rows
 */
[[nodiscard]] u32 infer_n_rows(const COOMatrix& coo) noexcept;

// ============================================================================
// Format Conversions
// ============================================================================

/**
 * Convert COO to CSC format.
 *
 * Algorithm: Bucket sort by column, then per-column merge of duplicates.
 *
 * @param coo     Input COO matrix
 * @param n_rows  Number of rows
 * @param n_cols  Number of columns
 * @return        CSC matrix with sorted, unique row indices per column
 */
[[nodiscard]] CSCMatrix coo_to_csc(
    const COOMatrix& coo,
    u32 n_rows,
    u32 n_cols
);

/**
 * Convert COO to CSR format.
 *
 * Algorithm: Bucket sort by row, then per-row merge of duplicates.
 *
 * @param coo     Input COO matrix
 * @param n_rows  Number of rows
 * @param n_cols  Number of columns
 * @return        CSR matrix with sorted, unique column indices per row
 */
[[nodiscard]] CSRMatrix coo_to_csr(
    const COOMatrix& coo,
    u32 n_rows,
    u32 n_cols
);

/**
 * Convert CSR to COO format (preserves row-major order).
 *
 * @param csr  Input CSR matrix
 * @return     COO matrix with entries in row-major order
 */
[[nodiscard]] COOMatrix csr_to_coo(const CSRMatrix& csr);

/**
 * CSR matrix addition: C = A + B (same shape).
 *
 * Algorithm: Row-wise two-pointer merge with two-pass (count + write).
 *
 * @param A      First matrix
 * @param B      Second matrix
 * @param thresh Drop elements with |val| ≤ thresh
 * @return       Sum matrix in CSR format
 * @throws       std::invalid_argument if shapes mismatch
 */
[[nodiscard]] CSRMatrix csr_add(
    const CSRMatrix& A,
    const CSRMatrix& B,
    double thresh = 0.0
);

} // namespace lever
