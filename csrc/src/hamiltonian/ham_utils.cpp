// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.cpp
 * @brief COO sparse matrix operations and format conversions.
 *
 * Provides utilities for:
 *   - Sorting and merging duplicate entries
 *   - Matrix addition in sorted COO format
 *   - Upper-to-full symmetric expansion
 *   - COO-to-CSC conversion with deduplication
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <lever/hamiltonian/ham_utils.hpp>
#include <algorithm>
#include <cmath>
#include <numeric> 
#include <stdexcept>

namespace lever {

// ============================================================================
// COO Matrix Operations
// ============================================================================

/**
 * Sort and merge duplicate entries in COO matrix.
 *
 * Algorithm: Indirect sort by (row, col), then accumulate duplicates.
 * Returns max |value| for numerical conditioning diagnostics.
 *
 * @param coo  Matrix to sort/merge (modified in-place)
 * @return     Maximum absolute value in merged matrix
 */
double sort_and_merge_coo(COOMatrix& coo) {
    if (coo.empty()) return 0.0;

    const size_t n = coo.nnz();

    // Indirect sort: preserve original data during comparison
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return (coo.rows[a] != coo.rows[b]) 
            ? (coo.rows[a] < coo.rows[b]) 
            : (coo.cols[a] < coo.cols[b]);
    });

    // Merge duplicates: accumulate consecutive (row, col) pairs
    COOMatrix merged;
    merged.reserve(n);
    merged.n_rows = coo.n_rows;
    merged.n_cols = coo.n_cols;

    double max_abs = 0.0;

    for (size_t i = 0; i < n; ) {
        const size_t idx = indices[i];
        const u32 r = coo.rows[idx];
        const u32 c = coo.cols[idx];
        double val = coo.vals[idx];

        // Accumulate all entries at (r, c)
        for (++i; i < n; ++i) {
            const size_t next_idx = indices[i];
            if (coo.rows[next_idx] != r || coo.cols[next_idx] != c) break;
            val += coo.vals[next_idx];
        }

        const double abs_val = std::abs(val);
        if (abs_val > 0.0) {
            max_abs = std::max(max_abs, abs_val);
            merged.push_back(r, c, val);
        }
    }

    coo = std::move(merged);
    return max_abs;
}

/**
 * Add two sorted COO matrices: C = A + B.
 *
 * Algorithm: Two-pointer merge with threshold-based filtering.
 * Assumes A and B are pre-sorted by (row, col).
 *
 * @param A      First matrix (sorted)
 * @param B      Second matrix (sorted)
 * @param thresh Drop entries with |value| ≤ thresh
 * @return       C = A + B (sorted, filtered)
 */
COOMatrix coo_add(const COOMatrix& A, const COOMatrix& B, double thresh) {
    COOMatrix C;
    C.reserve(A.nnz() + B.nnz());
    C.n_rows = std::max(A.n_rows, B.n_rows);
    C.n_cols = std::max(A.n_cols, B.n_cols);

    size_t i = 0, j = 0;
    const size_t m = A.nnz(), n = B.nnz();

    // Two-pointer merge
    while (i < m && j < n) {
        const u32 r_a = A.rows[i], c_a = A.cols[i];
        const u32 r_b = B.rows[j], c_b = B.cols[j];

        if (r_a < r_b || (r_a == r_b && c_a < c_b)) {
            // A entry comes first
            if (std::abs(A.vals[i]) > thresh) {
                C.push_back(r_a, c_a, A.vals[i]);
            }
            ++i;
        } else if (r_b < r_a || (r_b == r_a && c_b < c_a)) {
            // B entry comes first
            if (std::abs(B.vals[j]) > thresh) {
                C.push_back(r_b, c_b, B.vals[j]);
            }
            ++j;
        } else {
            // Same position: add values
            const double val = A.vals[i] + B.vals[j];
            if (std::abs(val) > thresh) {
                C.push_back(r_a, c_a, val);
            }
            ++i;
            ++j;
        }
    }

    // Append remaining entries from A
    while (i < m) {
        if (std::abs(A.vals[i]) > thresh) {
            C.push_back(A.rows[i], A.cols[i], A.vals[i]);
        }
        ++i;
    }

    // Append remaining entries from B
    while (j < n) {
        if (std::abs(B.vals[j]) > thresh) {
            C.push_back(B.rows[j], B.cols[j], B.vals[j]);
        }
        ++j;
    }

    return C;
}

/**
 * Expand upper-triangular matrix to full symmetric form.
 *
 * Algorithm: Copy (i,j) entries and add (j,i) for off-diagonal.
 * Input matrix need not be strictly upper-triangular.
 *
 * @param upper  Input matrix (typically upper-triangular)
 * @return       Full symmetric matrix (unsorted)
 */
COOMatrix mirror_upper_to_full(const COOMatrix& upper) {
    COOMatrix full;
    full.reserve(2 * upper.nnz());
    full.n_rows = upper.n_rows;
    full.n_cols = upper.n_cols;

    for (size_t k = 0; k < upper.nnz(); ++k) {
        const u32 i = upper.rows[k];
        const u32 j = upper.cols[k];
        const double v = upper.vals[k];

        full.push_back(i, j, v);  // Upper entry

        if (i != j) {
            full.push_back(j, i, v);  // Lower entry (transpose)
        }
    }

    return full;
}

/**
 * Infer number of rows from COO matrix.
 *
 * @param coo  Input matrix
 * @return     max(row_indices) + 1
 */
u32 infer_n_rows(const COOMatrix& coo) noexcept {
    if (coo.empty()) return 0;
    
    u32 max_row = 0;
    for (u32 r : coo.rows) {
        max_row = std::max(max_row, r);
    }
    return max_row + 1;
}

// ============================================================================
// Format Conversions
// ============================================================================

/**
 * Convert COO to CSC (Compressed Sparse Column) format.
 *
 * Algorithm: Bucket entries by column, sort each column, deduplicate.
 * Time complexity: O(nnz·log(nnz/n_cols)) for typical sparse matrices.
 *
 * @param coo     Input COO matrix
 * @param n_rows  Number of rows
 * @param n_cols  Number of columns
 * @return        Equivalent CSC matrix
 * @throws        std::out_of_range if column index exceeds n_cols
 */
CSCMatrix coo_to_csc(const COOMatrix& coo, u32 n_rows, u32 n_cols) {
    CSCMatrix csc;
    csc.n_rows = n_rows;
    csc.n_cols = n_cols;

    if (coo.empty()) {
        csc.col_ptrs.assign(n_cols + 1, 0);
        return csc;
    }

    // Distribute entries into column buckets
    std::vector<std::vector<std::pair<u32, double>>> col_buckets(n_cols);
    
    for (size_t k = 0; k < coo.nnz(); ++k) {
        const u32 c = coo.cols[k];
        if (c >= n_cols) {
            throw std::out_of_range("coo_to_csc: column index out of bounds");
        }
        col_buckets[c].emplace_back(coo.rows[k], coo.vals[k]);
    }

    // Build CSC with per-column sorting and deduplication
    csc.col_ptrs.resize(n_cols + 1);
    csc.col_ptrs[0] = 0;

    for (u32 j = 0; j < n_cols; ++j) {
        auto& bucket = col_buckets[j];

        if (bucket.empty()) {
            csc.col_ptrs[j + 1] = csc.col_ptrs[j];
            continue;
        }

        // Sort column entries by row index
        std::sort(bucket.begin(), bucket.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Merge consecutive duplicates
        for (size_t i = 0; i < bucket.size(); ) {
            const u32 row = bucket[i].first;
            double val = bucket[i].second;

            // Accumulate all entries at (row, j)
            for (++i; i < bucket.size() && bucket[i].first == row; ++i) {
                val += bucket[i].second;
            }

            if (val != 0.0) {
                csc.row_indices.push_back(row);
                csc.values.push_back(val);
            }
        }

        csc.col_ptrs[j + 1] = static_cast<u32>(csc.row_indices.size());
    }

    return csc;
}

} // namespace lever
