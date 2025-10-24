// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

#include <lever/hamiltonian/ham_utils.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lever {

// ============================================================================
// COO Operations
// ============================================================================

double sort_and_merge_coo(COOMatrix& coo) {
    if (coo.empty()) return 0.0;

    const size_t n = coo.nnz();

    // Create index array for indirect sort
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    // Sort indices by (row, col)
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return (coo.rows[a] != coo.rows[b]) 
            ? (coo.rows[a] < coo.rows[b]) 
            : (coo.cols[a] < coo.cols[b]);
    });

    // Merge duplicates
    COOMatrix merged;
    merged.reserve(n);
    merged.n_rows = coo.n_rows;
    merged.n_cols = coo.n_cols;

    double max_abs = 0.0;

    for (size_t i = 0; i < n; ) {
        const size_t idx = indices[i];
        const u32 r = coo.rows[idx];
        const u32 c = coo.cols[idx];
        double v = coo.vals[idx];

        // Accumulate duplicates
        for (++i; i < n; ++i) {
            const size_t next_idx = indices[i];
            if (coo.rows[next_idx] != r || coo.cols[next_idx] != c) break;
            v += coo.vals[next_idx];
        }

        const double abs_v = std::abs(v);
        if (abs_v > 0.0) {
            max_abs = std::max(max_abs, abs_v);
            merged.push_back(r, c, v);
        }
    }

    coo = std::move(merged);
    return max_abs;
}

COOMatrix coo_add(const COOMatrix& A, const COOMatrix& B, double thresh) {
    COOMatrix C;
    C.reserve(A.nnz() + B.nnz());
    C.n_rows = std::max(A.n_rows, B.n_rows);
    C.n_cols = std::max(A.n_cols, B.n_cols);

    size_t i = 0, j = 0;
    const size_t m = A.nnz(), n = B.nnz();

    while (i < m && j < n) {
        const u32 ra = A.rows[i], ca = A.cols[i];
        const u32 rb = B.rows[j], cb = B.cols[j];

        if (ra < rb || (ra == rb && ca < cb)) {
            if (std::abs(A.vals[i]) > thresh) {
                C.push_back(ra, ca, A.vals[i]);
            }
            ++i;
        } else if (rb < ra || (rb == ra && cb < ca)) {
            if (std::abs(B.vals[j]) > thresh) {
                C.push_back(rb, cb, B.vals[j]);
            }
            ++j;
        } else {
            // Same position: merge
            const double v = A.vals[i] + B.vals[j];
            if (std::abs(v) > thresh) {
                C.push_back(ra, ca, v);
            }
            ++i;
            ++j;
        }
    }

    // Append remaining entries
    while (i < m) {
        if (std::abs(A.vals[i]) > thresh) {
            C.push_back(A.rows[i], A.cols[i], A.vals[i]);
        }
        ++i;
    }
    while (j < n) {
        if (std::abs(B.vals[j]) > thresh) {
            C.push_back(B.rows[j], B.cols[j], B.vals[j]);
        }
        ++j;
    }

    return C;
}

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

CSCMatrix coo_to_csc(const COOMatrix& coo, u32 n_rows, u32 n_cols) {
    CSCMatrix csc;
    csc.n_rows = n_rows;
    csc.n_cols = n_cols;

    if (coo.empty()) {
        csc.col_ptrs.assign(n_cols + 1, 0);
        return csc;
    }

    // Bucket entries by column
    std::vector<std::vector<std::pair<u32, double>>> col_buckets(n_cols);
    
    for (size_t k = 0; k < coo.nnz(); ++k) {
        const u32 c = coo.cols[k];
        if (c >= n_cols) {
            throw std::out_of_range("coo_to_csc: column index out of bounds");
        }
        col_buckets[c].emplace_back(coo.rows[k], coo.vals[k]);
    }

    // Build CSC with per-column deduplication
    csc.col_ptrs.resize(n_cols + 1);
    csc.col_ptrs[0] = 0;

    for (u32 j = 0; j < n_cols; ++j) {
        auto& bucket = col_buckets[j];

        if (bucket.empty()) {
            csc.col_ptrs[j + 1] = csc.col_ptrs[j];
            continue;
        }

        // Sort by row index
        std::sort(bucket.begin(), bucket.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Merge consecutive duplicates
        for (size_t i = 0; i < bucket.size(); ) {
            const u32 row = bucket[i].first;
            double val = bucket[i].second;

            for (++i; i < bucket.size() && bucket[i].first == row; ++i) {
                val += bucket[i].second;
            }

            if (val != 0.0) {
                csc.row_indices.push_back(row);
                csc.values.push_back(val);
            }
        }

        csc.col_ptrs[j + 1] = csc.row_indices.size();
    }

    return csc;
}

} // namespace lever
