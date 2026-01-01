// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.hpp
 * @brief Sparse matrix utilities and shared Hamiltonian helpers.
 *
 * This header provides:
 *   - Numerical thresholds used by Hamiltonian builders
 *   - Sparse matrix formats (COO/CSR/CSC) and operations
 *   - HamBlocks container for H_SS / H_SC + C-space map
 *   - Common determinant/SO utilities for Heat-Bath screening
 *   - Screened complement generator for ASCI-style selection
 */

#pragma once

#include <detnqs/determinant/det_space.hpp>
#include <detnqs/utils/bit_utils.hpp>
#include <detnqs/integral/hb_table.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace detnqs {

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
// Screening mode
// ============================================================================

/**
 * Screening strategy for connectivity and complement generation.
 *
 * None   : pure combinatorial (no integral-based pruning)
 * Static : fixed ε₁ cutoff on integrals (Heat-Bath doubles)
 * Dynamic: amplitude-weighted cutoff based on ψ_S
 */
enum class ScreenMode : std::uint8_t {
    None,
    Static,
    Dynamic
};

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
 * @param coo  Input/output matrix
 * @return     Maximum absolute value: max_ij |H_ij|
 */
[[nodiscard]] double sort_and_merge_coo(COOMatrix& coo);

/**
 * Matrix addition C = A + B for sorted COO matrices.
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
 * @param A      First matrix
 * @param B      Second matrix
 * @param thresh Drop elements with |val| ≤ thresh
 * @return       Sum matrix in CSR format
 */
[[nodiscard]] CSRMatrix csr_add(
    const CSRMatrix& A,
    const CSRMatrix& B,
    double thresh = 0.0
);

// ============================================================================
// Determinant / SO helpers (shared by Hamiltonian code)
// ============================================================================

/**
 * Return occupied spin-orbitals of a determinant.
 * α: 2p, β: 2p+1. Order is deterministic (increasing index).
 */
[[nodiscard]] std::vector<int> get_occ_so(const Det& d);

/**
 * Check if a spin-orbital index is occupied in a determinant.
 * Out-of-range SO indices are treated as occupied to guard boundaries.
 */
[[nodiscard]] bool is_occ_so(const Det& d, int so_idx, int n_orb);

/**
 * Apply a double excitation i,j → a,b in spin-orbital representation.
 */
[[nodiscard]] Det exc2_so(
    const Det& ket,
    int i_so,
    int j_so,
    int a_so,
    int b_so
);

/**
 * Heuristic capacity estimate for connections from one determinant.
 */
[[nodiscard]] size_t est_conn_cap(int n_orb);

/**
 * Heat-Bath double excitations for a single determinant.
 *
 * For each occupied pair (i,j) and each integral row with
 * |⟨ij||ab⟩| ≥ cutoff, generates the excited determinant if a,b are virtual.
 */
template <class Visitor>
inline void for_each_double_hb(
    const Det& bra,
    int n_orb,
    const HeatBathTable& hb,
    double cutoff,
    Visitor&& visit
) {
    auto occ = get_occ_so(bra);
    const size_t n = occ.size();
    if (n < 2) return;

    for (size_t p = 1; p < n; ++p) {
        const int i_so = occ[p];
        for (size_t q = 0; q < p; ++q) {
            const int j_so = occ[q];

            auto row = hb.row_view(i_so, j_so).with_cutoff(cutoff);
            for (size_t k = 0; k < row.len; ++k) {
                const int a_so = row.a[k];
                const int b_so = row.b[k];

                if (is_occ_so(bra, a_so, n_orb) ||
                    is_occ_so(bra, b_so, n_orb)) {
                    continue;
                }

                std::forward<Visitor>(visit)(
                    exc2_so(bra, i_so, j_so, a_so, b_so)
                );
            }
        }
    }
}

// Forward declaration; full definition is in ham_eval.hpp.
class HamEval;

/**
 * Screened complement:
 *   C = (S_1 ∪ S_2) \ exclude
 *
 * where S_1, S_2 are singles/doubles from S, subject to screening:
 *   - ScreenMode::None:
 *       fallback to det_space::generate_complement (no integrals)
 *   - ScreenMode::Static:
 *       doubles via Heat-Bath with fixed ε₁; singles kept if |H_ik| > thresh
 *   - ScreenMode::Dynamic:
 *       doubles with τ_i = ε₁ / max(|ψ_S[i]|, δ);
 *       singles kept if |H_ik * ψ_S[i]| ≥ ε₁
 */
[[nodiscard]] std::vector<Det> generate_complement_screened(
    std::span<const Det> S,
    int n_orb,
    const DetMap& exclude,
    const HamEval& ham,
    const HeatBathTable* hb_table,
    ScreenMode mode,
    std::span<const double> psi_S = {},
    double eps1 = 1e-6
);

} // namespace detnqs