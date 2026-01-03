// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.hpp
 * @brief Sparse matrix utilities and Hamiltonian block structures.
 *
 * Provides:
 *   - Numerical thresholds for matrix element pruning and screening
 *   - Sparse matrix formats (COO/CSR/CSC) and conversion routines
 *   - HamBlocks container for H_VV / H_VP and P-space indexing
 *   - Determinant/spin-orbital helpers for Heat-Bath screening
 *   - Screened complement generation (static/dynamic cutoffs)
 *
 * File: detnqs/hamiltonian/ham_utils.hpp
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
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

/** Matrix element pruning: drop |H_ij| <= MAT_ELEMENT_THRESH */
inline constexpr double MAT_ELEMENT_THRESH = 1e-15;

/** Heat-Bath integral screening: discard |<ij||ab>| <= HEATBATH_THRESH */
inline constexpr double HEATBATH_THRESH = 1e-15;

/** Micro contribution filter: skip negligible amplitude contributions */
inline constexpr double MICRO_CONTRIB_THRESH = 1e-12;

// ============================================================================
// Screening Mode
// ============================================================================

/**
 * Screening strategy for complement generation.
 *
 * None    : Combinatorial (no integral pruning)
 * Static  : Fixed epsilon_1 cutoff on integrals
 * Dynamic : Amplitude-weighted cutoff: tau_i = epsilon_1 / max(|psi_V[i]|, delta)
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
 * Coordinate (COO) format: triplets (row, col, val).
 * May contain duplicates; canonicalize via sort_and_merge_coo().
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
    std::vector<u32>    row_indices;
    std::vector<double> values;
    std::vector<size_t> col_ptrs;  ///< Size = n_cols + 1
    u32 n_rows = 0;
    u32 n_cols = 0;

    struct ColView {
        std::span<const u32>    rows;
        std::span<const double> vals;
    };

    /** Access j-th column */
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
 * Compressed Sparse Row (CSR): row-major storage.
 * Each row has sorted, unique column indices.
 */
struct CSRMatrix {
    std::vector<size_t> row_ptrs;     ///< Size = n_rows + 1
    std::vector<u32>    col_indices;
    std::vector<double> values;
    u32 n_rows = 0;
    u32 n_cols = 0;

    struct RowView {
        std::span<const u32>    cols;
        std::span<const double> vals;
    };

    /** Access i-th row */
    [[nodiscard]] RowView row(u32 i) const noexcept {
        const size_t start = row_ptrs[i];
        const size_t end   = row_ptrs[i + 1];
        return {
            {col_indices.data() + start, end - start},
            {values.data() + start, end - start}
        };
    }
};

/**
 * Hamiltonian block decomposition:
 *
 *   H = [ H_VV  H_VP ]
 *       [ H_PV  H_PP ]
 *
 * where V = variational set, P = perturbative set (P = C \ V).
 *
 * H_VV : <V|H|V>
 * H_VP : <V|H|P>
 * map_P: indexing for P-space determinants
 */
struct HamBlocks {
    COOMatrix H_VV;
    COOMatrix H_VP;
    DetMap    map_P;
};

// ============================================================================
// COO Matrix Operations
// ============================================================================

/**
 * Canonicalize COO: sort by (row, col), merge duplicates, drop zeros.
 *
 * @param coo  Input/output matrix
 * @return     max_ij |H_ij|
 */
[[nodiscard]] double sort_and_merge_coo(COOMatrix& coo);

/**
 * Matrix addition C = A + B (both sorted).
 *
 * @param A      First matrix
 * @param B      Second matrix
 * @param thresh Drop |val| <= thresh
 * @return       Sum matrix (sorted, unique)
 */
[[nodiscard]] COOMatrix coo_add(
    const COOMatrix& A,
    const COOMatrix& B,
    double thresh = 0.0
);

/**
 * Mirror upper triangle to symmetric full matrix.
 * For each (i,j) with i < j, adds (j,i). Output is unsorted.
 */
[[nodiscard]] COOMatrix mirror_upper_to_full(const COOMatrix& upper);

/**
 * Infer row dimension: n_rows = max(rows) + 1.
 */
[[nodiscard]] u32 infer_n_rows(const COOMatrix& coo) noexcept;

// ============================================================================
// Format Conversions
// ============================================================================

/** Convert COO to CSC */
[[nodiscard]] CSCMatrix coo_to_csc(
    const COOMatrix& coo,
    u32 n_rows,
    u32 n_cols
);

/** Convert COO to CSR */
[[nodiscard]] CSRMatrix coo_to_csr(
    const COOMatrix& coo,
    u32 n_rows,
    u32 n_cols
);

/** Convert CSR to COO (row-major order) */
[[nodiscard]] COOMatrix csr_to_coo(const CSRMatrix& csr);

/**
 * CSR matrix addition: C = A + B.
 *
 * @param thresh Drop |val| <= thresh
 */
[[nodiscard]] CSRMatrix csr_add(
    const CSRMatrix& A,
    const CSRMatrix& B,
    double thresh = 0.0
);

// ============================================================================
// Determinant / Spin-Orbital Helpers
// ============================================================================

/**
 * Return occupied spin-orbitals: alpha = 2p, beta = 2p+1.
 * Output is sorted in ascending order.
 */
[[nodiscard]] std::vector<int> get_occ_so(const Det& d);

/**
 * Check if spin-orbital so_idx is occupied in d.
 * Out-of-range indices are treated as occupied (boundary guard).
 */
[[nodiscard]] bool is_occ_so(const Det& d, int so_idx, int n_orb);

/**
 * Apply double excitation i,j -> a,b in spin-orbital basis.
 */
[[nodiscard]] Det exc2_so(
    const Det& ket,
    int i_so,
    int j_so,
    int a_so,
    int b_so
);

/**
 * Heuristic capacity for connections from one determinant.
 */
[[nodiscard]] size_t est_conn_cap(int n_orb);

/**
 * Heat-Bath double excitations for a single determinant.
 *
 * For each occupied pair (i,j), generates excited determinants
 * with |<ij||ab>| >= cutoff and a,b virtual.
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

// Forward declaration
class HamEval;

/**
 * Generate screened perturbative set:
 *   P = (V_1 union V_2) \ exclude
 *
 * where V_1, V_2 are singles/doubles from V, subject to:
 *
 * ScreenMode::None:
 *   Fallback to det_space::generate_complement (no integral screening)
 *
 * ScreenMode::Static:
 *   Doubles via Heat-Bath with |<ij||ab>| >= eps1
 *   Singles kept if |H_ik| > thresh
 *
 * ScreenMode::Dynamic:
 *   Doubles with tau_i = eps1 / max(|psi_V[i]|, delta)
 *   Singles kept if |H_ik * psi_V[i]| >= eps1
 *
 * @param V         Variational set
 * @param n_orb     Number of spatial orbitals
 * @param exclude   Determinants to exclude (typically V itself)
 * @param ham       Hamiltonian evaluator
 * @param hb_table  Heat-Bath table (required for Static/Dynamic)
 * @param mode      Screening strategy
 * @param psi_V     Amplitudes on V (required for Dynamic mode)
 * @param eps1      Screening threshold
 */
[[nodiscard]] std::vector<Det> generate_complement_screened(
    std::span<const Det> V,
    int n_orb,
    const DetMap& exclude,
    const HamEval& ham,
    const HeatBathTable* hb_table,
    ScreenMode mode,
    std::span<const double> psi_V = {},
    double eps1 = 1e-6
);

} // namespace detnqs
