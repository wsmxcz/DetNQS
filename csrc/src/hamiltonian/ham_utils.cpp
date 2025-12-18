// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ham_utils.cpp
 * @brief Implementation of sparse matrix operations and shared helpers.
 */

#include <lever/hamiltonian/ham_utils.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/determinant/det_ops.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace lever {

// ============================================================================
// COO Matrix Operations
// ============================================================================

double sort_and_merge_coo(COOMatrix& coo) {
    if (coo.empty()) return 0.0;

    const size_t n = coo.nnz();

    // Indirect sort: create index permutation
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), size_t{0});

    std::ranges::sort(idx, [&](size_t a, size_t b) {
        if (coo.rows[a] != coo.rows[b]) return coo.rows[a] < coo.rows[b];
        return coo.cols[a] < coo.cols[b];
    });

    // Merge duplicates: single pass over sorted indices
    COOMatrix merged;
    merged.reserve(n);
    merged.n_rows = coo.n_rows;
    merged.n_cols = coo.n_cols;

    double max_abs = 0.0;

    for (size_t i = 0; i < n; ) {
        const size_t k = idx[i];
        const u32 row = coo.rows[k];
        const u32 col = coo.cols[k];
        double val = coo.vals[k];
        ++i;

        // Accumulate duplicates
        while (i < n) {
            const size_t k2 = idx[i];
            if (coo.rows[k2] != row || coo.cols[k2] != col) break;
            val += coo.vals[k2];
            ++i;
        }

        const double abs_val = std::abs(val);
        if (abs_val > 0.0) {
            max_abs = std::max(max_abs, abs_val);
            merged.push_back(row, col, val);
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
    const size_t m = A.nnz();
    const size_t n = B.nnz();

    // Two-pointer merge
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
            const double val = A.vals[i] + B.vals[j];
            if (std::abs(val) > thresh) {
                C.push_back(ra, ca, val);
            }
            ++i;
            ++j;
        }
    }

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

        full.push_back(i, j, v);
        if (i != j) {
            full.push_back(j, i, v);
        }
    }

    return full;
}

u32 infer_n_rows(const COOMatrix& coo) noexcept {
    if (coo.empty()) return 0;
    return std::ranges::max(coo.rows) + 1;
}

// ============================================================================
// Format Conversions
// ============================================================================

CSCMatrix coo_to_csc(const COOMatrix& coo, u32 n_rows, u32 n_cols) {
    CSCMatrix csc;
    csc.n_rows = n_rows;
    csc.n_cols = n_cols;
    csc.col_ptrs.assign(n_cols + 1, 0);

    if (coo.empty()) return csc;

    // Bucket sort by column
    std::vector<std::vector<std::pair<u32, double>>> buckets(n_cols);
    for (size_t k = 0; k < coo.nnz(); ++k) {
        const u32 c = coo.cols[k];
        if (c >= n_cols) {
            throw std::out_of_range("coo_to_csc: column index out of bounds");
        }
        buckets[c].emplace_back(coo.rows[k], coo.vals[k]);
    }

    size_t nnz = 0;
    for (u32 j = 0; j < n_cols; ++j) {
        auto& bucket = buckets[j];
        if (bucket.empty()) {
            csc.col_ptrs[j + 1] = nnz;
            continue;
        }

        std::ranges::sort(bucket, [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        u32 last_row = bucket[0].first;
        double acc = bucket[0].second;

        for (size_t t = 1; t < bucket.size(); ++t) {
            if (bucket[t].first == last_row) {
                acc += bucket[t].second;
            } else {
                if (acc != 0.0) {
                    csc.row_indices.push_back(last_row);
                    csc.values.push_back(acc);
                    ++nnz;
                }
                last_row = bucket[t].first;
                acc = bucket[t].second;
            }
        }

        if (acc != 0.0) {
            csc.row_indices.push_back(last_row);
            csc.values.push_back(acc);
            ++nnz;
        }

        csc.col_ptrs[j + 1] = nnz;
    }

    return csc;
}

CSRMatrix coo_to_csr(const COOMatrix& coo, u32 n_rows, u32 n_cols) {
    CSRMatrix csr;
    csr.n_rows = n_rows;
    csr.n_cols = n_cols;
    csr.row_ptrs.assign(n_rows + 1, 0);

    if (coo.empty()) return csr;

    // Bucket sort by row
    std::vector<std::vector<std::pair<u32, double>>> buckets(n_rows);
    for (size_t k = 0; k < coo.nnz(); ++k) {
        const u32 r = coo.rows[k];
        if (r >= n_rows) {
            throw std::out_of_range("coo_to_csr: row index out of bounds");
        }
        buckets[r].emplace_back(coo.cols[k], coo.vals[k]);
    }

    size_t nnz = 0;
    for (u32 i = 0; i < n_rows; ++i) {
        auto& bucket = buckets[i];
        if (bucket.empty()) {
            csr.row_ptrs[i + 1] = nnz;
            continue;
        }

        std::ranges::sort(bucket, [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        u32 last_col = bucket[0].first;
        double acc = bucket[0].second;

        for (size_t t = 1; t < bucket.size(); ++t) {
            if (bucket[t].first == last_col) {
                acc += bucket[t].second;
            } else {
                if (acc != 0.0) {
                    csr.col_indices.push_back(last_col);
                    csr.values.push_back(acc);
                    ++nnz;
                }
                last_col = bucket[t].first;
                acc = bucket[t].second;
            }
        }

        if (acc != 0.0) {
            csr.col_indices.push_back(last_col);
            csr.values.push_back(acc);
            ++nnz;
        }

        csr.row_ptrs[i + 1] = nnz;
    }

    return csr;
}

COOMatrix csr_to_coo(const CSRMatrix& csr) {
    COOMatrix coo;
    coo.n_rows = csr.n_rows;
    coo.n_cols = csr.n_cols;
    coo.reserve(csr.values.size());

    for (u32 i = 0; i < csr.n_rows; ++i) {
        const size_t start = csr.row_ptrs[i];
        const size_t end = csr.row_ptrs[i + 1];

        for (size_t p = start; p < end; ++p) {
            coo.push_back(i, csr.col_indices[p], csr.values[p]);
        }
    }

    return coo;
}

CSRMatrix csr_add(const CSRMatrix& A, const CSRMatrix& B, double thresh) {
    if (A.n_rows != B.n_rows || A.n_cols != B.n_cols) {
        throw std::invalid_argument("csr_add: matrix dimensions must match");
    }

    CSRMatrix C;
    C.n_rows = A.n_rows;
    C.n_cols = A.n_cols;
    C.row_ptrs.resize(C.n_rows + 1);
    C.row_ptrs[0] = 0;

    size_t total_nnz = 0;
    for (u32 i = 0; i < C.n_rows; ++i) {
        size_t row_nnz = 0;

        size_t pa = A.row_ptrs[i], qa = A.row_ptrs[i + 1];
        size_t pb = B.row_ptrs[i], qb = B.row_ptrs[i + 1];

        while (pa < qa && pb < qb) {
            const u32 ca = A.col_indices[pa];
            const u32 cb = B.col_indices[pb];

            if (ca < cb) {
                if (std::abs(A.values[pa]) > thresh) ++row_nnz;
                ++pa;
            } else if (cb < ca) {
                if (std::abs(B.values[pb]) > thresh) ++row_nnz;
                ++pb;
            } else {
                const double val = A.values[pa] + B.values[pb];
                if (std::abs(val) > thresh) ++row_nnz;
                ++pa;
                ++pb;
            }
        }

        while (pa < qa) {
            if (std::abs(A.values[pa]) > thresh) ++row_nnz;
            ++pa;
        }
        while (pb < qb) {
            if (std::abs(B.values[pb]) > thresh) ++row_nnz;
            ++pb;
        }

        total_nnz += row_nnz;
        C.row_ptrs[i + 1] = total_nnz;
    }

    C.col_indices.resize(total_nnz);
    C.values.resize(total_nnz);

    for (u32 i = 0; i < C.n_rows; ++i) {
        size_t write_pos = C.row_ptrs[i];

        size_t pa = A.row_ptrs[i], qa = A.row_ptrs[i + 1];
        size_t pb = B.row_ptrs[i], qb = B.row_ptrs[i + 1];

        while (pa < qa && pb < qb) {
            const u32 ca = A.col_indices[pa];
            const u32 cb = B.col_indices[pb];

            if (ca < cb) {
                const double val = A.values[pa];
                if (std::abs(val) > thresh) {
                    C.col_indices[write_pos] = ca;
                    C.values[write_pos] = val;
                    ++write_pos;
                }
                ++pa;
            } else if (cb < ca) {
                const double val = B.values[pb];
                if (std::abs(val) > thresh) {
                    C.col_indices[write_pos] = cb;
                    C.values[write_pos] = val;
                    ++write_pos;
                }
                ++pb;
            } else {
                const double val = A.values[pa] + B.values[pb];
                if (std::abs(val) > thresh) {
                    C.col_indices[write_pos] = ca;
                    C.values[write_pos] = val;
                    ++write_pos;
                }
                ++pa;
                ++pb;
            }
        }

        while (pa < qa) {
            const double val = A.values[pa];
            if (std::abs(val) > thresh) {
                C.col_indices[write_pos] = A.col_indices[pa];
                C.values[write_pos] = val;
                ++write_pos;
            }
            ++pa;
        }

        while (pb < qb) {
            const double val = B.values[pb];
            if (std::abs(val) > thresh) {
                C.col_indices[write_pos] = B.col_indices[pb];
                C.values[write_pos] = val;
                ++write_pos;
            }
            ++pb;
        }
    }

    return C;
}

// ============================================================================
// Determinant / SO helpers
// ============================================================================

std::vector<int> get_occ_so(const Det& d) {
    std::vector<int> occ;
    occ.reserve(popcount(d.alpha) + popcount(d.beta));

    for (u64 am = d.alpha; am; am = clear_lsb(am)) {
        occ.push_back(so_from_mo(ctz(am), 0));
    }
    for (u64 bm = d.beta; bm; bm = clear_lsb(bm)) {
        occ.push_back(so_from_mo(ctz(bm), 1));
    }

    return occ;
}

bool is_occ_so(const Det& d, int so_idx, int n_orb) {
    const int mo = mo_from_so(so_idx);
    if (mo < 0 || mo >= n_orb) return true;

    const u64 bit = (1ULL << mo);
    if (spin_from_so(so_idx) == 0) {
        return (d.alpha & bit) != 0;
    }
    return (d.beta & bit) != 0;
}

Det exc2_so(const Det& ket, int i_so, int j_so, int a_so, int b_so) {
    Det out = ket;

    auto flip = [&](int so, bool set_bit) {
        const int mo = mo_from_so(so);
        const int sp = spin_from_so(so);
        const u64 bit = (1ULL << mo);

        u64& mask = (sp == 0) ? out.alpha : out.beta;
        if (set_bit) {
            mask |= bit;
        } else {
            mask &= ~bit;
        }
    };

    flip(i_so, false);
    flip(j_so, false);
    flip(a_so, true);
    flip(b_so, true);

    return out;
}

size_t est_conn_cap(int n_orb) {
    // 1 diagonal + O(n_orb) singles + ~10% of n_orb^2 doubles
    if (n_orb <= 0) return 1;
    const size_t n = static_cast<size_t>(n_orb);
    return static_cast<size_t>(1 + 2 * n + (n * n) / 10);
}

// ============================================================================
// Screened complement
// ============================================================================

std::vector<Det> generate_complement_screened(
    std::span<const Det> S,
    int n_orb,
    const DetMap& exclude,
    const HamEval& ham,
    const HeatBathTable* hb_table,
    ScreenMode mode,
    std::span<const double> psi_S,
    double eps1
) {
    if (S.empty()) return {};

    // No screening: defer to pure combinatorial generator.
    if (mode == ScreenMode::None) {
        return det_space::generate_complement(S, n_orb, exclude, true);
    }

    if (!hb_table) {
        throw std::invalid_argument(
            "generate_complement_screened: Heat-Bath table is required "
            "for screened modes"
        );
    }

    if (mode == ScreenMode::Dynamic && psi_S.size() != S.size()) {
        throw std::invalid_argument(
            "generate_complement_screened: psi_S size must match |S| "
            "for dynamic screening"
        );
    }

    std::unordered_set<Det> uniq;
    uniq.reserve(S.size() * 16);

    const double num_thresh = MAT_ELEMENT_THRESH;
    const double delta = 1e-12;

    const size_t n_S = S.size();
    for (size_t i = 0; i < n_S; ++i) {
        const Det& bra = S[i];

        // --- Singles ---
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            if (exclude.contains(ket)) return;

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= num_thresh) return;

            if (mode == ScreenMode::Dynamic) {
                const double c = psi_S[i];
                const double w = std::abs(h * c);
                if (w < eps1) return;
            }
            // Static mode keeps all singles above numerical threshold.
            uniq.insert(ket);
        });

        // --- Doubles (Heat-Bath) ---
        double cutoff = eps1;
        if (mode == ScreenMode::Dynamic) {
            const double amp = std::max(std::abs(psi_S[i]), delta);
            cutoff = eps1 / amp;
        }

        for_each_double_hb(
            bra,
            n_orb,
            *hb_table,
            cutoff,
            [&](const Det& ket) {
                if (exclude.contains(ket)) return;
                uniq.insert(ket);
            }
        );
    }

    std::vector<Det> out;
    out.reserve(uniq.size());
    for (const auto& d : uniq) {
        out.push_back(d);
    }
    return det_space::canonicalize(std::move(out));
}

} // namespace lever