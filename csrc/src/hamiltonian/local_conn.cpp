// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.cpp
 * @brief Implementation of local connectivity and streaming energy.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#include <lever/hamiltonian/local_conn.hpp>

#include <lever/determinant/det_ops.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/ham_utils.hpp>
#include <lever/utils/bit_utils.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace lever {

namespace {

// ─── Helpers ──────────────────────────────────────────────────────────

inline std::vector<int> get_occupied_sos(const Det& d) {
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

inline bool is_so_occupied(const Det& d, int so_idx, int n_orb) {
    const int mo = mo_from_so(so_idx);
    if (mo < 0 || mo >= n_orb) return true;
    const u64 bit = (1ULL << mo);
    return (spin_from_so(so_idx) == 0) ? (d.alpha & bit) : (d.beta & bit);
}

inline Det apply_double_exc_so(Det d, int i, int j, int a, int b) {
    auto flip = [&](int so, bool val) {
        const int mo = mo_from_so(so);
        const u64 bit = (1ULL << mo);
        if (spin_from_so(so) == 0) {
            val ? (d.alpha |= bit) : (d.alpha &= ~bit);
        } else {
            val ? (d.beta |= bit) : (d.beta &= ~bit);
        }
    };
    flip(i, false); flip(j, false);
    flip(a, true);  flip(b, true);
    return d;
}

template <class Visitor>
inline void enumerate_doubles_hb(
    const Det& bra, int n_orb, const HeatBathTable& hb, double cutoff, Visitor&& visit
) {
    auto occ = get_occupied_sos(bra);
    const size_t n = occ.size();
    for (size_t p = 1; p < n; ++p) {
        for (size_t q = 0; q < p; ++q) {
            const int i = occ[p], j = occ[q];
            auto row = hb.row_view(i, j).with_cutoff(cutoff);
            for (size_t k = 0; k < row.len; ++k) {
                const int a = row.a[k], b = row.b[k];
                if (!is_so_occupied(bra, a, n_orb) && !is_so_occupied(bra, b, n_orb)) {
                    visit(apply_double_exc_so(bra, i, j, a, b));
                }
            }
        }
    }
}

} // anonymous namespace

// ─── Local Connectivity ───────────────────────────────────────────────

LocalConnRow get_local_conn(const Det& bra, const HamEval& ham, int n_orb,
                            const HeatBathTable* hb_table, double eps1,
                            bool use_heatbath) {
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument("get_local_conn: Heat-bath enabled but table is null");
    }

    LocalConnRow out;
    const size_t est = static_cast<size_t>(1 + 2 * n_orb + (n_orb * n_orb) / 10);
    out.dets.reserve(est);
    out.values.reserve(est);

    const double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath ? std::max(eps1, num_thresh) : num_thresh;

    // Diagonal
    double h = ham.compute_diagonal(bra);
    if (std::abs(h) > num_thresh) {
        out.dets.push_back(bra);
        out.values.push_back(h);
    }

    // Singles
    det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
        double val = ham.compute_elem(bra, ket);
        if (std::abs(val) > single_thresh) {
            out.dets.push_back(ket);
            out.values.push_back(val);
        }
    });

    // Doubles
    auto visitor = [&](const Det& ket) {
        double val = ham.compute_elem(bra, ket);
        if (std::abs(val) > num_thresh) {
            out.dets.push_back(ket);
            out.values.push_back(val);
        }
    };

    if (use_heatbath && hb_table) {
        enumerate_doubles_hb(bra, n_orb, *hb_table, eps1, visitor);
    } else {
        det_ops::for_each_double(bra, n_orb, visitor);
    }

    return out;
}

LocalConnBatch get_local_connections(std::span<const Det> samples,
                                     const HamEval& ham, int n_orb,
                                     const HeatBathTable* hb_table, double eps1,
                                     bool use_heatbath) {
    const size_t n = samples.size();
    LocalConnBatch out;
    if (n == 0) { out.offsets.push_back(0); return out; }

    std::vector<LocalConnRow> rows(n);
    #pragma omp parallel for schedule(dynamic, 32)
    for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
        rows[i] = get_local_conn(samples[i], ham, n_orb, hb_table, eps1, use_heatbath);
    }

    out.offsets.resize(n + 1);
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) {
        out.offsets[i] = static_cast<int>(total);
        total += rows[i].dets.size();
    }
    out.offsets[n] = static_cast<int>(total);

    out.dets.reserve(total);
    out.values.reserve(total);
    for (const auto& r : rows) {
        out.dets.insert(out.dets.end(), r.dets.begin(), r.dets.end());
        out.values.insert(out.values.end(), r.values.begin(), r.values.end());
    }
    return out;
}

// ─── Variational Energy ───────────────────────────────────────────────

VariationalResult compute_variational_energy(
    std::span<const Det> basis,
    std::span<const std::complex<double>> coeffs,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    const size_t N = basis.size();
    if (coeffs.size() != N) {
        throw std::invalid_argument("compute_variational_energy: size mismatch");
    }
    if (N == 0) return {0.0, 0.0};
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument("compute_variational_energy: HB table is null");
    }

    // 1. Build Map: Det -> Index
    // Use DetMap::from_ordered to respect input basis order.
    // We assume basis is unique; verify_unique=false for speed if caller guarantees it.
    // Note: DetMap owns the vector, so we copy from span.
    std::vector<Det> basis_vec(basis.begin(), basis.end());
    const DetMap det_map = DetMap::from_ordered(std::move(basis_vec), false);

    // 2. Compute Norm
    double norm = 0.0;
    for (const auto& c : coeffs) norm += std::norm(c);
    if (norm <= 1e-14) return {0.0, 0.0};

    // 3. Streaming Energy Accumulation
    double e_el = 0.0;
    const double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath ? std::max(eps1, num_thresh) : num_thresh;

    #pragma omp parallel reduction(+:e_el)
    {
        #pragma omp for schedule(dynamic, 32)
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(N); ++i) {
            const Det& bra = basis[i];
            const std::complex<double> bra_c_conj = std::conj(coeffs[i]);
            
            if (std::norm(bra_c_conj) < 1e-24) continue;

            std::complex<double> sigma_i = 0.0;

            // Helper to accumulate if connected ket is in basis
            auto accum = [&](const Det& ket, double val) {
                if (auto idx = det_map.get_idx(ket)) {
                    sigma_i += val * coeffs[*idx];
                }
            };

            // Diagonal
            double diag = ham.compute_diagonal(bra);
            if (std::abs(diag) > num_thresh) {
                // Optimization: we know index of bra is i
                sigma_i += diag * coeffs[i];
            }

            // Singles
            det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
                double val = ham.compute_elem(bra, ket);
                if (std::abs(val) > single_thresh) accum(ket, val);
            });

            // Doubles
            auto d_visit = [&](const Det& ket) {
                double val = ham.compute_elem(bra, ket);
                if (std::abs(val) > num_thresh) accum(ket, val);
            };

            if (use_heatbath && hb_table) {
                enumerate_doubles_hb(bra, n_orb, *hb_table, eps1, d_visit);
            } else {
                det_ops::for_each_double(bra, n_orb, d_visit);
            }

            // E += <Psi|bra> <bra|H|Psi> = conj(c_i) * sigma_i
            // Result must be real for Hermitian H
            e_el += std::real(bra_c_conj * sigma_i);
        }
    }

    return {e_el, norm};
}

} // namespace lever