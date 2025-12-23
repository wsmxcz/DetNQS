// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.cpp
 * @brief Implementation of local Hamiltonian connectivity.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#include <lever/hamiltonian/local_conn.hpp>

#include <lever/determinant/det_ops.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/hamiltonian/ham_utils.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace lever {

// ============================================================================
// Local connectivity
// ============================================================================

LocalConnRow get_local_conn(
    const Det& bra,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "get_local_conn: Heat-Bath requested but hb_table is null"
        );
    }

    LocalConnRow out;
    const size_t est = est_conn_cap(n_orb);
    out.dets.reserve(est);
    out.values.reserve(est);

    const double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

    // Diagonal element
    {
        const double h = ham.compute_diagonal(bra);
        if (std::abs(h) > num_thresh) {
            out.dets.push_back(bra);
            out.values.push_back(h);
        }
    }

    // Single excitations
    det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
        const double v = ham.compute_elem(bra, ket);
        if (std::abs(v) > single_thresh) {
            out.dets.push_back(ket);
            out.values.push_back(v);
        }
    });

    // Double excitations
    auto visit = [&](const Det& ket) {
        const double v = ham.compute_elem(bra, ket);
        if (std::abs(v) > num_thresh) {
            out.dets.push_back(ket);
            out.values.push_back(v);
        }
    };

    if (use_heatbath && hb_table) {
        for_each_double_hb(bra, n_orb, *hb_table, eps1, visit);
    } else {
        det_ops::for_each_double(bra, n_orb, visit);
    }

    return out;
}

LocalConnBatch get_local_connections(
    std::span<const Det> samples,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    const size_t n = samples.size();
    LocalConnBatch out;
    if (n == 0) {
        out.offsets.push_back(0);
        return out;
    }

    std::vector<LocalConnRow> rows(n);

#pragma omp parallel for schedule(dynamic, 32)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        rows[i] = get_local_conn(
            samples[static_cast<size_t>(i)],
            ham,
            n_orb,
            hb_table,
            eps1,
            use_heatbath
        );
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

// ============================================================================
// Variational energy
// ============================================================================

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
        throw std::invalid_argument(
            "compute_variational_energy: basis/coeff size mismatch"
        );
    }
    if (N == 0) return {0.0, 0.0};
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "compute_variational_energy: Heat-Bath requested but hb_table is null"
        );
    }

    std::vector<Det> basis_vec(basis.begin(), basis.end());
    const DetMap det_map = DetMap::from_ordered(std::move(basis_vec), false);

    double norm = 0.0;
    for (const auto& c : coeffs) norm += std::norm(c);
    if (norm <= 1e-14) return {0.0, 0.0};

    double e_el = 0.0;
    const double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

#pragma omp parallel reduction(+:e_el)
    {
#pragma omp for schedule(dynamic, 32)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N); ++i) {
            const size_t ii = static_cast<size_t>(i);
            const Det& bra = basis[ii];
            const std::complex<double> c_bra_conj = std::conj(coeffs[ii]);
            if (std::norm(c_bra_conj) < 1e-24) continue;

            std::complex<double> sigma_i = 0.0;

            auto accum = [&](const Det& ket, double v) {
                if (auto idx = det_map.get_idx(ket)) {
                    sigma_i += v * coeffs[*idx];
                }
            };

            const double diag = ham.compute_diagonal(bra);
            if (std::abs(diag) > num_thresh) {
                sigma_i += diag * coeffs[ii];
            }

            det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
                const double v = ham.compute_elem(bra, ket);
                if (std::abs(v) > single_thresh) accum(ket, v);
            });

            auto d_visit = [&](const Det& ket) {
                const double v = ham.compute_elem(bra, ket);
                if (std::abs(v) > num_thresh) accum(ket, v);
            };

            if (use_heatbath && hb_table) {
                for_each_double_hb(bra, n_orb, *hb_table, eps1, d_visit);
            } else {
                det_ops::for_each_double(bra, n_orb, d_visit);
            }

            e_el += std::real(c_bra_conj * sigma_i);
        }
    }

    return {e_el, norm};
}

// ============================================================================
// PT2 (Epstein–Nesbet)
// ============================================================================

Pt2Result compute_pt2(
    std::span<const Det> S,
    std::span<const std::complex<double>> coeffs_S,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    const std::size_t nS = S.size();
    if (coeffs_S.size() != nS) {
        throw std::invalid_argument("compute_pt2: S/coeff size mismatch");
    }
    if (nS == 0) return {0.0, 0.0, 0};
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument("compute_pt2: Heat-Bath requested but hb_table is null");
    }

    // Compute variational energy on S (normalized).
    const auto var = compute_variational_energy(
        S, coeffs_S, ham, n_orb,
        use_heatbath ? hb_table : nullptr,
        eps1, use_heatbath
    );
    if (var.norm <= 1e-14) return {0.0, 0.0, 0};

    const double e_var = var.e_el / var.norm;

    // Build an index map for S membership tests (preserve S order).
    std::vector<Det> S_vec(S.begin(), S.end());
    const DetMap map_S = DetMap::from_ordered(std::move(S_vec), false);

    const double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

    // Accumulate v(a) = <a|H|Psi_S> = sum_i H_{i,a} * c_i
    // using thread-local hash maps to reduce contention.
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    using AccMap = std::unordered_map<Det, std::complex<double>>;
    std::vector<AccMap> tl_maps(static_cast<std::size_t>(n_threads));
    for (auto& m : tl_maps) m.reserve(nS * 8);

#pragma omp parallel for schedule(dynamic, 32) if(nS > 128)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(nS); ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = tl_maps[static_cast<std::size_t>(tid)];

        const std::size_t ii = static_cast<std::size_t>(i);
        const Det& bra = S[ii];
        const std::complex<double> c_i = coeffs_S[ii];

        // Skip tiny coefficients.
        if (std::norm(c_i) < 1e-24) continue;

        // Singles: enumerate all, then threshold.
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            if (map_S.get_idx(ket)) return; // exclude internal space

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= single_thresh) return;

            acc[ket] += h * c_i;
        });

        // Doubles: either full enumeration or HB-screened enumeration.
        auto visit_double = [&](const Det& ket) {
            if (map_S.get_idx(ket)) return;

            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= num_thresh) return;

            acc[ket] += h * c_i;
        };

        if (use_heatbath && hb_table) {
            for_each_double_hb(bra, n_orb, *hb_table, eps1, visit_double);
        } else {
            det_ops::for_each_double(bra, n_orb, visit_double);
        }
    }

    // Merge thread-local maps.
    AccMap ext;
    ext.reserve(nS * 32);
    for (auto& m : tl_maps) {
        for (auto& kv : m) {
            ext[kv.first] += kv.second;
        }
    }

    // PT2 sum: sum_a |v(a)|^2 / (E_var - H_aa)
    double e_pt2 = 0.0;
    constexpr double denom_eps = 1e-12;

    for (const auto& kv : ext) {
        const Det& a = kv.first;
        const std::complex<double>& v = kv.second;

        const double haa = ham.compute_diagonal(a);
        double denom = e_var - haa;

        // Avoid numerical blow-ups for near-zero denominators.
        if (std::abs(denom) < denom_eps) {
            denom = (denom >= 0.0) ? denom_eps : -denom_eps;
        }

        e_pt2 += std::norm(v) / denom;
    }

    return {e_var, e_pt2, ext.size()};
}

} // namespace lever
