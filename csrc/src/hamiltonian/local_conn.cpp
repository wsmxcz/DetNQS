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

} // namespace lever
