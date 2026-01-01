// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.cpp
 * @brief Implementation of local Hamiltonian connectivity (real-only).
 */

#include <detnqs/hamiltonian/local_conn.hpp>

#include <detnqs/determinant/det_ops.hpp>
#include <detnqs/determinant/det_space.hpp>
#include <detnqs/hamiltonian/ham_utils.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace detnqs {

// ============================================================================
// Local connectivity (unchanged except includes)
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

    // Diagonal
    {
        const double h = ham.compute_diagonal(bra);
        if (std::abs(h) > num_thresh) {
            out.dets.push_back(bra);
            out.values.push_back(h);
        }
    }

    // Singles
    det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
        const double v = ham.compute_elem(bra, ket);
        if (std::abs(v) > single_thresh) {
            out.dets.push_back(ket);
            out.values.push_back(v);
        }
    });

    // Doubles
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
        rows[static_cast<size_t>(i)] = get_local_conn(
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
// Variational energy (real-only)
// ============================================================================

double compute_variational_energy(
    std::span<const Det> basis,
    std::span<const double> coeffs,
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
    if (N == 0) return 0.0;
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "compute_variational_energy: Heat-Bath requested but hb_table is null"
        );
    }

    // Build map for coefficient lookup in O(1)
    std::vector<Det> basis_vec(basis.begin(), basis.end());
    const DetMap det_map = DetMap::from_ordered(std::move(basis_vec), false);

    // Since coeffs are normalized in Python, use an absolute skip threshold.
    constexpr double rel_skip = 1e-24;

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
            const double c_i = coeffs[ii];

            // Skip tiny coefficients (coeffs are normalized already)
            if (c_i * c_i < rel_skip) continue;

            double sigma_i = 0.0;

            auto accum = [&](const Det& ket, double v) {
                if (auto idx = det_map.get_idx(ket)) {
                    sigma_i += v * coeffs[*idx];
                }
            };

            // Diagonal
            const double diag = ham.compute_diagonal(bra);
            if (std::abs(diag) > num_thresh) {
                sigma_i += diag * c_i;
            }

            // Singles
            det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
                const double v = ham.compute_elem(bra, ket);
                if (std::abs(v) > single_thresh) accum(ket, v);
            });

            // Doubles
            auto d_visit = [&](const Det& ket) {
                const double v = ham.compute_elem(bra, ket);
                if (std::abs(v) > num_thresh) accum(ket, v);
            };

            if (use_heatbath && hb_table) {
                for_each_double_hb(bra, n_orb, *hb_table, eps1, d_visit);
            } else {
                det_ops::for_each_double(bra, n_orb, d_visit);
            }

            e_el += c_i * sigma_i; // real-only: <c|H|c>
        }
    }

    return e_el; // energy for normalized coeffs
}

// ============================================================================
// PT2 (Epstein–Nesbet) (real-only, normalized internally)
// ============================================================================

Pt2Result compute_pt2(
    std::span<const Det> S,
    std::span<const double> coeffs_S,
    const HamEval& ham,
    int n_orb,
    double e_ref,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    const std::size_t nS = S.size();
    if (coeffs_S.size() != nS) {
        throw std::invalid_argument("compute_pt2: S/coeff size mismatch");
    }
    if (nS == 0) return {0.0, 0};
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument("compute_pt2: Heat-Bath requested but hb_table is null");
    }

    // Build index map for S membership
    std::vector<Det> S_vec(S.begin(), S.end());
    const DetMap map_S = DetMap::from_ordered(std::move(S_vec), false);

    const double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

    // Coeffs are normalized in Python, so use an absolute skip threshold.
    constexpr double rel_skip = 1e-24;

    // Accumulate v(a) = <a|H|Psi_S> (Psi_S normalized)
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    using AccMap = std::unordered_map<Det, double>;
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
        const double c_i = coeffs_S[ii];

        if (c_i * c_i < rel_skip) continue;

        // Singles
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            if (map_S.get_idx(ket)) return; // exclude internal
            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= single_thresh) return;
            acc[ket] += h * c_i;
        });

        // Doubles
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

    // Merge thread-local maps
    AccMap ext;
    ext.reserve(nS * 32);
    for (auto& m : tl_maps) {
        for (auto& kv : m) ext[kv.first] += kv.second;
    }

    double e_pt2 = 0.0;
    constexpr double denom_eps = 1e-12;

    for (const auto& kv : ext) {
        const Det& a = kv.first;
        const double v = kv.second;

        const double haa = ham.compute_diagonal(a);
        double denom = e_ref - haa; // use optimized reference energy

        if (std::abs(denom) < denom_eps) {
            denom = (denom >= 0.0) ? denom_eps : -denom_eps;
        }

        e_pt2 += (v * v) / denom; // real-only: |v|^2 = v^2
    }

    return {e_pt2, ext.size()};
}

} // namespace detnqs