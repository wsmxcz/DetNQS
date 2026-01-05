// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file local_conn.cpp
 * @brief Local Hamiltonian connectivity and energy evaluation (real-only).
 * 
 * Implements sparse Hamiltonian actions via deterministic excitation enumeration.
 * Supports Heat-Bath screening for double excitations (eps1 threshold).
 * 
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
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
// Local connectivity: H|bra> in sparse form
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
            "get_local_conn: Heat-Bath mode requires non-null hb_table"
        );
    }

    LocalConnRow out;
    const size_t est = est_conn_cap(n_orb);
    out.dets.reserve(est);
    out.values.reserve(est);

    constexpr double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

    // Diagonal element: <bra|H|bra>
    {
        const double h_diag = ham.compute_diagonal(bra);
        if (std::abs(h_diag) > num_thresh) {
            out.dets.push_back(bra);
            out.values.push_back(h_diag);
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

    // Double excitations (with optional Heat-Bath screening)
    auto visit_double = [&](const Det& ket) {
        const double v = ham.compute_elem(bra, ket);
        if (std::abs(v) > num_thresh) {
            out.dets.push_back(ket);
            out.values.push_back(v);
        }
    };

    if (use_heatbath && hb_table) {
        for_each_double_hb(bra, n_orb, *hb_table, eps1, visit_double);
    } else {
        det_ops::for_each_double(bra, n_orb, visit_double);
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

    // Build CSR-like structure
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
// Variational energy: E_var = <Psi_theta|H|Psi_theta> on V_k
// ============================================================================

double compute_variational_energy(
    std::span<const Det> var_basis,
    std::span<const double> coeffs,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    const size_t N_var = var_basis.size();
    if (coeffs.size() != N_var) {
        throw std::invalid_argument(
            "compute_variational_energy: var_basis/coeffs size mismatch"
        );
    }
    if (N_var == 0) return 0.0;
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "compute_variational_energy: Heat-Bath mode requires non-null hb_table"
        );
    }

    // Build index map for coefficient lookup in O(1)
    std::vector<Det> basis_vec(var_basis.begin(), var_basis.end());
    const DetMap det_map = DetMap::from_ordered(std::move(basis_vec), false);

    constexpr double coeff_skip = 1e-24;  // Skip negligible contributions
    constexpr double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

    double e_var = 0.0;

#pragma omp parallel reduction(+:e_var)
    {
#pragma omp for schedule(dynamic, 32)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N_var); ++i) {
            const size_t ii = static_cast<size_t>(i);
            const Det& bra = var_basis[ii];
            const double c_bra = coeffs[ii];

            if (c_bra * c_bra < coeff_skip) continue;

            double sigma_i = 0.0;

            auto accumulate = [&](const Det& ket, double h_elem) {
                if (auto idx = det_map.get_idx(ket)) {
                    sigma_i += h_elem * coeffs[*idx];
                }
            };

            // Diagonal contribution
            const double h_diag = ham.compute_diagonal(bra);
            if (std::abs(h_diag) > num_thresh) {
                sigma_i += h_diag * c_bra;
            }

            // Single excitations
            det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
                const double v = ham.compute_elem(bra, ket);
                if (std::abs(v) > single_thresh) accumulate(ket, v);
            });

            // Double excitations
            auto visit_double = [&](const Det& ket) {
                const double v = ham.compute_elem(bra, ket);
                if (std::abs(v) > num_thresh) accumulate(ket, v);
            };

            if (use_heatbath && hb_table) {
                for_each_double_hb(bra, n_orb, *hb_table, eps1, visit_double);
            } else {
                det_ops::for_each_double(bra, n_orb, visit_double);
            }

            e_var += c_bra * sigma_i;
        }
    }

    return e_var;
}

// ============================================================================
// Second-order Epstein–Nesbet / sPT correction for a non-linear reference |Psi0>.

// DeltaE2 = sum_k | <D_k|(H - E0)|Psi0> |^2 / (E0 - H_kk),
// where H_kk = <D_k|H|D_k>.

// If |Psi0> is an eigenvector of H restricted to the variational subspace (selected-CI case),
// the residual on V vanishes and the formula reduces to the usual external-space EN-PT2:
// DeltaE2 = sum_{a∉V} |<D_a|H|Psi0>|^2 / (E0 - H_aa).
// ============================================================================

Pt2Result compute_pt2(
    std::span<const Det> var_basis,
    std::span<const double> coeffs_var,
    const HamEval& ham,
    int n_orb,
    double e_ref,
    const HeatBathTable* hb_table,
    double eps1,
    bool use_heatbath
) {
    const std::size_t N_var = var_basis.size();
    if (coeffs_var.size() != N_var) {
        throw std::invalid_argument(
            "compute_pt2: var_basis/coeffs_var size mismatch"
        );
    }
    if (N_var == 0) return {0.0, 0};
    if (use_heatbath && !hb_table) {
        throw std::invalid_argument(
            "compute_pt2: Heat-Bath mode requires non-null hb_table"
        );
    }

    // Map for fast "is ket in V?" and index lookup
    std::vector<Det> basis_vec(var_basis.begin(), var_basis.end());
    const DetMap map_var = DetMap::from_ordered(std::move(basis_vec), false);

    constexpr double num_thresh = MAT_ELEMENT_THRESH;
    const double single_thresh = use_heatbath
        ? std::max(eps1, num_thresh)
        : num_thresh;

    constexpr double coeff_skip = 1e-24;
    constexpr double denom_eps  = 1e-12;

    // Thread-local accumulation of v(a) = <a|H|Psi0> for external determinants a ∉ V
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    using ExtMap = std::unordered_map<Det, double>;
    std::vector<ExtMap> thread_maps(static_cast<std::size_t>(n_threads));
    for (auto& m : thread_maps) m.reserve(N_var * 8);

    // Thread-local accumulation of the "internal residual" term on V
    std::vector<double> thread_internal(static_cast<std::size_t>(n_threads), 0.0);

#pragma omp parallel for schedule(dynamic, 32) if(N_var > 128)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(N_var); ++i) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& ext_acc = thread_maps[static_cast<std::size_t>(tid)];

        const std::size_t ii = static_cast<std::size_t>(i);
        const Det& bra = var_basis[ii];
        const double c_bra = coeffs_var[ii];

        if (c_bra * c_bra < coeff_skip) continue;

        // sigma_i = <D_i|H|Psi0> restricted to V coefficients
        double sigma_i = 0.0;

        // Diagonal (always needed for denom and contributes to sigma_i)
        const double h_ii = ham.compute_diagonal(bra);
        sigma_i += h_ii * c_bra;

        // Single excitations: contribute either to sigma_i (if ket in V) or v(a) (if ket ∉ V)
        det_ops::for_each_single(bra, n_orb, [&](const Det& ket) {
            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= single_thresh) return;

            if (auto idx = map_var.get_idx(ket)) {
                sigma_i += h * coeffs_var[*idx];
            } else {
                ext_acc[ket] += h * c_bra;
            }
        });

        // Double excitations (with optional Heat-Bath generation)
        auto visit_double = [&](const Det& ket) {
            const double h = ham.compute_elem(bra, ket);
            if (std::abs(h) <= num_thresh) return;

            if (auto idx = map_var.get_idx(ket)) {
                sigma_i += h * coeffs_var[*idx];
            } else {
                ext_acc[ket] += h * c_bra;
            }
        };

        if (use_heatbath && hb_table) {
            for_each_double_hb(bra, n_orb, *hb_table, eps1, visit_double);
        } else {
            det_ops::for_each_double(bra, n_orb, visit_double);
        }

        // Internal (V-space) residual contribution:
        //   r_i = <D_i|(H - E0)|Psi0> = sigma_i - E0 * c_i
        //   DeltaE2_int += r_i^2 / (E0 - H_ii)
        const double r_i = sigma_i - e_ref * c_bra;

        double denom = e_ref - h_ii;
        if (std::abs(denom) < denom_eps) {
            denom = (denom >= 0.0) ? denom_eps : -denom_eps;
        }

        thread_internal[static_cast<std::size_t>(tid)] += (r_i * r_i) / denom;
    }

    // Merge external maps into a single perturbative set P_k (a ∉ V)
    ExtMap perturb_set;
    perturb_set.reserve(N_var * 32);
    for (auto& m : thread_maps) {
        for (auto& [det, val] : m) {
            perturb_set[det] += val;
        }
    }

    // Sum internal term across threads
    double delta_int = 0.0;
    for (double x : thread_internal) delta_int += x;

    // External EN-PT2 term: sum_{a∉V} v(a)^2 / (E0 - H_aa)
    double delta_ext = 0.0;
    for (const auto& [a, v_a] : perturb_set) {
        const double h_aa = ham.compute_diagonal(a);

        double denom = e_ref - h_aa;
        if (std::abs(denom) < denom_eps) {
            denom = (denom >= 0.0) ? denom_eps : -denom_eps;
        }

        delta_ext += (v_a * v_a) / denom;
    }

    return {delta_int + delta_ext, perturb_set.size()};
}

} // namespace detnqs
