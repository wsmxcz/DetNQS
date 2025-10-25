// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file integral_mo.hpp
 * @brief Molecular orbital integrals storage and FCIDUMP parser.
 *
 * Manages one- and two-electron integrals with triangular packing
 * for efficient memory usage in post-HF calculations.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <cstddef>

namespace lever {

/**
 * Canonical key for one-electron integral h_pq via triangular packing.
 * Exploits permutational symmetry h_pq = h_qp.
 * Formula: key = max(p,q)·(max(p,q)+1)/2 + min(p,q)
 */
[[nodiscard]] constexpr size_t h1e_key(int p, int q) noexcept {
    if (p < q) std::swap(p, q);
    return static_cast<size_t>(p) * (p + 1) / 2 + q;
}

/**
 * Canonical key for two-electron integral (pq|rs) via compound packing.
 * Exploits 8-fold permutational symmetry.
 * Formula: key = K_pq·(K_pq+1)/2 + K_rs where K_ij = h1e_key(i,j)
 */
[[nodiscard]] constexpr size_t h2e_key(int p, int q, int r, int s) noexcept {
    if (p < q) std::swap(p, q);
    if (r < s) std::swap(r, s);
    
    size_t pq_key = h1e_key(p, q);
    size_t rs_key = h1e_key(r, s);
    
    if (pq_key < rs_key) std::swap(pq_key, rs_key);
    return pq_key * (pq_key + 1) / 2 + rs_key;
}

/**
 * Molecular orbital integral container with FCIDUMP I/O.
 *
 * Storage scheme:
 *   - h1e: Triangular packed (n²/2 elements)
 *   - h2e: Compound triangular packed (~n⁴/8 elements)
 */
class IntegralMO {
public:
    int n_orbs{0};      ///< Number of spatial MOs
    int n_elecs{0};     ///< Total electron count
    int spin_mult{1};   ///< Spin multiplicity 2S+1
    double e_nuc{0.0};  ///< Nuclear repulsion energy

    /**
     * Construct with pre-allocated storage for num_orbitals MOs.
     * @throws std::invalid_argument if num_orbitals ∉ [1,64]
     */
    explicit IntegralMO(int num_orbitals);

    /**
     * Load integrals from FCIDUMP file.
     * @throws std::runtime_error on I/O or parse failure
     */
    void load_from_fcidump(const std::string& filename);

    /**
     * Count non-zero integrals above threshold.
     * @return {n_h1e, n_h2e} counts
     */
    [[nodiscard]] std::pair<size_t, size_t> 
    get_nonzero_count(double threshold = 1e-12) const noexcept;

    /**
     * Retrieve one-electron integral h_pq.
     */
    [[nodiscard]] double get_h1e(int p, int q) const noexcept {
        return h1e_[h1e_key(p, q)];
    }

    /**
     * Retrieve two-electron integral (pq|rs) in Chemists' notation.
     *
     * Note: Chemists' (pq|rs) ≡ Physicists' ⟨pr|qs⟩
     * FCIDUMP uses Chemists' convention.
     */
    [[nodiscard]] double get_h2e(int p, int q, int r, int s) const noexcept {
        return h2e_[h2e_key(p, q, r, s)];
    }

private:
    std::vector<double> h1e_;  ///< One-electron integrals (packed)
    std::vector<double> h2e_;  ///< Two-electron integrals (packed)

    /// Parse FCIDUMP namelist section for system metadata
    void parse_namelist(std::ifstream& file);

    /// Extract integer parameter from namelist content
    bool extract_parameter(
        const std::string& content, 
        const std::string& param, 
        int& value
    );

    /// Check if line is comment or empty
    static bool is_comment_line(const std::string& line) noexcept;

    /// Parse single integral data line
    void parse_integral_line(std::string& line);
};

} // namespace lever
