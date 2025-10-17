// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file integral_mo.hpp
 * @brief Molecular orbital integrals management and FCIDUMP file parsing.
 * @author Zheng (Alex) Che, email: wsmxcz@gmail.com
 * @date July, 2025
 */
#pragma once

#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <cstddef>

namespace lever {

/**
 * @brief Computes canonical key for one-electron integral using triangular packing.
 *        Exploits symmetry: h(p,q) = h(q,p).
 * @param p First MO index (0-indexed).
 * @param q Second MO index (0-indexed).
 * @return Unique key for h(p,q).
 */
[[nodiscard]] constexpr size_t h1e_key(int p, int q) noexcept {
    if (p < q) {
        std::swap(p, q);
    }
    return static_cast<size_t>(p) * (p + 1) / 2 + q;
}

/**
 * @brief Computes canonical key for two-electron integral using compound triangular packing.
 *        Exploits 8-fold symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq) etc.
 * @param p,q,r,s MO indices (0-indexed).
 * @return Unique key for (pq|rs).
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
 * @class IntegralMO
 * @brief Manages molecular orbital integrals from FCIDUMP files.
 *
 * Provides memory-efficient storage using triangular packing and fast
 * integral lookup for post-Hartree-Fock calculations.
 */
class IntegralMO {
public:
    int n_orbs{0};      ///< Number of spatial orbitals.
    int n_elecs{0};     ///< Total number of electrons.
    int spin_mult{1};   ///< Spin multiplicity (2S+1).
    double e_nuc{0.0};  ///< Nuclear repulsion energy.

    /**
     * @brief Constructs IntegralMO with pre-allocated storage.
     * @param num_orbitals Number of spatial orbitals [1, 64].
     * @throws std::invalid_argument if num_orbitals is out of range.
     */
    explicit IntegralMO(int num_orbitals);

    /**
     * @brief Loads integrals from FCIDUMP file.
     * @param filename Path to FCIDUMP file.
     * @throws std::runtime_error if file cannot be opened or parsed.
     */
    void load_from_fcidump(const std::string& filename);

    /**
     * @brief Counts non-zero integrals.
     * @param threshold Numerical threshold for zero detection.
     * @return {h1e_count, h2e_count}.
     */
    [[nodiscard]] std::pair<size_t, size_t> get_nonzero_count(double threshold = 1e-12) const noexcept;

    /**
     * @brief Retrieves one-electron integral h(p,q).
     * @param p,q MO indices (0-indexed).
     * @return Integral value.
     */
    [[nodiscard]] double get_h1e(int p, int q) const noexcept {
        return h1e_[h1e_key(p, q)];
    }

    /**
     * @brief Retrieves two-electron integral in Chemists' notation: (pq|rs).
     *
     * This corresponds to the Physicists' notation <pr|qs>.
     * The FCIDUMP format standardly uses Chemists' notation.
     *
     * @param p,q,r,s MO indices (0-indexed).
     * @return Integral value.
     */
    [[nodiscard]] double get_h2e(int p, int q, int r, int s) const noexcept {
        return h2e_[h2e_key(p, q, r, s)];
    }

private:
    std::vector<double> h1e_;  ///< One-electron integrals (triangular packed).
    std::vector<double> h2e_;  ///< Two-electron integrals (compound triangular packed).

    /**
     * @brief Parses FCIDUMP namelist section for metadata.
     * @param file Input file stream.
     * @throws std::runtime_error if namelist parsing fails.
     */
    void parse_namelist(std::ifstream& file);

    /**
     * @brief Extracts integer parameter from namelist content.
     * @param content Namelist string content.
     * @param param Parameter name to extract.
     * @param value Reference to store extracted value.
     * @return True if parameter found and parsed successfully.
     */
    bool extract_parameter(const std::string& content, const std::string& param, int& value);

    /**
     * @brief Checks if line is a comment.
     * @param line Input line.
     * @return True if line is comment or empty.
     */
    static bool is_comment_line(const std::string& line) noexcept;

    /**
     * @brief Parses single integral data line from FCIDUMP.
     * @param line Integral data line.
     */
    void parse_integral_line(std::string& line);
};

} // namespace lever
