// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file integral_so.hpp
 * @brief Spin-orbital integral wrapper and conversion utilities.
 * @author Zheng (Alex) Che, email: wsmxcz@gmail.com
 * @date July, 2025
 */

#pragma once

#include <lever/integral/integral_mo.hpp>
#include <cassert>

namespace lever {

/**
 * @brief Converts molecular orbital index and spin to spin-orbital index.
 * @param mo_idx Molecular orbital index (0-indexed).
 * @param spin Spin (0 for alpha, 1 for beta).
 * @return Spin-orbital index using convention: even=alpha, odd=beta.
 */
[[nodiscard]] constexpr int so_from_mo(int mo_idx, int spin) noexcept {
    return (mo_idx << 1) | spin;
}

/**
 * @brief Extracts molecular orbital index from spin-orbital index.
 * @param so_idx Spin-orbital index.
 * @return Molecular orbital index.
 */
[[nodiscard]] constexpr int mo_from_so(int so_idx) noexcept {
    return so_idx >> 1;
}

/**
 * @brief Extracts spin from spin-orbital index.
 * @param so_idx Spin-orbital index.
 * @return Spin (0 for alpha, 1 for beta).
 */
[[nodiscard]] constexpr int spin_from_so(int so_idx) noexcept {
    return so_idx & 1;
}

/**
 * @brief Checks if two spin-orbital indices have the same spin.
 * @param so_idx1 First spin-orbital index.
 * @param so_idx2 Second spin-orbital index.
 * @return True if both have the same spin.
 */
[[nodiscard]] constexpr bool have_same_spin(int so_idx1, int so_idx2) noexcept {
    return (so_idx1 & 1) == (so_idx2 & 1);
}

/**
 * @class IntegralSO
 * @brief Spin-orbital integral wrapper for molecular orbital integrals.
 *
 * Provides spin-orbital basis access to MO integrals with automatic spin conservation.
 * Uses convention: even indices=alpha, odd indices=beta.
 *
 * Integral notations:
 * - Physicist's: <pq|rs> = (p_1 r_1 | q_1 s_1)
 * - Chemist's: [pq|rs] = <pr|qs>
 * - Antisymmetrized: <pq||rs> = <pq|rs> - <pq|sr>
 */
class IntegralSO {
public:
    /**
     * @brief Constructs spin-orbital integral wrapper.
     * @param mo_integrals Reference to molecular orbital integrals.
     */
    explicit IntegralSO(const IntegralMO& mo_integrals) noexcept
        : mo_ints_(mo_integrals), n_so_(mo_integrals.n_orbs << 1) {}

    /**
     * @brief Gets total number of spin-orbitals.
     * @return Number of spin-orbitals (2 × n_orbs).
     */
    [[nodiscard]] constexpr int get_n_so() const noexcept { return n_so_; }

    /**
     * @brief Gets total number of molecular orbitals.
     * @return Number of molecular orbitals.
     */
    [[nodiscard]] constexpr int get_n_mo() const noexcept { return mo_ints_.n_orbs; }

    /**
     * @brief Gets one-electron spin-orbital integral <μ|h|ν>.
     * @param mu First spin-orbital index.
     * @param nu Second spin-orbital index.
     * @return Integral value (0.0 if spins differ).
     */
    [[nodiscard]] double get_h1e_so(int mu, int nu) const noexcept {
        validate_so_index(mu);
        validate_so_index(nu);
        
        if (!have_same_spin(mu, nu)) [[unlikely]] return 0.0;
        return mo_ints_.get_h1e(mo_from_so(mu), mo_from_so(nu));
    }

    /**
     * @brief Gets two-electron integral in Physicist's notation: <μκ|νλ>.
     * @param mu First spin-orbital index.
     * @param kappa Second spin-orbital index.
     * @param nu Third spin-orbital index.
     * @param lambda Fourth spin-orbital index.
     * @return Integral value (0.0 if spin conservation violated).
     */
    [[nodiscard]] double get_h2e_phys(int mu, int kappa, int nu, int lambda) const noexcept {
        validate_so_indices(mu, kappa, nu, lambda);
        
        if (!have_same_spin(mu, nu) || !have_same_spin(kappa, lambda)) [[unlikely]] {
            return 0.0;
        }
        return mo_ints_.get_h2e(mo_from_so(mu), mo_from_so(nu),
                               mo_from_so(kappa), mo_from_so(lambda));
    }

    /**
     * @brief Gets two-electron integral in Chemist's notation: [μν|κλ].
     * @param mu First spin-orbital index.
     * @param nu Second spin-orbital index.
     * @param kappa Third spin-orbital index.
     * @param lambda Fourth spin-orbital index.
     * @return Integral value (0.0 if spin conservation violated).
     */
    [[nodiscard]] double get_h2e_chem(int mu, int nu, int kappa, int lambda) const noexcept {
        return get_h2e_phys(mu, kappa, nu, lambda);
    }

    /**
     * @brief Gets antisymmetrized two-electron integral: <μκ||νλ>.
     * @param mu First spin-orbital index.
     * @param kappa Second spin-orbital index.
     * @param nu Third spin-orbital index.
     * @param lambda Fourth spin-orbital index.
     * @return Antisymmetrized integral value.
     */
    [[nodiscard]] double get_h2e_anti(int mu, int kappa, int nu, int lambda) const noexcept {
        return get_h2e_phys(mu, kappa, nu, lambda) - 
               get_h2e_phys(mu, kappa, lambda, nu);
    }

    /**
     * @brief Gets Coulomb integral J_μν = <μμ|νν>.
     * @param mu First spin-orbital index.
     * @param nu Second spin-orbital index.
     * @return Coulomb integral value.
     */
    [[nodiscard]] double get_coulomb(int mu, int nu) const noexcept {
        return get_h2e_phys(mu, mu, nu, nu);
    }

    /**
     * @brief Gets exchange integral K_μν = <μν|νμ>.
     * @param mu First spin-orbital index.
     * @param nu Second spin-orbital index.
     * @return Exchange integral value.
     */
    [[nodiscard]] double get_exchange(int mu, int nu) const noexcept {
        return get_h2e_phys(mu, nu, nu, mu);
    }

    /**
     * @brief Gets nuclear repulsion energy.
     * @return Nuclear repulsion energy.
     */
    [[nodiscard]] double get_e_nuc() const noexcept {
        return mo_ints_.e_nuc;
    }

private:
    const IntegralMO& mo_ints_;  ///< Reference to molecular orbital integrals.
    const int n_so_;             ///< Total number of spin-orbitals.

    /**
     * @brief Debug validation for single spin-orbital index.
     * @param idx Spin-orbital index to validate.
     */
    void validate_so_index(int idx) const noexcept {
        assert(idx >= 0 && idx < n_so_ && "Spin-orbital index out of bounds.");
    }

    /**
     * @brief Debug validation for four spin-orbital indices.
     * @param mu First index.
     * @param nu Second index.
     * @param kappa Third index.
     * @param lambda Fourth index.
     */
    void validate_so_indices(int mu, int nu, int kappa, int lambda) const noexcept {
        assert(mu >= 0 && mu < n_so_ && "Spin-orbital index mu out of bounds.");
        assert(nu >= 0 && nu < n_so_ && "Spin-orbital index nu out of bounds.");
        assert(kappa >= 0 && kappa < n_so_ && "Spin-orbital index kappa out of bounds.");
        assert(lambda >= 0 && lambda < n_so_ && "Spin-orbital index lambda out of bounds.");
    }
};

} // namespace lever
