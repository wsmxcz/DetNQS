// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file integral_so.hpp
 * @brief Spin-orbital integral wrapper with automatic spin conservation.
 *
 * Converts MO integrals to SO basis using convention: even=α, odd=β.
 * Integral notations: <pq|rs> (Physicist), [pq|rs] (Chemist), <pq||rs> (antisymmetrized).
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <detnqs/integral/integral_mo.hpp>
#include <cassert>

namespace detnqs {

// ============================================================================
// Spin-orbital index conversion utilities
// ============================================================================

/**
 * Convert MO index and spin to SO index: so = 2·mo + σ.
 * @param mo_idx Molecular orbital index
 * @param spin   Spin (0=α, 1=β)
 */
[[nodiscard]] constexpr int so_from_mo(int mo_idx, int spin) noexcept {
    return (mo_idx << 1) | spin;
}

/** Extract MO index from SO index: mo = ⌊so/2⌋. */
[[nodiscard]] constexpr int mo_from_so(int so_idx) noexcept {
    return so_idx >> 1;
}

/** Extract spin from SO index: σ = so mod 2. */
[[nodiscard]] constexpr int spin_from_so(int so_idx) noexcept {
    return so_idx & 1;
}

/** Check if two SO indices share same spin. */
[[nodiscard]] constexpr bool have_same_spin(int so_idx1, int so_idx2) noexcept {
    return (so_idx1 & 1) == (so_idx2 & 1);
}

// ============================================================================
// Spin-orbital integral wrapper
// ============================================================================

/**
 * Spin-orbital integral wrapper with automatic spin conservation.
 *
 * Provides SO basis access to MO integrals. Returns 0 when spin conservation
 * is violated (e.g., ⟨α|h|β⟩ = 0, ⟨αβ|αβ⟩ = 0).
 *
 * Convention: Even indices → α spin, Odd indices → β spin
 */
class IntegralSO {
public:
    explicit IntegralSO(const IntegralMO& mo_integrals) noexcept
        : mo_ints_(mo_integrals), n_so_(mo_integrals.n_orbs << 1) {}

    /** Total number of spin-orbitals (2 × n_orbs). */
    [[nodiscard]] constexpr int get_n_so() const noexcept { return n_so_; }

    /** Total number of molecular orbitals. */
    [[nodiscard]] constexpr int get_n_mo() const noexcept { return mo_ints_.n_orbs; }

    /**
     * One-electron integral h_μν = ⟨μ|h|ν⟩.
     * Returns 0 if spin_μ ≠ spin_ν.
     */
    [[nodiscard]] double get_h1e_so(int mu, int nu) const noexcept {
        validate_so_index(mu);
        validate_so_index(nu);
        
        if (!have_same_spin(mu, nu)) [[unlikely]] return 0.0;
        return mo_ints_.get_h1e(mo_from_so(mu), mo_from_so(nu));
    }

    /**
     * Two-electron integral (Physicist's notation): ⟨μκ|νλ⟩.
     * Returns 0 if spin_μ ≠ spin_ν or spin_κ ≠ spin_λ.
     */
    [[nodiscard]] double get_h2e_phys(int mu, int kappa, int nu, int lambda) const noexcept {
        validate_so_indices(mu, kappa, nu, lambda);
        
        if (!have_same_spin(mu, nu) || !have_same_spin(kappa, lambda)) [[unlikely]] {
            return 0.0;
        }
        return mo_ints_.get_h2e(mo_from_so(mu), mo_from_so(nu),
                               mo_from_so(kappa), mo_from_so(lambda));
    }

    /** Two-electron integral (Chemist's notation): [μν|κλ] = ⟨μκ|νλ⟩. */
    [[nodiscard]] double get_h2e_chem(int mu, int nu, int kappa, int lambda) const noexcept {
        return get_h2e_phys(mu, kappa, nu, lambda);
    }

    /** Antisymmetrized integral: ⟨μκ||νλ⟩ = ⟨μκ|νλ⟩ - ⟨μκ|λν⟩. */
    [[nodiscard]] double get_h2e_anti(int mu, int kappa, int nu, int lambda) const noexcept {
        return get_h2e_phys(mu, kappa, nu, lambda) - 
               get_h2e_phys(mu, kappa, lambda, nu);
    }

    /** Coulomb integral: J_μν = ⟨μμ|νν⟩. */
    [[nodiscard]] double get_coulomb(int mu, int nu) const noexcept {
        return get_h2e_phys(mu, mu, nu, nu);
    }

    /** Exchange integral: K_μν = ⟨μν|νμ⟩. */
    [[nodiscard]] double get_exchange(int mu, int nu) const noexcept {
        return get_h2e_phys(mu, nu, nu, mu);
    }

    /** Nuclear repulsion energy. */
    [[nodiscard]] double get_e_nuc() const noexcept {
        return mo_ints_.e_nuc;
    }

private:
    const IntegralMO& mo_ints_;
    const int n_so_;

    void validate_so_index(int idx) const noexcept {
        assert(idx >= 0 && idx < n_so_ && "SO index out of range");
    }

    void validate_so_indices(int mu, int nu, int kappa, int lambda) const noexcept {
        assert(mu >= 0 && mu < n_so_ && "SO index μ out of range");
        assert(nu >= 0 && nu < n_so_ && "SO index ν out of range");
        assert(kappa >= 0 && kappa < n_so_ && "SO index κ out of range");
        assert(lambda >= 0 && lambda < n_so_ && "SO index λ out of range");
    }
};

} // namespace detnqs
