// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_enum.hpp
 * @brief Determinant space generators for FCI, CAS, and RAS.
 *
 * Generates complete configuration spaces via combinatorial enumeration.
 * Bitstring convention: bit i (LSB=0) → spatial orbital i (u64, ≤64 orbitals).
 * Output order is unspecified unless explicitly sorted by caller.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <detnqs/determinant/det.hpp>
#include <vector>
#include <cstddef>

namespace detnqs {

/**
 * @class FCISpace
 * @brief Full Configuration Interaction space generator.
 *
 * Enumerates all (n_orb choose n_α) × (n_orb choose n_β) determinants
 * for a given orbital count and electron numbers.
 */
class FCISpace {
public:
    /**
     * @brief Generate FCI space for given system.
     * @param n_orb    Spatial orbitals (1..64)
     * @param n_alpha  α electrons (0..n_orb)
     * @param n_beta   β electrons (0..n_orb)
     * @throws std::invalid_argument Invalid parameters
     */
    explicit FCISpace(int n_orb, int n_alpha, int n_beta);

    /// Immutable determinant view
    [[nodiscard]] const std::vector<Det>& dets() const noexcept;

    /// Total determinant count
    [[nodiscard]] size_t size() const noexcept;

    /// Iteration support
    [[nodiscard]] auto begin() const noexcept { return dets_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return dets_.end();   }

protected:
    /// Non-generating base constructor for CAS/RAS
    FCISpace() = default;

    std::vector<Det> dets_; ///< Generated determinants
};

/**
 * @class CASSpace
 * @brief Complete Active Space generator.
 *
 * Partitions orbitals into core (doubly occupied) | active (FCI) | virtual (empty).
 * Generates all configurations within the active space only.
 */
class CASSpace : public FCISpace {
public:
    /**
     * @brief Generate CAS(n_active, n_elec_active) space.
     * @param n_core_orb      Core orbitals (doubly occupied)
     * @param n_active_orb    Active orbitals (FCI region)
     * @param n_virtual_orb   Virtual orbitals (empty)
     * @param n_alpha_active  Active α electrons (0..n_active_orb)
     * @param n_beta_active   Active β electrons (0..n_active_orb)
     * @throws std::invalid_argument Invalid parameters
     */
    CASSpace(int n_core_orb, int n_active_orb, int n_virtual_orb,
             int n_alpha_active, int n_beta_active);
};

/**
 * @struct RASOrbitalPartition
 * @brief Contiguous orbital blocks: core | RAS1 | RAS2 | RAS3 | virtual.
 */
struct RASOrbitalPartition {
    int n_core    = 0;
    int n_ras1    = 0;
    int n_ras2    = 0;
    int n_ras3    = 0;
    int n_virtual = 0;
};

/**
 * @struct RASElectronConstraint
 * @brief Total electron constraints for RAS enumeration.
 */
struct RASElectronConstraint {
    int n_alpha_total  = 0;   ///< Total α electrons (includes core)
    int n_beta_total   = 0;   ///< Total β electrons (includes core)
    int max_holes_ras1 = -1;  ///< Max total holes in RAS1 (α+β), -1=unlimited
    int max_elecs_ras3 = -1;  ///< Max total electrons in RAS3 (α+β), -1=unlimited
};

/**
 * @class RASSpace
 * @brief Restricted Active Space generator.
 *
 * Tri-partition active space with hole/particle constraints:
 *   RAS1: ≤K holes w.r.t. double occupancy
 *   RAS2: Unrestricted (CAS-like)
 *   RAS3: ≤L electrons
 * Constraints apply to total (α+β) occupation.
 *
 * Algorithm: Constrained combinatorial enumeration over RAS1/RAS2/RAS3
 * orbital blocks, filtering by total hole/electron counts.
 */
class RASSpace : public FCISpace {
public:
    /**
     * @brief Generate RAS space with occupation constraints.
     * @param orb_part  Orbital partition (contiguous blocks)
     * @param elec_con  Electron counts and RAS1/RAS3 limits
     * @throws std::invalid_argument Invalid parameters
     */
    RASSpace(const RASOrbitalPartition& orb_part,
             const RASElectronConstraint& elec_con);
};

} // namespace detnqs
