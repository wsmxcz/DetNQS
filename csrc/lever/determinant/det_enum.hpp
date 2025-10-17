// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file det_enum.hpp
 * @brief Generators for complete determinant spaces (FCI, CAS, RAS).
 *
 * Notes:
 * - Bitstring convention: bit i (LSB=0) corresponds to spatial orbital i.
 * - All generators assume u64 bitmasks (<= 64 spatial orbitals).
 * - Outputs are not sorted unless the caller explicitly sorts them.
 */

#pragma once

#include <lever/determinant/det.hpp>
#include <vector>
#include <cstddef>

namespace lever {

/**
 * @class FCISpace
 * @brief Generates the complete Full Configuration Interaction (FCI) space.
 *
 * The FCI space consists of all Slater determinants for a given number of
 * spatial orbitals and alpha/beta electrons. The output order is unspecified.
 */
class FCISpace {
public:
    /**
     * @brief Generating constructor for FCI.
     * @param n_orb    Number of spatial orbitals (1..64).
     * @param n_alpha  Number of alpha electrons   (0..n_orb).
     * @param n_beta   Number of beta  electrons   (0..n_orb).
     * @throws std::invalid_argument on invalid parameters.
     */
    explicit FCISpace(int n_orb, int n_alpha, int n_beta);

    /// Immutable view of all determinants.
    [[nodiscard]] const std::vector<Det>& dets() const noexcept;

    /// Total number of determinants.
    [[nodiscard]] size_t size() const noexcept;

    /// Iteration helpers.
    [[nodiscard]] auto begin() const noexcept { return dets_.begin(); }
    [[nodiscard]] auto end()   const noexcept { return dets_.end();   }

protected:
    /// Non-generating base constructor for derived classes (CAS/RAS).
    FCISpace() = default;

    std::vector<Det> dets_; ///< Storage for generated determinants.
};

/**
 * @class CASSpace
 * @brief Complete Active Space (CAS): core orbitals are always doubly occupied,
 *        active orbitals form a full CI, virtual orbitals remain empty.
 */
class CASSpace : public FCISpace {
public:
    /**
     * @brief Construct a CAS space.
     * @param n_core_orb      # of core orbitals (always doubly occupied).
     * @param n_active_orb    # of active orbitals (FCI performed here).
     * @param n_virtual_orb   # of virtual orbitals (always empty).
     * @param n_alpha_active  # of alpha electrons in the active space (0..n_active_orb).
     * @param n_beta_active   # of beta  electrons in the active space (0..n_active_orb).
     * @throws std::invalid_argument on inconsistent parameters.
     */
    CASSpace(int n_core_orb, int n_active_orb, int n_virtual_orb,
             int n_alpha_active, int n_beta_active);
};

/**
 * @struct RASOrbitalPartition
 * @brief Contiguous partitioning for RAS: [core | ras1 | ras2 | ras3 | virtual].
 */
struct RASOrbitalPartition {
    int n_core   = 0;
    int n_ras1   = 0;
    int n_ras2   = 0;
    int n_ras3   = 0;
    int n_virtual= 0;
};

/**
 * @struct RASElectronConstraint
 * @brief Total electron constraints for RAS (applied to total spin population).
 */
struct RASElectronConstraint {
    int n_alpha_total = 0;   ///< total alpha electrons (includes core)
    int n_beta_total  = 0;   ///< total beta  electrons (includes core)
    int max_holes_ras1 = -1; ///< max total holes in RAS1 (across alpha+beta), -1 = no limit
    int max_elecs_ras3 = -1; ///< max total electrons in RAS3 (alpha+beta),    -1 = no limit
};

/**
 * @class RASSpace
 * @brief Restricted Active Space (RAS):
 *        - RAS1: at most K holes (w.r.t. double occupancy),
 *        - RAS2: CAS-like (unrestricted within RAS2),
 *        - RAS3: at most L electrons,
 *        constraints are applied to the total (alpha+beta) counts.
 */
class RASSpace : public FCISpace {
public:
    /**
     * @brief Construct a RAS space.
     * @param orb_part  Orbital partition (contiguous blocks).
     * @param elec_con  Total electron counts and RAS1/RAS3 constraints.
     * @throws std::invalid_argument on inconsistent parameters.
     */
    RASSpace(const RASOrbitalPartition& orb_part,
             const RASElectronConstraint& elec_con);
};

} // namespace lever