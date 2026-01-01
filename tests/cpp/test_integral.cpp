// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_integral.cpp
 * @brief Unit tests for MO and SO integral transformations.
 *
 * Tests integral loading from FCIDUMP, MO↔SO transformations,
 * and various integral notations (physicist's, chemist's, antisymmetrized).
 *
 * Test system: H₂/6-31G (R=2.0Å, 4 orbitals, 2 electrons, singlet)
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <detnqs/integral/integral_mo.hpp>
#include <detnqs/integral/integral_so.hpp>

#include <array>
#include <iostream>
#include <memory>
#include <vector>

using namespace detnqs;
using Catch::Approx;

/**
 * Test fixture for H₂/6-31G integral data.
 * Loads FCIDUMP and initializes MO/SO integral objects.
 */
struct IntegralTestFixture {
    std::shared_ptr<IntegralMO> mo_integrals;
    std::unique_ptr<IntegralSO> so_integrals;

    IntegralTestFixture() {
        constexpr auto fcidump_path = "H2_631g_2.00.FCIDUMP";
        constexpr int n_orbitals = 4;
        
        mo_integrals = std::make_shared<IntegralMO>(n_orbitals);
        
        try {
            mo_integrals->load_from_fcidump(fcidump_path);
        } catch (const std::runtime_error& e) {
            std::cerr << "FATAL: " << e.what() << '\n'
                      << "Ensure '" << fcidump_path << "' is in working directory\n";
            throw;
        }
        
        so_integrals = std::make_unique<IntegralSO>(*mo_integrals);
    }
};

// ============================================================================
// MO Integral Tests
// ============================================================================

/**
 * Verify FCIDUMP metadata (N_orb, N_elec, nuclear repulsion, etc.)
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralMO: Metadata validation", "[integral]") {
    REQUIRE(mo_integrals->n_orbs == 4);
    REQUIRE(mo_integrals->n_elecs == 2);
    REQUIRE(mo_integrals->spin_mult == 1);  // Singlet (2S+1=1)
    REQUIRE(mo_integrals->e_nuc == Approx(0.26458860546));
}

/**
 * Test one-electron integrals h_pq.
 * Verifies diagonal elements and hermiticity h_pq = h_qp.
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralMO: One-electron integrals h_pq", "[integral]") {
    SECTION("Diagonal elements") {
        REQUIRE(mo_integrals->get_h1e(0, 0) == Approx(-0.801833061063));
        REQUIRE(mo_integrals->get_h1e(1, 1) == Approx(-0.680884820395));
        REQUIRE(mo_integrals->get_h1e(2, 2) == Approx(0.202998161132));
        REQUIRE(mo_integrals->get_h1e(3, 3) == Approx(0.265010250063));
    }
    
    SECTION("Hermiticity: h_pq = h_qp") {
        REQUIRE(mo_integrals->get_h1e(2, 0) == Approx(-0.074391365158));
        REQUIRE(mo_integrals->get_h1e(0, 2) == Approx(-0.074391365158));
        REQUIRE(mo_integrals->get_h1e(3, 1) == Approx(0.120229115266));
        REQUIRE(mo_integrals->get_h1e(1, 3) == Approx(0.120229115266));
    }
}

/**
 * Test two-electron integrals [pq|rs] in chemist's notation.
 * Verifies key values and 8-fold permutational symmetry.
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralMO: Two-electron integrals [pq|rs]", "[integral]") {
    SECTION("Representative values") {
        REQUIRE(mo_integrals->get_h2e(0, 0, 0, 0) == Approx(0.422806268972));
        REQUIRE(mo_integrals->get_h2e(1, 1, 0, 0) == Approx(0.410534684646));
        REQUIRE(mo_integrals->get_h2e(1, 0, 1, 0) == Approx(0.163762656748));
    }
    
    SECTION("8-fold permutational symmetry") {
        const double ref_val = mo_integrals->get_h2e(1, 0, 3, 2);
        REQUIRE(ref_val == Approx(-0.180315206010));
        
        // [pq|rs] = [qp|sr] = [rs|pq] = [sr|qp] = [qp|rs] = [pq|sr] = [sr|pq] = [rs|qp]
        constexpr std::array<std::array<int, 4>, 8> perms = {{
            {1,0,3,2}, {0,1,2,3}, {3,2,1,0}, {2,3,0,1},
            {3,2,0,1}, {2,3,1,0}, {0,1,3,2}, {1,0,2,3}
        }};
        
        for (const auto& [p, q, r, s] : perms) {
            REQUIRE(mo_integrals->get_h2e(p, q, r, s) == Approx(ref_val));
        }
    }
}

// ============================================================================
// SO Indexing Tests
// ============================================================================

/**
 * Test MO↔SO index conversion utilities.
 * Convention: SO_idx = 2·MO_idx + spin, spin ∈ {0(α), 1(β)}
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralSO: Index conversion MO↔SO", "[integral]") {
    SECTION("MO+spin → SO") {
        REQUIRE(so_from_mo(0, 0) == 0);  // 0α
        REQUIRE(so_from_mo(0, 1) == 1);  // 0β
        REQUIRE(so_from_mo(3, 0) == 6);  // 3α
        REQUIRE(so_from_mo(3, 1) == 7);  // 3β
    }
    
    SECTION("SO → MO+spin") {
        REQUIRE(mo_from_so(0) == 0);
        REQUIRE(spin_from_so(0) == 0);  // α
        REQUIRE(mo_from_so(7) == 3);
        REQUIRE(spin_from_so(7) == 1);  // β
    }
    
    SECTION("Spin comparison") {
        REQUIRE(have_same_spin(0, 2));       // Both α
        REQUIRE_FALSE(have_same_spin(0, 1)); // α vs β
        REQUIRE(have_same_spin(3, 5));       // Both β
    }
}

// ============================================================================
// SO Integral Tests
// ============================================================================

/**
 * Test SO one-electron integrals h^SO_μν.
 * Should match MO values for same-spin pairs, vanish otherwise.
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralSO: One-electron integrals h^SO_μν", "[integral]") {
    SECTION("Same-spin elements match MO values") {
        const double h_01_mo = mo_integrals->get_h1e(0, 1);
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0,0), so_from_mo(1,0)) == Approx(h_01_mo));
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0,1), so_from_mo(1,1)) == Approx(h_01_mo));
        
        const double h_02_mo = mo_integrals->get_h1e(0, 2);
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0,0), so_from_mo(2,0)) == Approx(h_02_mo));
    }
    
    SECTION("Different-spin elements vanish") {
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0,0), so_from_mo(0,1)) == 0.0);
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(1,0), so_from_mo(2,1)) == 0.0);
    }
}

/**
 * Test SO two-electron integrals in multiple notations.
 *
 * Physicist's:     <μκ|νλ> = [μν|κλ] (MO chemist's)
 * Chemist's:       [μν|κλ]
 * Antisymmetrized: <μκ||νλ> = <μκ|νλ> - <μκ|λν>
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralSO: Two-electron integrals", "[integral]") {
    constexpr std::array mo_idx = {0, 1, 2, 3};
    const auto [pa, qa, ra, sa] = std::array{so_from_mo(mo_idx[0], 0),
                                            so_from_mo(mo_idx[1], 0),
                                            so_from_mo(mo_idx[2], 0),
                                            so_from_mo(mo_idx[3], 0)};
    const auto [pb, qb, rb, sb] = std::array{so_from_mo(mo_idx[0], 1),
                                            so_from_mo(mo_idx[1], 1),
                                            so_from_mo(mo_idx[2], 1),
                                            so_from_mo(mo_idx[3], 1)};
    
    SECTION("Physicist's notation: <μκ|νλ>") {
        // <pa qa | ra sa> = [pr|qs] in MO
        const double expected = mo_integrals->get_h2e(0, 2, 1, 3);
        REQUIRE(so_integrals->get_h2e_phys(pa, qa, ra, sa) == Approx(expected));
        
        // Diagonal case: <0α 1α | 0α 1α> = [00|11]
        const double expected_diag = mo_integrals->get_h2e(0, 0, 1, 1);
        REQUIRE(so_integrals->get_h2e_phys(
            so_from_mo(0,0), so_from_mo(1,0),
            so_from_mo(0,0), so_from_mo(1,0)
        ) == Approx(expected_diag));
        
        // Spin non-conservation → zero
        REQUIRE(so_integrals->get_h2e_phys(pa, qb, ra, sa) == 0.0);
        REQUIRE(so_integrals->get_h2e_phys(pa, qa, rb, sa) == 0.0);
    }
    
    SECTION("Chemist's notation: [μν|κλ]") {
        // [pa ra | qa sa] = [pr|qs] in MO
        const double expected = mo_integrals->get_h2e(0, 2, 1, 3);
        REQUIRE(so_integrals->get_h2e_chem(pa, ra, qa, sa) == Approx(expected));
        
        const double expected_0011 = mo_integrals->get_h2e(0, 0, 1, 1);
        REQUIRE(so_integrals->get_h2e_chem(
            so_from_mo(0,0), so_from_mo(0,0),
            so_from_mo(1,0), so_from_mo(1,0)
        ) == Approx(expected_0011));
    }
    
    SECTION("Antisymmetrized notation: <μκ||νλ>") {
        // Same-spin: <pa qa || ra sa> = <pa qa | ra sa> - <pa qa | sa ra>
        const double j_term = mo_integrals->get_h2e(0, 2, 1, 3);  // Coulomb
        const double k_term = mo_integrals->get_h2e(0, 3, 1, 2);  // Exchange
        REQUIRE(so_integrals->get_h2e_anti(pa, qa, ra, sa) == Approx(j_term - k_term));
        
        // Different-spin: <pa qb || ra sb> = <pa qb | ra sb> (Coulomb only)
        const double coulomb_only = mo_integrals->get_h2e(0, 2, 1, 3);
        REQUIRE(so_integrals->get_h2e_anti(pa, qb, ra, sb) == Approx(coulomb_only));
        
        // Diagonal case: <0α 1α || 0α 1α> = J - K
        const double j_int = mo_integrals->get_h2e(0, 0, 1, 1);
        const double k_int = mo_integrals->get_h2e(0, 1, 0, 1);
        REQUIRE(so_integrals->get_h2e_anti(
            so_from_mo(0,0), so_from_mo(1,0),
            so_from_mo(0,0), so_from_mo(1,0)
        ) == Approx(j_int - k_int));
    }
}
