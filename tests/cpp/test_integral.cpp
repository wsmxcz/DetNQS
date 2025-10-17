// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_integral.cpp
 * @brief Unit tests for
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <lever/integral/integral_mo.hpp>
#include <lever/integral/integral_so.hpp>
#include <memory>
#include <vector>
#include <iostream>
#include <array>

using namespace lever;
using Catch::Approx;

/**
 * @class IntegralTestFixture
 * @brief Test fixture providing H2/6-31G (R=2.0A) integral data for unit tests.
 * @details Loads FCIDUMP file and initializes both MO and SO integral objects.
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
            std::cerr << "FATAL ERROR: " << e.what() << std::endl;
            std::cerr << "Please ensure '" << fcidump_path << "' is in the execution directory" << std::endl;
            throw;
        }
        
        so_integrals = std::make_unique<IntegralSO>(*mo_integrals);
    }
};

/**
 * @test IntegralMO metadata validation
 * @brief Verifies FCIDUMP header information (orbital count, electron count, etc.)
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralMO: FCIDUMP metadata", "[integral]") {
    REQUIRE(mo_integrals->n_orbs == 4);
    REQUIRE(mo_integrals->n_elecs == 2);
    REQUIRE(mo_integrals->spin_mult == 1);  // MS2=0 -> S=0 -> 2S+1=1 (singlet)
    REQUIRE(mo_integrals->e_nuc == Approx(0.26458860546));
}

/**
 * @test One-electron integral verification
 * @brief Tests diagonal elements and off-diagonal symmetry h_pq = h_qp
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralMO: One-electron integrals", "[integral]") {
    SECTION("Diagonal elements") {
        REQUIRE(mo_integrals->get_h1e(0, 0) == Approx(-0.801833061063));
        REQUIRE(mo_integrals->get_h1e(1, 1) == Approx(-0.680884820395));
        REQUIRE(mo_integrals->get_h1e(2, 2) == Approx(0.202998161132));
        REQUIRE(mo_integrals->get_h1e(3, 3) == Approx(0.265010250063));
    }
    SECTION("Off-diagonal symmetry h_pq = h_qp") {
        REQUIRE(mo_integrals->get_h1e(2, 0) == Approx(-0.074391365158));
        REQUIRE(mo_integrals->get_h1e(0, 2) == Approx(-0.074391365158));
        REQUIRE(mo_integrals->get_h1e(3, 1) == Approx(0.120229115266));
        REQUIRE(mo_integrals->get_h1e(1, 3) == Approx(0.120229115266));
    }
}

/**
 * @test Two-electron integral verification
 * @brief Tests key integral values and 8-fold permutational symmetry
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralMO: Two-electron integrals", "[integral]") {
    SECTION("Key values") {
        REQUIRE(mo_integrals->get_h2e(0, 0, 0, 0) == Approx(0.422806268972));
        REQUIRE(mo_integrals->get_h2e(1, 1, 0, 0) == Approx(0.410534684646));
        REQUIRE(mo_integrals->get_h2e(1, 0, 1, 0) == Approx(0.163762656748));
    }
    SECTION("8-fold permutational symmetry") {
        const double reference_val = mo_integrals->get_h2e(1, 0, 3, 2);
        REQUIRE(reference_val == Approx(-0.180315206010));
        
        const std::array<std::array<int, 4>, 8> permutations = {{
            {1, 0, 3, 2}, {0, 1, 2, 3}, {3, 2, 1, 0}, {2, 3, 0, 1},
            {3, 2, 0, 1}, {2, 3, 1, 0}, {0, 1, 3, 2}, {1, 0, 2, 3}
        }};
        
        for (const auto& [p, q, r, s] : permutations) {
            REQUIRE(mo_integrals->get_h2e(p, q, r, s) == Approx(reference_val));
        }
    }
}

/**
 * @test Spin orbital indexing utilities
 * @brief Verifies SO-MO mapping functions and spin detection
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralSO: Indexing utilities", "[integral]") {
    SECTION("SO from MO and spin") {
        REQUIRE(so_from_mo(0, 0) == 0);  // 0α
        REQUIRE(so_from_mo(0, 1) == 1);  // 0β
        REQUIRE(so_from_mo(3, 0) == 6);  // 3α
        REQUIRE(so_from_mo(3, 1) == 7);  // 3β
    }
    SECTION("MO and spin from SO") {
        REQUIRE(mo_from_so(0) == 0);
        REQUIRE(spin_from_so(0) == 0);  // alpha
        REQUIRE(mo_from_so(7) == 3);
        REQUIRE(spin_from_so(7) == 1);  // beta
    }
    SECTION("Spin comparison") {
        REQUIRE(have_same_spin(0, 2));       // 0α, 1α
        REQUIRE_FALSE(have_same_spin(0, 1)); // 0α, 0β
        REQUIRE(have_same_spin(3, 5));       // 1β, 2β
    }
}

/**
 * @test Spin orbital one-electron integrals
 * @brief Tests spin-conserving and spin-flipping matrix elements
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralSO: One-electron integrals", "[integral]") {
    SECTION("Spin-conserving elements") {
        const double h_01 = mo_integrals->get_h1e(0, 1);
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0, 0), so_from_mo(1, 0)) == Approx(h_01));
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0, 1), so_from_mo(1, 1)) == Approx(h_01));
        
        const double h_02 = mo_integrals->get_h1e(0, 2);
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0, 0), so_from_mo(2, 0)) == Approx(h_02));
    }
    SECTION("Spin-flipping elements are zero") {
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(0, 0), so_from_mo(0, 1)) == 0.0);
        REQUIRE(so_integrals->get_h1e_so(so_from_mo(1, 0), so_from_mo(2, 1)) == 0.0);
    }
}

/**
 * @test Spin orbital two-electron integrals
 * @brief Tests physicist's, chemist's, and antisymmetrized notations
 */
TEST_CASE_METHOD(IntegralTestFixture, "IntegralSO: Two-electron integrals", "[integral]") {
    constexpr int p = 0, q = 1, r = 2, s = 3;
    const auto [pa, qa, ra, sa] = std::array{so_from_mo(p, 0), so_from_mo(q, 0), 
                                            so_from_mo(r, 0), so_from_mo(s, 0)};
    const auto [pb, qb, rb, sb] = std::array{so_from_mo(p, 1), so_from_mo(q, 1), 
                                            so_from_mo(r, 1), so_from_mo(s, 1)};
    
    SECTION("Physicist's notation <μκ|νλ>") {
        const double expected = mo_integrals->get_h2e(p, r, q, s);  // [pr|qs]
        REQUIRE(so_integrals->get_h2e_phys(pa, qa, ra, sa) == Approx(expected));
        
        const double expected_0011 = mo_integrals->get_h2e(0, 0, 1, 1);
        const double actual_0101 = so_integrals->get_h2e_phys(so_from_mo(0,0), so_from_mo(1,0), 
                                                              so_from_mo(0,0), so_from_mo(1,0));
        REQUIRE(actual_0101 == Approx(expected_0011));
        
        // Spin non-conserving cases are zero
        REQUIRE(so_integrals->get_h2e_phys(pa, qb, ra, sa) == 0.0);
        REQUIRE(so_integrals->get_h2e_phys(pa, qa, rb, sa) == 0.0);
    }
    
    SECTION("Chemist's notation [μν|κλ]") {
        const double expected = mo_integrals->get_h2e(p, r, q, s);  // [pr|qs]
        REQUIRE(so_integrals->get_h2e_chem(pa, ra, qa, sa) == Approx(expected));
        
        const double expected_0011 = mo_integrals->get_h2e(0, 0, 1, 1);
        const double actual_0011 = so_integrals->get_h2e_chem(so_from_mo(0,0), so_from_mo(0,0), 
                                                              so_from_mo(1,0), so_from_mo(1,0));
        REQUIRE(actual_0011 == Approx(expected_0011));
    }
    
    SECTION("Antisymmetrized notation <μκ||νλ>") {
        // Same spin: <pa qa || ra sa> = <pa qa | ra sa> - <pa qa | sa ra>
        const double coulomb_term = mo_integrals->get_h2e(p, r, q, s);  // [pr|qs]
        const double exchange_term = mo_integrals->get_h2e(p, s, q, r);  // [ps|qr]
        REQUIRE(so_integrals->get_h2e_anti(pa, qa, ra, sa) == Approx(coulomb_term - exchange_term));
        
        // Different spin (Coulomb only): <pa qb || ra sb> = <pa qb | ra sb>
        const double expected_coulomb = mo_integrals->get_h2e(p, r, q, s);
        REQUIRE(so_integrals->get_h2e_anti(pa, qb, ra, sb) == Approx(expected_coulomb));
        
        // Diagonal case: <0α 1α || 0α 1α> = J - K
        const double j_integral = mo_integrals->get_h2e(0, 0, 1, 1);  // Coulomb
        const double k_integral = mo_integrals->get_h2e(0, 1, 0, 1);  // Exchange
        const auto result = so_integrals->get_h2e_anti(so_from_mo(0,0), so_from_mo(1,0), 
                                                       so_from_mo(0,0), so_from_mo(1,0));
        REQUIRE(result == Approx(j_integral - k_integral));
    }
}
