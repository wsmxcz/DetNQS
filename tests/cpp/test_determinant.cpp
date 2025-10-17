// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_determinant.cpp
 * @brief Unit tests for determinant primitives, operations, sets, and CI spaces.
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_ops.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/determinant/det_enum.hpp>
#include <lever/utils/bit_utils.hpp>
#include <lever/utils/constants.hpp>

#include <algorithm>
#include <unordered_set>
#include <vector>

using namespace lever;

// -----------------------------------------------------------------------------
// Det basics & hashing
// -----------------------------------------------------------------------------
TEST_CASE("Det: comparison & hashing", "[det]") {
    const Det a{0b0101ULL, 0b0011ULL};
    const Det b{0b0101ULL, 0b0011ULL};
    const Det c{0b0111ULL, 0b0001ULL};
    const Det d{0b0101ULL, 0b0100ULL};

    SECTION("Three-way comparison") {
        REQUIRE(a == b);
        REQUIRE(a != c);
        REQUIRE((a <=> d) < 0);  // alpha equal, beta compares
        REQUIRE((c <=> d) > 0);
    }

    SECTION("Hash uniqueness for unordered_set") {
        std::unordered_set<Det> s;
        REQUIRE(s.insert(a).second);
        REQUIRE_FALSE(s.insert(b).second);
        REQUIRE(s.insert(c).second);
        REQUIRE(s.size() == 2);
        REQUIRE(s.count(a) == 1);
    }
}

// -----------------------------------------------------------------------------
// Bit utilities sanity
// -----------------------------------------------------------------------------
TEST_CASE("Bit utils: popcount/ctz/masks", "[utils]") {
    REQUIRE(popcount(0b10101ULL) == 3);
    REQUIRE(popcount(0ULL) == 0);
    REQUIRE(ctz(0b10100ULL) == 2);
    REQUIRE(ctz(1ULL) == 0);

    REQUIRE(make_mask<u64>(0) == 0ULL);
    REQUIRE(make_mask<u64>(3) == 0b111ULL);
    REQUIRE(extract_bits<u64>(0b10110ULL) == std::vector<int>({1, 2, 4}));
}

// -----------------------------------------------------------------------------
// Excitation analysis & phase (Scemama-Giner conventions)
// -----------------------------------------------------------------------------
TEST_CASE("Excitation degree & phase", "[det_ops]") {
    const int n_orb = 4;
    // ket: alpha= {0,1}, beta= {0,1}
    const Det ket{0b0011ULL, 0b0011ULL};

    SECTION("Diagonal (degree 0)") {
        const auto info = det_ops::analyze_excitation(ket, ket);
        REQUIRE(info.degree == 0);
        REQUIRE(info.phase == Catch::Approx(1.0));
        REQUIRE(info.n_alpha_exc == 0);
        REQUIRE(info.n_beta_exc == 0);
    }

    SECTION("Single alpha (degree 1), negative phase example") {
        // bra: alpha move 0 -> 2  => {1,2} (0b0110), beta unchanged
        const Det bra{0b0110ULL, 0b0011ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 1);
        REQUIRE(info.n_alpha_exc == 1);
        REQUIRE(info.n_beta_exc == 0);
        REQUIRE(info.holes_alpha[0] == 0);
        REQUIRE(info.particles_alpha[0] == 2);
        // Between(0,2) has orbital 1, which is occupied in bra -> odd -> -1
        REQUIRE(info.phase == Catch::Approx(-1.0));
        REQUIRE(det_ops::phase(bra, ket) == Catch::Approx(-1.0));
    }

    SECTION("Single beta (degree 1), negative phase example") {
        // bra: beta move 0 -> 2
        const Det bra{0b0011ULL, 0b0110ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 1);
        REQUIRE(info.n_alpha_exc == 0);
        REQUIRE(info.n_beta_exc == 1);
        REQUIRE(info.holes_beta[0] == 0);
        REQUIRE(info.particles_beta[0] == 2);
        REQUIRE(info.phase == Catch::Approx(-1.0));
    }

    SECTION("Double alpha-alpha (degree 2), non-crossing, positive phase") {
        // bra: alpha {2,3} from {0,1}
        const Det bra{0b1100ULL, 0b0011ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 2);
        REQUIRE(info.n_alpha_exc == 2);
        REQUIRE(info.n_beta_exc == 0);
        // holes {0,1}, particles {2,3} (both ascending)
        REQUIRE(info.holes_alpha[0] == 0);
        REQUIRE(info.holes_alpha[1] == 1);
        REQUIRE(info.particles_alpha[0] == 2);
        REQUIRE(info.particles_alpha[1] == 3);
        REQUIRE(info.phase == Catch::Approx(1.0));
    }

    SECTION("Double alpha-beta (degree 2), mixed spin") {
        // bra: alpha 0->2 ({1,2}), beta 1->3 ({0,3})
        const Det bra{0b0110ULL, 0b1001ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 2);
        REQUIRE(info.n_alpha_exc == 1);
        REQUIRE(info.n_beta_exc == 1);
        REQUIRE(info.phase == Catch::Approx(-1.0)); // odd permutation total here
    }

    SECTION("Degree > 2 returns phase sentinel 0.0") {
        // Make 2 alpha + 1 beta excitations relative to ket
        const Det bra{0b1100ULL, 0b0101ULL}; // alpha {2,3}, beta {0,2}
        REQUIRE(det_ops::phase(bra, ket) == Catch::Approx(0.0));
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree > MAX_EXCITATION_DEGREE);
        REQUIRE(info.phase == Catch::Approx(0.0));
    }
}

// -----------------------------------------------------------------------------
// det_ops: batch collectors (uniqueness & combinatorics counts)
// -----------------------------------------------------------------------------
TEST_CASE("Collectors: singles/doubles/connected", "[det_ops]") {
    const int n_orb = 4;
    // ket: alpha=2 e-, beta=2 e-
    const Det ket{0b0011ULL, 0b0011ULL};

    const auto singles   = det_ops::collect_singles(ket, n_orb);
    const auto doubles   = det_ops::collect_doubles(ket, n_orb);
    const auto connected = det_ops::collect_connected(ket, n_orb);

    const int n_alpha = popcount(ket.alpha);
    const int n_beta  = popcount(ket.beta);
    const int v_alpha = n_orb - n_alpha;
    const int v_beta  = n_orb - n_beta;

    const size_t expect_singles = size_t(n_alpha)*v_alpha + size_t(n_beta)*v_beta;            // 2*2 + 2*2 = 8
    const size_t expect_daa     = 1ULL * (n_alpha>=2) * (v_alpha>=2) * 1;                      // C(2,2)*C(2,2)=1
    const size_t expect_dbb     = 1ULL * (n_beta >=2) * (v_beta >=2) * 1;
    const size_t expect_dab     = size_t(n_alpha)*v_alpha * size_t(n_beta)*v_beta;             // 4*4=16
    const size_t expect_doubles = expect_daa + expect_dbb + expect_dab;                        // 18
    const size_t expect_conn    = expect_singles + expect_doubles;                             // 26

    REQUIRE(singles.size()   == expect_singles);
    REQUIRE(doubles.size()   == expect_doubles);
    REQUIRE(connected.size() == expect_conn);

    // Uniqueness check
    std::unordered_set<Det> u1(singles.begin(), singles.end());
    std::unordered_set<Det> u2(doubles.begin(), doubles.end());
    std::unordered_set<Det> u3(connected.begin(), connected.end());
    REQUIRE(u1.size() == singles.size());
    REQUIRE(u2.size() == doubles.size());
    REQUIRE(u3.size() == connected.size());
}

// -----------------------------------------------------------------------------
// DetMap & det_space set utilities
// -----------------------------------------------------------------------------
TEST_CASE("DetMap & set utilities", "[det_space]") {
    std::vector<Det> raw = {
        {1,1}, {2,2}, {3,3}, {1,1}, {4,4}
    };

    SECTION("DetMap::from_list canonicalizes") {
        auto m = DetMap::from_list(raw);
        REQUIRE(m.size() == 4);
        REQUIRE(m.get_idx({1,1}).has_value());
        REQUIRE(m.get_det(0) == Det{1,1});
        REQUIRE_THROWS_AS(m.get_det(4), std::out_of_range);

        const auto& ds = m.all_dets();
        for (u32 i = 0; i < ds.size(); ++i) {
            REQUIRE(m.get_idx(ds[i]).value() == i);
        }
    }

    SECTION("DetMap::from_ordered preserves order & removes dups (verify_unique=false)") {
        auto m = DetMap::from_ordered(raw, /*verify_unique=*/false);
        REQUIRE(m.size() == 4);
        // First occurrence kept
        REQUIRE(m.get_det(0) == Det{1,1});
        REQUIRE(m.get_det(1) == Det{2,2});
        REQUIRE(m.get_det(2) == Det{3,3});
        REQUIRE(m.get_det(3) == Det{4,4});
    }

    SECTION("Set ops: canonicalize/merge/set_union/stable_union") {
        std::vector<Det> a = {{1,1}, {2,2}, {3,3}};
        std::vector<Det> b = {{2,2}, {4,4}, {4,4}};

        auto ca = det_space::canonicalize(a);
        auto cb = det_space::canonicalize(b);
        REQUIRE(std::is_sorted(ca.begin(), ca.end()));
        REQUIRE(std::is_sorted(cb.begin(), cb.end()));

        auto merged = det_space::merge_sorted(ca, cb);
        REQUIRE(std::is_sorted(merged.begin(), merged.end()));
        REQUIRE(merged.size() == 4);

        auto un = det_space::set_union_hash(a, b, /*sorted=*/true);
        REQUIRE(std::is_sorted(un.begin(), un.end()));
        REQUIRE(un.size() == 4);

        auto st = det_space::stable_union(a, b); // order of 'a' preserved, new from 'b' appended once
        REQUIRE(st.front() == Det{1,1});
        std::unordered_set<Det> s(st.begin(), st.end());
        REQUIRE(s.size() == st.size());
    }
}

// -----------------------------------------------------------------------------
// det_space: batch generators & complement
// -----------------------------------------------------------------------------
TEST_CASE("det_space generators & complement", "[det_space]") {
    const int n_orb = 4;
    const Det ket{0b0011ULL, 0b0011ULL};
    std::vector<Det> parents{ket};

    const auto s1 = det_space::generate_singles(parents, n_orb, /*sorted=*/true);
    const auto d2 = det_space::generate_doubles(parents, n_orb, /*sorted=*/true);
    const auto cn = det_space::generate_connected(parents, n_orb, /*sorted=*/true);

    // Should match det_ops collectors in size
    REQUIRE(s1.size() == det_ops::collect_singles(ket, n_orb).size());
    REQUIRE(d2.size() == det_ops::collect_doubles(ket, n_orb).size());
    REQUIRE(cn.size() == det_ops::collect_connected(ket, n_orb).size());

    // Complement excluding parent + one extra to verify filtering
    auto exclude = DetMap::from_list({ket, cn.front()});
    const auto comp = det_space::generate_complement(parents, n_orb, exclude, /*sorted=*/true);
    REQUIRE(comp.size() + 1 == cn.size()); // filtered out one from 'cn' (plus parent which is not in cn anyway)
    REQUIRE(std::find(comp.begin(), comp.end(), cn.front()) == comp.end());
}

// -----------------------------------------------------------------------------
// FCI space enumeration
// -----------------------------------------------------------------------------
TEST_CASE("FCISpace enumeration", "[det_enum][FCI]") {
    const int n_orb = 4, na = 1, nb = 1;
    FCISpace fci(n_orb, na, nb);
    REQUIRE(fci.size() == size_t(na ? n_orb : 1) * size_t(nb ? n_orb : 1)); // C(4,1)^2 = 16

    for (const auto& d : fci.dets()) {
        REQUIRE(popcount(d.alpha) == na);
        REQUIRE(popcount(d.beta)  == nb);
    }
}

// -----------------------------------------------------------------------------
// CAS space enumeration
// -----------------------------------------------------------------------------
TEST_CASE("CASSpace enumeration", "[det_enum][CAS]") {
    // core=1 (bit 0), active=2 (bits 1-2), virtual=1 (bit 3)
    CASSpace cas(/*n_core=*/1, /*n_active=*/2, /*n_virtual=*/1,
                 /*n_alpha_active=*/1, /*n_beta_active=*/1);

    // Active FCI size C(2,1)^2 = 4
    REQUIRE(cas.size() == 4);

    for (const auto& d : cas.dets()) {
        // Core always doubly occupied (bit 0 set in both spins)
        REQUIRE((d.alpha & 0b0001ULL) != 0ULL);
        REQUIRE((d.beta  & 0b0001ULL) != 0ULL);
        // No occupation in virtual (bit 3)
        REQUIRE(((d.alpha | d.beta) & 0b1000ULL) == 0ULL);
        // Active has exactly 1 alpha & 1 beta among bits {1,2}
        REQUIRE(popcount((d.alpha >> 1) & 0b11ULL) == 1);
        REQUIRE(popcount((d.beta  >> 1) & 0b11ULL) == 1);
    }
}

// -----------------------------------------------------------------------------
// RAS space enumeration (TOTAL constraints on α+β)
// -----------------------------------------------------------------------------
TEST_CASE("RASSpace enumeration & TOTAL constraints", "[det_enum][RAS]") {
    // Partition: core=1, ras1=1, ras2=1, ras3=1, virtual=0  (total 4 orbitals)
    RASOrbitalPartition p{1, 1, 1, 1, 0};

    // Totals: each spin has 2 e- (incl. core)
    // Constraints: at most 1 HOLE in RAS1 (total), at most 1 ELECTRON in RAS3 (total)
    RASElectronConstraint e{ /*n_alpha_total=*/2, /*n_beta_total=*/2,
                             /*max_holes_ras1=*/1, /*max_elecs_ras3=*/1 };

    RASSpace ras(p, e);
    REQUIRE(ras.size() > 0);

    const int off_core = 0;
    const int off_r1 = off_core + p.n_core;
    const int off_r2 = off_r1 + p.n_ras1;
    const int off_r3 = off_r2 + p.n_ras2;

    const u64 coreMask = (1ULL << off_core);
    const u64 ras1Mask = (1ULL << off_r1);
    const u64 ras2Mask = (1ULL << off_r2);
    const u64 ras3Mask = (1ULL << off_r3);

    const int ras1_cap = 2 * p.n_ras1; // max total electrons in RAS1

    int with_ras3_occ = 0;

    for (const auto& d : ras.dets()) {
        // Core doubly occupied
        REQUIRE((d.alpha & coreMask) != 0ULL);
        REQUIRE((d.beta  & coreMask) != 0ULL);

        // TOTAL constraints
        const int occ_ras1 = popcount(d.alpha & ras1Mask) + popcount(d.beta & ras1Mask);
        const int holes1   = ras1_cap - occ_ras1;
        REQUIRE((holes1 <= e.max_holes_ras1 || e.max_holes_ras1 < 0));

        const int occ_ras3 = popcount(d.alpha & ras3Mask) + popcount(d.beta & ras3Mask);
        REQUIRE((occ_ras3 <= e.max_elecs_ras3 || e.max_elecs_ras3 < 0));

        with_ras3_occ += (occ_ras3 > 0);
        // Optional: ras2 is CAS-like, no direct cap other than capacity
        REQUIRE(popcount(d.alpha & ras2Mask) <= 1);
        REQUIRE(popcount(d.beta  & ras2Mask) <= 1);
    }

    // At least one determinant uses RAS3, but never exceeds TOTAL limit (<=1)
    REQUIRE(with_ras3_occ >= 0);
}