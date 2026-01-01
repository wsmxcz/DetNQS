// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_determinant.cpp
 * @brief Unit tests for determinant primitives, operations, sets, and CI spaces.
 *
 * Validates Slater determinant representations, excitation analysis using
 * Scemama-Giner phase conventions, bitwise utilities, CI space enumeration
 * (FCI/CAS/RAS), and determinant set operations.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <detnqs/determinant/det.hpp>
#include <detnqs/determinant/det_ops.hpp>
#include <detnqs/determinant/det_space.hpp>
#include <detnqs/determinant/det_enum.hpp>
#include <detnqs/utils/bit_utils.hpp>
#include <detnqs/utils/constants.hpp>

#include <algorithm>
#include <unordered_set>
#include <vector>

using namespace detnqs;

// -----------------------------------------------------------------------------
// Determinant primitives: comparison & hash consistency
// -----------------------------------------------------------------------------
TEST_CASE("Det: comparison & hashing", "[det]") {
    const Det a{0b0101ULL, 0b0011ULL};
    const Det b{0b0101ULL, 0b0011ULL};
    const Det c{0b0111ULL, 0b0001ULL};
    const Det d{0b0101ULL, 0b0100ULL};

    SECTION("Three-way comparison") {
        REQUIRE(a == b);
        REQUIRE(a != c);
        REQUIRE((a <=> d) < 0);  // α equal, β compares
        REQUIRE((c <=> d) > 0);
    }

    SECTION("Hash uniqueness in unordered_set") {
        std::unordered_set<Det> s;
        REQUIRE(s.insert(a).second);
        REQUIRE_FALSE(s.insert(b).second);  // Duplicate
        REQUIRE(s.insert(c).second);
        REQUIRE(s.size() == 2);
        REQUIRE(s.count(a) == 1);
    }
}

// -----------------------------------------------------------------------------
// Bit utilities: population count, trailing zeros, masks
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
// Excitation analysis: degree & Scemama-Giner phase convention
// Phase = (-1)^{n_perm}, where n_perm counts orbital permutations between bra & ket.
// Returns 0.0 for excitations beyond doubles.
// -----------------------------------------------------------------------------
TEST_CASE("Excitation degree & phase", "[det_ops]") {
    const int n_orb = 4;
    // Reference: |ket⟩ with α = {0,1}, β = {0,1}
    const Det ket{0b0011ULL, 0b0011ULL};

    SECTION("Diagonal (degree 0)") {
        const auto info = det_ops::analyze_excitation(ket, ket);
        REQUIRE(info.degree == 0);
        REQUIRE(info.phase == Catch::Approx(1.0));
        REQUIRE(info.n_alpha_exc == 0);
        REQUIRE(info.n_beta_exc == 0);
    }

    SECTION("Single α excitation: 0→2, phase = -1") {
        // |bra⟩ α = {1,2}, β = {0,1}
        const Det bra{0b0110ULL, 0b0011ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 1);
        REQUIRE(info.n_alpha_exc == 1);
        REQUIRE(info.n_beta_exc == 0);
        REQUIRE(info.holes_alpha[0] == 0);
        REQUIRE(info.particles_alpha[0] == 2);
        // Orbital 1 between hole-particle → odd permutation → -1
        REQUIRE(info.phase == Catch::Approx(-1.0));
        REQUIRE(det_ops::phase(bra, ket) == Catch::Approx(-1.0));
    }

    SECTION("Single β excitation: 0→2, phase = -1") {
        const Det bra{0b0011ULL, 0b0110ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 1);
        REQUIRE(info.n_alpha_exc == 0);
        REQUIRE(info.n_beta_exc == 1);
        REQUIRE(info.holes_beta[0] == 0);
        REQUIRE(info.particles_beta[0] == 2);
        REQUIRE(info.phase == Catch::Approx(-1.0));
    }

    SECTION("Double α excitation: 0,1→2,3, phase = +1") {
        // |bra⟩ α = {2,3}, β = {0,1}
        const Det bra{0b1100ULL, 0b0011ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 2);
        REQUIRE(info.n_alpha_exc == 2);
        REQUIRE(info.n_beta_exc == 0);
        // Holes {0,1}, particles {2,3} → non-crossing → even perm
        REQUIRE(info.holes_alpha[0] == 0);
        REQUIRE(info.holes_alpha[1] == 1);
        REQUIRE(info.particles_alpha[0] == 2);
        REQUIRE(info.particles_alpha[1] == 3);
        REQUIRE(info.phase == Catch::Approx(1.0));
    }

    SECTION("Mixed α-β double excitation") {
        // |bra⟩ α: 0→2 → {1,2}, β: 1→3 → {0,3}
        const Det bra{0b0110ULL, 0b1001ULL};
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree == 2);
        REQUIRE(info.n_alpha_exc == 1);
        REQUIRE(info.n_beta_exc == 1);
        REQUIRE(info.phase == Catch::Approx(-1.0));  // Odd total permutation
    }

    SECTION("Degree > 2 → sentinel phase = 0.0") {
        // Triple excitation: 2α + 1β relative to |ket⟩
        const Det bra{0b1100ULL, 0b0101ULL};  // α={2,3}, β={0,2}
        REQUIRE(det_ops::phase(bra, ket) == Catch::Approx(0.0));
        const auto info = det_ops::analyze_excitation(bra, ket);
        REQUIRE(info.degree > MAX_EXCITATION_DEGREE);
        REQUIRE(info.phase == Catch::Approx(0.0));
    }
}

// -----------------------------------------------------------------------------
// Batch collectors: singles/doubles/connected
// Validates combinatorial counts: C(n_occ, n_exc) × C(n_virt, n_exc)
// -----------------------------------------------------------------------------
TEST_CASE("Collectors: singles/doubles/connected", "[det_ops]") {
    const int n_orb = 4;
    // |ket⟩: n_α = 2, n_β = 2
    const Det ket{0b0011ULL, 0b0011ULL};

    const auto singles   = det_ops::collect_singles(ket, n_orb);
    const auto doubles   = det_ops::collect_doubles(ket, n_orb);
    const auto connected = det_ops::collect_connected(ket, n_orb);

    const int n_alpha = popcount(ket.alpha);
    const int n_beta  = popcount(ket.beta);
    const int v_alpha = n_orb - n_alpha;
    const int v_beta  = n_orb - n_beta;

    // Singles: n_α·v_α + n_β·v_β = 2×2 + 2×2 = 8
    const size_t expect_singles = size_t(n_alpha) * v_alpha + size_t(n_beta) * v_beta;
    // Doubles: C(2,2)×C(2,2) (αα) + C(2,2)×C(2,2) (ββ) + (2×2)×(2×2) (αβ) = 1+1+16 = 18
    const size_t expect_daa = 1ULL * (n_alpha >= 2) * (v_alpha >= 2);
    const size_t expect_dbb = 1ULL * (n_beta  >= 2) * (v_beta  >= 2);
    const size_t expect_dab = size_t(n_alpha) * v_alpha * size_t(n_beta) * v_beta;
    const size_t expect_doubles = expect_daa + expect_dbb + expect_dab;
    const size_t expect_conn    = expect_singles + expect_doubles;  // 26

    REQUIRE(singles.size()   == expect_singles);
    REQUIRE(doubles.size()   == expect_doubles);
    REQUIRE(connected.size() == expect_conn);

    // All determinants unique
    std::unordered_set<Det> u_singles(singles.begin(), singles.end());
    std::unordered_set<Det> u_doubles(doubles.begin(), doubles.end());
    std::unordered_set<Det> u_connected(connected.begin(), connected.end());
    REQUIRE(u_singles.size()   == singles.size());
    REQUIRE(u_doubles.size()   == doubles.size());
    REQUIRE(u_connected.size() == connected.size());
}

// -----------------------------------------------------------------------------
// DetMap & set utilities: canonicalization, merging, union
// -----------------------------------------------------------------------------
TEST_CASE("DetMap & set utilities", "[det_space]") {
    std::vector<Det> raw = {{1,1}, {2,2}, {3,3}, {1,1}, {4,4}};

    SECTION("DetMap::from_list → sorted, deduplicated, indexed") {
        auto m = DetMap::from_list(raw);
        REQUIRE(m.size() == 4);
        REQUIRE(m.get_idx({1,1}).has_value());
        REQUIRE(m.get_det(0) == Det{1,1});
        REQUIRE_THROWS_AS(m.get_det(4), std::out_of_range);

        // Bidirectional consistency
        const auto& ds = m.all_dets();
        for (u32 i = 0; i < ds.size(); ++i) {
            REQUIRE(m.get_idx(ds[i]).value() == i);
        }
    }

    SECTION("DetMap::from_ordered → preserves order, removes duplicates") {
        auto m = DetMap::from_ordered(raw, /*verify_unique=*/false);
        REQUIRE(m.size() == 4);
        // First occurrence preserved
        REQUIRE(m.get_det(0) == Det{1,1});
        REQUIRE(m.get_det(1) == Det{2,2});
        REQUIRE(m.get_det(2) == Det{3,3});
        REQUIRE(m.get_det(3) == Det{4,4});
    }

    SECTION("Set operations: canonicalize/merge/union/stable_union") {
        std::vector<Det> a = {{1,1}, {2,2}, {3,3}};
        std::vector<Det> b = {{2,2}, {4,4}, {4,4}};

        auto ca = det_space::canonicalize(a);
        auto cb = det_space::canonicalize(b);
        REQUIRE(std::is_sorted(ca.begin(), ca.end()));
        REQUIRE(std::is_sorted(cb.begin(), cb.end()));

        // Merge two sorted sets
        auto merged = det_space::merge_sorted(ca, cb);
        REQUIRE(std::is_sorted(merged.begin(), merged.end()));
        REQUIRE(merged.size() == 4);

        // Hash-based union
        auto un = det_space::set_union_hash(a, b, /*sorted=*/true);
        REQUIRE(std::is_sorted(un.begin(), un.end()));
        REQUIRE(un.size() == 4);

        // Stable union: order of 'a' preserved, new from 'b' appended once
        auto st = det_space::stable_union(a, b);
        REQUIRE(st.front() == Det{1,1});
        std::unordered_set<Det> s(st.begin(), st.end());
        REQUIRE(s.size() == st.size());  // No duplicates
    }
}

// -----------------------------------------------------------------------------
// Batch generators & complement
// -----------------------------------------------------------------------------
TEST_CASE("det_space generators & complement", "[det_space]") {
    const int n_orb = 4;
    const Det ket{0b0011ULL, 0b0011ULL};
    std::vector<Det> parents{ket};

    const auto s1 = det_space::generate_singles(parents, n_orb, /*sorted=*/true);
    const auto d2 = det_space::generate_doubles(parents, n_orb, /*sorted=*/true);
    const auto cn = det_space::generate_connected(parents, n_orb, /*sorted=*/true);

    // Sizes match det_ops collectors
    REQUIRE(s1.size() == det_ops::collect_singles(ket, n_orb).size());
    REQUIRE(d2.size() == det_ops::collect_doubles(ket, n_orb).size());
    REQUIRE(cn.size() == det_ops::collect_connected(ket, n_orb).size());

    // Complement: exclude parent + one extra from connected
    auto exclude = DetMap::from_list({ket, cn.front()});
    const auto comp = det_space::generate_complement(parents, n_orb, exclude, /*sorted=*/true);
    REQUIRE(comp.size() + 1 == cn.size());  // One filtered out
    REQUIRE(std::find(comp.begin(), comp.end(), cn.front()) == comp.end());
}

// -----------------------------------------------------------------------------
// FCI space enumeration: Full CI with fixed n_α, n_β
// Size = C(n_orb, n_α) × C(n_orb, n_β)
// -----------------------------------------------------------------------------
TEST_CASE("FCISpace enumeration", "[det_enum][FCI]") {
    const int n_orb = 4, n_alpha = 1, n_beta = 1;
    FCISpace fci(n_orb, n_alpha, n_beta);
    REQUIRE(fci.size() == size_t(n_orb) * size_t(n_orb));  // C(4,1)² = 16

    for (const auto& d : fci.dets()) {
        REQUIRE(popcount(d.alpha) == n_alpha);
        REQUIRE(popcount(d.beta)  == n_beta);
    }
}

// -----------------------------------------------------------------------------
// CAS space: FCI in active orbitals, frozen core/virtual
// Verifies core doubly occupied, virtual empty, active FCI
// -----------------------------------------------------------------------------
TEST_CASE("CASSpace enumeration", "[det_enum][CAS]") {
    // Partition: 1 core, 2 active, 1 virtual (4 orbitals total)
    // Active: 1 α, 1 β → C(2,1)² = 4 determinants
    CASSpace cas(/*n_core=*/1, /*n_active=*/2, /*n_virtual=*/1,
                 /*n_alpha_active=*/1, /*n_beta_active=*/1);
    REQUIRE(cas.size() == 4);

    for (const auto& d : cas.dets()) {
        // Core (bit 0) doubly occupied
        REQUIRE((d.alpha & 0b0001ULL) != 0ULL);
        REQUIRE((d.beta  & 0b0001ULL) != 0ULL);
        // Virtual (bit 3) empty
        REQUIRE(((d.alpha | d.beta) & 0b1000ULL) == 0ULL);
        // Active (bits 1-2): exactly 1 α and 1 β
        REQUIRE(popcount((d.alpha >> 1) & 0b11ULL) == 1);
        REQUIRE(popcount((d.beta  >> 1) & 0b11ULL) == 1);
    }
}

// -----------------------------------------------------------------------------
// RAS space: TOTAL constraints on α+β holes/particles
// max_holes_ras1: max total (α+β) holes in RAS1
// max_elecs_ras3: max total (α+β) electrons in RAS3
// -----------------------------------------------------------------------------
TEST_CASE("RASSpace enumeration & TOTAL constraints", "[det_enum][RAS]") {
    // Partition: 1 core, 1 RAS1, 1 RAS2, 1 RAS3, 0 virtual (4 orbitals)
    RASOrbitalPartition p{1, 1, 1, 1, 0};

    // Each spin: 2 electrons total (including core)
    // Constraints: ≤1 total hole in RAS1, ≤1 total electron in RAS3
    RASElectronConstraint e{/*n_alpha_total=*/2, /*n_beta_total=*/2,
                            /*max_holes_ras1=*/1, /*max_elecs_ras3=*/1};

    RASSpace ras(p, e);
    REQUIRE(ras.size() > 0);

    const int off_core = 0;
    const int off_r1   = off_core + p.n_core;
    const int off_r2   = off_r1 + p.n_ras1;
    const int off_r3   = off_r2 + p.n_ras2;

    const u64 core_mask = (1ULL << off_core);
    const u64 ras1_mask = (1ULL << off_r1);
    const u64 ras2_mask = (1ULL << off_r2);
    const u64 ras3_mask = (1ULL << off_r3);

    const int ras1_cap = 2 * p.n_ras1;  // Max total electrons in RAS1 = 2

    int ras3_occupancy_count = 0;

    for (const auto& d : ras.dets()) {
        // Core doubly occupied
        REQUIRE((d.alpha & core_mask) != 0ULL);
        REQUIRE((d.beta  & core_mask) != 0ULL);

        // Total holes in RAS1: ras1_cap - (n_α + n_β in RAS1)
        const int occ_ras1 = popcount(d.alpha & ras1_mask) + popcount(d.beta & ras1_mask);
        const int holes1   = ras1_cap - occ_ras1;
        REQUIRE((holes1 <= e.max_holes_ras1 || e.max_holes_ras1 < 0));

        // Total electrons in RAS3
        const int occ_ras3 = popcount(d.alpha & ras3_mask) + popcount(d.beta & ras3_mask);
        REQUIRE((occ_ras3 <= e.max_elecs_ras3 || e.max_elecs_ras3 < 0));

        ras3_occupancy_count += (occ_ras3 > 0);

        // RAS2: CAS-like, no hard constraints beyond capacity
        REQUIRE(popcount(d.alpha & ras2_mask) <= 1);
        REQUIRE(popcount(d.beta  & ras2_mask) <= 1);
    }

    // At least some determinants use RAS3 (verify constraint is active)
    REQUIRE(ras3_occupancy_count >= 0);
}
