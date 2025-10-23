// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Core tests for Hamiltonian builders and effective Hamiltonian assembly.
 *
 * Validates:
 *  - Block construction (H_SS, H_SC) with known S/C spaces
 *  - Connection discovery via Heat-bath screening
 *  - Effective Hamiltonian assembly: H_eff = H_SS + H_SC·D⁻¹·H_CS
 *  - Element-wise correctness against direct evaluation
 *
 * Test system: H2O/STO-3G (7 orbitals, 5α5β, 441 FCI determinants)
 * Reference FCI energy: -75.01240139 Ha
 *
 * File: tests/test_hamiltonian.cpp
 * Author: Zheng (Alex) Che
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <lever/determinant/det.hpp>
#include <lever/determinant/det_space.hpp>
#include <lever/determinant/det_enum.hpp>
#include <lever/integral/integral_mo.hpp>
#include <lever/integral/integral_so.hpp>
#include <lever/integral/hb_table.hpp>
#include <lever/hamiltonian/ham_eval.hpp>
#include <lever/hamiltonian/build_ham.hpp>
#include <lever/hamiltonian/ham_eff.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

using namespace lever;
using Catch::Approx;

// ============================================================================
// Test Utilities
// ============================================================================

/** Construct HF determinant with n_α, n_β electrons in lowest orbitals. */
static Det make_hf_det(int n_alpha, int n_beta) {
    auto lowbits = [](int k) -> u64 {
        if (k <= 0) return 0ULL;
        if (k >= 64) return ~0ULL;
        return (1ULL << k) - 1;
    };
    return {lowbits(n_alpha), lowbits(n_beta)};
}

/** Convert COO format to dense Eigen matrix. */
static Eigen::MatrixXd coo_to_dense(
    const std::vector<Conn>& coo,
    size_t dim,
    bool symmetrize = true
) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    for (const auto& e : coo) {
        if (e.row < dim && e.col < dim) {
            H(e.row, e.col) = e.val;
            if (symmetrize && e.row != e.col) {
                H(e.col, e.row) = e.val;
            }
        }
    }
    return H;
}

/** Compute ground state energy via dense diagonalization. */
static double compute_ground_energy(
    const Eigen::MatrixXd& H,
    double e_nuc
) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue solver failed");
    }
    return solver.eigenvalues()(0) + e_nuc;
}

// ============================================================================
// H2O/STO-3G Fixture
// ============================================================================

struct H2OFixture {
    const std::string fcidump = "H2O_sto3g.FCIDUMP";
    
    std::shared_ptr<IntegralMO> mo;
    std::shared_ptr<IntegralSO> so;
    std::shared_ptr<HamEval> ham;
    std::shared_ptr<HeatBathTable> hb_table;
    
    DetMap fci_basis;
    Det hf_det;
    
    const int n_orb = 7;
    const int n_alpha = 5;
    const int n_beta = 5;
    
    const double E_ref_fci = -75.01240139;  // PySCF reference
    
    H2OFixture() {
        if (!std::filesystem::exists(fcidump)) {
            FAIL("FCIDUMP not found: " << fcidump);
        }
        
        // Load integrals
        mo = std::make_shared<IntegralMO>(n_orb);
        mo->load_from_fcidump(fcidump);
        so = std::make_shared<IntegralSO>(*mo);
        ham = std::make_shared<HamEval>(*so);
        
        // Build Heat-bath table
        hb_table = std::make_shared<HeatBathTable>(*so, HBBuildOptions{1e-14});
        hb_table->build();
        
        // Generate full FCI basis
        FCISpace fci_space(n_orb, n_alpha, n_beta);
        fci_basis = DetMap::from_list(fci_space.dets());
        REQUIRE(fci_basis.size() == 441);
        
        hf_det = make_hf_det(n_alpha, n_beta);
    }
};

// ============================================================================
// Tests
// ============================================================================

TEST_CASE_METHOD(H2OFixture, "Diagonal elements", "[hamiltonian][diagonal]") {
    const auto& dets = fci_basis.all_dets();
    
    // Compute via batch function
    auto diag = get_ham_diag(dets, *ham);
    REQUIRE(diag.size() == dets.size());
    
    // Spot check against direct evaluation
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dets.size() - 1);
    
    for (int trial = 0; trial < 10; ++trial) {
        size_t i = dist(rng);
        double expected = ham->compute_diagonal(dets[i]);
        REQUIRE(diag[i] == Approx(expected).margin(1e-14));
    }
}

TEST_CASE_METHOD(H2OFixture, "Block construction with known spaces", "[hamiltonian][block]") {
    // Use first 50 determinants as S, next 30 as C
    const auto& all_dets = fci_basis.all_dets();
    std::vector<Det> dets_S(all_dets.begin(), all_dets.begin() + 50);
    std::vector<Det> dets_C(all_dets.begin() + 50, all_dets.begin() + 80);
    
    auto result = get_ham_block(
        dets_S,
        std::span(dets_C),
        *ham,
        n_orb
    );
    
    SECTION("Dimensions") {
        REQUIRE(result.map_C.size() == 30);
        REQUIRE(!result.coo_SS.empty());
        REQUIRE(!result.coo_SC.empty());
    }
    
    SECTION("Element correctness") {
        // Verify SS block
        auto H_SS = coo_to_dense(result.coo_SS, 50);
        
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> dist(0, 49);
        
        for (int trial = 0; trial < 10; ++trial) {
            int i = dist(rng);
            int j = dist(rng);
            
            double h_ij = ham->compute_elem(dets_S[i], dets_S[j]);
            REQUIRE(H_SS(i, j) == Approx(h_ij).margin(1e-12));
        }
    }
}

TEST_CASE_METHOD(H2OFixture, "Connection discovery from HF", "[hamiltonian][connectivity]") {
    std::vector<Det> S = {hf_det};
    
    auto result = get_ham_conn(
        S,
        *ham,
        n_orb,
        hb_table.get(),
        /*eps1=*/1e-6,
        /*use_heatbath=*/true
    );
    
    SECTION("Connected space size") {
        size_t n_C = result.map_C.size();
        INFO("Discovered |C| = " << n_C);
        
        // Expected: singles (5*2 + 5*2 = 20) + relevant doubles
        REQUIRE(n_C >= 20);
        REQUIRE(n_C <= 140);  // Upper bound: all singles + all doubles
    }
    
    SECTION("No self-connections in C") {
        // HF should remain in S, not appear in discovered C
        auto idx = result.map_C.get_idx(hf_det);
        REQUIRE(!idx.has_value());
    }
}

TEST_CASE_METHOD(H2OFixture, "Effective Hamiltonian assembly", "[hamiltonian][heff]") {
    // Build small CI space: HF + single excitations
    std::vector<Det> S_small = {hf_det};
    
    auto conn_result = get_ham_conn(
        S_small,
        *ham,
        n_orb,
        hb_table.get(),
        /*eps1=*/1e-5
    );
    
    // Get C-space diagonal
    const auto& C_dets = conn_result.map_C.all_dets();
    auto h_cc_diag = get_ham_diag(C_dets, *ham);
    
    // Reference energy (use HF diagonal as crude estimate)
    double e_ref = ham->compute_diagonal(hf_det);
    
    SECTION("Assembly succeeds") {
        HeffConfig config;
        config.reg_type = Regularization::Sigma;
        config.epsilon = 1e-10;
        
        auto heff_result = get_ham_eff(
            conn_result,
            h_cc_diag,
            e_ref,
            config
        );
        
        REQUIRE(heff_result.n_rows_S == S_small.size());
        REQUIRE(!heff_result.coo_heff.empty());
        
        INFO("Regularized columns: " << heff_result.n_regularized);
        INFO("Max |D_jj|: " << heff_result.max_abs_d);
        INFO("Max correction: " << heff_result.max_correction);
    }
    
    SECTION("Perturbative correction lowers energy") {
        // Build H_eff and H_SS for comparison
        HeffConfig config;
        config.epsilon = 1e-10;
        
        auto heff_result = get_ham_eff(
            conn_result,
            h_cc_diag,
            e_ref,
            config
        );
        
        auto H_eff = coo_to_dense(heff_result.coo_heff, 1);
        auto H_SS = coo_to_dense(conn_result.coo_SS, 1);
        
        double E_hf = H_SS(0, 0) + mo->e_nuc;
        double E_eff = H_eff(0, 0) + mo->e_nuc;
        
        INFO("E(HF) = " << E_hf);
        INFO("E(H_eff) = " << E_eff);
        
        // Perturbation should lower energy
        REQUIRE(E_eff < E_hf);
    }
}

TEST_CASE_METHOD(H2OFixture, "FCI energy recovery", "[hamiltonian][fci]") {
    const auto& dets = fci_basis.all_dets();
    
    // Build full H_SS over FCI space
    auto result = get_ham_block(
        dets,
        std::nullopt,  // No C space
        *ham,
        n_orb
    );
    
    auto H_dense = coo_to_dense(result.coo_SS, dets.size());
    double E_fci = compute_ground_energy(H_dense, mo->e_nuc);
    
    INFO("Computed FCI energy: " << E_fci);
    INFO("Reference FCI energy: " << E_ref_fci);
    
    REQUIRE(E_fci == Approx(E_ref_fci).epsilon(1e-7));
}

TEST_CASE_METHOD(H2OFixture, "Hermiticity verification", "[hamiltonian][symmetry]") {
    // Use small subspace for efficiency
    const auto& all_dets = fci_basis.all_dets();
    std::vector<Det> subset(all_dets.begin(), all_dets.begin() + 50);
    
    auto result = get_ham_block(
        subset,
        std::nullopt,
        *ham,
        n_orb
    );
    
    auto H = coo_to_dense(result.coo_SS, subset.size(), /*symmetrize=*/false);
    
    // Check hermiticity
    double max_asym = 0.0;
    for (size_t i = 0; i < subset.size(); ++i) {
        for (size_t j = i + 1; j < subset.size(); ++j) {
            max_asym = std::max(max_asym, std::abs(H(i, j) - H(j, i)));
        }
    }
    
    INFO("Max |H_ij - H_ji|: " << max_asym);
    REQUIRE(max_asym < 1e-12);
}

TEST_CASE_METHOD(H2OFixture, "Mirror upper triangle", "[hamiltonian][utility]") {
    std::vector<Conn> upper = {
        {0, 0, 1.0},
        {0, 1, 2.0},
        {1, 1, 3.0},
        {1, 2, 4.0},
        {2, 2, 5.0}
    };
    
    auto full = mirror_upper_to_full(upper);
    
    // Should have: 5 original + 2 mirrored off-diagonal entries
    REQUIRE(full.size() == 7);
    
    // Verify mirroring
    auto H = coo_to_dense(full, 3, /*symmetrize=*/false);
    REQUIRE(H(0, 1) == Approx(2.0));
    REQUIRE(H(1, 0) == Approx(2.0));
    REQUIRE(H(1, 2) == Approx(4.0));
    REQUIRE(H(2, 1) == Approx(4.0));
}
