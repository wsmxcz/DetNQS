// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_hamiltonian.cpp
 * @brief Core Hamiltonian builder and effective Hamiltonian tests.
 *
 * Validates:
 *  - Block construction: H_SS, H_SC with known S/C spaces
 *  - Heat-bath connection discovery
 *  - Effective Hamiltonian: H_eff = H_SS + H_SC·D⁻¹·H_CS
 *  - Element-wise correctness vs direct evaluation
 *
 * Test system: H₂O/STO-3G (7 orb, 5α5β, 441 FCI dets)
 * Reference FCI: -75.01240139 Ha (PySCF)
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <detnqs/determinant/det.hpp>
#include <detnqs/determinant/det_space.hpp>
#include <detnqs/determinant/det_enum.hpp>
#include <detnqs/integral/integral_mo.hpp>
#include <detnqs/integral/integral_so.hpp>
#include <detnqs/integral/hb_table.hpp>
#include <detnqs/hamiltonian/ham_eval.hpp>
#include <detnqs/hamiltonian/build_ham.hpp>
#include <detnqs/hamiltonian/ham_eff.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <random>
#include <vector>

using namespace detnqs;
using Catch::Approx;

// ============================================================================
// Test Utilities
// ============================================================================

/**
 * Construct HF determinant with n_α/n_β electrons in lowest orbitals.
 * Returns |00...011...1⟩ with n bits set.
 */
[[nodiscard]] static Det make_hf_det(int n_alpha, int n_beta) {
    auto lowbits = [](int k) -> u64 {
        if (k <= 0) return 0ULL;
        if (k >= 64) return ~0ULL;
        return (1ULL << k) - 1;
    };
    return {lowbits(n_alpha), lowbits(n_beta)};
}

/**
 * Convert COO sparse matrix to dense Eigen format.
 * Optionally symmetrize by mirroring upper triangle.
 */
[[nodiscard]] static Eigen::MatrixXd coo_to_dense(
    const COOMatrix& coo,
    size_t dim,
    bool symmetrize = true
) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    for (size_t k = 0; k < coo.nnz(); ++k) {
        u32 i = coo.rows[k];
        u32 j = coo.cols[k];
        double v = coo.vals[k];
        
        if (i < dim && j < dim) {
            H(i, j) = v;
            if (symmetrize && i != j) H(j, i) = v;
        }
    }
    return H;
}

/**
 * Compute ground state energy via dense diagonalization.
 * Returns E₀ = λ_min(H) + E_nuc.
 */
[[nodiscard]] static double compute_ground_energy(
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
// H₂O/STO-3G Test Fixture
// ============================================================================

struct H2OFixture {
    static constexpr const char* fcidump = "H2O_sto3g.FCIDUMP";
    static constexpr int n_orb = 7;
    static constexpr int n_alpha = 5;
    static constexpr int n_beta = 5;
    static constexpr double E_ref_fci = -75.01240139;  ///< PySCF reference
    
    std::shared_ptr<IntegralMO> mo;
    std::shared_ptr<IntegralSO> so;
    std::shared_ptr<HamEval> ham;
    std::shared_ptr<HeatBathTable> hb_table;
    
    DetMap fci_basis;
    Det hf_det;
    
    H2OFixture() {
        if (!std::filesystem::exists(fcidump)) {
            FAIL("FCIDUMP not found: " << fcidump);
        }
        
        // Load integrals and build SO representation
        mo = std::make_shared<IntegralMO>(n_orb);
        mo->load_from_fcidump(fcidump);
        so = std::make_shared<IntegralSO>(*mo);
        ham = std::make_shared<HamEval>(*so);
        
        // Build Heat-bath screening table
        hb_table = std::make_shared<HeatBathTable>(
            *so, HBBuildOptions{HEATBATH_THRESH}
        );
        hb_table->build();
        
        // Generate full FCI space: (7 choose 5)² = 441 determinants
        FCISpace fci_space(n_orb, n_alpha, n_beta);
        fci_basis = DetMap::from_list(fci_space.dets());
        REQUIRE(fci_basis.size() == 441);
        
        hf_det = make_hf_det(n_alpha, n_beta);
    }
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(H2OFixture, "Diagonal element evaluation", "[hamiltonian][diagonal]") {
    const auto& dets = fci_basis.all_dets();
    
    // Batch computation of all diagonal elements
    auto diag = get_ham_diag(dets, *ham);
    REQUIRE(diag.size() == dets.size());
    
    // Verify against direct evaluation (spot check)
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, dets.size() - 1);
    
    for (int trial = 0; trial < 10; ++trial) {
        size_t i = dist(rng);
        double expected = ham->compute_diagonal(dets[i]);
        REQUIRE(diag[i] == Approx(expected).margin(1e-14));
    }
}

TEST_CASE_METHOD(H2OFixture, "Block construction with known spaces", "[hamiltonian][block]") {
    // Partition FCI space: S = first 50 dets, C = next 30 dets
    const auto& all_dets = fci_basis.all_dets();
    std::vector<Det> dets_S(all_dets.begin(), all_dets.begin() + 50);
    std::vector<Det> dets_C(all_dets.begin() + 50, all_dets.begin() + 80);
    
    auto result = get_ham_block(dets_S, std::span(dets_C), *ham, n_orb);
    
    SECTION("Dimensions") {
        REQUIRE(result.map_C.size() == 30);
        REQUIRE(!result.H_SS.empty());
        REQUIRE(!result.H_SC.empty());
    }
    
    SECTION("Element-wise correctness") {
        auto H_SS = coo_to_dense(result.H_SS, 50);
        
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> dist(0, 49);
        
        // Spot check random matrix elements
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
    
    // Discover connected C-space via Heat-bath screening
    auto result = get_ham_conn(
        S, *ham, n_orb, hb_table.get(),
        /*eps1=*/1e-6,
        /*use_heatbath=*/true
    );
    
    SECTION("Connected space size") {
        size_t n_C = result.map_C.size();
        INFO("Discovered |C| = " << n_C);
        
        // Expected: singles (≈20) + relevant doubles
        REQUIRE(n_C >= 20);
        REQUIRE(n_C <= 140);
    }
    
    SECTION("No self-connections") {
        // HF determinant remains in S, not in C
        auto idx = result.map_C.get_idx(hf_det);
        REQUIRE(!idx.has_value());
    }
}

TEST_CASE_METHOD(H2OFixture, "Effective Hamiltonian assembly", "[hamiltonian][heff]") {
    // Build CI space: HF + Heat-bath connections
    std::vector<Det> S_small = {hf_det};
    
    auto conn_result = get_ham_conn(
        S_small, *ham, n_orb, hb_table.get(), /*eps1=*/1e-5
    );
    
    // Compute C-space diagonal: D_jj = E_ref - H_CC[j,j]
    const auto& C_dets = conn_result.map_C.all_dets();
    auto h_cc_diag = get_ham_diag(C_dets, *ham);
    
    // Use HF energy as reference
    double e_ref = ham->compute_diagonal(hf_det);
    
    SECTION("Assembly succeeds") {
        HeffConfig config{
            .reg_type = Regularization::Sigma,
            .epsilon = 1e-10
        };
        
        auto H_eff = get_ham_eff(
            conn_result.H_SS, conn_result.H_SC,
            h_cc_diag, e_ref, config
        );
        
        REQUIRE(H_eff.n_rows == S_small.size());
        REQUIRE(!H_eff.empty());
    }
    
    SECTION("Perturbative correction lowers energy") {
        HeffConfig config{.epsilon = 1e-10};
        
        auto H_eff = get_ham_eff(
            conn_result.H_SS, conn_result.H_SC,
            h_cc_diag, e_ref, config
        );
        
        // Extract energies: ΔE = H_SC·D⁻¹·H_CS should be negative
        auto H_eff_dense = coo_to_dense(H_eff, 1);
        auto H_SS_dense = coo_to_dense(conn_result.H_SS, 1);
        
        double E_hf = H_SS_dense(0, 0) + mo->e_nuc;
        double E_eff = H_eff_dense(0, 0) + mo->e_nuc;
        
        INFO("E(HF) = " << E_hf);
        INFO("E(H_eff) = " << E_eff);
        
        REQUIRE(E_eff < E_hf);
    }
}

TEST_CASE_METHOD(H2OFixture, "FCI energy recovery", "[hamiltonian][fci]") {
    const auto& dets = fci_basis.all_dets();
    
    // Build full Hamiltonian matrix over FCI space
    auto result = get_ham_block(dets, std::nullopt, *ham, n_orb);
    
    auto H_dense = coo_to_dense(result.H_SS, dets.size());
    double E_fci = compute_ground_energy(H_dense, mo->e_nuc);
    
    INFO("Computed FCI: " << E_fci);
    INFO("Reference FCI: " << E_ref_fci);
    
    REQUIRE(E_fci == Approx(E_ref_fci).epsilon(1e-7));
}

TEST_CASE_METHOD(H2OFixture, "Hermiticity verification", "[hamiltonian][symmetry]") {
    // Use small subspace for efficiency
    const auto& all_dets = fci_basis.all_dets();
    std::vector<Det> subset(all_dets.begin(), all_dets.begin() + 50);
    
    auto result = get_ham_block(subset, std::nullopt, *ham, n_orb);
    
    // Check H = H† without symmetrization
    auto H = coo_to_dense(result.H_SS, subset.size(), /*symmetrize=*/false);
    
    double max_asym = 0.0;
    for (size_t i = 0; i < subset.size(); ++i) {
        for (size_t j = i + 1; j < subset.size(); ++j) {
            max_asym = std::max(max_asym, std::abs(H(i, j) - H(j, i)));
        }
    }
    
    INFO("Max |H_ij - H_ji|: " << max_asym);
    REQUIRE(max_asym < 1e-12);
}

TEST_CASE_METHOD(H2OFixture, "Upper triangle mirroring utility", "[hamiltonian][utility]") {
    // Construct upper-triangle COO matrix
    COOMatrix upper{
        .rows = {0, 0, 1, 1, 2},
        .cols = {0, 1, 1, 2, 2},
        .vals = {1.0, 2.0, 3.0, 4.0, 5.0},
        .n_rows = 3,
        .n_cols = 3
    };
    
    auto full = mirror_upper_to_full(upper);
    
    // 5 original + 2 mirrored off-diagonal = 7 total
    REQUIRE(full.nnz() == 7);
    
    // Verify symmetry
    auto H = coo_to_dense(full, 3, /*symmetrize=*/false);
    REQUIRE(H(0, 1) == Approx(2.0));
    REQUIRE(H(1, 0) == Approx(2.0));
    REQUIRE(H(1, 2) == Approx(4.0));
    REQUIRE(H(2, 1) == Approx(4.0));
}
