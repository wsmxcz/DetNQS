// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_hamiltonian.cpp
 * @brief Core Hamiltonian block builder and effective Hamiltonian validation.
 *
 * Tests cover:
 *  - Block matrix assembly: H_VV, H_VP for known V/P subspaces
 *  - Heat-bath driven connection discovery (V → C → P)
 *  - Effective Hamiltonian downfolding: H_eff = H_VV + H_VP D^{-1} H_PV
 *  - Element-wise accuracy and Hermiticity
 *
 * Test system: H₂O/STO-3G (7 orb, 5α + 5β, 441 FCI determinants)
 * Reference FCI energy: -75.01240139 Ha (PySCF)
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
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
 * Construct Hartree-Fock determinant: lowest n_α + n_β orbitals occupied.
 * Returns |00...011...1⟩ with n_α alpha-spin and n_β beta-spin bits set.
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
 * Optionally symmetrize by copying upper triangle to lower.
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
 * Ground state energy via dense diagonalization.
 * Returns E_0 = min(eigenvalues(H)) + E_nuc.
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
    static constexpr double E_ref_fci = -75.01240139;  ///< PySCF benchmark
    
    std::shared_ptr<IntegralMO> mo;
    std::shared_ptr<IntegralSO> so;
    std::shared_ptr<HamEval> ham;
    std::shared_ptr<HeatBathTable> hb_table;
    
    DetMap fci_basis;  ///< Full CI space: (7 choose 5)^2 = 441 determinants
    Det hf_det;
    
    H2OFixture() {
        if (!std::filesystem::exists(fcidump)) {
            FAIL("FCIDUMP not found: " << fcidump);
        }
        
        // Load MO integrals and build spin-orbital representation
        mo = std::make_shared<IntegralMO>(n_orb);
        mo->load_from_fcidump(fcidump);
        so = std::make_shared<IntegralSO>(*mo);
        ham = std::make_shared<HamEval>(*so);
        
        // Build Heat-bath screening table for connection discovery
        hb_table = std::make_shared<HeatBathTable>(
            *so, HBBuildOptions{HEATBATH_THRESH}
        );
        hb_table->build();
        
        // Generate full FCI basis
        FCISpace fci_space(n_orb, n_alpha, n_beta);
        fci_basis = DetMap::from_list(fci_space.dets());
        REQUIRE(fci_basis.size() == 441);
        
        hf_det = make_hf_det(n_alpha, n_beta);
    }
};

// ============================================================================
// Test Cases
// ============================================================================

TEST_CASE_METHOD(H2OFixture, "Diagonal element batch evaluation", "[hamiltonian][diagonal]") {
    const auto& dets = fci_basis.all_dets();
    
    // Batch compute all diagonal matrix elements: H_ii = ⟨i|Ĥ|i⟩
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

TEST_CASE_METHOD(H2OFixture, "Block matrix assembly with explicit V/P partition", "[hamiltonian][block]") {
    // Partition FCI space: V = first 50 dets, P = next 30 dets
    const auto& all_dets = fci_basis.all_dets();
    std::vector<Det> dets_V(all_dets.begin(), all_dets.begin() + 50);
    std::vector<Det> dets_P(all_dets.begin() + 50, all_dets.begin() + 80);
    
    // Build H_VV and H_VP blocks (API may use legacy names internally)
    auto result = get_ham_block(dets_V, std::span(dets_P), *ham, n_orb);
    
    SECTION("Dimension consistency") {
        REQUIRE(result.map_P.size() == 30);  // Perturbative space indexing
        REQUIRE(!result.H_VV.empty());       // H_VV block
        REQUIRE(!result.H_VP.empty());       // H_VP block
    }
    
    SECTION("Element-wise accuracy") {
        auto H_VV = coo_to_dense(result.H_VV, 50);
        
        std::mt19937 rng(123);
        std::uniform_int_distribution<int> dist(0, 49);
        
        // Random sampling of H_VV[i,j] vs direct ⟨i|Ĥ|j⟩
        for (int trial = 0; trial < 10; ++trial) {
            int i = dist(rng);
            int j = dist(rng);
            
            double h_ij = ham->compute_elem(dets_V[i], dets_V[j]);
            REQUIRE(H_VV(i, j) == Approx(h_ij).margin(1e-12));
        }
    }
}

TEST_CASE_METHOD(H2OFixture, "Heat-bath driven connection discovery", "[hamiltonian][connectivity]") {
    std::vector<Det> V_init = {hf_det};
    
    // Discover connected space C via Heat-bath, extract P = C \ V
    auto result = get_ham_conn(
        V_init, *ham, n_orb, hb_table.get(),
        /*eps1=*/1e-6,
        /*use_heatbath=*/true
    );
    
    SECTION("Perturbative space size") {
        size_t n_P = result.map_P.size();  // P-space after V removal
        INFO("Discovered |P| = " << n_P);
        
        // Expected: HF singles (~20) + relevant doubles
        REQUIRE(n_P >= 20);
        REQUIRE(n_P <= 140);
    }
    
    SECTION("HF determinant excluded from P-space") {
        // HF remains in V, should not appear in P
        auto idx = result.map_P.get_idx(hf_det);
        REQUIRE(!idx.has_value());
    }
}

TEST_CASE_METHOD(H2OFixture, "Effective Hamiltonian downfolding", "[hamiltonian][heff]") {
    // Initialize V = {HF}, discover P via Heat-bath
    std::vector<Det> V_hf = {hf_det};
    
    auto conn_result = get_ham_conn(
        V_hf, *ham, n_orb, hb_table.get(), /*eps1=*/1e-5
    );
    
    // Compute P-space diagonal: D_jj = E_ref - H_PP[j,j]
    const auto& P_dets = conn_result.map_P.all_dets();
    auto h_PP_diag = get_ham_diag(P_dets, *ham);
    
    double e_ref = ham->compute_diagonal(hf_det);
    
    SECTION("Assembly completes without error") {
        HeffConfig config{
            .reg_type = Regularization::Sigma,
            .epsilon = 1e-10
        };
        
        // H_eff = H_VV + H_VP D^{-1} H_PV
        auto H_eff = get_ham_eff(
            conn_result.H_VV,   // H_VV
            conn_result.H_VP,   // H_VP
            h_PP_diag,          // diagonal of H_PP
            e_ref,
            config
        );
        
        REQUIRE(H_eff.n_rows == V_hf.size());
        REQUIRE(!H_eff.empty());
    }
    
    SECTION("Perturbative correction lowers energy") {
        HeffConfig config{.epsilon = 1e-10};
        
        auto H_eff = get_ham_eff(
            conn_result.H_VV, conn_result.H_VP,
            h_PP_diag, e_ref, config
        );
        
        // Compare E(HF) vs E(H_eff): downfolding should lower energy
        auto H_eff_dense = coo_to_dense(H_eff, 1);
        auto H_VV_dense = coo_to_dense(conn_result.H_VV, 1);
        
        double E_hf = H_VV_dense(0, 0) + mo->e_nuc;
        double E_eff = H_eff_dense(0, 0) + mo->e_nuc;
        
        INFO("E(HF)    = " << E_hf);
        INFO("E(H_eff) = " << E_eff);
        
        REQUIRE(E_eff < E_hf);
    }
}

TEST_CASE_METHOD(H2OFixture, "FCI energy benchmark", "[hamiltonian][fci]") {
    const auto& dets = fci_basis.all_dets();
    
    // Build full Hamiltonian over FCI space (V = full space, P = empty)
    auto result = get_ham_block(dets, std::nullopt, *ham, n_orb);
    
    auto H_dense = coo_to_dense(result.H_VV, dets.size());
    double E_fci = compute_ground_energy(H_dense, mo->e_nuc);
    
    INFO("Computed E_FCI:  " << E_fci);
    INFO("Reference E_FCI: " << E_ref_fci);
    
    REQUIRE(E_fci == Approx(E_ref_fci).epsilon(1e-7));
}

TEST_CASE_METHOD(H2OFixture, "Hermiticity check", "[hamiltonian][symmetry]") {
    // Use 50-determinant subspace for efficiency
    const auto& all_dets = fci_basis.all_dets();
    std::vector<Det> subset(all_dets.begin(), all_dets.begin() + 50);
    
    auto result = get_ham_block(subset, std::nullopt, *ham, n_orb);
    
    // Check H = H† without manual symmetrization
    auto H = coo_to_dense(result.H_VV, subset.size(), /*symmetrize=*/false);
    
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
    // Construct upper-triangle COO matrix (3×3)
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
    
    // Verify symmetry: H_ij = H_ji
    auto H = coo_to_dense(full, 3, /*symmetrize=*/false);
    REQUIRE(H(0, 1) == Approx(2.0));
    REQUIRE(H(1, 0) == Approx(2.0));
    REQUIRE(H(1, 2) == Approx(4.0));
    REQUIRE(H(2, 1) == Approx(4.0));
}
