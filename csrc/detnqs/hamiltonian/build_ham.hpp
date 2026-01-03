// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file build_ham.hpp
 * @brief Streaming Hamiltonian construction for variational subspaces.
 *
 * Constructs H_VV (variational-variational) and H_VP (variational-perturbative)
 * blocks under different screening policies:
 *
 *  - KnownSets:  Pre-defined V and P spaces
 *  - StaticHB:   Heat-Bath screening with fixed ε_1 cutoff
 *  - DynamicAmp: Amplitude-weighted screening τ_i = ε_1/max(|ψ_V[i]|, δ)
 *
 * Output format: Deterministic sorted COO with merged duplicates.
 * Parallelism:   OpenMP with thread-local accumulation.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#pragma once

#include <detnqs/determinant/det.hpp>
#include <detnqs/hamiltonian/ham_eval.hpp>
#include <detnqs/hamiltonian/ham_utils.hpp>
#include <detnqs/integral/hb_table.hpp>

#include <optional>
#include <span>
#include <vector>

namespace detnqs {

/**
 * Compute diagonal Hamiltonian elements ⟨D|H|D⟩.
 *
 * @param dets Determinant list
 * @param ham  Hamiltonian evaluator
 * @return Diagonal elements aligned with input determinants
 */
[[nodiscard]] std::vector<double> get_ham_diag(
    std::span<const Det> dets,
    const HamEval& ham
);

/**
 * Build H_VV via full enumeration within variational space.
 *
 * Enumerates all single and double excitations within V.
 *
 * @param dets_V Variational determinants
 * @param ham    Hamiltonian evaluator
 * @param n_orb  Number of spatial orbitals
 * @return COO matrix for H_VV
 */
[[nodiscard]] COOMatrix get_ham_vv(
    std::span<const Det> dets_V,
    const HamEval& ham,
    int n_orb
);

/**
 * Build H_VV and H_VP with pre-defined spaces.
 *
 * Policy: KnownSets
 *  - Computes ⟨V|H|V⟩ and ⟨V|H|P⟩ for given V and P spaces
 *  - No screening applied
 *
 * @param dets_V Variational determinants
 * @param dets_P Optional perturbative determinants; nullopt yields empty H_VP
 * @param ham    Hamiltonian evaluator
 * @param n_orb  Number of spatial orbitals
 * @return HamBlocks{H_VV, H_VP, P_index_map}
 */
[[nodiscard]] HamBlocks get_ham_block(
    std::span<const Det> dets_V,
    std::optional<std::span<const Det>> dets_P,
    const HamEval& ham,
    int n_orb
);

/**
 * Build H_VV and discover connected space via static Heat-Bath screening.
 *
 * Policy: StaticHB
 *  - Doubles: Retain if |⟨ij||ab⟩| ≥ ε_1 using Heat-Bath table
 *  - Singles: Evaluate and retain if |H_ik| ≥ 1e-12
 *  - Returns H_VP where P = discovered connected \ V
 *
 * @param dets_V       Variational determinants
 * @param ham          Hamiltonian evaluator
 * @param n_orb        Number of spatial orbitals
 * @param hb_table     Heat-Bath integral table (required if use_heatbath=true)
 * @param eps1         Heat-Bath threshold ε_1
 * @param use_heatbath Enable Heat-Bath pruning
 * @return HamBlocks{H_VV, H_VP, P_index_map}
 */
[[nodiscard]] HamBlocks get_ham_conn(
    std::span<const Det> dets_V,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1 = 1e-6,
    bool use_heatbath = true
);

/**
 * Build H_VV and discover connected space via amplitude-weighted screening.
 *
 * Policy: DynamicAmp (per-row adaptive cutoff)
 *  - Doubles: Retain if |⟨ij||ab⟩| ≥ τ_i where τ_i = ε_1/max(|ψ_V[i]|, 1e-10)
 *  - Singles: Retain if |H_ik · ψ_V[i]| ≥ ε_1
 *  - Prioritizes excitations from large-amplitude configurations
 *
 * @param dets_V   Variational determinants
 * @param psi_V    Wavefunction amplitudes aligned with dets_V
 * @param ham      Hamiltonian evaluator
 * @param n_orb    Number of spatial orbitals
 * @param hb_table Heat-Bath integral table (required)
 * @param eps1     Amplitude criterion threshold ε_1
 * @return HamBlocks{H_VV, H_VP, P_index_map}
 */
[[nodiscard]] HamBlocks get_ham_conn_amp(
    std::span<const Det> dets_V,
    std::span<const double> psi_V,
    const HamEval& ham,
    int n_orb,
    const HeatBathTable* hb_table,
    double eps1 = 1e-6
);

} // namespace detnqs
