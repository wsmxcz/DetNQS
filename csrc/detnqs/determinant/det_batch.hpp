// Copyright 2025 The DetNQS Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * CPU batch preparation for determinant features used by neural models.
 *
 * Computes features for determinant batches in S∪C space:
 *  - occ: occupied spin-orbital indices (alpha ascending, beta ascending + n_orb)
 *  - k: excitation rank k = |holes| = |particles| relative to reference
 *  - phase: fermionic sign computed via canonical ordering:
 *           annihilate holes (DESC) then create particles (ASC)
 *  - holes/particles: k-excitation orbital indices, zero-padded to kmax
 *  - hp_mask: validity mask with first k entries true
 *  - holes_pos/parts_pos: position indices in reference occ/virt lists
 *
 * This file contains core C++ logic with no nanobind bindings.
 *
 * File: detnqs/det_batch/det_batch.hpp
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: December, 2025
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <detnqs/determinant/det.hpp>
#include <detnqs/utils/types.hpp>

namespace detnqs::det_batch {

struct PrepareOptions {
    int  kmax = 0;           // Maximum excitation rank to support
    bool need_k = false;      // Compute excitation rank k
    bool need_phase = false;  // Compute fermionic phase (±1)
    bool need_hp = false;     // Compute holes/particles + hp_mask
    bool need_hp_pos = false; // Compute position indices (requires need_hp)
};

/**
 * Prepare determinant batch features into caller-provided buffers.
 *
 * Input:
 *  - det_pairs: [alpha_0, beta_0, alpha_1, beta_1, ...] of length 2B
 *  - ref: reference determinant for excitation analysis
 *  - n_orb, n_alpha, n_beta: system parameters
 *
 * Required output:
 *  - occ_out: int32[B, n_alpha + n_beta]
 *
 * Optional outputs (pass nullptr to skip):
 *  - k_out:         int8[B]
 *  - phase_out:     int8[B]
 *  - holes_out:     int32[B, kmax]
 *  - parts_out:     int32[B, kmax]
 *  - hp_mask_out:   bool[B, kmax]
 *  - holes_pos_out: int32[B, kmax]
 *  - parts_pos_out: int32[B, kmax]
 *
 * Throws std::invalid_argument on invalid electron counts or buffer shapes.
 */
void prepare_det_batch(
    const u64* det_pairs,
    std::size_t B,
    Det ref,
    int n_orb,
    int n_alpha,
    int n_beta,
    const PrepareOptions& opt,
    int32_t* occ_out,
    int8_t*  k_out,
    int8_t*  phase_out,
    int32_t* holes_out,
    int32_t* parts_out,
    bool*    hp_mask_out,
    int32_t* holes_pos_out,
    int32_t* parts_pos_out
);

} // namespace detnqs::det_batch
