// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file constants.hpp
 * @brief Framework-wide compile-time constants.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include "types.hpp"
#include <limits>

namespace lever {

// ============================================================================
// Orbital Indexing Limits
// ============================================================================

/**
 * Maximum spatial orbitals encodable in 64-bit bitmask.
 * Orbital indices: [0, 63]
 */
inline constexpr int MAX_ORBITALS_U64 = 64;

/**
 * Sentinel for invalid determinant index.
 * Used in lookup failures and uninitialized states.
 */
inline constexpr u32 INVALID_INDEX = std::numeric_limits<u32>::max();

// ============================================================================
// Slater-Condon Rules
// ============================================================================

/**
 * Maximum excitation degree with non-zero ⟨Φ_i|Ĥ|Φ_j⟩.
 *
 * Degree classification:
 *   0 → Diagonal element  ⟨Φ|Ĥ|Φ⟩
 *   1 → Single excitation ⟨Φ|Ĥ|Φ_i^a⟩
 *   2 → Double excitation ⟨Φ|Ĥ|Φ_ij^ab⟩
 *
 * Higher degrees yield zero by Slater-Condon rules.
 */
inline constexpr u8 MAX_EXCITATION_DEGREE = 2;

} // namespace lever
