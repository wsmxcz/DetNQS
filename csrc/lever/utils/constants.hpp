// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file constants.hpp
 * @brief Compile-time constants for LEVER framework
 */

#pragma once

#include "types.hpp"
#include <limits>

namespace lever {

// ============================================================================
// Orbital and Basis Limits
// ============================================================================

/// Maximum spatial orbitals supported by u64 bitmask (0-63)
inline constexpr int MAX_ORBITALS_U64 = 64;

/// Sentinel value for invalid/not-found determinant index
inline constexpr u32 INVALID_INDEX = std::numeric_limits<u32>::max();

// ============================================================================
// Excitation Degree Limits (Slater-Condon Rules)
// ============================================================================

/// Maximum excitation degree with non-zero Hamiltonian matrix elements
/// (0: diagonal, 1: singles, 2: doubles)
inline constexpr u8 MAX_EXCITATION_DEGREE = 2;

} // namespace lever
