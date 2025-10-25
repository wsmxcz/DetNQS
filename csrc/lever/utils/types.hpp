// Copyright 2025 The LEVER Authors - All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file types.hpp
 * @brief Platform-independent type aliases and compatibility checks.
 *
 * Provides fixed-width integer/float types for deterministic
 * numerical behavior across platforms. Enforces little-endian
 * requirement for JAX/XLA interoperability.
 *
 * Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
 * Date: November, 2025
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <bit>

namespace lever {

// Unsigned integers - Bitwise operations and masks
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Signed integers - Array indices and offsets
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// Floating-point - Numerical computations
using f32 = float;
using f64 = double;

// Platform compatibility checks
static_assert(sizeof(f64) == 8, "64-bit double required");
static_assert(sizeof(i32) >= 4, "32-bit int minimum required");
static_assert(std::endian::native == std::endian::little,
              "Little-endian required for JAX/XLA compatibility");

} // namespace lever
