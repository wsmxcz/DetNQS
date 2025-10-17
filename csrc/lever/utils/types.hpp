// Copyright 2025 The LEVER Authors
// SPDX-License-Identifier: Apache-2.0

/**
 * @file types.hpp
 * @brief Platform-independent type definitions for LEVER
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <bit>

namespace lever {

// Fixed-width unsigned integers (bitmask operations)
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// Fixed-width signed integers (indices, offsets)
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

// Floating-point types
using f32 = float;
using f64 = double;

// Platform compatibility assertions
static_assert(sizeof(f64) == 8, "64-bit double required");
static_assert(sizeof(i32) >= 4, "32-bit int minimum required");
static_assert(std::endian::native == std::endian::little, 
              "Little-endian platform required for JAX/XLA compatibility");

} // namespace lever
