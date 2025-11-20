# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build configuration for LEVER C++ core and Python bindings.
#
# Components:
#   - lever_core: Static library with computational kernels
#   - _lever_cpp: Python extension via nanobind
#
# Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
# Date: November, 2025

# ============================================================================
# Core Static Library
# ============================================================================
# Computational kernels for determinants, integrals, and Hamiltonians.
# Position-independent code required for Python extension linkage.

add_library(lever_core STATIC
    src/determinant/det_enum.cpp
    src/determinant/det_ops.cpp
    src/determinant/det_space.cpp
    src/integral/integral_mo.cpp
    src/integral/hb_table.cpp
    src/hamiltonian/ham_eval.cpp
    src/hamiltonian/build_ham.cpp
    src/hamiltonian/local_conn.cpp
    src/hamiltonian/ham_eff.cpp
    src/hamiltonian/ham_utils.cpp
)
add_library(lever::core ALIAS lever_core)

target_include_directories(lever_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(lever_core PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

set_target_properties(lever_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# ============================================================================
# Python Extension Module
# ============================================================================
# Nanobind-based bindings exposing lever_core to Python.

nanobind_add_module(_lever_cpp SHARED
    bridge/bridge.cpp
)

target_link_libraries(_lever_cpp PRIVATE lever::core)

target_include_directories(_lever_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lever
)

# ============================================================================
# Installation
# ============================================================================
# Install extension into in-tree package for `import lever._lever_cpp`.

include(GNUInstallDirs)

set(LEVER_PY_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/lever" CACHE PATH
    "Python package installation directory")

install(TARGETS _lever_cpp
    LIBRARY DESTINATION "${LEVER_PY_PACKAGE_DIR}"
    RUNTIME DESTINATION "${LEVER_PY_PACKAGE_DIR}"
    ARCHIVE DESTINATION "${LEVER_PY_PACKAGE_DIR}"
)
