# Copyright 2025 The LEVER Authors - All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CMake configuration for LEVER C++ extensions and JAX FFI modules.

# Builds:
#   - lever_core: Static library (determinant, integrals, Hamiltonian)
#   - _lever_cpp: Nanobind Python bindings
#   - _lever_ffi: JAX FFI module (CPU + optional CUDA kernels)

# File: csrc/CMakeLists.txt
# Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
# Date: December, 2025


# -----------------------------------------------------------------------------
# Build Options
# -----------------------------------------------------------------------------
option(LEVER_ENABLE_CUDA "Enable CUDA kernels for JAX FFI" ON)
option(LEVER_ENABLE_CUSOLVERDX "Enable cuSolverDx fused kernels" ON)

# -----------------------------------------------------------------------------
# Core Static Library
# -----------------------------------------------------------------------------
# Provides determinant algebra, MO integrals, Hamiltonian construction
add_library(lever_core STATIC
    src/determinant/det_enum.cpp
    src/determinant/det_ops.cpp
    src/determinant/det_space.cpp
    src/determinant/det_batch.cpp
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

target_link_libraries(lever_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

set_target_properties(lever_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# -----------------------------------------------------------------------------
# Python Bindings (Nanobind)
# -----------------------------------------------------------------------------
nanobind_add_module(_lever_cpp SHARED bridge/bridge.cpp)
target_link_libraries(_lever_cpp PRIVATE lever::core)
target_include_directories(_lever_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/lever)

# -----------------------------------------------------------------------------
# JAX FFI Extension
# -----------------------------------------------------------------------------
set(FFI_SOURCES ffi/ffi_module.cpp ffi/logdet_cpu.cc)

# Configure CUDA support
if(LEVER_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc" CACHE FILEPATH "" FORCE)
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "" FORCE)
  
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DLEVER_WITH_CUDA=1)

    # Optional cuSolverDx integration (fused batched LU for log|A|)
    if(LEVER_ENABLE_CUSOLVERDX)
        if(DEFINED ENV{CONDA_PREFIX})
            list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
        endif()

        find_package(mathdx REQUIRED COMPONENTS cusolverdx CONFIG)
        add_definitions(-DLEVER_WITH_CUSOLVERDX=1)

        message(STATUS "MathDx version: ${mathdx_VERSION}")
        message(STATUS "cuSolverDx include: ${cusolverdx_INCLUDE_DIRS}")
        message(STATUS "cuSolverDx fatbin: ${cusolverdx_FATBIN}")
    endif()

    list(APPEND FFI_SOURCES ffi/logdet_cuda.cu)
endif()

# Build FFI module
nanobind_add_module(_lever_ffi SHARED ${FFI_SOURCES})

target_include_directories(_lever_ffi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JAXLIB_INCLUDE_DIR}
)

target_link_libraries(_lever_ffi PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)

# CUDA-specific configuration
if(LEVER_ENABLE_CUDA)
    set(LEVER_CUDA_ARCHS "native" CACHE STRING "CUDA architectures (e.g., 80;90;120 or native)")

    set_target_properties(_lever_ffi PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${LEVER_CUDA_ARCHS}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ON
    )

    target_compile_options(_lever_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --expt-extended-lambda
            $<$<BOOL:${LEVER_ENABLE_CUSOLVERDX}>:-dlto>
        >
    )

    target_link_options(_lever_ffi PRIVATE
        $<$<AND:$<LINK_LANGUAGE:CUDA>,$<BOOL:${LEVER_ENABLE_CUSOLVERDX}>>:-dlto>
    )

    target_link_libraries(_lever_ffi PRIVATE CUDA::cudart)

    if(LEVER_ENABLE_CUSOLVERDX)
        if(TARGET mathdx::cusolverdx_fatbin)
            target_link_libraries(_lever_ffi PRIVATE mathdx::cusolverdx_fatbin)
        else()
            target_link_libraries(_lever_ffi PRIVATE mathdx::cusolverdx)
        endif()
    else()
        target_link_libraries(_lever_ffi PRIVATE CUDA::cublas)
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

set(LEVER_PY_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/lever" CACHE PATH
    "Python package installation directory")

install(TARGETS _lever_cpp _lever_ffi
    LIBRARY DESTINATION "${LEVER_PY_PACKAGE_DIR}"
    RUNTIME DESTINATION "${LEVER_PY_PACKAGE_DIR}"
    ARCHIVE DESTINATION "${LEVER_PY_PACKAGE_DIR}"
)