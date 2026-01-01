# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

# CMake configuration for DetNQS C++ extensions and JAX FFI modules.
#
# Components:
#   - detnqs_core: Determinant algebra, MO integrals, Hamiltonian
#   - _detnqs_cpp: Python bindings (nanobind)
#   - _detnqs_ffi: JAX FFI module (CUDA kernels, optional cuSolverDx)
#
# File: csrc/CMakeLists.txt
# Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
# Date: December, 2025

# -----------------------------------------------------------------------------
# Build Options
# -----------------------------------------------------------------------------
option(detnqs_ENABLE_CUDA "Enable CUDA kernels for JAX FFI" ON)
option(detnqs_ENABLE_CUSOLVERDX "Enable cuSolverDx fused kernels" ON)

# -----------------------------------------------------------------------------
# Core Static Library
# -----------------------------------------------------------------------------
add_library(detnqs_core STATIC
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
add_library(detnqs::core ALIAS detnqs_core)

target_include_directories(detnqs_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(detnqs_core PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)

set_target_properties(detnqs_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# -----------------------------------------------------------------------------
# Python Bindings (Nanobind)
# -----------------------------------------------------------------------------
nanobind_add_module(_detnqs_cpp SHARED bridge/bridge.cpp)
target_link_libraries(_detnqs_cpp PRIVATE detnqs::core)
target_include_directories(_detnqs_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/detnqs)

# -----------------------------------------------------------------------------
# JAX FFI Extension (CUDA-only; CPU uses pure JAX fallback)
# -----------------------------------------------------------------------------
set(FFI_SOURCES ffi/ffi_module.cpp)

if(detnqs_ENABLE_CUDA)
    list(APPEND FFI_SOURCES ffi/logdet_cuda.cu)
endif()

nanobind_add_module(_detnqs_ffi SHARED ${FFI_SOURCES})

target_include_directories(_detnqs_ffi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${JAXLIB_INCLUDE_DIR}
)

if(detnqs_ENABLE_CUDA)
    # CUDA compiler setup
    set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc" CACHE FILEPATH "" FORCE)
    set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "" FORCE)

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-Ddetnqs_WITH_CUDA=1)

    # Single-arch build enforcement
    set(detnqs_CUDA_ARCHS "native" CACHE STRING "CUDA arch (80/90/120 or native)")

    if(detnqs_CUDA_ARCHS MATCHES ";")
        message(FATAL_ERROR "detnqs_CUDA_ARCHS must be a single arch (e.g., 90) or 'native'.")
    endif()

    set_target_properties(_detnqs_ffi PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${detnqs_CUDA_ARCHS}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(_detnqs_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )

    target_link_libraries(_detnqs_ffi PRIVATE CUDA::cudart)

    # cuSolverDx integration (optional fused kernels)
    if(detnqs_ENABLE_CUSOLVERDX)
        if(DEFINED ENV{CONDA_PREFIX})
            list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/nvidia/mathdx/25.12")
            set(mathdx_ROOT "$ENV{CONDA_PREFIX}/nvidia/mathdx/25.12" CACHE PATH "" FORCE)
        endif()

        find_package(mathdx REQUIRED COMPONENTS cusolverdx CONFIG)
        add_definitions(-Ddetnqs_WITH_CUSOLVERDX=1)

        # Determine SM code for cuSolverDx template instantiation
        set(detnqs_CUSOLVERDX_SM "" CACHE STRING "cuSolverDx SM code (800/900/1200)")

        if(detnqs_CUSOLVERDX_SM STREQUAL "")
            if(detnqs_CUDA_ARCHS STREQUAL "native")
                # Detect local GPU SM at configure-time
                file(WRITE "${CMAKE_BINARY_DIR}/detnqs_detect_sm.cu"
"#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 2;
    std::printf(\"%d\\n\", p.major * 100 + p.minor * 10);
    return 0;
}
")
                try_run(_RUN_RES _COMPILE_RES
                    "${CMAKE_BINARY_DIR}"
                    "${CMAKE_BINARY_DIR}/detnqs_detect_sm.cu"
                    RUN_OUTPUT_VARIABLE _SM_OUT
                )
                if(_COMPILE_RES AND _RUN_RES EQUAL 0)
                    string(STRIP "${_SM_OUT}" detnqs_CUSOLVERDX_SM)
                else()
                    message(FATAL_ERROR
                        "Failed to detect GPU SM. "
                        "Set -Ddetnqs_CUSOLVERDX_SM=900 or -Ddetnqs_CUDA_ARCHS=90.")
                endif()
            else()
                # Map CUDA arch (e.g., 90) to SM code (900)
                if(detnqs_CUDA_ARCHS MATCHES "^[0-9]+$")
                    math(EXPR detnqs_CUSOLVERDX_SM "${detnqs_CUDA_ARCHS} * 10")
                else()
                    message(FATAL_ERROR
                        "detnqs_CUDA_ARCHS must be numeric (80/90/120) or 'native'. "
                        "Alternatively, set detnqs_CUSOLVERDX_SM explicitly.")
                endif()
            endif()
        endif()

        target_compile_definitions(_detnqs_ffi PRIVATE
            DETNQS_CUSOLVERDX_SM=${detnqs_CUSOLVERDX_SM}
        )

        # Link cuSolverDx library
        if(TARGET mathdx::cusolverdx_fatbin)
            target_link_libraries(_detnqs_ffi PRIVATE mathdx::cusolverdx_fatbin)
        else()
            target_link_libraries(_detnqs_ffi PRIVATE mathdx::cusolverdx)
        endif()

        # Enable device-side LTO for fused kernels
        target_compile_options(_detnqs_ffi PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-dlto>
        )
        target_link_options(_detnqs_ffi PRIVATE "$<DEVICE_LINK:-dlto>")
    endif()
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

set(detnqs_PY_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/detnqs" CACHE PATH
    "Python package installation directory")

install(TARGETS _detnqs_cpp _detnqs_ffi
    LIBRARY DESTINATION "${detnqs_PY_PACKAGE_DIR}"
    RUNTIME DESTINATION "${detnqs_PY_PACKAGE_DIR}"
    ARCHIVE DESTINATION "${detnqs_PY_PACKAGE_DIR}"
)