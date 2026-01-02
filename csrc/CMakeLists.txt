# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

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
# JAX FFI Extension
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
    # -------------------------------------------------------------------------
    # CUDA Compiler/Toolkit Setup
    # -------------------------------------------------------------------------
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
            set(CMAKE_CUDA_COMPILER "$ENV{CONDA_PREFIX}/bin/nvcc"
                CACHE FILEPATH "CUDA compiler" FORCE)
        endif()
    endif()

    if(NOT DEFINED CUDAToolkit_ROOT)
        if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/include/cuda.h")
            set(CUDAToolkit_ROOT "$ENV{CONDA_PREFIX}"
                CACHE PATH "CUDA toolkit root" FORCE)
        endif()
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(_detnqs_ffi PRIVATE detnqs_WITH_CUDA=1)

    # -------------------------------------------------------------------------
    # GPU Architecture Selection (User-Specified)
    # -------------------------------------------------------------------------
    set(detnqs_CUDA_ARCHS "" CACHE STRING
        "CUDA architecture (80/90/120). Must be explicitly set.")

    if(detnqs_CUDA_ARCHS STREQUAL "")
        message(FATAL_ERROR
            "detnqs_CUDA_ARCHS not set. Specify target arch (e.g., -Ddetnqs_CUDA_ARCHS=80).")
    endif()

    if(detnqs_CUDA_ARCHS MATCHES ";")
        message(FATAL_ERROR
            "detnqs_CUDA_ARCHS must be a single value, not a list.")
    endif()

    # -------------------------------------------------------------------------
    # CUDA Standard: C++17 for cuSolverDx, C++20 otherwise
    # -------------------------------------------------------------------------
    if(detnqs_ENABLE_CUSOLVERDX)
        # cuSolverDx requires C++17 and LTO
        set(_cuda_std 17)
    else()
        set(_cuda_std 20)
    endif()

    set_target_properties(_detnqs_ffi PROPERTIES
        CUDA_STANDARD ${_cuda_std}
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${detnqs_CUDA_ARCHS}"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_options(_detnqs_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )

    target_link_libraries(_detnqs_ffi PRIVATE CUDA::cudart)

    # -------------------------------------------------------------------------
    # cuSolverDx Integration (Requires LTO)
    # -------------------------------------------------------------------------
    if(detnqs_ENABLE_CUSOLVERDX)
        if(DEFINED ENV{CONDA_PREFIX})
            list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/nvidia/mathdx/25.12")
            set(mathdx_ROOT "$ENV{CONDA_PREFIX}/nvidia/mathdx/25.12" CACHE PATH "" FORCE)
        endif()

        find_package(mathdx REQUIRED COMPONENTS cusolverdx CONFIG)
        target_compile_definitions(_detnqs_ffi PRIVATE detnqs_WITH_CUSOLVERDX=1)

        # ---------------------------------------------------------------------
        # SM Code for cuSolverDx Template Instantiation
        # ---------------------------------------------------------------------
        set(detnqs_CUSOLVERDX_SM "" CACHE STRING
            "cuSolverDx SM code (800/900/1200). Must match 10x CUDA arch.")

        if(detnqs_CUSOLVERDX_SM STREQUAL "")
            message(FATAL_ERROR
                "detnqs_CUSOLVERDX_SM not set. Specify SM code (e.g., -Ddetnqs_CUSOLVERDX_SM=800).")
        endif()

        if(NOT detnqs_CUSOLVERDX_SM MATCHES "^[0-9]+$")
            message(FATAL_ERROR "detnqs_CUSOLVERDX_SM must be numeric (800/900/1200).")
        endif()

        # Consistency check with CUDA_ARCHS
        if(detnqs_CUDA_ARCHS MATCHES "^[0-9]+$")
            math(EXPR _expected_sm "${detnqs_CUDA_ARCHS} * 10")
            if(NOT detnqs_CUSOLVERDX_SM EQUAL _expected_sm)
                message(WARNING
                    "SM mismatch: detnqs_CUSOLVERDX_SM=${detnqs_CUSOLVERDX_SM}, "
                    "expected ${_expected_sm} (10x of detnqs_CUDA_ARCHS=${detnqs_CUDA_ARCHS}).")
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

        # Enable IPO/LTO for cuSolverDx (required for proper linking)
        set_property(TARGET _detnqs_ffi PROPERTY INTERPROCEDURAL_OPTIMIZATION ON)

        # Device link with LTO
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