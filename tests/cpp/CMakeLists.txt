# Copyright 2025 The DetNQS Authors
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# C++ Unit Tests Configuration
# Author: Zheng (Alex) Che, email: wsmxcz@gmail.com
# Date: November, 2025
# ==============================================================================

# Test executable with source files
add_executable(detnqs_cpp_tests
    test_determinant.cpp
    test_integral.cpp
    test_hamiltonian.cpp
)

# ==============================================================================
# Test Data Files
# ==============================================================================

set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/benchmark/FCIDUMP)
set(TEST_DATA_FILES
    ${TEST_DATA_DIR}/H2_631g_2.00.FCIDUMP
    ${TEST_DATA_DIR}/H2O_sto3g.FCIDUMP
)

# Validate and copy test data files
foreach(TEST_FILE ${TEST_DATA_FILES})
    if(NOT EXISTS ${TEST_FILE})
        message(FATAL_ERROR "Missing test data: ${TEST_FILE}")
    endif()

    add_custom_command(
        TARGET detnqs_cpp_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${TEST_FILE}"
                "$<TARGET_FILE_DIR:detnqs_cpp_tests>"
        COMMENT "Copying ${TEST_FILE}"
    )
endforeach()

# ==============================================================================
# Dependencies and Includes
# ==============================================================================

target_link_libraries(detnqs_cpp_tests PRIVATE
    detnqs::core
    Catch2::Catch2WithMain
)

target_include_directories(detnqs_cpp_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/detnqs
)

# ==============================================================================
# CTest Integration
# ==============================================================================

include(CTest)
include(Catch)
catch_discover_tests(detnqs_cpp_tests
    PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:detnqs_cpp_tests>"
)
