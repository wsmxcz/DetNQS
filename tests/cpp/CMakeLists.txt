# file: tests/cpp/CMakeLists.txt

add_executable(lever_cpp_tests
    test_determinant.cpp
    test_integral.cpp
    test_hamiltonian.cpp
)

# --- Define a list of all required test data files ---
set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/benchmark/FCIDUMP)
set(TEST_DATA_FILES
    ${TEST_DATA_DIR}/H2_631g_2.00.FCIDUMP
    ${TEST_DATA_DIR}/H2O_sto3g.FCIDUMP
    # Add any other required FCIDUMP files here in the future
)

# --- Loop over the list to check for existence and set up copy commands ---
foreach(TEST_FILE ${TEST_DATA_FILES})
    # Check if each file exists
    if(NOT EXISTS ${TEST_FILE})
        message(FATAL_ERROR "Required test data file not found: ${TEST_FILE}")
    endif()

    # Add a command to copy each file to the build directory
    # where the test executable will run.
    add_custom_command(
        TARGET lever_cpp_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "${TEST_FILE}"
                "$<TARGET_FILE_DIR:lever_cpp_tests>"
        COMMENT "Copying test data: ${TEST_FILE}"
    )
endforeach()

# Link the test executable against our core library and Catch2.
target_link_libraries(lever_cpp_tests PRIVATE
    lever::core
    Catch2::Catch2WithMain
)

# The tests need to find the public headers of our library.
target_include_directories(lever_cpp_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/csrc/lever
)

# Automatically discover and add tests from the executable to CTest.
include(CTest)
include(Catch)
catch_discover_tests(lever_cpp_tests
    PROPERTIES
        # The working directory is where the test executable is located,
        # which is also where we copied the data files.
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:lever_cpp_tests>"
)