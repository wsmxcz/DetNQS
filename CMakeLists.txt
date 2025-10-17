cmake_minimum_required(VERSION 3.28)
project(lever LANGUAGES CXX)

# ---- Global C++ standard ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# This variable will be inherited by subdirectories.
set(LEVER_BENCHMARK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmark" CACHE PATH "Path to benchmark data")
message(STATUS "Benchmark data directory set to: ${LEVER_BENCHMARK_DIR}")

# --- Find Dependencies ---
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
    RESULT_VARIABLE nanobind_exec_ret
)
if (nanobind_exec_ret EQUAL 0)
    message(STATUS "nanobind CMake dir detected at: ${nanobind_ROOT}")
    list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}")
else()
    message(WARNING "Could not detect nanobind CMake dir via `python -m nanobind --cmake_dir`")
endif()

find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(Catch2 3 REQUIRED)

# include(FetchContent)
# # --- Fetch ankerl::unordered_dense for high-performance hash maps ---
# FetchContent_Declare(
#     unordered_dense
#     GIT_REPOSITORY https://github.com/martinus/unordered_dense.git
#     GIT_TAG        v4.5.0 # Pin to a specific, known-good version
# )
# FetchContent_MakeAvailable(unordered_dense)

# --- Subdirectories ---
add_subdirectory(csrc)

# --- Enable and Add Tests ---
enable_testing()
add_subdirectory(tests/cpp)