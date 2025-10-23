# file: csrc/CMakeLists.txt

# ============================================================================
# 1. Define the core static library: lever_core
#    This contains all the computational logic and is linked by both the
#    Python module and the C++ tests.
# ============================================================================
add_library(lever_core STATIC
    src/determinant/det_enum.cpp
    src/determinant/det_ops.cpp
    src/determinant/det_space.cpp
    src/determinant/det_enum.cpp
    src/integral/integral_mo.cpp
    src/integral/hb_table.cpp
    src/hamiltonian/ham_eval.cpp
    src/hamiltonian/build_ham.cpp
    src/hamiltonian/ham_eff.cpp
)
add_library(lever::core ALIAS lever_core)

target_include_directories(lever_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link lever_core against its dependencies.
target_link_libraries(lever_core PUBLIC
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
)

# Ensure Position Independent Code, required for linking a static lib
# into a shared library (the Python extension).
set_target_properties(lever_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# 2. Define the Python extension module: _lever_cpp
#    This is a shared library that nanobind uses to expose C++ to Python.
# ============================================================================
# Assuming your bridge code is in csrc/bridge/bridge.cpp
nanobind_add_module(_lever_cpp SHARED
    bridge/bridge.cpp
)

# Link the Python module against our static core library.
target_link_libraries(_lever_cpp PRIVATE
    lever::core
)

# The Python module needs to know where to find the public headers.
target_include_directories(_lever_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lever
)

# --- Install the Python extension directly into the in-tree package dir ---
# This keeps `import lever._lever_cpp` working after a simple `cmake --install .`

include(GNUInstallDirs)

# Allow overrides: e.g. -DLEVER_PY_PACKAGE_DIR=/some/other/dir
set(LEVER_PY_PACKAGE_DIR "${CMAKE_SOURCE_DIR}/lever" CACHE PATH
    "Where to place the Python extension when installing")

# nanobind_add_module() creates a MODULE library suitable as a Python extension.
# Install to the package dir for all platforms (LIBRARY for *nix, RUNTIME for Windows).
install(TARGETS _lever_cpp
    LIBRARY DESTINATION "${LEVER_PY_PACKAGE_DIR}"
    RUNTIME DESTINATION "${LEVER_PY_PACKAGE_DIR}"
    ARCHIVE DESTINATION "${LEVER_PY_PACKAGE_DIR}"
)