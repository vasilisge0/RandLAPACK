include(CMakeFindDependencyMacro)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

set(RandLAPACK_VERSION "0.0.0")
set(RandLAPACK_VERSION_MAJOR "0")
set(RandLAPACK_VERSION_MINOR "0")
set(RandLAPACK_VERSION_PATCH "0")

# randblas
if (NOT RandBLAS_DIR)
    # RandBLAS currently installs to lib/cmake/ not lib/cmake/RandBLAS/
    set(RandBLAS_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
endif()
find_dependency(RandBLAS)

# lapack++
if (NOT lapackpp_DIR)
    set(lapackpp_DIR "/global/homes/v/vgeorgio/lapackpp_install/lib64/cmake/lapackpp")
endif()
find_dependency(lapackpp)

include("${CMAKE_CURRENT_LIST_DIR}/RandLAPACKTargets.cmake")
