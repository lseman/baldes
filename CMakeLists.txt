cmake_minimum_required(VERSION 3.10)

# Project name
project(BALDES)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable Interprocedural Optimization (Link Time Optimization)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# Include directories
include_directories(include)

# Build types and flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING "Choose the type of build." FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -march=native -flto -funroll-loops")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -Wall -Wextra")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Define a config.h.in template
configure_file("config.h.in" "config.h")

# Include the generated config.h file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

option(RIH "Enable RIH compilation option" OFF)
option(RCC "Enable RCC compilation option" OFF)
option(SRC3 "Enable 3SRC compilation option" OFF)
option(SRC "Enable SRC compilation option" OFF)
option(GET_TBB "Enable TBB compilation option" OFF)
option(UNREACHABLE_DOMINANCE "Enable Unreachable Dominance compilation option"
       OFF)
option(SORTED_LABELS "Enable Sorted Labels compilation option" OFF)
option(MCD "Enable MCD compilation option" OFF)
option(FIX_BUCKETS "Enable Fixed Buckets compilation option" OFF)
option(AVX "Enable AVX compilation option" OFF)
option(IPM "Enable IPM compilation option" OFF)
option(TR "Enable TR compilation option" OFF)
option(STAB "Enable STAB compilation option" ON)
option(AUGMENTED "Enable Augmented compilation option" ON)
option(GET_SUITESPARSE "Enable SuiteSparse compilation option" OFF)
option(EXACT_RCC "Enable Exact RCC compilation option" OFF)
# Define the size of resources without using cache
set(R_SIZE 1)
set(N_SIZE 102)
set(MAX_SRC_CUTS 50)
set(BUCKET_CAPACITY 50)
set(TIME_INDEX 0)
set(DEMAND_INDEX 1)
set(MAIN_RESOURCES 1)

# Define resources and whether they are disposable or not
set(RESOURCES
    "Time"
    CACHE STRING "List of resources")
set(RESOURCES_DISPOSABLE
    "1"
    CACHE
      STRING
      "List of flags indicating if each resource is disposable (1 for disposable, 0 for non-disposable)"
)

# Ensure that both SRC3 and SRC are not selected at the same time
if(SRC AND SRC3)
  message(
    FATAL_ERROR
      "Both SRC and SRC3 cannot be enabled at the same time. Please choose only one."
  )
endif()

if(STAB AND IPM)
  message(
    FATAL_ERROR
      "Both STAB and IPM cannot be enabled at the same time. Please choose only one."
  )
endif()

# Attempt to include CPM.cmake from provided path
include(cmake/CPM.cmake OPTIONAL)

# Check if CPM was included, if not, fetch and include CPM
if(NOT COMMAND CPMAddPackage)
  # Include FetchContent module
  include(FetchContent)
  # Declare CPM.cmake as a FetchContent
  FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
    GIT_TAG v0.40.2)
  # Fetch CPM
  FetchContent_MakeAvailable(CPM)
  # Include CPM.cmake after it has been fetched
  include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

cpmaddpackage(
  NAME
  stdexec
  GITHUB_REPOSITORY
  NVIDIA/stdexec
  GIT_TAG
  main
  OPTIONS
  "STDEXEC_BUILD_TESTS OFF"
  "STDEXEC_BUILD_EXAMPLES OFF"
  "STDEXEC_BUILD_BENCHMARKS OFF"
  "STDEXEC_BUILD_DOCS OFF"
  "BUILD_TESTING OFF")

cpmaddpackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.0.2)

if(GET_SUITESPARSE)
  cpmaddpackage(
    NAME
    SuiteSparse
    GITHUB_REPOSITORY
    DrTimothyAldenDavis/SuiteSparse
    GIT_TAG
    v7.8.2
    OPTIONS
    "BUILD_METIS OFF")
  # Specify only CHOLMOD build
  set(SUITESPARSE_BUILDS "CHOLMOD")
  set(SUITESPARSE_CONFIG_ONLY TRUE)
endif()

find_package(TBB)
if(TBB_FOUND)
  message(STATUS "TBB found: ${TBB_VERSION}")
else()
  message(STATUS "TBB not found, will download it")
  # set GET_TBB to true
  set(GET_TBB ON)
endif()

if(GET_TBB)
  cpmaddpackage(
    NAME
    TBB
    GITHUB_REPOSITORY
    oneapi-src/oneTBB
    GIT_TAG
    v2021.13.0
    OPTIONS
    "TBB_TEST OFF"
    "TBB_EXAMPLES OFF"
    "TBB_BUILD_SHARED OFF")
endif()

if(IPM)
  include_directories(/usr/include/suitesparse/)

  cpmaddpackage(NAME Eigen GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG 3.4.0)
  include_directories(${EIGEN_SOURCE_DIR})
endif()

set(GUROBI_HOME $ENV{GUROBI_HOME})

include_directories(${GUROBI_HOME}/include)

# Source files
set(SOURCES examples/VRPTW.cpp src/BucketGraph.cpp src/SRC.cpp
            extra/Heuristic.cpp src/RIH.cpp)

set(CVRPSEP_SOURCES
    cvrpsep/basegrph.cpp
    cvrpsep/capsep.cpp
    cvrpsep/cnstrmgr.cpp
    cvrpsep/memmod.cpp
    cvrpsep/compcuts.cpp
    cvrpsep/strngcmp.cpp
    cvrpsep/compress.cpp
    cvrpsep/cutbase.cpp
    cvrpsep/fcapfix.cpp
    cvrpsep/mxf.cpp
    cvrpsep/twomatch.cpp
    cvrpsep/glmsep.cpp
    cvrpsep/binpack.cpp
    cvrpsep/combsep.cpp
    cvrpsep/fcits.cpp
    cvrpsep/grsearch.cpp
    cvrpsep/hpmstar.cpp
    cvrpsep/strcomb.cpp
    cvrpsep/blocks.cpp
    cvrpsep/sort.cpp)

set(IPM_SOURCES ipm/IPSolver.cpp)

# Define a list for the source files that will be included
set(EXECUTABLE_SOURCES ${SOURCES})

# Append sources based on the flags
if(RCC)
  message(STATUS "RCC is set, adding CVRPSEP_SOURCES.")
  list(APPEND EXECUTABLE_SOURCES ${CVRPSEP_SOURCES})
endif()

if(IPM)
  message(STATUS "IPM is set, adding IPM_SOURCES.")
  list(APPEND EXECUTABLE_SOURCES ${IPM_SOURCES})
endif()

# Add the executable with the combined sources
add_executable(vrptw ${EXECUTABLE_SOURCES})

# Linking libraries
target_link_libraries(vrptw PRIVATE STDEXEC::stdexec)
target_link_libraries(vrptw PRIVATE fmt::fmt)
if(GET_TBB)
  target_link_libraries(vrptw PRIVATE TBB::tbb)
endif()
if(IPM)
  target_link_libraries(vrptw PRIVATE Eigen3::Eigen)
  target_link_libraries(vrptw PRIVATE cholmod)
  find_package(OpenMP)
  # target_link_libraries(vrptw PRIVATE OpenMP::OpenMP_CXX)
endif()

set(GUROBI_VERSION_MAJOR 110) # Adjust based on Gurobi version
message(STATUS "Gurobi version: ${GUROBI_VERSION_MAJOR}")

# Link Gurobi libraries
target_link_libraries(
  vrptw PRIVATE ${GUROBI_HOME}/lib/libgurobi_c++.a
                ${GUROBI_HOME}/lib/libgurobi${GUROBI_VERSION_MAJOR}.so tbb)
