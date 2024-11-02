cmake_minimum_required(VERSION 3.16)
project(BALDES LANGUAGES C CXX)

# Check if Clang is available
find_program(CLANG_C_COMPILER clang)
find_program(CLANG_CXX_COMPILER clang++)

if(CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
  # Set Clang as the default compiler
  set(CMAKE_C_COMPILER
      "${CLANG_C_COMPILER}"
      CACHE STRING "" FORCE)
  set(CMAKE_CXX_COMPILER
      "${CLANG_CXX_COMPILER}"
      CACHE STRING "" FORCE)
  message(STATUS "Clang found: using Clang as the default compiler")
else()
  message(STATUS "Clang not found: using default compiler")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  string(REPLACE "-fconcepts-diagnostics-depth=10" "" CMAKE_CXX_FLAGS
                 "${CMAKE_CXX_FLAGS}")
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable Interprocedural Optimization (Link Time Optimization)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API_BOTH TRUE)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "ccache found: using ccache for compilation")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
endif()

# Include directories
include_directories(include)
include_directories(include/bucket)
include_directories(include/algebra)
include_directories(include/cuts)
include_directories(include/extra)
include_directories(include/utils)
include_directories(third_party/hgs_vrptw)

# Build types and flags
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "RelWithDebInfo"
      CACHE STRING "Choose the type of build." FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -Wall -Wextra")

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++20")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include the generated config.h file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

option(RIH "Enable RIH compilation option" OFF)
option(RCC "Enable RCC compilation option" ON)
option(SRC "Enable SRC compilation option" ON)
# option(GET_TBB "Enable TBB compilation option" OFF)
option(UNREACHABLE_DOMINANCE "Enable Unreachable Dominance compilation option"
       OFF)
option(SORTED_LABELS "Enable Sorted Labels compilation option" OFF)
option(MCD "Enable MCD compilation option" OFF)
option(FIX_BUCKETS "Enable Fixed Buckets compilation option" ON)
option(AVX "Enable AVX compilation option" OFF)
option(IPM "Enable IPM compilation option" ON)
option(TR "Enable TR compilation option" OFF)
option(GET_SUITESPARSE "Enable SuiteSparse compilation option" OFF)
option(EXACT_RCC "Enable Exact RCC compilation option" OFF)
option(EVRP "Enable EVRPTW compilation option" OFF)
option(WITH_PYTHON "Enable Python bindings" OFF)
option(MTW "Enable MTW compilation option" OFF)
option(PSTEP "Enable PSTEP compilation option" OFF)
option(SCHRODINGER "Enable Schrodinger compilation option" OFF)
option(CLIQUER "Enable Cliquer compilation option" OFF)
option(VERBOSE "Enable verbose output" OFF)
option(JEMALLOC "Enable jemalloc" ON)
option(GUROBI "Enable GUROBI" OFF)
option(HIGHS "Enable HIGHS" ON)
option(NSYNC "Enable nsync" OFF)
option(CHOLMOD "Enable cholmod" OFF)
option(IPM_ACEL "Enable ipm acelleration" OFF)
option(BALDES "Enable baldes compilation" ON)
option(HGS "Enable HGS" OFF)

# Resource Definitions
set(R_SIZE
    1
    CACHE STRING "Number of resources")
set(N_SIZE
    102
    CACHE STRING "Number of nodes")
set(MAX_SRC_CUTS
    50
    CACHE STRING "Maximum number of SRC cuts")
set(BUCKET_CAPACITY
    50
    CACHE STRING "Bucket capacity")
set(TIME_INDEX
    0
    CACHE STRING "Time index")
set(N_ADD
    10
    CACHE STRING "Number of additional nodes")
set(HGS_TIME
    2
    CACHE STRING "HGS max running time")

# Git Commit Hash
execute_process(
  COMMAND git rev-parse --short HEAD
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-DGIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

add_definitions(-DN_SIZE=${N_SIZE})
add_definitions(-DR_SIZE=${R_SIZE})
add_definitions(-DMAX_SRC_CUTS=${MAX_SRC_CUTS})
add_definitions(-DBUCKET_CAPACITY=${BUCKET_CAPACITY})
add_definitions(-DTIME_INDEX=${TIME_INDEX})
add_definitions(-DN_ADD=${N_ADD})

# if IPM is disabled, force STAB to be enabled
if(NOT IPM)
  message(STATUS "IPM is disabled, enabling STAB")
  set(STAB ON)
else()
  message(STATUS "IPM is enabled, disabling STAB")
  set(STAB OFF)
endif()

# Configure the config.h file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in # Input file (config.h.in)
  ${CMAKE_CURRENT_BINARY_DIR}/config.h # Output file (config.h)
)

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

find_package(xxhash)
if(NOT xxhash_FOUND)
  cpmaddpackage(
    NAME
    xxhash
    GITHUB_REPOSITORY
    Cyan4973/xxHash
    GIT_TAG
    origin/dev
    SOURCE_SUBDIRECTORY
    cmake_unofficial # Specify the folder with CMake configuration
  )
  include_directories(${xxhash_SOURCE_DIR})
  add_subdirectory(${xxhash_SOURCE_DIR}/cmake_unofficial)
endif()

cpmaddpackage(NAME ankerl_unordered_dense GITHUB_REPOSITORY
              martinus/unordered_dense GIT_TAG origin/main)
include_directories(${ankerl_unordered_dense_SOURCE_DIR}/include)

if(NSYNC)
  set(nsync_ADDED ON)
  cpmaddpackage(
    NAME
    nsync
    GITHUB_REPOSITORY
    google/nsync
    GIT_TAG
    origin/master
    OPTIONS
    "NSYNC_ENABLE_TESTS OFF")
endif()

find_package(FMT)
if(NOT FMT_FOUND)
  message(STATUS "FMT not found, using bundled version")
  cpmaddpackage(NAME fmt GITHUB_REPOSITORY fmtlib/fmt GIT_TAG 11.0.2)
endif()

if(IPM)

endif()

if(HIGHS)
  cpmaddpackage(
    NAME
    Highs
    GITHUB_REPOSITORY
    ERGO-Code/HiGHS
    GIT_TAG
    origin/master
    OPTIONS
    "BUILD_SHARED_LIBS ON"
    "BUILD_EXAMPLES OFF"
    "BUILD_CXX_EXAMPLE OFF"
    "BUILD_CSHARP_EXAMPLE OFF"
    "BUILD_TESTING OFF"
    "USE_DOTNET_STD_21 OFF")
  include_directories(${Highs_SOURCE_DIR}/src ${Highs_INCLUDE_DIR})

  # Path to your patch file
  set(PATCH_FILE "${CMAKE_CURRENT_LIST_DIR}/highs.patch")
  # echo the patch file
  message(STATUS "Patch file: ${PATCH_FILE}")

  # Apply the patch using execute_process
  execute_process(
    COMMAND patch -p1 -i ${PATCH_FILE}
    WORKING_DIRECTORY ${Highs_SOURCE_DIR}
    RESULT_VARIABLE PATCH_RESULT
    ERROR_VARIABLE PATCH_ERROR)

endif()

if(WITH_PYTHON)
  cpmaddpackage(NAME pybind11 GITHUB_REPOSITORY pybind/pybind11 GIT_TAG v2.13.6)
endif()

set(GUROBI_HOME $ENV{GUROBI_HOME})
include_directories(${GUROBI_HOME}/include)

# Source files
set(SOURCES src/BCP.cpp src/BucketGraph.cpp src/MIPHandler.cpp
            src/Branching.cpp)

set(CVRPSEP_SOURCES
    third_party/cvrpsep/basegrph.cpp
    third_party/cvrpsep/capsep.cpp
    third_party/cvrpsep/cnstrmgr.cpp
    third_party/cvrpsep/memmod.cpp
    third_party/cvrpsep/compcuts.cpp
    third_party/cvrpsep/strngcmp.cpp
    third_party/cvrpsep/compress.cpp
    third_party/cvrpsep/cutbase.cpp
    third_party/cvrpsep/fcapfix.cpp
    third_party/cvrpsep/mxf.cpp
    # third_party/cvrpsep/twomatch.cpp third_party/cvrpsep/glmsep.cpp
    # third_party/cvrpsep/binpack.cpp
    third_party/cvrpsep/combsep.cpp
    third_party/cvrpsep/fcits.cpp
    third_party/cvrpsep/grsearch.cpp
    # third_party/cvrpsep/hpmstar.cpp third_party/cvrpsep/strcomb.cpp
    # third_party/cvrpsep/blocks.cpp
    third_party/cvrpsep/sort.cpp)

set(HGS_SOURCES
    third_party/hgs_vrptw/Genetic.cpp third_party/hgs_vrptw/Individual.cpp
    third_party/hgs_vrptw/LocalSearch.cpp third_party/hgs_vrptw/Params.cpp
    third_party/hgs_vrptw/Population.cpp third_party/hgs_vrptw/Split.cpp)

set(FPSOURCE
    third_party/fpmax/fpmax.cpp third_party/fpmax/buffer.cpp
    third_party/fpmax/data.cpp third_party/fpmax/fsout.cpp
    third_party/fpmax/fp_tree.cpp third_party/fpmax/fp_node.cpp)

set(IPM_SOURCES src/IPSolver.cpp)
set(SRC_SOURCES src/SRC.cpp src/HighOrderSRC.cpp)

# Define a list for the source files that will be included
set(EXECUTABLE_SOURCES ${SOURCES} ${HGS_SOURCES} ${FPSOURCE})

# Append sources based on the flags
if(RCC)
  message(STATUS "RCC is set, adding CVRPSEP_SOURCES.")
  list(APPEND EXECUTABLE_SOURCES ${CVRPSEP_SOURCES})
  list(APPEND EXECUTABLE_SOURCES src/RCC.cpp)
endif()

if(IPM OR IPM_ACEL)
  message(STATUS "IPM is set, adding IPM_SOURCES.")
  list(APPEND EXECUTABLE_SOURCES ${IPM_SOURCES})
endif()

if(SRC)
  message(STATUS "SRC is set, adding SRC_SOURCES.")
  list(APPEND EXECUTABLE_SOURCES ${SRC_SOURCES})
endif()

if(IPM OR IPM_ACEL)
  cpmaddpackage(
    NAME
    Eigen
    GIT_REPOSITORY
    https://gitlab.com/libeigen/eigen.git
    GIT_TAG
    3.4.0
    DOWNLOAD_ONLY
    YES # Only download Eigen, don't configure or build it
  )
endif()

# Common Libraries List
set(COMMON_LIBRARIES STDEXEC::stdexec fmt::fmt xxhash)

if(HIGHS)
  list(APPEND COMMON_LIBRARIES highs::highs)
endif()

if(JEMALLOC)
  find_library(JEMALLOC_LIBRARY jemalloc)
  if(JEMALLOC_LIBRARY)
    message(STATUS "jemalloc found: ${JEMALLOC_LIBRARY}")
    list(APPEND COMMON_LIBRARIES ${JEMALLOC_LIBRARY})
  else()
    message(STATUS "jemalloc not found")
  endif()
endif()

if(NSYNC)
  list(APPEND COMMON_LIBRARIES nsync)
  include_directories(${nsync_SOURCE_DIR}/public)
endif()

set(GUROBI_VERSION_MAJOR 110) # Adjust based on Gurobi version

if(APPLE)
  # macOS detected, use .dylib for linking
  list(APPEND COMMON_LIBRARIES ${GUROBI_HOME}/lib/libgurobi_c++.a
       ${GUROBI_HOME}/lib/libgurobi100.dylib)
else()
  # Non-macOS (Linux or other platforms), use .so and .a files

  list(APPEND COMMON_LIBRARIES ${GUROBI_HOME}/lib/libgurobi_c++.a
       ${GUROBI_HOME}/lib/libgurobi${GUROBI_VERSION_MAJOR}.so)
endif()

if(BALDES)
  # Add the executable with the combined sources
  add_executable(baldes examples/Baldes.cpp ${EXECUTABLE_SOURCES})

  message(STATUS "Gurobi version: ${GUROBI_VERSION_MAJOR}")
  target_compile_options(baldes PRIVATE -Wno-c++26-extensions
                                        -Wno-deprecated-declarations)

  target_precompile_headers(
    baldes
    PRIVATE
    include/Common.h
    ${CVRPSEP_HEADERS}
    ${FPMAX_HEADERS}
    include/Arc.h
    include/Dual.h
    include/VRPNode.h
    third_party/pdqsort.h)

  if(IPM)
    target_include_directories(baldes PRIVATE ${Eigen_SOURCE_DIR})
    target_include_directories(baldes PRIVATE ${Eigen_SOURCE_DIR}/unsupported)
  endif()

  target_link_libraries(baldes PRIVATE ${COMMON_LIBRARIES})

endif()

if(CLIQUER)
  add_executable(cliquer src/Original.cpp src/BucketGraph.cpp)
endif()

if(MTW)
  add_executable(vrpmtw mtw/VRPMTW.cpp ${EXECUTABLE_SOURCES} ${HGS_SOURCES}
                        ${FPSOURCE})
  target_compile_options(vrpmtw PRIVATE -Wno-c++26-extensions
                                        -Wno-deprecated-declarations)
  target_link_libraries(vrpmtw PRIVATE ${COMMON_LIBRARIES})
endif()

if(WITH_PYTHON)
  pybind11_add_module(pybaldes src/BucketBindings.cpp src/BucketGraph.cpp
                      ${EXECUTABLE_SOURCES})
  target_link_libraries(pybaldes PRIVATE ${COMMON_LIBRARIES})
  set_target_properties(pybaldes PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(pybaldes PRIVATE -Wno-c++26-extensions
                                          -Wno-deprecated-declarations)
endif()

if(HGS)
  add_executable(hgs src/HGS.cpp ${HGS_SOURCES} ${FPSOURCE})
  target_link_libraries(hgs PRIVATE ${COMMON_LIBRARIES})
  target_compile_options(hgs PRIVATE -Wno-c++26-extensions
                                     -Wno-deprecated-declarations)
endif()
