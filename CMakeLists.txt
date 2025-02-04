cmake_minimum_required(VERSION 3.16)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "${CCACHE_PROGRAM}")
    message(STATUS "Using ccache for faster compilation: ${CCACHE_PROGRAM}")
else()
    message(WARNING "ccache not found, compilation will not be cached.")
endif()

project(BALDES LANGUAGES C CXX)

#-------------------------------------------------------------------------------
# Compiler Settings
#-------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CUDA_STANDARD "")
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Check for Clang
find_program(CLANG_C_COMPILER clang)
find_program(CLANG_CXX_COMPILER clang++)

if(CLANG_C_COMPILER AND CLANG_CXX_COMPILER)
  set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}" CACHE STRING "" FORCE)
  set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}" CACHE STRING "" FORCE)
  message(STATUS "Clang found: using Clang as the default compiler")
else()
  message(STATUS "Clang not found: using default compiler")
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  string(REPLACE "-fconcepts-diagnostics-depth=10" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -Xclang -Wno-unknown-warning-option")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -march=native -flto -funroll-loops -ftree-vectorize -fomit-frame-pointer -ffunction-sections -fdata-sections -fno-signed-zeros -fno-trapping-math -mtune=native -fvectorize -Xclang -Wno-unknown-warning-option")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g -Wall -Wextra -Xclang -Wno-unknown-warning-option")

if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -std=c++20")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unknown-argument -Wno-error=unused-command-line-argument")

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
include(FetchContent)

# CPM.cmake
if(NOT COMMAND CPMAddPackage)
  FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
    GIT_TAG v0.40.5)
  FetchContent_MakeAvailable(CPM)
  include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

# stdexec
cpmaddpackage(
  NAME stdexec
  GITHUB_REPOSITORY NVIDIA/stdexec
  GIT_TAG main
  OPTIONS "STDEXEC_BUILD_TESTS OFF" "STDEXEC_BUILD_EXAMPLES OFF" "STDEXEC_BUILD_BENCHMARKS OFF" "STDEXEC_BUILD_DOCS OFF" "BUILD_TESTING OFF"
)

# fmt
cpmaddpackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 11.1.3
  OPTIONS "FMT_DOC OFF" "FMT_TEST OFF" "FMT_INSTALL OFF"
)

# # mdspan
# cpmaddpackage(
#   NAME mdspan
#   GITHUB_REPOSITORY kokkos/mdspan
#   GIT_TAG stable
#   OPTIONS "MDSPAN_BUILD_TESTS OFF" "MDSPAN_BUILD_EXAMPLES OFF" "MDSPAN_BUILD_BENCHMARKS OFF" "MDSPAN_BUILD_DOCS OFF" "BUILD_TESTING OFF"
# )

# xxhash
find_package(xxhash QUIET)
if(NOT xxhash_FOUND)
  cpmaddpackage(
    NAME xxhash
    GITHUB_REPOSITORY Cyan4973/xxHash
    GIT_TAG origin/dev
    SOURCE_SUBDIRECTORY cmake_unofficial
  )
endif()

# unordered_dense
cpmaddpackage(
  NAME ankerl_unordered_dense
  GITHUB_REPOSITORY martinus/unordered_dense
  GIT_TAG origin/main
)

# libcuckoo
cpmaddpackage(
  NAME libcuckoo
  GITHUB_REPOSITORY efficient/libcuckoo
  GIT_TAG origin/master
)

# Highs
if(HIGHS)
  cpmaddpackage(
    NAME Highs
    GITHUB_REPOSITORY ERGO-Code/HiGHS
    GIT_TAG origin/master
    OPTIONS "BUILD_SHARED_LIBS ON" "BUILD_EXAMPLES OFF" "BUILD_CXX_EXAMPLE OFF" "BUILD_CSHARP_EXAMPLE OFF" "BUILD_TESTING OFF" "USE_DOTNET_STD_21 OFF"
  )
endif()

# pybind11
if(WITH_PYTHON)
  cpmaddpackage(
    NAME pybind11
    GITHUB_REPOSITORY pybind/pybind11
    GIT_TAG v2.13.6
  )
endif()

if(IPM OR IPM_ACEL)
  cpmaddpackage(
    NAME
    Eigen
    GIT_REPOSITORY
    https://gitlab.com/libeigen/eigen.git
    GIT_TAG
    nightly
    DOWNLOAD_ONLY
    YES # Only download Eigen, don't configure or build it
  )
endif()

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


#-------------------------------------------------------------------------------
# Project Configuration
#-------------------------------------------------------------------------------
option(RIH "Enable RIH compilation option" OFF)
option(RCC "Enable RCC compilation option" ON)
option(SRC "Enable SRC compilation option" ON)
option(UNREACHABLE_DOMINANCE "Enable Unreachable Dominance compilation option" OFF)
option(SORTED_LABELS "Enable Sorted Labels compilation option" ON)
option(MCD "Enable MCD compilation option" OFF)
option(FIX_BUCKETS "Enable Fixed Buckets compilation option" ON)
option(IPM "Enable IPM compilation option" ON)
option(TR "Enable TR compilation option" OFF)
option(EXACT_RCC "Enable Exact RCC compilation option" OFF)
option(EVRP "Enable EVRPTW compilation option" OFF)
option(WITH_PYTHON "Enable Python bindings" OFF)
option(MTW "Enable MTW compilation option" OFF)
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
option(CUSTOM_COST "Enable custom cost" OFF)

# Resource Definitions
set(R_SIZE 1 CACHE STRING "Number of resources")
set(N_SIZE 102 CACHE STRING "Number of nodes")
set(MAX_SRC_CUTS 50 CACHE STRING "Maximum number of SRC cuts")
set(BUCKET_CAPACITY 100 CACHE STRING "Bucket capacity")
set(TIME_INDEX 0 CACHE STRING "Time index")
set(N_ADD 10 CACHE STRING "Number of additional nodes")
set(HGS_TIME 2 CACHE STRING "HGS max running time")
set(VERBOSE_LEVEL 0 CACHE STRING "Verbose level")

# Git Commit Hash
execute_process(
  COMMAND git rev-parse --short HEAD
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-DGIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

# Configure the config.h file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

#-------------------------------------------------------------------------------
# Source Files
#-------------------------------------------------------------------------------
set(SOURCES
    src/BCP.cpp
    src/BucketGraph.cpp
    src/MIPHandler.cpp
    src/Branching.cpp
    src/BucketPSTEP.cpp
)

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
    third_party/cvrpsep/combsep.cpp
    third_party/cvrpsep/fcits.cpp
    third_party/cvrpsep/grsearch.cpp
    third_party/cvrpsep/sort.cpp
)

set(HGS_SOURCES
    third_party/hgs_vrptw/Genetic.cpp
    third_party/hgs_vrptw/Individual.cpp
    third_party/hgs_vrptw/LocalSearch.cpp
    third_party/hgs_vrptw/Params.cpp
    third_party/hgs_vrptw/Population.cpp
    third_party/hgs_vrptw/Split.cpp
)

set(FPSOURCE
    third_party/fpmax/fpmax.cpp
    third_party/fpmax/buffer.cpp
    third_party/fpmax/data.cpp
    third_party/fpmax/fsout.cpp
    third_party/fpmax/fp_tree.cpp
    third_party/fpmax/fp_node.cpp
)

set(MUSSPSOURCE
    third_party/muSSP/Graph.cpp
    third_party/muSSP/Sink.cpp
)

set(IPM_SOURCES src/IPSolver.cpp)
set(SRC_SOURCES src/SRC.cpp src/HighOrderSRC.cpp)

# Define a list for the source files that will be included
set(EXECUTABLE_SOURCES ${SOURCES} ${HGS_SOURCES} ${FPSOURCE} ${MUSSPSOURCE})

if(RCC)
  list(APPEND EXECUTABLE_SOURCES ${CVRPSEP_SOURCES} src/RCC.cpp)
endif()

if(IPM OR IPM_ACEL)
  list(APPEND EXECUTABLE_SOURCES ${IPM_SOURCES})
endif()

if(SRC)
  list(APPEND EXECUTABLE_SOURCES ${SRC_SOURCES})
endif()

#-------------------------------------------------------------------------------
# Include Directories
#-------------------------------------------------------------------------------

# Include directories for third-party libraries
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/bucket
    ${CMAKE_CURRENT_SOURCE_DIR}/include/algebra
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuts
    ${CMAKE_CURRENT_SOURCE_DIR}/include/extra
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hgs_vrptw  # Add this line
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cvrpsep
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpmax
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/muSSP
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in  # Input template file
  ${CMAKE_CURRENT_BINARY_DIR}/config.h     # Output file
)

# Include the generated config.h file
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${ankerl_unordered_dense_SOURCE_DIR}/include)

set(GUROBI_HOME $ENV{GUROBI_HOME})
include_directories(${GUROBI_HOME}/include)

# -----------------------------------------------------------------------------
# CUDA
# -----------------------------------------------------------------------------

# # Find CUDA
# # find_package(CUDA REQUIRED)
# find_package(CUDAToolkit REQUIRED)

# # Set CUDA compilation flags
# set(CUDA_NVCC_FLAGS
#     -O3
#     -arch=sm_75  # Set the compute capability of your GPU
#     --default-stream per-thread
#     -Xcompiler -fPIC
# )

# # Add CUDA source files
# set(CUDA_SOURCES
#     include/algebra/EvaluatorGPU.cu
# )

# # Append CUDA sources to the list of executable sources
# list(APPEND EXECUTABLE_SOURCES ${CUDA_SOURCES})


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------
set(COMMON_LIBRARIES stdexec fmt xxhash libcuckoo)

set(GUROBI_VERSION_MAJOR 120) # Adjust based on Gurobi version
if(APPLE)
  # macOS detected, use .dylib for linking
  list(APPEND COMMON_LIBRARIES ${GUROBI_HOME}/lib/libgurobi_c++.a
       ${GUROBI_HOME}/lib/libgurobi100.dylib)
else()
  # Non-macOS (Linux or other platforms), use .so and .a files

  list(APPEND COMMON_LIBRARIES ${GUROBI_HOME}/lib/libgurobi_c++.a
       ${GUROBI_HOME}/lib/libgurobi${GUROBI_VERSION_MAJOR}.so tbb)
endif()


if(NSYNC)
  list(APPEND COMMON_LIBRARIES nsync)
  include_directories(${nsync_SOURCE_DIR}/public)
endif()

# -----------------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------------

if(BALDES)
  add_executable(baldes src/Baldes.cpp ${EXECUTABLE_SOURCES})
  target_compile_options(baldes PRIVATE -Wno-c++26-extensions -Wno-deprecated-declarations)
target_link_libraries(baldes PRIVATE ${COMMON_LIBRARIES})

    target_precompile_headers(baldes PRIVATE include/Common.h)
# target_link_libraries(baldes PRIVATE CUDA::cudart CUDA::cusparse)

  if(JEMALLOC)
    find_library(MIMALLOC_LIBRARY MIMALLOC)
    if(MIMALLOC_LIBRARY)
      target_link_libraries(baldes PRIVATE ${MIMALLOC_LIBRARY})
    endif()
  endif()

  if(IPM OR IPM_ACEL)
    target_include_directories(baldes PRIVATE ${Eigen_SOURCE_DIR} ${Eigen_SOURCE_DIR}/unsupported)
  endif()

if(GUROBI)
target_link_libraries(baldes PRIVATE gurobi_c++)
endif()

      # Set CUDA-specific properties
  set_target_properties(baldes PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES 75  # Set the compute capability of your GPU
  )
endif()

if(CLIQUER)
  add_executable(cliquer src/Original.cpp src/BucketGraph.cpp)
endif()

if(MTW)
  add_executable(vrpmtw mtw/VRPMTW.cpp ${EXECUTABLE_SOURCES} ${HGS_SOURCES} ${FPSOURCE})
  target_compile_options(vrpmtw PRIVATE -Wno-c++26-extensions -Wno-deprecated-declarations)
  target_link_libraries(vrpmtw PRIVATE stdexec fmt xxhash mdspan libcuckoo)
endif()

if(WITH_PYTHON)
  pybind11_add_module(pybaldes src/BucketBindings.cpp src/BCP.cpp src/BucketGraph.cpp src/MIPHandler.cpp src/Branching.cpp ${EXECUTABLE_SOURCES})
  target_link_libraries(pybaldes PRIVATE stdexec fmt xxhash mdspan libcuckoo pybind11::module)
  set_target_properties(pybaldes PROPERTIES POSITION_INDEPENDENT_CODE ON PREFIX "")
  target_compile_options(pybaldes PRIVATE -Wno-c++26-extensions -Wno-deprecated-declarations)

  if(IPM OR IPM_ACEL)
    target_include_directories(pybaldes PRIVATE ${Eigen_SOURCE_DIR} ${Eigen_SOURCE_DIR}/unsupported)
  endif()
endif()

if(HGS)
  add_executable(hgs src/HGS.cpp ${HGS_SOURCES} ${FPSOURCE})
  target_link_libraries(hgs PRIVATE stdexec fmt xxhash mdspan libcuckoo)
  target_compile_options(hgs PRIVATE -Wno-c++26-extensions -Wno-deprecated-declarations)
endif()
