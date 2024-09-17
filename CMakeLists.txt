cmake_minimum_required(VERSION 3.10)

# Project name
project(BALDES)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Enable release build
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RIH "Enable RIH compilation option" OFF)
option(RCC "Enable RCC compilation option" OFF)
option(SRC3 "Enable 3SRC compilation option" OFF)
option(SRC "Enable SRC compilation option" OFF)
option(GET_TBB "Enable TBB compilation option" OFF)
option(UNREACHABLE_DOMINANCE "Enable Unreachable Dominance compilation option" OFF)
option(SORTED_LABELS "Enable Sorted Labels compilation option" OFF)
option(MCD "Enable MCD compilation option" OFF)

# Define an option to specify the size of resources
set(R_SIZE
    1
    CACHE STRING "Number of resources")
set(N_SIZE
    102
    CACHE STRING "Number of customers")
set(MAX_SRC_CUTS
    50
    CACHE STRING "Number of allowed SRC cuts")

# Ensure that both SRC3 and SRC are not selected at the same time
if(SRC AND SRC3)
  message(FATAL_ERROR "Both SRC and SRC3 cannot be enabled at the same time. Please choose only one.")
endif()

if(RIH)
  add_definitions(-DRIH)
endif()

if(RCC)
  add_definitions(-DRCC)
endif()

if(SRC)
  add_definitions(-DSRC)
endif()

if(SRC3)
  add_definitions(-DSRC3)
endif()

if(FIXED_BUCKETS)
  add_definitions(-DFIXED_BUCKETS)
endif()

if(UNREACHABLE_DOMINANCE)
  add_definitions(-DUNREACHABLE_DOMINANCE)
endif()

if(SORTED_LABELS)
  add_definitions(-DSORTED_LABELS)
endif()

if(MCD)
  add_definitions(-DMCD)
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
    GIT_TAG v0.40.2 # You can specify a specific version of CPM.cmake
  )
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

set(GUROBI_HOME $ENV{GUROBI_HOME})

include_directories(${GUROBI_HOME}/include)

# Source files
set(SOURCES examples/VRPTW.cpp src/BucketGraph.cpp src/SRC.cpp
            extra/Heuristic.cpp src/BucketJump.cpp src/RIH.cpp)

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

# Executable
add_executable(vrptw ${SOURCES} ${CVRPSEP_SOURCES})

# Linking libraries
target_link_libraries(vrptw PRIVATE STDEXEC::stdexec)
if(GET_TBB)
  target_link_libraries(vrptw PRIVATE TBB::tbb)
endif()

set(GUROBI_VERSION_MAJOR 110) # Adjust based on Gurobi version

# Link Gurobi libraries
target_link_libraries(
  vrptw PRIVATE ${GUROBI_HOME}/lib/libgurobi_c++.a
                ${GUROBI_HOME}/lib/libgurobi${GUROBI_VERSION_MAJOR}.so tbb)

# Set compile-time definition for R_SIZE
target_compile_definitions(vrptw PUBLIC R_SIZE=${R_SIZE})
target_compile_definitions(vrptw PUBLIC N_SIZE=${N_SIZE})
target_compile_definitions(vrptw PUBLIC MAX_SRC_CUTS=${MAX_SRC_CUTS})
