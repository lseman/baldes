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
set(BUCKET_CAPACITY
    50
    CACHE STRING "Bucket max capacity")
set(TIME_INDEX
    0
    CACHE STRING "Time index")
set(DEMAND_INDEX
    1
    CACHE STRING "Demand index")
set(MAIN_RESOURCES
    1
    CACHE STRING "Main resources")

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

# Executable
if(RCC)
  add_executable(vrptw ${SOURCES} ${CVRPSEP_SOURCES})
else()
  add_executable(vrptw ${SOURCES})
endif()

# Linking libraries
target_link_libraries(vrptw PRIVATE STDEXEC::stdexec)
target_link_libraries(vrptw PRIVATE fmt::fmt)
if(GET_TBB)
  target_link_libraries(vrptw PRIVATE TBB::tbb)
endif()

set(GUROBI_VERSION_MAJOR 110) # Adjust based on Gurobi version

# Link Gurobi libraries
target_link_libraries(
  vrptw PRIVATE ${GUROBI_HOME}/lib/libgurobi_c++.a
                ${GUROBI_HOME}/lib/libgurobi${GUROBI_VERSION_MAJOR}.so tbb)

target_compile_definitions(vrptw PUBLIC R_SIZE=${R_SIZE})
target_compile_definitions(vrptw PUBLIC N_SIZE=${N_SIZE})
target_compile_definitions(vrptw PUBLIC MAX_SRC_CUTS=${MAX_SRC_CUTS})
target_compile_definitions(vrptw PUBLIC BUCKET_CAPACITY=${BUCKET_CAPACITY})
target_compile_definitions(vrptw PUBLIC TIME_INDEX=${TIME_INDEX})
target_compile_definitions(vrptw PUBLIC DEMAND_INDEX=${DEMAND_INDEX})
target_compile_definitions(vrptw PUBLIC MAIN_RESOURCES=${MAIN_RESOURCES})
target_compile_definitions(vrptw PUBLIC RESOURCES="${RESOURCES}")
target_compile_definitions(
  vrptw PUBLIC RESOURCES_DISPOSABLE="${RESOURCES_DISPOSABLE}")
